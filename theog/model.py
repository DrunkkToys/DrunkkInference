"""Lesson 4: the model, a decoder-only transformer in the Qwen3 layout.

Every block:  x = x + Attention(RMSNorm(x))
              x = x + SwiGLU(RMSNorm(x))
Attention normalises q and k per head (QK-norm) and then rotates them (RoPE).
The output head reuses the embedding table (tied embeddings).

This layout is exactly what llama.cpp, Ollama and LM Studio run as `qwen3`,
so lesson 8 can export the weights without changing a number.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 16_000
    hidden: int = 512
    layers: int = 10
    heads: int = 4
    head_dim: int = 128
    ffn: int = 2048          # SwiGLU inner width, 4 x hidden
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0
    eps: float = 1e-5

    def to_dict(self) -> dict:
        return asdict(self)


def theog_50m() -> Config:
    """theOG-50M: 10 layers x 512, 4 heads x 128. 50,148,352 parameters."""
    return Config()


def laptop() -> Config:
    """The laptop track: about 8M parameters, trains in about 35 minutes on an
    Apple silicon laptop. Same layout as theOG-50M, only smaller."""
    return Config(vocab_size=8_000, hidden=256, layers=6, heads=4, head_dim=64, ffn=1024, max_seq_len=512)


def tiny() -> Config:
    """A toy size for tests and for reading the code on a laptop."""
    return Config(vocab_size=512, hidden=64, layers=2, heads=2, head_dim=32, ffn=256, max_seq_len=128)


class RMSNorm(nn.Module):
    """Scale every vector to unit root-mean-square, then by a learned weight."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def causal_attention(q, k, v):
    """Attention written out by hand (lesson 4, step 3). q, k, v: (..., T, d).

    The model itself calls F.scaled_dot_product_attention, which computes the
    same thing faster; the test checks that the two agree."""
    T, d = q.shape[-2], q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d ** 0.5                     # (..., T, T)
    future = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
    scores = scores.masked_fill(future, float("-inf"))              # no peeking ahead
    return scores.softmax(-1) @ v


def rope_tables(cfg: Config):
    """cos/sin for every position and every pair of dimensions."""
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
    freqs = torch.outer(torch.arange(cfg.max_seq_len, dtype=torch.float32), inv_freq)  # (T, d/2)
    return torch.cat([freqs, freqs], -1).cos(), torch.cat([freqs, freqs], -1).sin()   # (T, d)


def rotate(x, cos, sin):
    """RoPE, "NEOX" style: dimension i is paired with i + d/2 and the pair is
    turned by an angle that grows with the position."""
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.head_dim
        inner = cfg.heads * cfg.head_dim
        self.q = nn.Linear(cfg.hidden, inner, bias=False)
        self.k = nn.Linear(cfg.hidden, inner, bias=False)
        self.v = nn.Linear(cfg.hidden, inner, bias=False)
        self.o = nn.Linear(inner, cfg.hidden, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = (p(x).view(B, T, self.heads, self.head_dim).transpose(1, 2) for p in (self.q, self.k, self.v))
        q, k = self.q_norm(q), self.k_norm(k)               # QK-norm first...
        q, k = rotate(q, cos[:T], sin[:T]), rotate(k, cos[:T], sin[:T])  # ...then RoPE
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)    # no peeking ahead
        return self.o(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.up = nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden, cfg.eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden, cfg.eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))  # the MLP reads the post-attention x


class TheOG(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        nn.init.normal_(self.embed.weight, std=0.02)  # small: the same table makes the logits
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.hidden, cfg.eps)
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        return self.norm(x) @ self.embed.weight.T  # tied output head: (B, T, vocab)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = TheOG(theog_50m())
    print(f"theOG-50M: {count_params(m):,} parameters")
    for name, p in list(m.named_parameters())[:12]:
        print(f"  {name:28s} {tuple(p.shape)}")
