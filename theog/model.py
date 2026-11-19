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
        """x / sqrt(mean(x**2 over the last dimension) + eps) * weight."""
        # EXERCISE 4.2: lessons/04-model/2-rmsnorm.md
        # Hint: torch.rsqrt is 1/sqrt. Use keepdim=True so the shapes broadcast.
        raise NotImplementedError("exercise 4.2: write RMSNorm.forward")


def causal_attention(q, k, v):
    """Attention written out by hand. q, k, v: (..., T, d).

    scores = q @ k^T / sqrt(d); positions in the future get -inf; softmax over
    the last dimension; then the weighted sum of v.
    """
    # EXERCISE 4.3: lessons/04-model/3-attention.md
    # Hint: k.transpose(-2, -1). torch.ones(T, T, dtype=torch.bool).triu(1) is
    #       True exactly at the future positions. masked_fill(mask, float('-inf')).
    raise NotImplementedError("exercise 4.3: write causal_attention")


def rope_tables(cfg: Config):
    """cos and sin tables, each (max_seq_len, head_dim).

    Pair i (i = 0 .. head_dim/2 - 1) turns at frequency
    1 / rope_theta ** (2i / head_dim). At position p its angle is p * frequency.
    Both halves of a row hold the same angles (NEOX layout).
    """
    # EXERCISE 4.4a: lessons/04-model/4-rope.md
    # Hint: torch.outer(positions, inv_freq) gives (T, d/2) angles. torch.cat([a, a], -1).
    raise NotImplementedError("exercise 4.4a: write rope_tables")


def rotate(x, cos, sin):
    """RoPE, NEOX style: dimension i is paired with i + d/2 and each pair is
    turned by its angle. x * cos + rotate_half(x) * sin, where
    rotate_half([a, b]) = [-b, a].
    """
    # EXERCISE 4.4b: lessons/04-model/4-rope.md
    # Hint: half = x.shape[-1] // 2; x[..., :half] and x[..., half:].
    raise NotImplementedError("exercise 4.4b: write rotate")


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
        """x: (B, T, hidden) -> (B, T, hidden).

    Project to q, k, v and split into heads: (B, heads, T, head_dim).
    QK-norm (self.q_norm, self.k_norm), then RoPE on q and k with cos[:T], sin[:T],
    then causal attention (F.scaled_dot_product_attention(..., is_causal=True)),
    then merge the heads back and apply self.o.
        """
        # EXERCISE 4.5: lessons/04-model/5-heads-and-qk-norm.md
        # Hint: p(x).view(B, T, self.heads, self.head_dim).transpose(1, 2) splits heads;
        #       out.transpose(1, 2).reshape(B, T, -1) merges them.
        raise NotImplementedError("exercise 4.5: write Attention.forward")


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.up = nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.hidden, bias=False)

    def forward(self, x):
        """down(silu(gate(x)) * up(x))."""
        # EXERCISE 4.6: lessons/04-model/6-swiglu.md
        # Hint: F.silu.
        raise NotImplementedError("exercise 4.6: write SwiGLU.forward")


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden, cfg.eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden, cfg.eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        """x + attention(norm(x)), then that + mlp(norm(that))."""
        # EXERCISE 4.7: lessons/04-model/7-block.md
        # Hint: Two lines. The MLP reads the x that already has the attention added.
        raise NotImplementedError("exercise 4.7: write Block.forward")


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
        """ids (B, T) -> logits (B, T, vocab_size).

    Embed, run every block (with self.cos, self.sin), final norm, then multiply
    by the embedding table transposed (tied output head).
        """
        # EXERCISE 4.8: lessons/04-model/8-full-model.md
        # Hint: self.embed.weight.T
        raise NotImplementedError("exercise 4.8: write TheOG.forward")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = TheOG(theog_50m())
    print(f"theOG-50M: {count_params(m):,} parameters")
    for name, p in list(m.named_parameters())[:12]:
        print(f"  {name:28s} {tuple(p.shape)}")
