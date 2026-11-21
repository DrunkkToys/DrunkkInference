"""Attention experiments: the same model with one thing in its attention changed.

`apply(model, spec)` swaps every block's attention for `LabAttention`, before training, the way
`latent.add_head` adds the latent head. Lesson 4's `Attention` stays as you wrote it.

spec keys (anything left out is lesson 4's attention):
  kv=N       N key/value heads shared by all query heads (GQA; kv=1 is MQA). llama.cpp's qwen3 runs it.
  gate=1     a sigmoid gate per head on the attention output, read from the block input
             (Qiu et al., 2025, "Gated Attention for Large Language Models"). Not in qwen3.
  qknorm=0   no QK-norm. Not in qwen3.

Head count, head size and the RoPE base are plain config fields: `theog.train --set heads=8,head_dim=32`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from theog.model import Config, RMSNorm, rotate

QWEN3 = {"kv"}  # spec keys lesson 8's export can write


def parse(text: str) -> dict:
    """'kv=2,gate=1' -> {'kv': 2, 'gate': 1}."""
    return {k: int(v) for k, v in (p.split("=") for p in text.split(",") if p)} if text else {}


class LabAttention(nn.Module):
    def __init__(self, cfg: Config, spec: dict):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.head_dim
        self.kv = spec.get("kv", cfg.heads)
        if cfg.heads % self.kv:
            raise ValueError(f"kv={self.kv} must divide heads={cfg.heads}")
        self.q = nn.Linear(cfg.hidden, cfg.heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.hidden, self.kv * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.hidden, self.kv * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.heads * cfg.head_dim, cfg.hidden, bias=False)
        self.qknorm = bool(spec.get("qknorm", 1))
        if self.qknorm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.eps)
        self.gate = nn.Linear(cfg.hidden, cfg.heads, bias=True) if spec.get("gate") else None

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k, v = (p(x).view(B, T, self.kv, self.head_dim).transpose(1, 2) for p in (self.k, self.v))
        if self.qknorm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = rotate(q, cos[:T], sin[:T]), rotate(k, cos[:T], sin[:T])
        if self.kv != self.heads:  # every group of heads reads the same key/value head
            k, v = (t.repeat_interleave(self.heads // self.kv, dim=1) for t in (k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.gate is not None:  # (B, heads, T, 1): how much of each head gets through
            out = out * torch.sigmoid(self.gate(x)).transpose(1, 2)[..., None]
        return self.o(out.transpose(1, 2).reshape(B, T, -1))


def apply(model, spec: dict) -> None:
    for b in model.blocks:
        b.attn = LabAttention(model.cfg, spec).to(model.embed.weight.device)
    model.attn_spec = dict(spec)
