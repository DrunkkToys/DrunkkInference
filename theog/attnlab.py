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


# ---- Looking inside a trained model (lesson 4, step 9) ----

FIT = "none,sig,shared,skeleton,plan,pause16"  # the rows lessons 6 and 4 score


@torch.no_grad()
def attention_maps(model, E) -> list[torch.Tensor]:
    """Attention weights of every layer, (heads, T, T) each, for the input embeddings E (1, T, hidden)."""
    x, out, T = E, [], E.shape[1]
    future = torch.ones(T, T, dtype=torch.bool, device=E.device).triu(1)
    for b in model.blocks:
        h, a = b.attn_norm(x), b.attn
        kv = getattr(a, "kv", a.heads)
        q = a.q(h).view(1, T, a.heads, a.head_dim).transpose(1, 2)
        k = a.k(h).view(1, T, kv, a.head_dim).transpose(1, 2)
        if getattr(a, "qknorm", True):
            q, k = a.q_norm(q), a.k_norm(k)
        q, k = rotate(q, model.cos[:T], model.sin[:T]), rotate(k, model.cos[:T], model.sin[:T])
        k = k.repeat_interleave(a.heads // kv, dim=1)
        s = (q @ k.transpose(-2, -1) / a.head_dim ** 0.5).masked_fill(future, float("-inf"))
        out.append(s.softmax(-1)[0].float().cpu())
        x = b(x, model.cos, model.sin)
    return out


def mass(model, tok, design: str, rows) -> torch.Tensor:
    """(layers, 4): where answer tokens look. Columns: position 0, the prompt, the think span
    (<|Think|> to <|/Think|>), the answer so far. Mean over answer tokens, heads and rows."""
    from theog import latent
    from theog.think import build
    from theog.tokens import ANSWER, THINK, THINK_CLOSE
    dev = model.embed.weight.device
    acc = torch.zeros(len(model.blocks), 4)
    for r in rows:
        ids = build(tok, design, r)["input_ids"]
        t, c, a = ids.index(THINK), ids.index(THINK_CLOSE), ids.index(ANSWER)
        if design in ("lat8", "lat8p"):
            E = latent.think(model, ids[:t + 1])
            E = torch.cat([E, model.embed(torch.tensor([ids[t + 1 + latent.K:]], device=dev))], 1)
        else:
            E = model.embed(torch.tensor([ids], device=dev))
        for l, p in enumerate(attention_maps(model, E)):
            p = p[:, a + 1:].mean(1).mean(0)
            acc[l] += torch.stack([p[0], p[1:t].sum(), p[t:c + 1].sum(), p[c + 1:].sum()])
    return acc / len(rows)


@torch.no_grad()
def head_ablation(model, tok, rows) -> tuple[float, list[list[float]]]:
    """Answer nats per row with every head on, and how much each head adds when it's switched off
    (its slice of `o`'s input zeroed). Big = the head carries work; about 0 = spare."""
    from theog.cost import nats
    from theog.think import build
    seqs = [build(tok, "none", r) for r in rows]
    cost = lambda: sum(sum(nats(model, s["input_ids"], s["labels"])) for s in seqs) / len(seqs)
    base, table = cost(), []
    for b in model.blocks:
        W, d, row = b.attn.o.weight, b.attn.head_dim, []
        for h in range(b.attn.heads):
            keep = W.data[:, h * d:(h + 1) * d].clone()
            W.data[:, h * d:(h + 1) * d] = 0
            row.append(cost() - base)
            W.data[:, h * d:(h + 1) * d] = keep
        table.append(row)
    return base, table


def main() -> None:
    import argparse
    from pathlib import Path
    from tokenizers import Tokenizer
    from theog.cost import rows_for
    from theog.train import load_model, pick_device
    ap = argparse.ArgumentParser(description="Look inside a trained model's attention (lesson 4, step 9).")
    ap.add_argument("what", choices=["mass", "heads"])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    ap.add_argument("--think", default="none", help="the design the model was trained with")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(a.tokenizer))
    model = load_model(a.ckpt, pick_device())
    if a.think == "lat8p":
        model.latent.own = True
    rows = rows_for(tok, Path("data/rows.clean.jsonl"), FIT.split(","), a.n, 512)
    if a.what == "mass":
        m = mass(model, tok, a.think, rows)
        print(f"{a.ckpt}: {len(rows)} rows | where answer tokens look")
        print("layer   first  prompt  span  answer")
        for l, r in enumerate(m):
            print(f"  {l}     {r[0]:.3f}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}")
        r = m.mean(0)
        print(f"  all   {r[0]:.3f}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}")
    else:
        base, table = head_ablation(model, tok, rows)
        print(f"{a.ckpt}: {len(rows)} rows | answer cost {base:.2f} nats per row; one head off adds:")
        for l, row in enumerate(table):
            print(f"  layer {l}: " + "  ".join(f"h{h} {x:+7.2f}" for h, x in enumerate(row)))


if __name__ == "__main__":
    main()
