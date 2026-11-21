"""Lesson 6, latent interpolation: the think span is K continuous vectors, not tokens.

Slot k of K holds the point k/(K+1) of the way from the prompt to the code:

    z_k = (1 - a_k) * mean(embed(prompt)) + a_k * mean(embed(code)),    a_k = k / (K + 1)

rescaled to the mean norm of a token embedding. Training feeds these points at the slot positions
(teacher forcing) and adds a loss that makes a small head (`model.latent`) predict z_k from the
hidden state one position earlier. At inference no code exists yet: the model writes the K
vectors itself, each from the hidden state of the previous one, and only then writes the code.
So the scored cost and the benchmarks use the model's own vectors, never the answer's.
In the spirit of Coconut (Hao et al. 2024), with a fixed target path instead of free thoughts.

lat8p ("own" mode, `model.latent.own = True`) closes the gap between training and inference. lat8
trains on the answer-built targets and the model learns to read the answer out of them; with its own
vectors it falls apart. lat8p never feeds a target: a first pass sees the prompt and K placeholder
slots and predicts all K vectors at once; a second pass reads those vectors and writes the code. The
token loss is on the second pass and its gradient flows back through the vectors; the targets only
enter the cosine loss. Inference runs the same two passes, so training and inference match exactly.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from theog.tokens import ANSWER, ANSWER_CLOSE, ASSISTANT, END, IGNORE, THINK, THINK_CLOSE, USER, USER_CLOSE

K = 8
SLOT = 102          # the "~" token: a placeholder id, its embedding is never used at a slot
LAMBDA = 1.0        # weight of the vector loss next to the token loss


def sequence(tok, user: str, answer: str) -> dict:
    prompt = [USER] + tok.encode(f"\n{user.strip()} /think\n").ids + [USER_CLOSE, ASSISTANT]
    think = [THINK] + [SLOT] * K + [THINK_CLOSE]
    body = [ANSWER] + tok.encode(f"\n```python\n{answer}\n```\n").ids + [ANSWER_CLOSE, END]
    ids = prompt + think + body
    P = len(prompt)  # the <|Think|> position
    labels = [IGNORE] * len(ids)
    for i in range(P, len(ids) - 1):
        labels[i] = ids[i + 1]
    for i in range(P, P + K):  # the positions before each slot: slots are vectors, no token loss
        labels[i] = IGNORE
    return {"input_ids": ids, "labels": labels}


class Head(nn.Module):
    """Predicts the next slot vector. `mu` is a running mean of the prompt and code means: the
    targets are centered on it, or every row's target would point the same way (the mean of many
    embeddings is mostly the global mean; measured: cosine 0.8 between different rows)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.register_buffer("mu", torch.zeros(hidden))
        self.register_buffer("seen", torch.zeros(()))
        self.own = False  # lat8p: set by train / cost / evaluate, not saved

    def forward(self, h):
        return self.proj(h)


def add_head(model) -> None:
    model.latent = Head(model.cfg.hidden).to(model.embed.weight.device)


def _hidden(model, x):
    """TheOG.forward without the embedding and the output head: inputs are vectors, not ids."""
    for block in model.blocks:
        x = block(x, model.cos, model.sin)
    return model.norm(x)


def _norm(model):
    return model.embed.weight.detach().norm(dim=-1).mean()


def targets(model, x: torch.Tensor):
    """(inputs with the slot vectors in place, slot mask, slot targets) for a packed batch."""
    E = model.embed(x)
    xs = x.cpu().numpy()
    Ed, n = E.detach(), _norm(model)
    mask = torch.zeros(x.shape, dtype=torch.bool)
    Z = torch.zeros_like(E)
    pairs = []
    for b in range(x.shape[0]):
        row = xs[b]
        users, ucl, ans, acl = (np.where(row == t)[0] for t in (USER, USER_CLOSE, ANSWER, ANSWER_CLOSE))
        for t in np.where(row == THINK)[0]:
            u, uc = users[users < t], ucl[ucl < t]
            a = ans[ans > t]
            if not len(u) or not len(uc) or not len(a) or t + K >= x.shape[1]:
                continue
            a = a[0]
            ac = acl[acl > a]
            ac = ac[0] if len(ac) else x.shape[1]
            pairs.append((b, t, Ed[b, u[-1] + 1:uc[-1]].mean(0), Ed[b, a + 1:ac].mean(0)))
    head = model.latent
    if pairs and model.training:  # running mean of the prompt and code means (0.99 decay)
        m = torch.stack([v for _, _, p, c in pairs for v in (p, c)]).mean(0)
        head.mu.mul_(0.99 if head.seen else 0.0).add_(m * (0.01 if head.seen else 1.0))
        head.seen.fill_(1)
    for b, t, p, c in pairs:
        for k in range(1, K + 1):
            al = k / (K + 1)
            z = (1 - al) * p + al * c - head.mu
            Z[b, t + k] = z / z.norm() * n
            mask[b, t + k] = True
    mask = mask.to(x.device)
    return torch.where(mask[..., None], Z, E), mask, Z


def loss(model, x, y):
    """Token cross-entropy + LAMBDA * (1 - cos) between predicted and target slot vectors."""
    if model.latent.own:
        return loss_own(model, x, y)
    inp, mask, Z = targets(model, x)
    h = _hidden(model, inp)
    logits = h @ model.embed.weight.T
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), ignore_index=IGNORE)
    prev = torch.zeros_like(mask)
    prev[:, :-1] = mask[:, 1:]                 # the position before each slot predicts it
    if mask.any():
        v = model.latent(h[prev])
        cos = 1 - F.cosine_similarity(v.float(), Z[mask].float(), dim=-1).mean()
    else:
        cos = h.sum() * 0
    return ce + LAMBDA * cos, ce, cos


def loss_own(model, x, y):
    """lat8p: pass 1 (prompt + placeholders) predicts the vectors, pass 2 reads them."""
    _, mask, Z = targets(model, x)
    E = model.embed(x)                          # the slots hold the placeholder's embedding
    prev = torch.zeros_like(mask)
    prev[:, :-1] = mask[:, 1:]
    if not mask.any():
        logits = _hidden(model, E) @ model.embed.weight.T
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), ignore_index=IGNORE)
        return ce, ce, ce * 0
    v = model.latent(_hidden(model, E)[prev])     # same row-major order as E[mask]
    vn = v / v.norm(dim=-1, keepdim=True) * _norm(model)
    inp = E.masked_scatter(mask[..., None], vn.to(E.dtype))
    logits = _hidden(model, inp) @ model.embed.weight.T
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), ignore_index=IGNORE)
    cos = 1 - F.cosine_similarity(v.float(), Z[mask].float(), dim=-1).mean()
    return ce + LAMBDA * cos, ce, cos


@torch.no_grad()
def think(model, ids: list[int]) -> torch.Tensor:
    """Embeddings of `ids` (ending with <|Think|>) followed by the K vectors the model writes.
    A prompt longer than the window keeps its last tokens, as generate() does."""
    dev = model.embed.weight.device
    ids = ids[-(model.cfg.max_seq_len - K):]
    if model.latent.own:  # lat8p: one pass over K placeholders, as in training
        E = model.embed(torch.tensor([ids + [SLOT] * K], device=dev))
        v = model.latent(_hidden(model, E)[0, len(ids) - 1:len(ids) - 1 + K])
        v = v / v.norm(dim=-1, keepdim=True) * _norm(model)
        return torch.cat([E[:, :len(ids)], v[None]], 1)
    E = model.embed(torch.tensor([ids], device=dev))
    n = _norm(model)
    for _ in range(K):
        v = model.latent(_hidden(model, E)[0, -1])
        E = torch.cat([E, (v / v.norm() * n)[None, None]], 1)
    return E


@torch.no_grad()
def nats(model, seq: dict) -> tuple[float, float]:
    """(span nats, answer nats) with the model's OWN slot vectors (nothing from the answer)."""
    ids, labels = seq["input_ids"], seq["labels"]
    t = ids.index(THINK)
    E = think(model, ids[:t + 1])
    rest = model.embed(torch.tensor([ids[t + 1 + K:]], device=E.device))
    lp = torch.log_softmax((_hidden(model, torch.cat([E, rest], 1)) @ model.embed.weight.T)[0].float(), -1)
    a = ids.index(ANSWER)
    span = answer = 0.0
    for i, y in enumerate(labels):
        if y == IGNORE:
            continue
        if i < a:
            span -= float(lp[i, y])
        else:
            answer -= float(lp[i, y])
    return span, answer


@torch.no_grad()
def generate(model, tok, task: str, max_new: int = 1200) -> str:
    """Greedy: the prompt, K vectors of the model's own, then <|/Think|><|Answer|> and the code."""
    from theog.format import prompt_ids
    from theog.tokens import STOP_IDS
    ids = prompt_ids(tok, task, True)
    E = think(model, ids)
    E = torch.cat([E, model.embed(torch.tensor([[THINK_CLOSE, ANSWER]], device=E.device))], 1)
    out = []
    for _ in range(max_new):
        h = _hidden(model, E[:, -model.cfg.max_seq_len:])
        t = int((h[0, -1] @ model.embed.weight.T).argmax())
        if t in STOP_IDS:
            break
        out.append(t)
        E = torch.cat([E, model.embed(torch.tensor([[t]], device=E.device))], 1)
    return "<|/Think|><|Answer|>" + tok.decode(out, skip_special_tokens=False)
