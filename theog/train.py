"""Lesson 5: train the model.

  uv run python -m theog.train --packed data/packed --out runs/theog-50m

One pass over the training windows (one epoch) with the recipe theOG-50M v3 used:
AdamW (lr 3e-4, betas 0.9/0.95, weight decay 0.01), 5% linear warmup, cosine
decay to 10% of the peak, gradient clipping at 1.0, batch 4 x 8 accumulation
steps x 2048 tokens. Every --eval-every steps it measures the validation loss,
keeps best.pt, and writes resume.pt; run the same command again to continue.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from theog import model as M
from theog.pack import load
from theog.tokens import IGNORE

PRESETS = {"theog_50m": M.theog_50m, "laptop": M.laptop, "tiny": M.tiny}


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_factor(update: int, total: int, warmup: int) -> float:
    """0 -> 1 linearly over `warmup` updates, then a cosine from 1 down to 0.1."""
    if update < warmup:
        return update / warmup
    progress = (update - warmup) / max(1, total - warmup)
    return 0.10 + 0.90 * 0.5 * (1 + math.cos(math.pi * progress))


def loss_fn(logits, labels):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=IGNORE)


def batch(ids, lbl, rows, device):
    x = torch.from_numpy(np.asarray(ids[rows], dtype=np.int64)).to(device)
    y = torch.from_numpy(np.asarray(lbl[rows], dtype=np.int64)).to(device)
    return x, y


@torch.no_grad()
def evaluate(model, ids, lbl, n_windows, batch_size, device) -> float:
    """Mean loss per supervised token over the first n_windows val windows."""
    model.eval()
    total, count = 0.0, 0
    for s in range(0, min(n_windows, len(ids)), batch_size):
        rows = list(range(s, min(s + batch_size, n_windows, len(ids))))
        x, y = batch(ids, lbl, rows, device)
        logits = model(x)
        n = int((y != IGNORE).sum())
        total += float(loss_fn(logits, y)) * n
        count += n
    model.train()
    return total / max(1, count)


def train(packed: Path, out: Path, preset: str = "theog_50m", batch_size: int = 4, accum: int = 8,
          lr: float = 3e-4, eval_every: int = 2000, eval_windows: int = 400, log_every: int = 50,
          seed: int = 42, max_steps: int = 0, epochs: float = 1.0, device: torch.device | None = None) -> dict:
    device = device or pick_device()
    cfg = PRESETS[preset]()
    meta = json.loads((packed / "meta.json").read_text())
    if meta["seq_len"] > cfg.max_seq_len or meta["vocab_size"] > cfg.vocab_size:
        raise SystemExit(f"data (seq_len {meta['seq_len']}, vocab {meta['vocab_size']}) does not fit "
                         f"preset {preset} (max_seq_len {cfg.max_seq_len}, vocab {cfg.vocab_size})")
    tr_ids, tr_lbl = load(packed, "train")
    va_ids, va_lbl = load(packed, "val")

    torch.manual_seed(seed)
    model = M.TheOG(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)

    steps = math.ceil(epochs * len(tr_ids) / batch_size)  # micro-batches
    if max_steps:
        steps = min(steps, max_steps)
    updates = max(1, steps // accum)                  # optimizer updates
    warmup = max(1, int(updates * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: lr_factor(u, updates, warmup))
    print(f"{preset}: {M.count_params(model):,} params on {device} | {len(tr_ids):,} train windows "
          f"({meta['train_tokens']:,} tokens) | {steps:,} steps = {updates:,} updates", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    start, best = 0, float("inf")
    if (out / "resume.pt").exists():
        ck = torch.load(out / "resume.pt", map_location=device, weights_only=False)
        if ck["config"] != cfg.to_dict():
            raise SystemExit(f"{out}/resume.pt was trained with another config; use another --out")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start, best = ck["step"] + 1, ck["best_val"]
        print(f"resumed at step {start:,} (best val {best:.4f})", flush=True)

    # Every window once per epoch, reshuffled each epoch. The order is a fixed
    # function of the seed, so a resumed run reads exactly the batches the
    # interrupted one would have.
    rng, order = random.Random(seed + 1), []
    while len(order) < steps * batch_size:
        epoch = list(range(len(tr_ids)))
        rng.shuffle(epoch)
        order += epoch

    def save(path, step, full=False):
        state = {"step": step, "model": model.state_dict(), "config": cfg.to_dict()}
        if full:
            state.update(optimizer=opt.state_dict(), scheduler=sched.state_dict(), best_val=best)
        torch.save(state, path)

    model.train()
    t0 = time.time()
    for step in range(start, steps):
        x, y = batch(tr_ids, tr_lbl, sorted(order[step * batch_size:(step + 1) * batch_size]), device)
        loss = loss_fn(model(x), y)
        (loss / accum).backward()  # gradients add up over `accum` micro-batches
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
        if step % log_every == 0:
            print(f"step {step:>7,}/{steps:,} | loss {loss.item():.4f} | lr {sched.get_last_lr()[0]:.2e} "
                  f"| {time.time() - t0:,.0f}s", flush=True)
        if step > 0 and step % eval_every == 0:
            val = evaluate(model, va_ids, va_lbl, eval_windows, batch_size, device)
            mark = ""
            if val < best:
                best, mark = val, " *best*"
                save(out / "best.pt", step)
            save(out / "resume.pt", step, full=True)
            print(f"eval step {step:,}: val loss {val:.4f} (best {best:.4f}){mark}", flush=True)
    save(out / "final.pt", steps)
    val = evaluate(model, va_ids, va_lbl, eval_windows, batch_size, device)
    if val < best:  # the last steps, at the smallest learning rate, are often the best
        save(out / "best.pt", steps)
    print(f"done: {steps:,} steps in {time.time() - t0:,.0f}s | final val loss {val:.4f}", flush=True)
    return {"steps": steps, "final_val": val, "best_val": min(best, val)}


def load_model(ckpt: Path, device=None) -> M.TheOG:
    state = torch.load(ckpt, map_location=device or "cpu", weights_only=False)
    model = M.TheOG(M.Config(**state["config"]))
    model.load_state_dict(state["model"])
    return model.to(device or "cpu").eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preset", choices=PRESETS, default="theog_50m")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-windows", type=int, default=400)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the training windows")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (0 = run every epoch)")
    a = ap.parse_args()
    train(a.packed, a.out, a.preset, a.batch_size, a.accum, a.lr, a.eval_every, a.eval_windows,
          a.log_every, a.seed, a.max_steps, a.epochs)


if __name__ == "__main__":
    main()
