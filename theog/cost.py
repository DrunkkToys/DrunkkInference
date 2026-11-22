"""Lesson 7: what a think design costs the model, in nats per row.

  uv run python -m theog.cost --ckpt runs/think-sig/final.pt --tokenizer data/tokenizer.json --think sig

A span computed from the prompt and the code adds no information: it can only
split "prompt -> code" into steps the model finds easier. So the measure of a
design is the whole cost of writing span and answer,

    -log p(span | prompt) - log p(answer | prompt, span)

summed over the row's tokens and averaged over validation rows. Every design is
scored on the same rows: the first --n validation rows that fit --seq-len
tokens in every design named in --fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from theog.format import sequence
from theog.pack import split_of
from theog.think import DESIGNS, build, span_answer

LESSON_DESIGNS = "none,sig,shared,skeleton,plan"  # the rows lesson 7 scores
from theog.tokens import ANSWER, IGNORE
from theog.train import load_model, pick_device


@torch.no_grad()
def nats(model, ids: list[int], labels: list[int]) -> tuple[float, float]:
    """(span nats, answer nats) of one training sequence.

    Run the model on `ids`, take log_softmax of the logits, and add up
    -log p(label) at every position whose label isn't IGNORE. Positions before
    the one holding <|Answer|> predict the span (and its closing tokens); that
    position and every later one predict the answer.
    """
    device = next(model.parameters()).device
    lp = torch.log_softmax(model(torch.tensor([ids], device=device))[0].float(), -1)
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


def rows_for(tok: Tokenizer, rows: Path, fit: list[str], n: int, seq_len: int) -> list[dict]:
    out = []
    with open(rows, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if split_of(r["user"]) != "val":
                continue
            if all(len(sequence(tok, r["user"], *span_answer(d, r)[::-1])["input_ids"]) <= seq_len for d in fit):
                out.append(r)
                if len(out) == n:
                    break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--rows", type=Path, default=Path("data/rows.clean.jsonl"))
    ap.add_argument("--think", choices=DESIGNS, required=True, help="the design the model was trained with")
    ap.add_argument("--fit", default=LESSON_DESIGNS,
                    help="designs every scored row must fit in (keep the default to compare with lesson 7)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--swap", action="store_true",
                    help="lat8/lat8p: give each row the vectors written for the next row's prompt")
    ap.add_argument("--seq-len", type=int, default=512)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(a.tokenizer))
    dev = pick_device()
    model = load_model(a.ckpt, dev)
    if a.think == "lat8p":
        model.latent.own = True
    rows = rows_for(tok, a.rows, a.fit.split(","), a.n, a.seq_len)
    span = answer = 0.0
    for i, r in enumerate(rows):
        s = build(tok, a.think, r, a.seq_len)
        if a.think in ("lat8", "lat8p"):
            from theog import latent
            other = build(tok, a.think, rows[(i + 1) % len(rows)], a.seq_len) if a.swap else None
            sn, an = latent.nats(model, s, other)
        else:
            sn, an = nats(model, s["input_ids"], s["labels"])
        span += sn
        answer += an
    n = len(rows)
    print(f"{a.think}{' (swapped vectors)' if a.swap else ''}: {n} rows | span {span / n:.1f} + answer {answer / n:.1f} = {(span + answer) / n:.1f} nats per row")


if __name__ == "__main__":
    main()
