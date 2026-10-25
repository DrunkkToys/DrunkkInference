"""Lessons 5-7: write code with a trained checkpoint.

  uv run python -m theog.generate --ckpt runs/theog-50m/best.pt --tokenizer data/tokenizer.json \
      "Write a function that returns the n-th Fibonacci number."
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from theog.format import prompt_ids
from theog.tokens import STOP_IDS


def sample(logits, temperature: float = 0.2, top_k: int = 40) -> int:
    """temperature 0 = always the most likely token (greedy). Otherwise draw
    from the top_k most likely tokens, sharpened by the temperature."""
    if temperature <= 0:
        return int(logits.argmax())
    vals, idx = torch.topk(logits / temperature, min(top_k, logits.numel()))
    return int(idx[torch.multinomial(torch.softmax(vals, -1), 1)])


@torch.no_grad()
def generate(model, tok: Tokenizer, task: str, think: bool = False, max_new: int = 1200,
             temperature: float = 0.2, top_k: int = 40) -> str:
    """The raw text the model writes after the prompt, special tokens shown."""
    device = next(model.parameters()).device
    ids = prompt_ids(tok, task, think)
    out = []
    for _ in range(max_new):
        window = (ids + out)[-model.cfg.max_seq_len:]
        logits = model(torch.tensor([window], device=device))[0, -1].float()
        t = sample(logits, temperature, top_k)
        if t in STOP_IDS:
            break
        out.append(t)
    return tok.decode(out, skip_special_tokens=False)


_CODE = re.compile(r"```(?:python)?\n(.*?)(?:```|$)", re.DOTALL)


def extract_code(text: str) -> str:
    """The code inside the answer's ```python block (or the whole text)."""
    m = _CODE.search(text)
    return (m.group(1) if m else text).strip()


def main() -> None:
    from theog.train import load_model, pick_device
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--think", action="store_true", help="open the think span instead of closing it")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=1200)
    a = ap.parse_args()
    model = load_model(a.ckpt, pick_device())
    tok = Tokenizer.from_file(str(a.tokenizer))
    print(generate(model, tok, a.task, a.think, a.max_new, a.temperature, a.top_k))


if __name__ == "__main__":
    main()
