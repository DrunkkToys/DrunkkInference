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
    """The next token id. temperature 0: the most likely token (greedy).
    Otherwise: keep the top_k logits, divide by the temperature, softmax, draw one.
    """
    # EXERCISE 5.3: lessons/05-training/README.md
    # Hint: torch.topk returns (values, indices). torch.multinomial(probs, 1).
    raise NotImplementedError("exercise 5.3: write sample")


@torch.no_grad()
def generate(model, tok: Tokenizer, task: str, think: bool = False, max_new: int = 1200,
             temperature: float = 0.2, top_k: int = 40) -> str:
    """The text the model writes after the prompt, special tokens shown.

    Start from prompt_ids(tok, task, think). Repeat up to max_new times: run the
    model on the last model.cfg.max_seq_len ids, take the logits of the last
    position, sample a token, stop if it is in STOP_IDS, else append it.
    """
    # EXERCISE 5.5: lessons/05-training/README.md
    # Hint: model(torch.tensor([window], device=device))[0, -1].float().
    #       tok.decode(out, skip_special_tokens=False).
    raise NotImplementedError("exercise 5.5: write generate")


_CODE = re.compile(r"```(?:python)?\n(.*?)(?:```|$)", re.DOTALL)


def extract_code(text: str) -> str:
    """The code inside the answer's ```python block, stripped. A block cut
    off before its closing fence is kept as is. No block: the whole text.
    """
    # EXERCISE 7.1: lessons/07-evaluation/README.md
    # Hint: The compiled pattern _CODE above does the matching: _CODE.search(text).
    raise NotImplementedError("exercise 7.1: write extract_code")


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
