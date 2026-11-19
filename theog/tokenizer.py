"""Lesson 1: train a byte-level BPE tokenizer.

  uv run python -m theog.tokenizer --rows data/rows.jsonl --out data/tokenizer.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from theog.tokens import SPECIAL_TOKENS

VOCAB_SIZE = 16_000


# ── BPE by hand (lesson 1, part 1) ────────────────────────────────────────
# Slow but readable. The real tokenizer below uses the Rust `tokenizers`
# library, which runs the same algorithm thousands of times faster.

def most_frequent_pair(ids: list[int]) -> tuple[int, int] | None:
    """The pair of neighbouring ids that occurs most often, or None when
    there are fewer than two ids. On a tie, the pair seen first wins.

    most_frequent_pair([1, 2, 3, 1, 2]) == (1, 2)
    """
    # EXERCISE 1.1: lessons/01-tokenizer/README.md
    # Hint: zip(ids, ids[1:]) gives every neighbouring pair. Count them in a dict;
    #       max(counts, key=counts.get) returns the first key with the top count.
    raise NotImplementedError("exercise 1.1: write most_frequent_pair")


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair`, left to right, with `new_id`.

    merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    merge([7, 7, 7], (7, 7), 50) == [50, 7]
    """
    # EXERCISE 1.2: lessons/01-tokenizer/README.md
    # Hint: Walk with an index i. If ids[i], ids[i + 1] is the pair, append new_id
    #       and jump 2; otherwise append ids[i] and step 1.
    raise NotImplementedError("exercise 1.2: write merge")


def bpe_by_hand(text: str, n_merges: int) -> tuple[list[int], dict[int, bytes]]:
    """Start from the UTF-8 bytes of `text` and do `n_merges` merges.
    Returns (final ids, vocab) where vocab[id] is the bytes that id stands for.
    Ids 0-255 are the single bytes; merge number k gets id 256 + k.
    """
    # EXERCISE 1.3: lessons/01-tokenizer/README.md
    # Hint: list(text.encode('utf-8')) gives the starting ids. For each new id:
    #       find the most frequent pair, merge it, and record
    #       vocab[new_id] = vocab[a] + vocab[b]. Stop early if there is no pair.
    raise NotImplementedError("exercise 1.3: write bpe_by_hand")


# ── the real tokenizer (lesson 1, part 2) ─────────────────────────────────

def build_tokenizer() -> Tokenizer:
    """An empty byte-level BPE tokenizer: GPT-2's split regex, every byte
    representable, and a decoder that turns byte-level tokens back into text.
    """
    # EXERCISE 1.4: lessons/01-tokenizer/README.md
    # Hint: Tokenizer(models.BPE()), then set .pre_tokenizer to
    #       pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    #       and .decoder to decoders.ByteLevel().
    raise NotImplementedError("exercise 1.4: write build_tokenizer")


def train(texts: Iterable[str], vocab_size: int = VOCAB_SIZE, min_frequency: int = 2) -> Tokenizer:
    """Train build_tokenizer() on `texts`. SPECIAL_TOKENS must get ids 0-8,
    in order, and all 256 bytes must be in the vocabulary from the start.
    """
    # EXERCISE 1.5: lessons/01-tokenizer/README.md
    # Hint: trainers.BpeTrainer(vocab_size=..., min_frequency=...,
    #       special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    #       show_progress=False), then tok.train_from_iterator(texts, trainer=trainer).
    raise NotImplementedError("exercise 1.5: write train")


def row_texts(path: Path) -> Iterator[str]:
    """The text the model will read: every prompt and every answer."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield row["user"]
            yield row["answer"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True, help="rows.jsonl from lesson 2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    a = ap.parse_args()
    tok = train(row_texts(a.rows), a.vocab_size)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(a.out))
    sample = "def add(a, b):\n    return a + b\n"
    ids = tok.encode(sample).ids
    print(f"{a.out}: {tok.get_vocab_size()} tokens")
    print(f"{sample!r} -> {len(ids)} tokens: {[tok.id_to_token(i) for i in ids]}")


if __name__ == "__main__":
    main()
