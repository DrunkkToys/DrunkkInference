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
    """The pair of neighbouring ids that occurs most often (first seen wins a
    tie), or None when there are fewer than two ids."""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return max(counts, key=counts.get) if counts else None


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair`, left to right, with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def bpe_by_hand(text: str, n_merges: int) -> tuple[list[int], dict[int, bytes]]:
    """Start from the UTF-8 bytes of `text` and do `n_merges` merges.
    Returns the final ids and what each id stands for."""
    ids = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, 256 + n_merges):
        pair = most_frequent_pair(ids)
        if pair is None:
            break
        ids = merge(ids, pair, new_id)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return ids, vocab


# ── the real tokenizer (lesson 1, part 2) ─────────────────────────────────

def build_tokenizer() -> Tokenizer:
    """An empty byte-level BPE: GPT-2's split regex, every byte representable."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    return tok


def train(texts: Iterable[str], vocab_size: int = VOCAB_SIZE, min_frequency: int = 2) -> Tokenizer:
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,                      # ids 0..8, in order
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


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
