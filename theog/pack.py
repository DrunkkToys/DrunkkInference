"""Lesson 3: tokenize every row and pack the sequences into fixed windows.

  uv run python -m theog.pack --rows data/rows.jsonl --tokenizer data/tokenizer.json --out data/packed

Writes {train,val}_{ids,labels}.bin (int16, one window = seq_len values) and
meta.json. A window holds whole rows only: a row is never cut in two, and a
row longer than the window is dropped (a cut answer teaches code that never
ends).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from theog.format import sequence
from theog.tokens import IGNORE, PAD

SEQ_LEN = 2048
VAL_PCT = 5


def split_of(user: str) -> str:
    """'val' for about VAL_PCT % of prompts, 'train' for the rest, decided by
    a hash of the stripped prompt so the same task always lands on the same side.
    """
    # EXERCISE 3.2: lessons/03-format-and-packing/README.md
    # Hint: int(hashlib.sha256(user.strip().encode()).hexdigest()[:8], 16) % 100.
    raise NotImplementedError("exercise 3.2: write split_of")


class Packer:
    """Greedy fill: add rows to the window while they fit, then pad and write."""

    def __init__(self, out: Path, name: str, seq_len: int):
        self.seq_len = seq_len
        self.ids_f = open(out / f"{name}_ids.bin", "wb")
        self.lbl_f = open(out / f"{name}_labels.bin", "wb")
        self.ids, self.lbl = [], []
        self.windows = self.tokens = 0

    def add(self, ids: list[int], labels: list[int]) -> None:
        """Add one row to the current window. If it does not fit, flush the
    window first. Count the row's tokens in self.tokens.
        """
        # EXERCISE 3.3: lessons/03-format-and-packing/README.md
        # Hint: len(self.ids) + len(ids) > self.seq_len means it does not fit.
        raise NotImplementedError("exercise 3.3: write Packer.add")

    def flush(self) -> None:
        """Write the current window, padded to seq_len (PAD in ids, IGNORE in
    labels), as int16 to the two files. Do nothing if the window is empty.
        """
        # EXERCISE 3.4: lessons/03-format-and-packing/README.md
        # Hint: np.asarray(list, dtype=np.int16).tofile(self.ids_f). Then reset the
        #       lists and add 1 to self.windows.
        raise NotImplementedError("exercise 3.4: write Packer.flush")

    def close(self) -> None:
        self.flush()
        self.ids_f.close()
        self.lbl_f.close()


def pack(rows: Path, tokenizer: Path, out: Path, seq_len: int = SEQ_LEN) -> dict:
    tok = Tokenizer.from_file(str(tokenizer))
    if tok.get_vocab_size() > 32767:
        raise ValueError("int16 storage needs a vocabulary under 32768 tokens")
    out.mkdir(parents=True, exist_ok=True)
    packers = {s: Packer(out, s, seq_len) for s in ("train", "val")}
    n_rows = n_long = 0
    with open(rows, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            seq = sequence(tok, row["user"], row["answer"], row.get("think", ""))
            if len(seq["input_ids"]) > seq_len:
                n_long += 1
                continue
            packers[split_of(row["user"])].add(seq["input_ids"], seq["labels"])
            n_rows += 1
    for p in packers.values():
        p.close()
    meta = {"seq_len": seq_len, "rows": n_rows, "dropped_longer_than_seq_len": n_long,
            "tokenizer": str(tokenizer), "vocab_size": tok.get_vocab_size(),
            **{f"{s}_windows": p.windows for s, p in packers.items()},
            **{f"{s}_tokens": p.tokens for s, p in packers.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load(packed: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(ids, labels) as (windows, seq_len) int16 arrays, memory-mapped: the
    file stays on disk and only the windows a batch reads come into RAM."""
    meta = json.loads((packed / "meta.json").read_text())
    shape = (meta[f"{split}_windows"], meta["seq_len"])
    ids = np.memmap(packed / f"{split}_ids.bin", dtype=np.int16, mode="r", shape=shape)
    lbl = np.memmap(packed / f"{split}_labels.bin", dtype=np.int16, mode="r", shape=shape)
    return ids, lbl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    a = ap.parse_args()
    meta = pack(a.rows, a.tokenizer, a.out, a.seq_len)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
