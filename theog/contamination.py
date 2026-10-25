"""Lesson 2: remove training rows that contain a benchmark task.

  uv run python -m theog.contamination --rows data/rows.jsonl --out data/rows.clean.jsonl

A model that saw a test task during training scores it from memory. A row is
dropped when it holds at least half of the 13-word sequences (13-grams) of
any HumanEval or MBPP-500 task's prompt or reference solution.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

N = 13
THRESHOLD = 0.5
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


def grams(text: str, n: int = N) -> set[tuple]:
    words = _WORD.findall(text)
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


class Index:
    """Every 13-gram of every benchmark text -> the texts it belongs to."""

    def __init__(self, texts: dict[str, str]):
        self.size, self.where = {}, defaultdict(set)
        for key, text in texts.items():
            g = grams(text)
            if g:
                self.size[key] = len(g)
                for x in g:
                    self.where[x].add(key)

    def hits(self, text: str) -> list[str]:
        """Benchmark texts that `text` covers to THRESHOLD or more."""
        count = Counter()
        for x in grams(text):
            for key in self.where.get(x, ()):
                count[key] += 1
        return [k for k, c in count.items() if c / self.size[k] >= THRESHOLD]


def benchmark_texts() -> dict[str, str]:
    from theog.evaluate import load_tasks
    texts = {}
    for ds in ("humaneval", "mbpp500"):
        for t in load_tasks(ds):
            texts[f"{t['task_id']}:prompt"] = t["prompt"]
            texts[f"{t['task_id']}:solution"] = t["solution"]
    return texts


def clean(rows: Path, out: Path, index: Index) -> Counter:
    flagged = Counter()
    kept = 0
    with open(rows, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as w:
        for line in f:
            row = json.loads(line)
            hit = index.hits(row["user"] + "\n" + row["answer"])
            if hit:
                for h in hit:
                    flagged[h.split(":")[0]] += 1
                continue
            w.write(line)
            kept += 1
    print(f"kept {kept:,} rows; dropped rows matching {len(flagged)} benchmark tasks")
    for task, n in flagged.most_common(20):
        print(f"  {n:>6,}  {task}")
    return flagged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    clean(a.rows, a.out, Index(benchmark_texts()))


if __name__ == "__main__":
    main()
