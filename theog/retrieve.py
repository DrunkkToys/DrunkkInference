"""Lesson 6, retrieval span: the code of the closest TRAINING task, found from the prompt alone.

Every span so far was built from the prompt or from the row's own code, so it carried nothing the
model didn't already have. This one brings in information from outside the weights: the solution
of the training task whose prompt shares the most rare words with this one (TF-IDF over words).
It is deterministic and computed without the row's answer, so at inference the harness writes it
into the think span for free, like psig. RETRO (Borgeaud et al. 2022) and kNN-LM (Khandelwal et
al. 2020) make small models match much larger ones this way.

Only train-split rows are indexed, and a row never retrieves itself or a row with the same prompt.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

WORD = re.compile(r"[a-z_][a-z0-9_]+")
MAX_DF = 0.05      # words in more than 5% of prompts carry no signal
MAX_QUERY = 24     # the rarest query words only
MAX_CHARS = 400    # retrieved code is cut to this many characters (whole lines)
CACHE = Path("data/retrieve.json")


def _key(user: str) -> str:
    return hashlib.sha256(user.strip().encode()).hexdigest()[:16]


def _words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def _cut(code: str) -> str:
    out, n = [], 0
    for line in code.split("\n"):
        if n + len(line) + 1 > MAX_CHARS:
            out.append("...")
            break
        out.append(line)
        n += len(line) + 1
    return "\n".join(out)


class Index:
    def __init__(self, rows_path: Path = Path("data/rows.clean.jsonl")):
        from theog.pack import split_of
        self.keys, self.code, post = [], [], defaultdict(list)
        with open(rows_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if split_of(r["user"]) != "train":
                    continue
                i = len(self.keys)
                self.keys.append(_key(r["user"]))
                self.code.append(r["answer"])
                for w in _words(r["user"]):
                    post[w].append(i)
        n = len(self.keys)
        self.idf = {w: math.log(n / len(p)) for w, p in post.items() if len(p) <= MAX_DF * n}
        self.post = {w: post[w] for w in self.idf}

    def nearest(self, user: str) -> str:
        me = _key(user)
        q = sorted((w for w in _words(user) if w in self.idf), key=lambda w: -self.idf[w])[:MAX_QUERY]
        score = defaultdict(float)
        for w in q:
            for i in self.post[w]:
                score[i] += self.idf[w]
        for i, _ in sorted(score.items(), key=lambda kv: -kv[1]):
            if self.keys[i] != me:
                return _cut(self.code[i])
        return ""


_TABLE: dict[str, str] | None = None
_INDEX: Index | None = None


def retr(row: dict) -> str:
    """The think span: 'similar task:' then the nearest training solution."""
    global _TABLE, _INDEX
    if _TABLE is None and CACHE.exists():
        _TABLE = json.loads(CACHE.read_text())
    k = _key(row["user"])
    if _TABLE is not None and k in _TABLE:
        code = _TABLE[k]
    else:  # a prompt outside the dataset (a benchmark task)
        _INDEX = _INDEX or Index()
        code = _INDEX.nearest(row["user"])
    return "similar task:\n" + code if code else "similar task: none"


def build(rows_path: Path = Path("data/rows.clean.jsonl")) -> None:
    """Precompute the nearest training solution of every row into data/retrieve.json."""
    idx, table = Index(rows_path), {}
    with open(rows_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            r = json.loads(line)
            table[_key(r["user"])] = idx.nearest(r["user"])
            if n % 10000 == 0:
                print(f"{n:,} rows", flush=True)
    CACHE.write_text(json.dumps(table))
    print(f"wrote {CACHE}: {len(table):,} prompts")


if __name__ == "__main__":
    build()
