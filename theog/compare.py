"""Lesson 6: is model B really better than model A? Compare two results files task by task.

  uv run python -m theog.compare results/a_humaneval_*.jsonl results/b_humaneval_*.jsonl

Two models scored on the same tasks are not two independent samples: most tasks
pass for both or fail for both, and those say nothing about which is better. Only
the tasks that pass for exactly one of them count. If B is no better than A, each
of those is a coin flip, so the chance of a split at least this lopsided is an
exact binomial tail. That is McNemar's test, exact form.

  p < 0.05   the gap would rarely come from coin flips: worth believing
  p > 0.05   coin flips do this often: not a result yet, whatever the totals say
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def load(path: str | Path) -> dict[str, bool]:
    """task_id -> passed, from a results file written by theog.evaluate."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["task_id"]] = bool(r["passed"])
    return out


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided p-value for a split of b against c among n = b + c discordant tasks."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(a: dict[str, bool], b: dict[str, bool]) -> dict:
    """Count both / only A / only B / neither, and the p-value. Both files must cover the same tasks."""
    if a.keys() != b.keys():
        missing = sorted(a.keys() ^ b.keys())
        raise ValueError(f"the two files cover different tasks, e.g. {missing[:3]}")
    only_a = sorted(t for t in a if a[t] and not b[t])
    only_b = sorted(t for t in a if b[t] and not a[t])
    return {
        "tasks": len(a),
        "a": sum(a.values()), "b": sum(b.values()),
        "both": sum(a[t] and b[t] for t in a),
        "only_a": only_a, "only_b": only_b,
        "p": mcnemar_p(len(only_a), len(only_b)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="results file of model A")
    ap.add_argument("b", help="results file of model B")
    ap.add_argument("--show", action="store_true", help="list the tasks only one of them passes")
    args = ap.parse_args()
    r = compare(load(args.a), load(args.b))
    p = "p < 0.001" if r["p"] < 0.001 else f"p = {r['p']:.3f}"
    print(f"A {r['a']}/{r['tasks']}  B {r['b']}/{r['tasks']}  both {r['both']}  "
          f"only A {len(r['only_a'])}  only B {len(r['only_b'])}  {p}")
    if args.show:
        print("only A:", " ".join(r["only_a"]) or "-")
        print("only B:", " ".join(r["only_b"]) or "-")


if __name__ == "__main__":
    main()
