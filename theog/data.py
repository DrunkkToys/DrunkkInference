"""Lesson 2: build the training rows from public Hugging Face datasets.

  uv run python -m theog.data download --raw data/raw
  uv run python -m theog.data build --raw data/raw --out data/rows.jsonl

A row is {"source", "user", "answer", "entry_point", "tests"}: one task in
plain English and one Python solution. Every source is downloaded at a pinned
revision, so the same command gives the same rows a year from now.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterator

# (repo id, revision, files, licence). Both licences allow training.
SOURCES = {
    "opencodeinstruct": (
        "nvidia/OpenCodeInstruct", "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
        [f"data/train-{i:05d}-of-00050.parquet" for i in range(50)], "CC BY 4.0"),
    "opc_educational": (
        "OpenCoder-LLM/opc-sft-stage2", "7d28f40d579edd7c24402d17d0c7639f991e6f8d",
        ["educational_instruct/train-00000-of-00001.parquet"], "MIT"),
}

MAX_PER_SKELETON = 3  # rows kept per solution shape (same code, renamed)

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.DOTALL)
_WS = re.compile(r"\s+")


# ── download ──────────────────────────────────────────────────────────────

def download(raw: Path, parts: int = 0) -> None:
    """Every file of every source, or only the first `parts` files of each
    (the laptop track uses --parts 1)."""
    from huggingface_hub import hf_hub_download
    for name, (repo, rev, files, licence) in SOURCES.items():
        files = files[:parts or None]
        print(f"{repo} @ {rev[:8]} ({licence}): {len(files)} file(s)", flush=True)
        for f in files:
            hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=raw / name)


# ── reading the sources ───────────────────────────────────────────────────

def parse(code: str):
    """The AST of `code`, or None when it is not valid Python 3."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None


def _defines(tree) -> bool:
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in tree.body)


def _is_test_block(tree) -> bool:
    """Only asserts and test functions: a test block, not a solution."""
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("test"):
            return False
        if isinstance(n, ast.ClassDef) and not n.name.startswith("Test"):
            return False
    return any(isinstance(n, ast.Assert) for n in ast.walk(tree))


def single_solution(text: str) -> tuple[str | None, str]:
    """(solution, why-not). The solution is the one fenced Python block that
    defines a function or class. Zero or several such blocks: the row is
    ambiguous and dropped."""
    blocks = _FENCE.findall(text or "")
    if not blocks:
        return None, "no code block"
    if any(lang.lower() not in ("python", "py", "python3", "") for lang, _ in blocks):
        return None, "non-Python code block"
    sols = []
    for _, body in blocks:
        tree = parse(body)
        if tree is not None and _defines(tree) and not _is_test_block(tree):
            sols.append(body.strip())
    if len(sols) != 1:
        return None, f"{'no' if not sols else 'several'} def/class code blocks"
    return sols[0], ""


def _parquet(paths, columns):
    import pyarrow.parquet as pq
    for p in paths:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=2000, columns=columns):
            yield from batch.to_pylist()


def _as_list(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return []
    return list(v) if isinstance(v, (list, tuple)) else []


def read_opencodeinstruct(raw: Path, drops: Counter) -> Iterator[dict]:
    files = sorted((raw / "opencodeinstruct").rglob("*.parquet"))
    for r in _parquet(files, ["input", "output", "tests_execution_status", "unit_tests"]):
        states = _as_list(r["tests_execution_status"])
        if not states or any(s != "pass" for s in states):
            drops["a unit test of the source did not pass"] += 1
            continue
        sol, why = single_solution(r["output"])
        if sol is None:
            drops[why] += 1
            continue
        yield {"source": "opencodeinstruct", "user": r["input"], "answer": sol,
               "entry_point": "", "tests": "\n".join(map(str, _as_list(r["unit_tests"])))}


def read_opc_educational(raw: Path, drops: Counter) -> Iterator[dict]:
    files = sorted((raw / "opc_educational").rglob("*.parquet"))
    for r in _parquet(files, ["instruction", "code", "entry_point", "testcase"]):
        yield {"source": "opc_educational", "user": r["instruction"], "answer": (r["code"] or "").strip(),
               "entry_point": r["entry_point"] or "", "tests": "\n".join(_as_list(r["testcase"]))}


# ── cleaning one row ──────────────────────────────────────────────────────

_KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
         ast.Assign, ast.AnnAssign)


def _drop_docstring(body: list) -> list:
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:] or [ast.Pass()]
    return body


def solution_only(code: str) -> str | None:
    """The solution without driver code, docstrings or comments.

    Top level keeps imports, functions, classes and constants; prints, calls,
    loops and `if __name__ == "__main__":` go (they are usage examples, not the
    answer). ast.unparse writes the code back in one canonical style, which
    also drops every comment."""
    tree = parse(code)
    if tree is None:
        return None
    tree.body = [n for n in _drop_docstring(tree.body) if isinstance(n, _KEEP)]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.body = _drop_docstring(node.body)
    if not _defines(tree):
        return None
    return ast.unparse(tree)


def answer_in_prompt(user: str, answer: str, min_len: int = 20) -> bool:
    """True when a real line of the answer is already written in the prompt:
    the model would learn to copy, not to solve."""
    prompt = _WS.sub(" ", user)
    for line in answer.splitlines():
        s = _WS.sub(" ", line).strip()
        if len(s) >= min_len and not s.startswith(("def ", "class ", "import ", "from ", "@")) and s in prompt:
            return True
    return False


def function_name(answer: str, entry_point: str = "") -> str:
    """The name the tests call: the row's entry_point, else the first def or
    class at the top level of the answer."""
    if entry_point:
        return entry_point
    for node in ast.parse(answer).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
    return ""


def with_tests(user: str, answer: str, entry_point: str, tests: str, n: int = 3) -> str:
    """The prompt, plus up to n of the row's own assert lines when the prompt
    never says what the function is called. Same wording as MBPP's prompts.

    Without this the model has to invent a name, and a test that calls
    `is_palindrome` fails on a correct `check_palindrome`."""
    name = function_name(answer, entry_point)
    if not name or name in user:
        return user
    asserts = [t.strip() for t in tests.splitlines() if t.strip().startswith("assert") and name in t][:n]
    if not asserts:
        return user
    return user + "\nYour code should pass these tests:\n\n" + "\n".join(asserts)


def clean(row: dict, drops: Counter) -> dict | None:
    user = (row["user"] or "").strip()
    if not user:
        drops["prompt is empty"] += 1
        return None
    answer = solution_only(row["answer"])
    if answer is None:
        drops["answer is not Python 3 with a def/class"] += 1
        return None
    if answer_in_prompt(user, answer):
        drops["answer code is inside the prompt"] += 1
        return None
    return {**row, "user": user, "answer": answer}


# ── dedup ─────────────────────────────────────────────────────────────────

def sha(text: str) -> str:
    return hashlib.sha256(_WS.sub(" ", text).strip().encode()).hexdigest()


class _Rename(ast.NodeTransformer):
    """Every name becomes `_`: two solutions that differ only in names match."""

    def visit_Name(self, n):
        return ast.copy_location(ast.Name(id="_", ctx=n.ctx), n)

    def visit_arg(self, n):
        n.arg = "_"
        return n

    def visit_FunctionDef(self, n):
        n.name = "_"
        self.generic_visit(n)
        return n

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef


def skeleton(answer: str) -> str:
    return sha(ast.dump(_Rename().visit(ast.parse(answer))))


class Dedup:
    """Exact dedup on prompt and on answer (first copy wins), then at most
    MAX_PER_SKELETON rows per solution shape. Renamed clones of one solution
    are useful variety up to a point; past it they crowd out other tasks."""

    def __init__(self, max_per_skeleton: int = MAX_PER_SKELETON):
        self.prompts, self.answers, self.shapes = set(), set(), Counter()
        self.max = max_per_skeleton

    def keep(self, row: dict, drops: Counter) -> bool:
        p, a, s = sha(row["user"]), sha(row["answer"]), skeleton(row["answer"])
        if p in self.prompts:
            drops["duplicate prompt"] += 1
            return False
        if a in self.answers:
            drops["duplicate answer"] += 1
            return False
        if self.shapes[s] >= self.max:
            drops[f"more than {self.max} rows with the same solution shape"] += 1
            return False
        self.prompts.add(p)
        self.answers.add(a)
        self.shapes[s] += 1
        return True


def build(raw: Path, out: Path, limit: int = 0) -> Counter:
    drops, kept, dedup = Counter(), Counter(), Dedup()
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".part")  # renamed when complete: never a half file
    readers = (read_opencodeinstruct, read_opc_educational)  # tested sources first: they win a dup
    with open(part, "w", encoding="utf-8") as w:
        for reader in readers:
            for i, row in enumerate(reader(raw, drops)):
                if limit and i >= limit:
                    break
                row = clean(row, drops)
                if row is None or not dedup.keep(row, drops):
                    continue
                # after dedup, which must compare the tasks, not their test lists
                row["user"] = with_tests(row["user"], row["answer"], row["entry_point"], row["tests"])
                w.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept[row["source"]] += 1
    part.rename(out)
    for src, n in kept.items():
        print(f"kept {n:>9,}  {src}")
    for why, n in drops.most_common():
        print(f"drop {n:>9,}  {why}")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--raw", type=Path, required=True)
    d.add_argument("--parts", type=int, default=0, help="first N files per source (0 = all); the laptop track uses 1")
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--limit", type=int, default=0, help="rows read per source (0 = all), for a quick sample")
    a = ap.parse_args()
    if a.cmd == "download":
        download(a.raw, a.parts)
    else:
        build(a.raw, a.out, a.limit)


if __name__ == "__main__":
    main()
