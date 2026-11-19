"""Where you are in the course: runs each lesson's tests and prints a table.

  uv run python -m theog.progress
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LESSONS = [
    ("1", "tokenizer", "tests/test_01_tokenizer.py"),
    ("2", "data", "tests/test_02_data.py"),
    ("3", "format + packing", "tests/test_03_format_pack.py"),
    ("4", "model", "tests/test_04_model.py"),
    ("5", "training + generation", "tests/test_05_train.py"),
    ("7", "evaluation", "tests/test_07_evaluate.py"),
    ("8", "export", "tests/test_08_export.py"),
]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    try:
        import pytest  # noqa: F401
    except ImportError:
        sys.exit("pytest is not installed. Run: uv sync --extra dev --extra export")
    print(f"{'lesson':<28} tests")
    for n, name, path in LESSONS:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", path],
                           cwd=root, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["?"])[-1]
        mark = "done" if r.returncode == 0 else "to do"
        print(f"{n}  {name:<25} {mark:<6} {last}")
    print("\nRun one lesson's tests with details: uv run pytest tests/test_04_model.py -x")


if __name__ == "__main__":
    main()
