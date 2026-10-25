import json

import pytest

from theog.tokenizer import train as train_tokenizer


def toy_rows(n=300):
    """Small synthetic tasks: enough text to train a 512-token tokenizer."""
    ops = [("adds", "+"), ("subtracts", "-"), ("multiplies by", "*")]
    rows = []
    for i in range(n):
        word, op = ops[i % len(ops)]
        k = i % 97
        rows.append({"source": "toy", "user": f"Write a function f{i} that {word} {k} to its argument x.",
                     "answer": f"def f{i}(x):\n    return x {op} {k}", "entry_point": f"f{i}", "tests": ""})
    return rows


@pytest.fixture(scope="session")
def rows_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in toy_rows()))
    return path


@pytest.fixture(scope="session")
def tokenizer_file(rows_file, tmp_path_factory):
    from theog.tokenizer import row_texts
    tok = train_tokenizer(row_texts(rows_file), vocab_size=512)
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    tok.save(str(path))
    return path
