import json

import numpy as np
from tokenizers import Tokenizer

from theog.format import prompt_ids, sequence
from theog.pack import load, pack, split_of
from theog.tokens import ANSWER, ANSWER_CLOSE, ASSISTANT, END, IGNORE, THINK, THINK_CLOSE, USER


def test_sequence_layout_and_labels(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    s = sequence(tok, "Add one.", "def f(x):\n    return x + 1")
    ids, lbl = s["input_ids"], s["labels"]
    assert ids[0] == USER and ids[-2:] == [ANSWER_CLOSE, END]
    a = ids.index(ASSISTANT)
    assert ids[a + 1:a + 4] == [THINK, THINK_CLOSE, ANSWER]  # empty think span
    assert all(x == IGNORE for x in lbl[:a + 1])               # the prompt is never learned
    assert all(lbl[i] == ids[i + 1] for i in range(a + 1, len(ids) - 1))  # the rest: next token
    assert lbl[-1] == IGNORE


def test_prompt_ids_match_training_prefix(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    s = sequence(tok, "Add one.", "def f(x):\n    return x + 1")
    p = prompt_ids(tok, "Add one.")
    assert s["input_ids"][:len(p)] == p


def test_pack_never_splits_a_row(rows_file, tokenizer_file, tmp_path):
    meta = pack(rows_file, tokenizer_file, tmp_path, seq_len=128)
    ids, _ = load(tmp_path, "train")
    assert meta["rows"] == 300 and ids.shape[1] == 128
    starts = int((np.asarray(ids) == USER).sum())
    ends = int((np.asarray(ids) == END).sum())
    assert starts == ends  # every row that starts in a window ends in it


def test_long_rows_are_dropped_not_cut(tokenizer_file, tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"user": "x", "answer": "def f():\n    return " + " + ".join(["1"] * 200)}) + "\n")
    meta = pack(rows, tokenizer_file, tmp_path / "p", seq_len=64)
    assert meta["dropped_longer_than_seq_len"] == 1 and meta["rows"] == 0


def test_split_is_by_prompt():
    assert split_of("same task") == split_of("same task ")
