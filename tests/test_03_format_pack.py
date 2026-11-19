import json

import numpy as np
from tokenizers import Tokenizer

from theog.format import sequence
from theog.pack import Packer, load, pack, split_of
from theog.tokens import ANSWER, ANSWER_CLOSE, ASSISTANT, END, IGNORE, PAD, THINK, THINK_CLOSE, USER


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


def test_think_text_is_learned(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    s = sequence(tok, "Add one.", "def f(x):\n    return x + 1", think_text="add 1")
    ids, lbl = s["input_ids"], s["labels"]
    t, tc = ids.index(THINK), ids.index(THINK_CLOSE)
    assert tc > t + 1                                   # the think text sits between the markers
    assert all(lbl[i] == ids[i + 1] for i in range(t, tc))


def test_packer_pads_and_masks_the_tail(tmp_path):
    p = Packer(tmp_path, "train", seq_len=8)
    p.add([1, 10, 11], [IGNORE, 11, 2])
    p.add([1, 12, 13], [IGNORE, 13, 2])
    p.add([1, 14, 15], [IGNORE, 15, 2])                  # does not fit: the first window is written
    p.close()
    ids = np.fromfile(tmp_path / "train_ids.bin", dtype=np.int16).reshape(-1, 8)
    lbl = np.fromfile(tmp_path / "train_labels.bin", dtype=np.int16).reshape(-1, 8)
    assert ids.tolist() == [[1, 10, 11, 1, 12, 13, PAD, PAD], [1, 14, 15, PAD, PAD, PAD, PAD, PAD]]
    assert lbl[0, 6:].tolist() == [IGNORE, IGNORE]


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
    sides = [split_of(f"task {i}") for i in range(2000)]
    assert 40 < sides.count("val") < 160                 # about 5%
