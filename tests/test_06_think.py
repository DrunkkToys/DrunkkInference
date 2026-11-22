"""Lesson 6: every think design is a fixed rule (row) -> text, and its cost is measured in nats."""
import io
import json
import re
import tokenize
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from theog.cost import nats
from theog.format import sequence
from theog.model import TheOG, tiny
from theog.pack import load, pack
from theog.think import DESIGNS, RETRIEVED, grounded, plan, shared, sig, skeleton, think_text
from theog.tokens import ANSWER, IGNORE, THINK, THINK_CLOSE

ANSWER_CODE = '''def count_char_freq(s):
    """Count each character."""
    freq = {}
    for char in s.lower():
        if char in freq:
            freq[char] += 1
        else:
            freq[char] = 1
    return freq'''
TESTS = 'assert count_char_freq("aA") == {"a": 2}\nassert count_char_freq("") == {}'
ROW = {"user": "Write a function count_char_freq that counts how often each character appears "
               "in the string s, ignoring case. Return a dict.",
       "answer": ANSWER_CODE, "entry_point": "count_char_freq", "tests": TESTS}


def _tok(code):
    return [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.string.strip()]


# --- exercise 6.1: grounded ---------------------------------------------------

def test_grounded_names_numbers_and_strings():
    prompt = "Return 'yes' if n is above 10, else 'no'. Use a loop."
    words = set(re.findall(r"\w+", prompt))
    found = {t.string: grounded(t, prompt, words) for t in _tok("if n > 10 and k > 1:\n    return 'yes'\nreturn ''")}
    assert found["n"] and found["10"] and found["'yes'"]      # in the prompt: copy it
    assert not found["k"]                                       # a name the model must invent
    assert not found["1"]                                       # "1" is inside "10", not a word of the prompt
    assert not found["if"] and not found["and"]                 # keywords are never "grounded"
    assert not found["''"]                                      # an empty string carries nothing
    assert not found[">"] and not found[":"]                    # operators neither


# --- the designs ----------------------------------------------------------------

def test_every_design_on_one_row():
    assert DESIGNS["none"](ROW) == ""
    assert sig(ROW) == "def count_char_freq(s):"
    assert shared(ROW) == "count_char_freq s"                   # "char" is not a word of the prompt
    assert skeleton(ROW) == "def count_char_freq s\n  for in s\n    if in\n    else\n  return"


def test_designs_are_deterministic_and_never_crash():
    broken = dict(ROW, answer="def f(:\n  return")
    for name, design in DESIGNS.items():
        if name in RETRIEVED and not Path("data/rows.clean.jsonl").exists():
            continue  # retrieval searches lesson 2's rows: nothing to search yet
        assert design(ROW) == design(dict(ROW))
        assert isinstance(design(broken), str), name


def test_think_text_has_name_test_and_plan():
    assert think_text(ANSWER_CODE, "count_char_freq", TESTS) == (
        "name: count_char_freq(s)\n"
        'test: count_char_freq("aA") == {"a": 2}\n'
        "plan:\n"
        "- freq = {}\n"
        "- for char in s.lower(): branch, updates freq\n"
        "- return freq")
    assert think_text(ANSWER_CODE, "other_name", TESTS) == ""   # no function to plan, no span


def test_plan_keeps_short_conditions_and_marks_recursion():
    code = '''def convert(num, base):
    digits = '0123456789'
    if num < base:
        return digits[num]
    return convert(num // base, base) + digits[num % base]'''
    assert plan(code, "convert") == [
        "- digits = '0123456789'",
        "- if num < base: return ...",
        "- return ... using recursion"]


def test_plan_lists_helpers_and_class_methods():
    code = ("def helper(x):\n    return x\n\nclass Stack:\n    def __init__(self):\n        self.items = []\n"
            "    def push(self, x):\n        self.items.append(x)")
    assert plan(code, "Stack") == [
        "- def __init__(self): ...",
        "- def push(self, x): .append()",
        "- also: def helper"]


def test_pack_writes_the_design_into_the_span(tokenizer_file, tmp_path):
    tok = Tokenizer.from_file(str(tokenizer_file))
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(dict(ROW, user=f"{ROW['user']} Task {i}.")) + "\n" for i in range(40)))
    meta = pack(rows, tokenizer_file, tmp_path / "p", seq_len=512, think="sig")
    assert meta["think"] == "sig"
    split = "train" if meta["train_windows"] else "val"
    ids = load(tmp_path / "p", split)[0][0].tolist()
    t = ids.index(THINK)
    assert tok.decode(ids[t + 1:ids.index(THINK_CLOSE, t)]).strip() == "def count_char_freq(s):"


# --- exercise 6.2: nats -----------------------------------------------------------

def test_nats_splits_the_loss_at_answer(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    torch.manual_seed(0)
    model = TheOG(tiny()).eval()
    s = sequence(tok, "Add one.", "def f(x):\n    return x + 1", think_text="add 1")
    ids, lbl = s["input_ids"], s["labels"]
    span, answer = nats(model, ids, lbl)
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([ids]))[0].float(), -1)
    every = [-float(lp[i, y]) for i, y in enumerate(lbl) if y != IGNORE]
    assert span + answer == pytest.approx(sum(every), rel=1e-5)  # nothing lost, nothing counted twice
    a = ids.index(ANSWER)
    assert answer == pytest.approx(sum(-float(lp[i, lbl[i]]) for i in range(a, len(ids)) if lbl[i] != IGNORE), rel=1e-5)
    assert span > 0 and answer > 0


def test_empty_span_still_costs_its_markers(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    torch.manual_seed(0)
    model = TheOG(tiny()).eval()
    s = sequence(tok, "Add one.", "def f(x):\n    return x + 1")
    span, _ = nats(model, s["input_ids"], s["labels"])
    assert span > 0                                             # <|Think|><|/Think|><|Answer|> are predicted too
