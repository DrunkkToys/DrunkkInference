"""Lesson 7, part 2: the interpolation designs (pause, c2f, psig, retr, lat8, lat8p)."""
from dataclasses import replace

import torch
from tokenizers import Tokenizer

from theog import latent
from theog.model import TheOG, tiny
from theog.think import DESIGNS, _def_sig, build, coarse_fine, fine_into_coarse, mask_prefill, psig, shrink
from theog.tokens import IGNORE, THINK, THINK_CLOSE

CODE = '''def count_pos(xs):
    n = 0
    for x in xs:
        if x > 0:
            n += 1
    return n'''
ROW = {"user": "Count the positive numbers in a list.\nYour code should pass these tests:\n\n"
               "assert count_pos([1, -2, 3]) == 2", "answer": CODE}


def test_coarse_fine_rebuilds_the_code():
    for depth in (0, 1):
        span, answer = coarse_fine(CODE, depth)
        assert "..." in span
        assert fine_into_coarse(span, answer) == CODE


def test_psig_reads_the_assert_or_the_def_line():
    assert psig(ROW) == "def count_pos(1 arg) -> int"
    he = 'from typing import List\n\n\ndef below_zero(operations: List[int]) -> bool:\n    """..."""\n'
    assert psig({"user": he}) == "def below_zero(1 arg) -> bool"
    assert _def_sig("no code here") is None
    assert psig({"user": "Write something."}) == "def ?"


def test_shrink_cuts_lines_until_it_fits():
    text = "similar task:\n" + "\n".join(f"line {i}" for i in range(20))
    out = shrink(text, lambda t: len(t) < 60)
    assert len(out) < 60 and out.endswith("...") and out.startswith("similar task:")


def test_prefill_has_no_loss(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    seq = build(tok, "psig", ROW)
    t = seq["input_ids"].index(THINK)
    close = seq["input_ids"].index(THINK_CLOSE)
    assert all(y == IGNORE for y in seq["labels"][t:close - 1])
    assert seq["labels"][close - 1] == THINK_CLOSE  # the model still learns to close the span
    assert not mask_prefill(dict(seq), tok, "text that is not in the span")


def test_latent_sequence_has_k_slots_and_no_slot_loss(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    seq = build(tok, "lat8", ROW)
    ids, labels = seq["input_ids"], seq["labels"]
    t = ids.index(THINK)
    assert ids[t + 1:t + 1 + latent.K] == [latent.SLOT] * latent.K
    assert all(labels[i] == IGNORE for i in range(t, t + latent.K))
    assert labels[t + latent.K] == THINK_CLOSE
    assert DESIGNS["lat8"](ROW) == DESIGNS["lat8p"](ROW) == ""


def test_lat8p_trains_on_the_vectors_inference_writes(tokenizer_file):
    """Own mode: the vectors pass 1 feeds pass 2 in training are the ones inference writes, and
    they don't depend on the code (no answer leaks into them)."""
    torch.manual_seed(0)
    tok = Tokenizer.from_file(str(tokenizer_file))
    model = TheOG(replace(tiny(), max_seq_len=512)).eval()
    latent.add_head(model)
    model.latent.own = True
    a, b = build(tok, "lat8p", ROW), build(tok, "lat8p", {**ROW, "answer": "def f():\n    return 1"})
    t = a["input_ids"].index(THINK)
    with torch.no_grad():
        h = latent._hidden(model, model.embed(torch.tensor([a["input_ids"]])))[0, t:t + latent.K]
        v = model.latent(h)
        v = v / v.norm(dim=-1, keepdim=True) * latent._norm(model)
        own = latent.think(model, a["input_ids"][:t + 1])
    assert own.shape[1] == t + 1 + latent.K
    assert torch.allclose(own[0, t + 1:], v, atol=1e-5)
    assert torch.allclose(own, latent.think(model, b["input_ids"][:t + 1]))
    model.train()
    total, ce, cos = latent.loss(model, torch.tensor([a["input_ids"]]), torch.tensor([a["labels"]]))
    assert torch.isfinite(total) and 0 <= float(cos.detach()) <= 2


def test_psr_is_the_signature_then_the_neighbor(tokenizer_file, monkeypatch):
    from theog import think
    monkeypatch.setattr(think, "retr", lambda row: "similar task:\ndef g(x):\n    return x")
    monkeypatch.setitem(think.PREFILL, "psr", lambda row: think.psig(row) + "\n" + think.retr(row))
    assert think.PREFILL["psr"](ROW) == "def count_pos(1 arg) -> int\nsimilar task:\ndef g(x):\n    return x"
