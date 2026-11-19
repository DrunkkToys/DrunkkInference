"""Lesson 5: the learning-rate schedule, the loss, sampling, the prompt, and a real (tiny) training run."""
import math

import torch
from tokenizers import Tokenizer

from theog.format import prompt_ids, sequence
from theog.generate import generate, sample
from theog.pack import pack
from theog.tokens import ANSWER, ASSISTANT, IGNORE, THINK, THINK_CLOSE, USER, USER_CLOSE
from theog.train import load_model, loss_fn, lr_factor, train


def test_lr_schedule():
    assert lr_factor(0, 100, 5) == 0
    assert lr_factor(5, 100, 5) == 1.0
    assert abs(lr_factor(100, 100, 5) - 0.10) < 1e-9
    assert abs(lr_factor(2, 100, 5) - 0.4) < 1e-9                    # warmup is a straight line
    assert 0.1 < lr_factor(50, 100, 5) < 1.0                         # decaying in the middle


def test_loss_ignores_masked_positions():
    logits = torch.zeros(1, 3, 4)                                    # uniform guess over 4 tokens
    labels = torch.tensor([[2, IGNORE, IGNORE]])
    assert abs(loss_fn(logits, labels).item() - math.log(4)) < 1e-6
    logits[0, 1, 0] = 100.0                                          # a confident guess at an ignored spot...
    assert abs(loss_fn(logits, labels).item() - math.log(4)) < 1e-6  # ...changes nothing


def test_sample_greedy_and_top_k():
    logits = torch.tensor([0.0, 5.0, 1.0, 4.0])
    assert sample(logits, temperature=0) == 1
    torch.manual_seed(0)
    picks = {sample(logits, temperature=1.0, top_k=2) for _ in range(200)}
    assert picks == {1, 3}                                           # only the two best ever come out


def test_no_think_prompt_closes_the_span(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    p = prompt_ids(tok, "Add one.")
    assert p[0] == USER and p[-5:] == [USER_CLOSE, ASSISTANT, THINK, THINK_CLOSE, ANSWER]
    assert "/no_think" in tok.decode(p[1:-5])


def test_think_prompt_leaves_the_span_open(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    p = prompt_ids(tok, "Add one.", think=True)
    assert p[-3:] == [USER_CLOSE, ASSISTANT, THINK]
    assert "/think" in tok.decode(p[1:-3]) and "/no_think" not in tok.decode(p[1:-3])


def test_prompt_is_exactly_the_start_of_a_training_sequence(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    s = sequence(tok, "  Add one.  ", "def f(x):\n    return x + 1")
    p = prompt_ids(tok, "  Add one.  ")
    assert s["input_ids"][:len(p)] == p


def test_tiny_model_learns_and_resumes(rows_file, tokenizer_file, tmp_path):
    """A few hundred CPU steps on the toy tasks: the loss must fall well below
    the uniform guess (ln 512 = 6.2), and a second run resumes, not restarts."""
    packed = tmp_path / "packed"
    pack(rows_file, tokenizer_file, packed, seq_len=128)
    run = tmp_path / "run"
    cpu = torch.device("cpu")
    first = train(packed, run, preset="tiny", batch_size=8, accum=1, lr=3e-3, eval_every=100,
                  eval_windows=8, log_every=1000, max_steps=101, epochs=10, device=cpu)
    assert first["final_val"] < 2.5
    assert (run / "resume.pt").exists() and (run / "best.pt").exists()
    again = train(packed, run, preset="tiny", batch_size=8, accum=1, lr=3e-3, eval_every=100,
                  eval_windows=8, log_every=1000, max_steps=101, epochs=10, device=cpu)
    assert again["steps"] == 101

    model = load_model(run / "final.pt")
    tok = Tokenizer.from_file(str(tokenizer_file))
    raw = generate(model, tok, "Write a function f5 that subtracts 5 to its argument x.",
                   temperature=0, max_new=40)
    assert raw.startswith("\n```python\ndef f")                    # it learned the shape of an answer
