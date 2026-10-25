import torch
from tokenizers import Tokenizer

from theog.generate import extract_code, generate
from theog.pack import pack
from theog.train import load_model, lr_factor, train


def test_lr_schedule():
    assert lr_factor(0, 100, 5) == 0
    assert lr_factor(5, 100, 5) == 1.0
    assert abs(lr_factor(100, 100, 5) - 0.10) < 1e-9


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
    assert "<|/Think|>" not in raw  # the prompt already closed the think span
    assert isinstance(extract_code(raw), str)
