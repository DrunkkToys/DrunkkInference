"""Attention experiments (theog/attnlab.py): lesson 4's attention with one thing changed."""
from dataclasses import replace

import pytest
import torch

from theog import attnlab
from theog.model import Attention, TheOG, count_params, tiny
from theog.train import _overrides, load_model


def _x(cfg, T=16):
    torch.manual_seed(0)
    return torch.randn(2, T, cfg.hidden)


def test_full_kv_is_lesson_4_attention():
    cfg = replace(tiny(), heads=4, head_dim=16)
    ref, lab = Attention(cfg), attnlab.LabAttention(cfg, {})
    lab.load_state_dict(ref.state_dict())
    m = TheOG(cfg)
    x = _x(cfg)
    assert torch.allclose(ref(x, m.cos, m.sin), lab(x, m.cos, m.sin), atol=1e-6)


@pytest.mark.parametrize("spec", [{"kv": 2}, {"kv": 1}, {"gate": 1}, {"qknorm": 0}, {"kv": 2, "gate": 1}])
def test_variants_run_and_stay_causal(spec):
    cfg = replace(tiny(), heads=4, head_dim=16)
    m = TheOG(cfg)
    lab = attnlab.LabAttention(cfg, spec)
    x = _x(cfg)
    y = lab(x, m.cos, m.sin)
    assert y.shape == x.shape and torch.isfinite(y).all()
    x2 = x.clone(); x2[:, -1] += 1.0   # change the last token: earlier outputs must not move
    assert torch.allclose(lab(x2, m.cos, m.sin)[:, :-1], y[:, :-1], atol=1e-6)


def test_kv_saves_key_value_weights():
    cfg = replace(tiny(), heads=4, head_dim=16)
    full, gqa = TheOG(cfg), TheOG(cfg)
    attnlab.apply(gqa, {"kv": 2})
    saved = cfg.layers * 2 * cfg.hidden * (cfg.heads - 2) * cfg.head_dim
    assert count_params(full) - count_params(gqa) == saved
    with pytest.raises(ValueError):
        attnlab.LabAttention(cfg, {"kv": 3})


def test_checkpoint_keeps_the_spec(tmp_path):
    cfg = replace(tiny(), heads=4, head_dim=16)
    m = TheOG(cfg)
    attnlab.apply(m, {"kv": 2, "gate": 1})
    torch.save({"step": 0, "model": m.state_dict(), "config": cfg.to_dict(), "attn": m.attn_spec}, tmp_path / "c.pt")
    back = load_model(tmp_path / "c.pt")
    assert back.attn_spec == {"kv": 2, "gate": 1}
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    assert torch.allclose(back(ids), m.eval()(ids), atol=1e-6)


def test_overrides_are_typed():
    assert _overrides("heads=8,head_dim=32,rope_theta=1e6") == {"heads": 8, "head_dim": 32, "rope_theta": 1e6}
    assert attnlab.parse("kv=2,gate=1") == {"kv": 2, "gate": 1} and attnlab.parse("") == {}


@pytest.mark.parametrize("spec", [{}, {"kv": 2}, {"qknorm": 0}])
def test_attention_maps_are_causal_distributions(spec):
    cfg = replace(tiny(), heads=4, head_dim=16)
    torch.manual_seed(0)
    m = TheOG(cfg)
    if spec:
        attnlab.apply(m, spec)
    E = m.embed(torch.randint(0, cfg.vocab_size, (1, 12)))
    maps = attnlab.attention_maps(m, E)
    assert len(maps) == cfg.layers
    for p in maps:
        assert p.shape == (cfg.heads, 12, 12)
        assert torch.allclose(p.sum(-1), torch.ones(cfg.heads, 12), atol=1e-5)
        assert p.triu(1).abs().max() == 0  # nothing looks ahead


def test_gatemask_gates_only_the_set_layers():
    cfg = replace(tiny(), heads=4, head_dim=16, layers=3)
    m = TheOG(cfg)
    attnlab.apply(m, {"gate": 1, "gatemask": 1})
    assert [b.attn.gate is not None for b in m.blocks] == [True, False, False]
    attnlab.apply(m, {"gate": 1, "gatemask": 6})
    assert [b.attn.gate is not None for b in m.blocks] == [False, True, True]


def test_ensemble_is_the_mean_of_log_probabilities():
    from theog.evaluate import ensemble
    cfg = replace(tiny(), heads=4, head_dim=16)
    torch.manual_seed(0); a = TheOG(cfg).eval()
    torch.manual_seed(1); b = TheOG(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        both, self_ = ensemble(a, b)(ids), ensemble(a, a)(ids)
        la, lb = torch.log_softmax(a(ids), -1), torch.log_softmax(b(ids), -1)
    assert torch.allclose(both, (la + lb) / 2, atol=1e-5)
    assert torch.allclose(self_, la, atol=1e-5)  # a model with itself is the model
    assert ensemble(a, b).cfg is a.cfg
