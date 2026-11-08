"""Lesson 4: one test per step."""
import math

import torch
import torch.nn.functional as F

from theog.model import (Attention, Block, RMSNorm, SwiGLU, TheOG, count_params, rope_tables, rotate,
                         theog_50m, tiny)


def test_step1_embedding_is_a_table_lookup():
    model = TheOG(tiny())
    x = model.embed(torch.tensor([[5, 17, 5]]))
    assert x.shape == (1, 3, 64)
    assert torch.equal(x[0, 0], model.embed.weight[5]) and torch.equal(x[0, 0], x[0, 2])


def test_step2_rmsnorm_gives_unit_rms_and_ignores_scale():
    norm = RMSNorm(4, eps=1e-5)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    with torch.no_grad():
        assert torch.allclose(norm(x).pow(2).mean(-1).sqrt(), torch.ones(1), atol=1e-4)
        assert torch.allclose(norm(x), norm(100 * x), atol=1e-4)


def test_step3_attention_by_hand_matches_sdpa():
    torch.manual_seed(0)
    T, d = 5, 8
    q, k, v = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)
    scores = (q @ k.T / math.sqrt(d)).masked_fill(~torch.ones(T, T).tril().bool(), float("-inf"))
    ref = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
    assert torch.allclose(scores.softmax(-1) @ v, ref, atol=1e-6)


def test_step4_attention_keeps_the_shape():
    cfg = tiny()
    cos, sin = rope_tables(cfg)
    assert Attention(cfg)(torch.randn(2, 7, cfg.hidden), cos, sin).shape == (2, 7, cfg.hidden)


def test_step5_rope_scores_depend_only_on_distance():
    cfg = tiny()
    cos, sin = rope_tables(cfg)
    torch.manual_seed(0)
    q, k = torch.randn(cfg.head_dim), torch.randn(cfg.head_dim)
    score = lambda i, j: rotate(q, cos[i], sin[i]) @ rotate(k, cos[j], sin[j])  # noqa: E731
    assert torch.allclose(score(3, 1), score(50, 48), atol=1e-4)
    assert not torch.allclose(score(3, 1), score(3, 0), atol=1e-2)
    assert torch.allclose(rotate(q, cos[7], sin[7]).norm(), q.norm())


def test_step6_swiglu_is_down_of_silu_gate_times_up():
    torch.manual_seed(0)
    mlp = SwiGLU(tiny())
    x = torch.randn(1, 6, 64)
    assert torch.allclose(mlp(x), mlp.down(F.silu(mlp.gate(x)) * mlp.up(x)))


def test_step7_block_adds_to_the_residual_stream():
    torch.manual_seed(0)
    cfg = tiny()
    block = Block(cfg)
    cos, sin = rope_tables(cfg)
    x = torch.randn(1, 6, cfg.hidden)
    with torch.no_grad():
        h = x + block.attn(block.attn_norm(x), cos, sin)
        assert torch.allclose(block(x, cos, sin), h + block.mlp(block.mlp_norm(h)))


def test_step8_theog_50m_has_theogs_parameter_count():
    with torch.device("meta"):
        model = TheOG(theog_50m())
    assert count_params(model) == 50_148_352  # theOG-50M v3, DrunkkSet/release/v3/config.json


def test_step8_output_head_is_the_embedding_table():
    model = TheOG(tiny())
    ids = torch.randint(0, 512, (1, 8))
    x = model.norm(model.blocks[1](model.blocks[0](model.embed(ids), model.cos, model.sin), model.cos, model.sin))
    assert torch.allclose(model(ids), x @ model.embed.weight.T)


def test_step8_causal_the_future_does_not_change_the_past():
    torch.manual_seed(0)
    model = TheOG(tiny()).eval()
    a = torch.randint(0, 512, (1, 20))
    b = a.clone()
    b[0, 10:] = torch.randint(0, 512, (10,))
    with torch.no_grad():
        assert torch.allclose(model(a)[0, :10], model(b)[0, :10], atol=1e-5)


def test_step8_initial_loss_is_near_uniform_guess():
    """A fresh model knows nothing: its loss is about ln(vocab)."""
    torch.manual_seed(0)
    model = TheOG(tiny())
    ids = torch.randint(0, 512, (4, 64))
    loss = F.cross_entropy(model(ids).reshape(-1, 512), ids.reshape(-1))
    assert abs(loss.item() - math.log(512)) < 0.5
