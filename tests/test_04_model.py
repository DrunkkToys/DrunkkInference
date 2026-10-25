import torch

from theog.model import TheOG, count_params, theog_50m, tiny


def test_theog_50m_has_theogs_parameter_count():
    with torch.device("meta"):
        model = TheOG(theog_50m())
    assert count_params(model) == 50_148_352  # theOG-50M v2, release/config.json


def test_output_head_is_the_embedding_table():
    model = TheOG(tiny())
    ids = torch.randint(0, 512, (1, 8))
    x = model.norm(model.blocks[1](model.blocks[0](model.embed(ids), model.cos, model.sin), model.cos, model.sin))
    assert torch.allclose(model(ids), x @ model.embed.weight.T)


def test_causal_the_future_does_not_change_the_past():
    torch.manual_seed(0)
    model = TheOG(tiny()).eval()
    a = torch.randint(0, 512, (1, 20))
    b = a.clone()
    b[0, 10:] = torch.randint(0, 512, (10,))
    with torch.no_grad():
        assert torch.allclose(model(a)[0, :10], model(b)[0, :10], atol=1e-5)


def test_initial_loss_is_near_uniform_guess():
    """A fresh model knows nothing: its loss is about ln(vocab)."""
    torch.manual_seed(0)
    model = TheOG(tiny())
    ids = torch.randint(0, 512, (4, 64))
    loss = torch.nn.functional.cross_entropy(model(ids).reshape(-1, 512), ids.reshape(-1))
    assert abs(loss.item() - torch.log(torch.tensor(512.0)).item()) < 0.5
