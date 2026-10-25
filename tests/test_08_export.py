import numpy as np
import pytest
import torch

from theog.model import TheOG, tiny

gguf = pytest.importorskip("gguf")


def test_gguf_holds_every_weight(tokenizer_file, tmp_path):
    from theog.export import export_gguf
    torch.manual_seed(0)
    model = TheOG(tiny())
    out = tmp_path / "tiny.gguf"
    export_gguf(model, tokenizer_file, out, quant="F16")
    r = gguf.GGUFReader(str(out))
    names = {t.name: t for t in r.tensors}
    assert len(names) == len(model.state_dict())
    emb = names["token_embd.weight"].data.astype(np.float32)
    assert np.allclose(emb, model.embed.weight.detach().numpy(), atol=1e-3)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    assert arch == "qwen3"


def test_modelfile_points_at_the_gguf():
    from theog.export import modelfile
    mf = modelfile("./theog.gguf", "no_think")
    assert mf.startswith("FROM ./theog.gguf\n") and 'PARAMETER stop "<|/Answer|>"' in mf
