# Step 8: the whole model

```python
class TheOG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.hidden, cfg.eps)
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        return self.norm(x) @ self.embed.weight.T
```

Embedding (step 1), 10 blocks (step 7), a final RMSNorm (step 2), and then the output.

- `register_buffer(..., persistent=False)`: the RoPE tables move to the GPU with the model, but
  they are not saved in checkpoints, because they can always be recomputed.

## The output: tied embeddings

The last line compares each position's final vector with **every row of the embedding table**: one
dot product per token in the vocabulary, 16,000 scores. These are the **logits**. The highest one is
the model's best guess for the next token. Lesson 5's loss turns them into probabilities.

Using the input table again for the output is called **tied embeddings**. A token's row means the
same thing on the way in and on the way out, and it saves a second 8.2M-parameter table. That's
16% of theOG.

It is also why the table starts small (`std=0.02`, step 1). Large rows would give large logits from
the first step, so the untrained model would be confidently wrong instead of evenly unsure.

## Run it

```python
import torch
from theog.model import TheOG, tiny
torch.manual_seed(0)
model = TheOG(tiny())
ids = torch.tensor([[1, 2, 3, 4]])
with torch.no_grad():
    logits = model(ids)
print(logits.shape)
probs = logits[0, -1].softmax(-1)
print(round(float(probs.max()), 4), round(1 / 512, 4))
```

```text
torch.Size([1, 4, 512])
0.0032 0.002
```

One score per vocabulary token at each of the 4 positions. Untrained, the top token gets 0.0032,
close to 1/512: evenly unsure, as it should be. `test_04_model.py` checks the same through the loss,
which should start near ln(512) ≈ 6.24.

## The size

```python
@dataclass
class Config:
    vocab_size: int = 16_000
    hidden: int = 512
    layers: int = 10
    heads: int = 4
    head_dim: int = 128
    ffn: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0
    eps: float = 1e-5
```

| part | parameters |
|---|--:|
| embedding, shared with the output (step 1) | 8,192,000 |
| 10 blocks (step 7) | 41,955,840 |
| final norm | 512 |
| **total** | **50,148,352** |

```bash
uv run python -m theog.model
```

```text
theOG-50M: 50,148,352 parameters
```

The same total as theOG-50M v3's published count.

## Why this exact layout

Every choice in steps 1 to 8 is what Qwen3 uses: pre-norm RMSNorm, QK-norm then NEOX RoPE, SwiGLU,
tied embeddings, no biases. llama.cpp, Ollama, LM Studio and MLX already run `qwen3`, so lesson 8 can
export theOG with no custom code. theOG is not Qwen: it has no Qwen weights, only the same
arrangement of layers.

theOG-50M v3's research model has two extra pieces that `qwen3` has no place for, removed at export
(see "Export notes" on the [model card](https://huggingface.co/DrunkkToys/theOG-50M)). The course
model is the exported one: with the same weights loaded, it gives the same logits as theOG's own
model with those pieces off.

## Try this

- Remove `is_causal=True` in `Attention` and run `uv run pytest tests/test_04_model.py`. Read which
  test fails, and why.
- Change `Config` to 12 layers and hidden 640. How many parameters is that?

Next: [lesson 5, training](../05-training/README.md).
