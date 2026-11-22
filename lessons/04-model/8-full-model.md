# Step 8: the whole model

Put it together. `__init__` is written for you:

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
```

Embedding (step 1), the blocks (step 7; 6 on the laptop model, 10 on theOG), a final RMSNorm (step 2), and then the output.

- `register_buffer(..., persistent=False)`: the RoPE tables move to the GPU with the model, but
  they are not saved in checkpoints, because they can always be recomputed.

## The output: tied embeddings

The last line compares each position's final vector with **every row of the embedding table**: one
dot product per token in the vocabulary: 8,000 scores on the laptop model, 16,000 on theOG. These are the **logits**. The highest one is
the model's best guess for the next token. Lesson 5's loss turns them into probabilities.

Using the input table again for the output is called **tied embeddings**. A token's row means the
same thing on the way in and on the way out, and it saves a second 8.2M-parameter table. That's
16% of theOG.

## Exercise 4.8: `TheOG.forward`

`forward(ids)` takes a (B, T) tensor of token ids and returns (B, T, vocab_size) logits:

1. Look the ids up in `self.embed`.
2. Run every block in order, each with `self.cos` and `self.sin`.
3. Apply the final `self.norm`, then multiply by the transposed embedding table:
   `@ self.embed.weight.T`.

```bash
uv run pytest tests/test_04_model.py
```

All 15 should pass now. The last ones check the parameter counts and that an untrained model's
loss starts near ln(vocab).

<details><summary>Solution</summary>

```python
def forward(self, ids):
    x = self.embed(ids)
    for block in self.blocks:
        x = block(x, self.cos, self.sin)
    return self.norm(x) @ self.embed.weight.T
```

</details>

The small starting table (`std=0.02`, step 1) matters here too. Large rows would give large logits from
the first step, so the untrained model would be confidently wrong instead of evenly unsure.

## Run it

Before you run it, predict: an untrained model, 512 possible tokens. What probability should it
give its favourite next token? Write a number down.

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
  embed.weight                 (16000, 512)
  blocks.0.attn_norm.weight    (512,)
  ...
```

The first line is the one to check. Then every weight of block 0 with its shape: compare them with
steps 2 to 7. The same total as theOG-50M v3's published count.

The laptop preset (`laptop()` in the same file) is the same layout, smaller:

| | laptop | theOG |
|---|--:|--:|
| vocab | 8,000 | 16,000 |
| hidden | 256 | 512 |
| layers | 6 | 10 |
| heads × head_dim | 4 × 64 | 4 × 128 |
| ffn | 1,024 | 2,048 |
| window | 512 | 2,048 |
| embedding | 2,048,000 | 8,192,000 |
| blocks | 6,295,296 | 41,955,840 |
| **total** | **8,343,552** | **50,148,352** |

On the laptop model the embedding is a quarter of all parameters. Small models spend a big share
on their vocabulary.

## Why this exact layout

Every choice in steps 1 to 8 is what Qwen3 uses: pre-norm RMSNorm, QK-norm then NEOX RoPE, SwiGLU,
tied embeddings, no biases. llama.cpp, Ollama, LM Studio and MLX already run `qwen3`, so lesson 8 can
export theOG with no custom code. theOG is not Qwen: it has no Qwen weights, only the same
arrangement of layers.

theOG-50M v3's research model has two extra pieces that `qwen3` has no place for, removed at export
(see "Export notes" on the [model card](https://huggingface.co/DrunkkToys/theOG-50M)). The course
model is the exported one: with the same weights loaded, it gives the same logits as theOG's own
model with those pieces off.

## Going further

1. Remove `is_causal=True` in your `Attention.forward` and run the tests. Which one fails, and why?
2. Change `Config` to 12 layers and hidden 640 (ffn 2,560, head_dim 160). How many parameters?
3. Tied embeddings save a table. On which model does that save a bigger share, laptop or theOG?

<details><summary>Answers</summary>

1. `test_step8_causal_the_future_does_not_change_the_past` fails: without the mask, changing a later token changes the
   output at earlier positions. In training that's cheating: the model would read the token it is
   supposed to predict, the loss would drop fast, and generation would be garbage, since at
   generation time the future isn't there.
2. Embedding 16,000 × 640 = 10,240,000. Per block: 4 × 640² = 1,638,400 attention, 3 × 640 × 2,560
   = 4,915,200 MLP, plus 2 × 640 + 2 × 160 norms = 6,555,200. 12 blocks: 78,662,400. Plus the final
   norm: **88,903,040**. Check it with `sum(p.numel() for p in TheOG(cfg).parameters())`.
3. The laptop model: an untied output table would add 2,048,000 to 8.3M (25%), against 8,192,000
   to 50.1M (16%) for theOG.

</details>

Next: [lesson 5, training](../05-training/README.md). Steps 9 and 10 need a trained model: lesson 5
sends you back to step 9, lesson 7 to step 10.
