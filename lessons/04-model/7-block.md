# Step 7: a block and the residual stream

A **block** puts steps 2 to 6 together. `__init__` made `self.attn_norm`, `self.attn`,
`self.mlp_norm` and `self.mlp`.

## The residual stream

No layer replaces the token's vector. Each one reads a normalised copy and **adds** its result
back:

```
x = x + attention(attn_norm(x))
x = x + mlp(mlp_norm(x))
```

The vector flowing from block to block is called the **residual stream**. Every layer writes a
small correction into it.

This matters for training. In lesson 5 the gradient has to travel from the loss back to the first
layer. The `+` gives it a direct path through every block. Without it, even a 6-block model is much
harder to train.

The order is fixed: attention first, then the MLP reads the stream *after* attention has added to
it, so the MLP can use what attention just gathered.

## Exercise 4.7: `Block.forward`

Two lines, `forward(x, cos, sin)`. Attention needs `cos` and `sin`; the MLP doesn't.

```bash
uv run pytest tests/test_04_model.py -k step7
```

<details><summary>Solution</summary>

```python
def forward(self, x, cos, sin):
    x = x + self.attn(self.attn_norm(x), cos, sin)
    return x + self.mlp(self.mlp_norm(x))
```

</details>

## Run it

```python
import torch
from theog.model import Block, tiny, rope_tables
torch.manual_seed(0)
cfg = tiny()
block = Block(cfg)
cos, sin = rope_tables(cfg)
x = torch.randn(1, 6, cfg.hidden)
with torch.no_grad():
    y = block(x, cos, sin)
print(y.shape)
print(round(float((y - x).norm() / x.norm()), 2))
```

```text
torch.Size([1, 6, 64])
0.26
```

Same shape, so blocks stack. The second number is the size of the block's change next to `x`
itself: a correction, not a replacement.

Per block on the laptop model: 262,144 (attention) + 786,432 (MLP) + 640 (norms: 256 + 256 + 64 +
64) = **1,049,216 parameters**. 6 blocks: **6,295,296**.

Next: [step 8, the whole model](8-full-model.md).
