# Step 7: a block and the residual stream

A **block** puts steps 2 to 6 together:

```python
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden, cfg.eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.hidden, cfg.eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))
```

## The residual stream

Look at the `x = x + ...`. No layer replaces the token's vector. Each one reads a normalised copy and
**adds** its result back. The vector flowing from block to block is called the **residual stream**.
Every layer writes a small correction into it.

This matters for training. In step 3 of lesson 5, the gradient has to travel from the loss back to
the first layer. The `+` gives it a direct path through all 10 blocks. Without it, a 10-block model
is much harder to train.

The order is fixed: attention first, then the MLP reads the stream *after* attention has added to
it. So the MLP can use what attention just gathered.

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

Same shape, so blocks can be stacked. The second number is how big the block's change is next to `x`
itself: a correction, not a replacement.

theOG-50M, per block: 1,048,576 (attention) + 3,145,728 (MLP) + 1,280 (norms: 512 + 512 + 128 +
128) = **4,195,584 parameters**. 10 blocks: **41,955,840**.

Next: [step 8, the whole model](8-full-model.md).
