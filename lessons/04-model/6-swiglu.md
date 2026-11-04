# Step 6: the SwiGLU MLP

Attention moves information *between* tokens. The **MLP** then works on each token on its own: it
looks at what the token now knows and transforms it. Most of the model's parameters are here.

```python
class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.up = nn.Linear(cfg.hidden, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

Read `forward` from the inside out:

1. `gate(x)` and `up(x)`: two separate projections from 512 to 2048 numbers. The wider space gives the
   layer room to work.
2. `F.silu(...)`: a curve that lets positive numbers through almost unchanged and squashes negatives
   to near 0. Without a curve like this, stacking layers would still only compute one big linear
   function, and the model couldn't learn anything a single matrix can't do.
3. `* up(x)`: the gate, multiplied position by position with the other projection, decides how much
   of each of the 2048 values passes. This gating is the "GLU" part. The gate uses SiLU, also called
   Swish, so the whole thing is "SwiGLU".
4. `down(...)`: back to 512, so it can be added back to `x`.

## Run it

```python
import torch
import torch.nn.functional as F
from theog.model import SwiGLU, tiny
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(F.silu(x).round(decimals=2))
mlp = SwiGLU(tiny())
print(mlp(torch.randn(1, 6, 64)).shape)
```

```text
tensor([-0.1400, -0.2700,  0.0000,  0.7300,  2.8600])
torch.Size([1, 6, 64])
```

Negatives are squashed near 0 and positives pass almost unchanged. Same shape in and out.

theOG-50M, per block: gate, up and down are 3 × 512 × 2048 = **3,145,728 parameters**, three times
the attention.

Next: [step 7, the block](7-block.md).
