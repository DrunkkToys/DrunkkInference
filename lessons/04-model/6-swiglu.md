# Step 6: the SwiGLU MLP

Attention moves information *between* tokens. The **MLP** then works on each token on its own: it
looks at what the token now knows and transforms it. Most of the model's parameters are here.

`__init__` made three linear layers: `gate` and `up` (hidden → ffn, 4× wider) and `down`
(ffn → hidden). The forward pass, from the inside out:

1. `gate(x)` and `up(x)`: two separate projections into the wider space, which gives the layer room
   to work.
2. `F.silu(gate(x))`: a curve that lets positive numbers through almost unchanged and squashes
   negatives towards 0. Without some curve like this, stacking layers would still only compute one
   big linear function, and the model couldn't learn anything a single matrix can't.
3. `silu(gate(x)) * up(x)`: multiplied position by position, the gated half decides how much of
   each of `up`'s values passes. This gating is the "GLU" part. SiLU is also called Swish, hence
   "SwiGLU".
4. `down(...)`: back to the hidden size, so it can be added back to `x`.

## Exercise 4.6: `SwiGLU.forward`

One line.

```bash
uv run pytest tests/test_04_model.py -k step6
```

<details><summary>Solution</summary>

```python
def forward(self, x):
    return self.down(F.silu(self.gate(x)) * self.up(x))
```

</details>

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

Negatives squashed near 0, positives almost unchanged. Same shape in and out.

Per block: 3 × 256 × 1024 = **786,432 parameters** on the laptop model, three times the attention
(3 × 512 × 2048 = 3,145,728 for theOG).

Next: [step 7, the block](7-block.md).
