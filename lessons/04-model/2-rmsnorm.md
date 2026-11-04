# Step 2: RMSNorm

Each block adds its result to the token's vector (step 7). After 10 blocks the numbers could have
grown very large, or shrunk to almost nothing, and training breaks when they do.

**RMSNorm** rescales a vector so its root-mean-square is 1, then multiplies each position by a learned
weight. RMS means: square every number, take the mean, take the square root.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```

`rsqrt` is 1 / √. `eps` (1e-5) prevents a division by zero for an all-zero vector. `weight` starts at
1, so a new RMSNorm only rescales.

## Run it

```python
import torch
from theog.model import RMSNorm
norm = RMSNorm(4, eps=1e-5)
x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [100.0, 200.0, 300.0, 400.0]])
with torch.no_grad():
    y = norm(x)
print(y)
print(y.pow(2).mean(-1).sqrt())
```

```text
tensor([[0.3651, 0.7303, 1.0954, 1.4606],
        [0.3651, 0.7303, 1.0954, 1.4606]])
tensor([1.0000, 1.0000])
```

A vector and the same vector ×100 give the same output. The **direction** is kept and the **size** is
removed.

## Where it goes

theOG normalises *before* each layer, not after: `x + attn(norm(x))`. This is "pre-norm". The vector
carried from block to block is never normalised itself, only the copy each layer reads. That's what
lets deep transformers train stably. There are 2 per block, plus one at the very end.

Older transformers use LayerNorm, which also subtracts the mean. RMSNorm skips that step, is
simpler, and works as well. Qwen3 and Llama use it.

Next: [step 3, attention](3-attention.md).
