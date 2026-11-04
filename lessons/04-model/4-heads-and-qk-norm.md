# Step 4: heads and QK-norm

Step 3 was one attention. A token may need several things at once: the matching `(`, the name of the
variable being returned, the indentation level. One set of weights per token can't point at all of
them.

## Heads

**Multi-head attention** runs several attentions side by side, each on its own slice. theOG has 4
heads of 128 numbers. Each head has its own queries, keys and values and its own weights. Their
outputs are put side by side again (4 × 128 = 512) and mixed by one more matrix, `o`.

```python
q, k, v = (p(x).view(B, T, self.heads, self.head_dim).transpose(1, 2) for p in (self.q, self.k, self.v))
q, k = self.q_norm(q), self.k_norm(k)
q, k = rotate(q, cos[:T], sin[:T]), rotate(k, cos[:T], sin[:T])
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
return self.o(out.transpose(1, 2).reshape(B, T, -1))
```

`view(B, T, heads, head_dim).transpose(1, 2)` cuts each 512-number vector into 4 pieces of 128 and
moves the head axis forward: shape (batch, heads, tokens, head_dim). Attention then runs for every
head at once.

## QK-norm

The second line applies an RMSNorm (step 2) to every query and every key, per head, before scoring.
Without it, training can make queries and keys keep growing. Then the scores grow, the softmax puts
all the weight on one token, and learning stalls. QK-norm caps that. Qwen3 uses it.

The third line is RoPE: [step 5](5-rope.md).

## Run it

```python
import torch
from theog.model import Attention, tiny, rope_tables
cfg = tiny()
attn = Attention(cfg)
cos, sin = rope_tables(cfg)
x = torch.randn(1, 6, cfg.hidden)
B, T, _ = x.shape
q = attn.q(x).view(B, T, cfg.heads, cfg.head_dim).transpose(1, 2)
print(q.shape)
print(attn(x, cos, sin).shape)
```

```text
torch.Size([1, 2, 6, 32])
torch.Size([1, 6, 64])
```

The tiny size has 2 heads of 32. Attention goes in and out at the same shape, (1, 6, 64), so it can be
added back to `x` (step 7).

theOG-50M, per block: q, k, v, o are 4 × 512 × 512 = **1,048,576 parameters**, plus 128 + 128 for
the two QK-norms. No biases.

Next: [step 5, RoPE](5-rope.md).
