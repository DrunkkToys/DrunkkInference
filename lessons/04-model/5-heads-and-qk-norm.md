# Step 5: heads and QK-norm

Step 3 was one attention. A token may need several things at once: the matching `(`, the name of
the variable being returned, the indentation level. One query per token can't point at all of them.

## Heads

**Multi-head attention** runs several attentions side by side, each on its own slice of the
vector. The laptop model has 4 heads of 64 numbers; theOG has 4 heads of 128. Each head has its own
queries, keys and values. Their outputs are put side by side again and mixed by one more matrix,
`o`.

The reshaping is the only tricky part. A projection gives (B, T, heads × head_dim). Cut the last
axis into (heads, head_dim) with `.view(B, T, heads, head_dim)`, then move the head axis forward
with `.transpose(1, 2)`: (B, heads, T, head_dim). Now attention runs for every head at once, since
your step 3 function didn't care about leading axes. Undo it at the end:
`.transpose(1, 2).reshape(B, T, -1)`.

## QK-norm

Before scoring, apply an RMSNorm (step 2) to every query and every key, per head. Without it,
training can make queries and keys keep growing. Then the scores grow, the softmax puts all its
weight on one token, and learning stalls. QK-norm caps that. Qwen3 uses it.

## Exercise 4.5: `Attention.forward`

`__init__` already made `self.q`, `self.k`, `self.v`, `self.o` (linear layers) and `self.q_norm`,
`self.k_norm`. In `forward(x, cos, sin)`, with `B, T, _ = x.shape`:

1. Project `x` to q, k, v and split each into heads: (B, heads, T, head_dim).
2. QK-norm: `q = self.q_norm(q)`, same for k.
3. RoPE on q and k, with the first T rows of the tables: `rotate(q, cos[:T], sin[:T])`.
   Values are not rotated: position decides *who* gets read, not *what*.
4. `F.scaled_dot_product_attention(q, k, v, is_causal=True)`: the fast version of your step 3.
5. Merge the heads back to (B, T, heads × head_dim) and apply `self.o`.

The order of 2 and 3 matters: normalise, then rotate. That's what `qwen3` does, and lesson 8's
export relies on it.

```bash
uv run pytest tests/test_04_model.py -k step5
```

<details><summary>Solution</summary>

```python
def forward(self, x, cos, sin):
    B, T, _ = x.shape
    q, k, v = (p(x).view(B, T, self.heads, self.head_dim).transpose(1, 2) for p in (self.q, self.k, self.v))
    q, k = self.q_norm(q), self.k_norm(k)
    q, k = rotate(q, cos[:T], sin[:T]), rotate(k, cos[:T], sin[:T])
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.o(out.transpose(1, 2).reshape(B, T, -1))
```

</details>

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

The tiny size has 2 heads of 32. Attention goes in and out at the same shape, (1, 6, 64), so it can
be added back to `x` (step 7).

Per block: q, k, v and o are 4 × 256 × 256 = **262,144 parameters** on the laptop model
(4 × 512 × 512 = 1,048,576 for theOG), plus the two QK-norms. No biases anywhere.

Next: [step 6, SwiGLU](6-swiglu.md).
