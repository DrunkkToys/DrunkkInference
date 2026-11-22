# Step 3: attention

After step 1, each token's vector only knows the token itself. To write `return a + b`, the model
must know that `a` and `b` are the arguments a few tokens back. **Attention** is how a token reads
the tokens before it.

## Query, key, value

Each token makes three vectors from its own, with three learned matrices:

- **query**: what this token is looking for
- **key**: what this token offers to others
- **value**: what this token passes on if it is chosen

Then, for every token *i*:

1. **Score** every token *j*: `query_i · key_j` (a dot product: multiply position by position, add
   up). A high score means *j* has what *i* is looking for. Divide the scores by √d (d = vector
   size), so they don't grow just because vectors got longer.
2. **Mask** the future: token *i* may only look at tokens 0 to *i*. During training the whole
   sequence is there at once, and without the mask, token *i* could read token *i* + 1, the very
   token it's supposed to predict. The model would learn to cheat, perfectly, and be useless.
3. **Softmax** each row of scores into weights that are positive and add up to 1. A score of −∞
   becomes a weight of exactly 0.
4. **Mix**: token *i* gets the weighted sum of the values.

For 4 tokens with random queries and keys (the same `q` and `k` as in "Run it" below), the weights
look like this. Row *i* is how much token *i* reads from each token:

```text
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.4900, 0.5100, 0.0000, 0.0000],
        [0.6200, 0.0300, 0.3500, 0.0000],
        [0.2000, 0.6200, 0.1000, 0.0700]])
```

Everything above the diagonal is 0: that's the mask. Each row adds up to 1. The first token can
only read itself.

## Exercise 4.3: `causal_attention`

In `theog/model.py`. `q`, `k`, `v` have shape (..., T, d): there may be extra leading axes (batch,
heads), and your code shouldn't care. Return shape (..., T, d).

You already have the recipe: the four steps above, score, mask, softmax, mix. Try to write them
from that alone. What shape is the score table? Which cells of it are "the future"?

<details><summary>Hint: the PyTorch calls</summary>

- `k.transpose(-2, -1)` swaps the last two axes, so `q @ k.transpose(-2, -1)` is every query
  against every key: shape (..., T, T).
- `torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)` is True exactly above the diagonal:
  the future.
- `scores.masked_fill(mask, float("-inf"))`, then `.softmax(-1)`, then `@ v`.

</details>

```bash
uv run pytest tests/test_04_model.py -k step3
```

The test compares your function with PyTorch's `F.scaled_dot_product_attention(q, k, v,
is_causal=True)`, which computes the same thing.

<details><summary>Solution</summary>

```python
def causal_attention(q, k, v):
    T, d = q.shape[-2], q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d ** 0.5
    future = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
    scores = scores.masked_fill(future, float("-inf"))
    return scores.softmax(-1) @ v
```

</details>

## Run it

```python
import torch
import torch.nn.functional as F
from theog.model import causal_attention
torch.manual_seed(0)
q, k, v = torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)
out = causal_attention(q, k, v)
ref = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
print(out.shape, torch.allclose(out, ref, atol=1e-6))
print(torch.allclose(out[0], v[0]))
```

```text
torch.Size([4, 8]) True
True
```

The second line: token 0 can only read itself, so its output is exactly its own value.

To see the weight table from the top of this page, add one line:

```python
print((q @ k.T / 8 ** 0.5).masked_fill(torch.ones(4, 4, dtype=torch.bool).triu(1), float("-inf")).softmax(-1).round(decimals=2))
```

## Why the model doesn't use your version

Yours builds the full T × T score matrix. At 2048 tokens that's 4 million numbers per head per
sequence, written to memory and read back. `F.scaled_dot_product_attention` computes the same
result in tiles without ever storing the whole matrix (this is what "flash attention" means), so
it's faster and needs far less memory. `Attention` in step 5 calls it. Your version exists so you
know exactly what that one line does, and the test proves the two agree.

Next: [step 4, RoPE](4-rope.md).
