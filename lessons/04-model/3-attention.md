# Step 3: attention

After step 1, each token's vector only knows the token itself. To write `return a + b`, the model
must know that `a` and `b` are the arguments a few tokens back. **Attention** is how a token reads the
tokens before it.

## Query, key, value

Each token makes three vectors from its own, with three learned matrices:

- **query**: what this token is looking for
- **key**: what this token offers to others
- **value**: what this token passes on if it is chosen

Then, for every token *i*:

1. **Score** every token *j*: `query_i · key_j` (a dot product: multiply position by position, add up).
   A high score means *j* has what *i* is looking for. Scores are divided by √d (d = vector size), so
   they don't grow with d.
2. **Mask** the future: token *i* may only look at tokens 0 to *i*. In training the whole sequence is
   there at once, and without the mask, token *i* could read token *i*+1, the very token it must
   predict.
3. **Softmax** turns the scores of each row into weights that are positive and add up to 1.
4. **Mix**: token *i* gets the weighted sum of the values.

## Run it, by hand

```python
import torch
torch.manual_seed(0)
T, d = 4, 8
q, k, v = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)
scores = q @ k.T / d ** 0.5
mask = torch.ones(T, T).tril().bool()
scores = scores.masked_fill(~mask, float("-inf"))
weights = scores.softmax(-1)
print(weights.round(decimals=2))
out = weights @ v
ref = torch.nn.functional.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
print(torch.allclose(out, ref, atol=1e-6))
```

```text
tensor([[1.0000, 0.0000, 0.0000, 0.0000],
        [0.4900, 0.5100, 0.0000, 0.0000],
        [0.6200, 0.0300, 0.3500, 0.0000],
        [0.2000, 0.6200, 0.1000, 0.0700]])
True
```

Row *i* is how much token *i* reads from each token. Everything above the diagonal is 0: that's the
mask. Each row adds up to 1. The first token can only read itself.

`True` means PyTorch's `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, the one line
`theog/model.py` uses, computes exactly these four steps. It just does them faster and with less
memory.

Next: [step 4, heads and QK-norm](4-heads-and-qk-norm.md).
