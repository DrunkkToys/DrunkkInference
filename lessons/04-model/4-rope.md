# Step 4: RoPE, word order

Step 1 showed that both 5s in `[5, 17, 5]` get the same vector, and attention by itself doesn't
know the order either: shuffle the tokens and every score just moves with them. But `a - b` is not
`b - a`. The model needs positions.

**Rotary position embedding (RoPE)** turns each query and key by an angle that depends on its
position. Picture each pair of numbers in the vector as an arrow on a flat plane. RoPE rotates the
arrow by *position × speed*, with a different speed for each pair: fast pairs track nearby order,
slow pairs track long distances.

Rotating both vectors changes their dot product only through the **difference** of their angles.
So the attention score of token *i* on token *j* depends on how far apart they are (*i* − *j*),
not on where they sit. A `(` five tokens back looks the same at position 10 as at position 1000.

RoPE has no parameters. Nothing to learn, just two tables computed once.

## Exercise 4.4a: `rope_tables`

Return `(cos, sin)`, each of shape `(max_seq_len, head_dim)`.

- There are `head_dim / 2` pairs. Pair *i* turns at speed `1 / rope_theta ** (2i / head_dim)`: from
  1 for the first pair down to almost 1/10,000 for the last.
  `torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim` gives the `2i / head_dim` values.
- At position *p* the angle is *p* × speed. `torch.outer(positions, speeds)` gives every angle at
  once: shape `(max_seq_len, head_dim / 2)`.
- Both halves of each row hold the same angles, because in step 4b dimension *i* is paired with
  dimension *i* + `head_dim / 2`. `torch.cat([angles, angles], -1)`, then `.cos()` and `.sin()`.

<details><summary>Solution</summary>

```python
def rope_tables(cfg):
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
    freqs = torch.outer(torch.arange(cfg.max_seq_len, dtype=torch.float32), inv_freq)
    return torch.cat([freqs, freqs], -1).cos(), torch.cat([freqs, freqs], -1).sin()
```

</details>

## Exercise 4.4b: `rotate`

Rotating a 2D point `(a, b)` by angle θ gives `(a cos θ − b sin θ, b cos θ + a sin θ)`. Do that
for every pair at once: split `x` into halves `x1 = x[..., :half]` and `x2 = x[..., half:]`, build
`[-x2, x1]` with `torch.cat(..., -1)`, and return `x * cos + that * sin`.

This pairing (*i* with *i* + half) is called the **NEOX** layout. Pairing *i* with *i* + 1 also
exists, and llama.cpp has to be told which one, so lesson 8 relies on this being NEOX, which is
what `qwen3` uses.

```bash
uv run pytest tests/test_04_model.py -k step4
```

<details><summary>Solution</summary>

```python
def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos + rotated * sin
```

</details>

## Run it

```python
import torch
from theog.model import rotate, rope_tables, tiny
cfg = tiny()
cos, sin = rope_tables(cfg)
torch.manual_seed(0)
q, k = torch.randn(cfg.head_dim), torch.randn(cfg.head_dim)
def score(i, j):
    return float(rotate(q, cos[i], sin[i]) @ rotate(k, cos[j], sin[j]))
print(round(score(3, 1), 4), round(score(10, 8), 4), round(score(50, 48), 4))
print(round(score(3, 0), 4))
print(torch.allclose(rotate(q, cos[7], sin[7]).norm(), q.norm()))
```

```text
-4.7229 -4.7229 -4.7229
-5.1072
True
```

- The same query and key, 2 apart, score the same at positions 3, 10 and 50.
- 3 apart scores differently.
- A rotation never changes a vector's length. Only its direction carries the position.

Next: [step 5, heads and QK-norm](5-heads-and-qk-norm.md).
