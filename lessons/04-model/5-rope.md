# Step 5: RoPE, word order

Step 1 showed that both 5s in `[5, 17, 5]` get the same vector, and attention by itself doesn't know
the order either. But `a - b` is not `b - a`. The model needs positions.

**Rotary position embedding (RoPE)** turns each query and key by an angle that depends on its
position. Picture each pair of numbers in the vector as an arrow on a flat plane. RoPE rotates
the arrow by *position × speed*, with a different speed for each pair: fast pairs track nearby
order, slow pairs track long distances.

Rotating both vectors changes their dot product only through the **difference** of their angles. So
the attention score of token *i* on token *j* depends on how far apart they are (*i − j*), not on
where they sit in the sequence. A `(` five tokens back looks the same at position 10 as at 1000.

```python
def rope_tables(cfg):
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2) / cfg.head_dim))
    freqs = torch.outer(torch.arange(cfg.max_seq_len), inv_freq)        # position × speed
    return torch.cat([freqs, freqs], -1).cos(), torch.cat([freqs, freqs], -1).sin()

def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos + rotated * sin
```

(`rope_tables` is shown slightly shortened. The file also sets `dtype=torch.float32`.)

- `inv_freq` holds the speeds: from 1 down to 1/10,000 (`rope_theta`).
- The tables are computed once for all 2048 positions, at build time.
- `rotate` pairs number *i* with number *i* + half. This is the "NEOX" layout. Pairing *i* with
  *i* + 1 also exists, and llama.cpp has to be told which one, so lesson 8 relies on it being NEOX,
  which is what `qwen3` uses.

RoPE has no parameters: nothing to learn.

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

Next: [step 6, SwiGLU](6-swiglu.md).
