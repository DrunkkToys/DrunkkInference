# Lesson 4: the model

theOG is a **decoder-only transformer**: the same family as GPT and Qwen. The whole model is
`theog/model.py`, about 130 lines. This lesson builds it one piece at a time, in eight steps. Each
step explains one piece, shows its code, and gives a snippet to run. Its output is printed in the
step, so you can compare.

```
token ids
   │
   ▼
1 embedding            each id becomes a vector of 512 numbers
   │
   ▼  ┌──────────────── block, ×10 ─────────────────┐
      │ 2 RMSNorm                                    │
      │ 3 attention  ◄── 4 heads + QK-norm, 5 RoPE   │
      │   + residual                                 │
      │ 2 RMSNorm                                    │
      │ 6 SwiGLU MLP                                 │
      │   + residual                       7 block   │
      └──────────────────────────────────────────────┘
   │
   ▼
8 RMSNorm, × embedding table   one score per vocabulary token
```

| step | piece | code in `theog/model.py` |
|---|---|---|
| [1](1-embedding.md) | embedding: ids to vectors | `nn.Embedding` in `TheOG` |
| [2](2-rmsnorm.md) | RMSNorm: keep vectors the same size | `RMSNorm` |
| [3](3-attention.md) | attention: tokens read earlier tokens | `F.scaled_dot_product_attention` in `Attention` |
| [4](4-heads-and-qk-norm.md) | heads and QK-norm | `Attention` |
| [5](5-rope.md) | RoPE: word order | `rope_tables`, `rotate` |
| [6](6-swiglu.md) | the SwiGLU MLP | `SwiGLU` |
| [7](7-block.md) | a block and the residual stream | `Block` |
| [8](8-full-model.md) | the whole model, tied output, 50,148,352 parameters | `TheOG`, `Config` |

Every snippet uses the `tiny()` size (hidden 64, 2 layers, 2 heads of 32, vocabulary 512) so it
runs instantly on a CPU. theOG-50M is the same code with bigger numbers. Run a snippet from the repo
root with `uv run python -c '...'` or paste it into `uv run python`.

## Check it

```bash
uv run pytest tests/test_04_model.py
```

There is one test per step.
