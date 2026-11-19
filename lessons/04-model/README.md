# Lesson 4: the model

**You write:** the transformer. RMSNorm, attention by hand, RoPE, multi-head attention with
QK-norm, the SwiGLU MLP, the block, and the forward pass of the whole model: 8 functions, most of
them 2 to 5 lines. **You get:** `theog/model.py`, which lesson 5 trains. **Time:** two to three hours.
This is the longest lesson, and the one people remember.

theOG is a **decoder-only transformer**, the same family as GPT and Qwen. The whole model is
`theog/model.py`, about 150 lines. The class skeletons (which layers exist, their sizes) are
written for you. What's missing is what each piece *computes*.

```
token ids
   │
   ▼
1 embedding            each id becomes a vector (256 numbers laptop, 512 theOG)
   │
   ▼  ┌──────────────── block, ×6 laptop / ×10 theOG ──────┐
      │ 2 RMSNorm                                           │
      │ 3 attention  ◄── 4 RoPE, 5 heads + QK-norm          │
      │   + residual                                        │
      │ 2 RMSNorm                                           │
      │ 6 SwiGLU MLP                                        │
      │   + residual                              7 block   │
      └─────────────────────────────────────────────────────┘
   │
   ▼
8 RMSNorm, × embedding table   one score per vocabulary token
```

| step | piece | you write | test |
|---|---|---|---|
| [1](1-embedding.md) | embedding: ids to vectors | nothing: read it | `-k step1` |
| [2](2-rmsnorm.md) | RMSNorm: keep vectors the same size | `RMSNorm.forward` | `-k step2` |
| [3](3-attention.md) | attention: tokens read earlier tokens | `causal_attention` | `-k step3` |
| [4](4-rope.md) | RoPE: word order | `rope_tables`, `rotate` | `-k step4` |
| [5](5-heads-and-qk-norm.md) | heads and QK-norm | `Attention.forward` | `-k step5` |
| [6](6-swiglu.md) | the SwiGLU MLP | `SwiGLU.forward` | `-k step6` |
| [7](7-block.md) | a block and the residual stream | `Block.forward` | `-k step7` |
| [8](8-full-model.md) | the whole model, tied output | `TheOG.forward` | `-k step8` |

Run one step's tests with:

```bash
uv run pytest tests/test_04_model.py -k step2
```

Every snippet uses the `tiny()` size (hidden 64, 2 layers, 2 heads of 32, vocabulary 512), so it
runs instantly on a CPU. The laptop and theOG-50M sizes are the same code with bigger numbers. Run
a snippet from the repo root by pasting it into `uv run python`. Each step prints the output you
should get. If yours differs, your function differs.

Do the steps in order. Each test only uses the steps before it.

Start: [step 1, the embedding](1-embedding.md).
