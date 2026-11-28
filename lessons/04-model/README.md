# Lesson 4: the model

**You write:** the transformer. RMSNorm, attention by hand, RoPE, multi-head attention with
QK-norm, the SwiGLU MLP, the block, and the forward pass of the whole model: 8 functions, most of
them 2 to 5 lines. **You get:** `theog/model.py`, which lesson 5 trains. **You learn:** what every
piece of a transformer computes, and why it's there. **Time:** about three hours, more if PyTorch
is new to you. This is the longest lesson, and the one people remember.

New to PyTorch? Read [the PyTorch you need](../00-what-we-build/pytorch.md) first. Fifteen
minutes, and every operation in this lesson is on it.

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

Two parts. **Part 1, build it**, is now: steps 1 to 8, every one with a test. **Part 2, look
inside**, is step 9, and it waits for a trained model, so you come back to it after lesson 5.

## Part 1: build it, now

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

## Part 2: look inside, after lesson 5

| step | piece | you write | test |
|---|---|---|---|
| [9](9-reading-attention.md) | reading the attention of a trained model | nothing: run it | `test_04_attnlab.py` |

Changing the attention, the part every paper changes, is a lesson of its own:
[lesson 10](../10-attention/README.md), after the course.

Run one step's tests with:

```bash
uv run pytest tests/test_04_model.py -k step2
```

Every snippet uses the `tiny()` size (hidden 64, 2 layers, 2 heads of 32, vocabulary 512), so it
runs instantly on a CPU. The laptop and theOG-50M sizes are the same code with bigger numbers. Run
a snippet from the repo root by pasting it into `uv run python`. Each step prints the output you
should get. If yours differs, your function differs.

Each exercise gives the idea first and folds the exact PyTorch calls into a **Hint**. Try without
it. The calls are easy to look up; knowing which ones you need is the part that sticks.

Do the steps in order. Each test only uses the steps before it.

Step 9 is the odd one out. It needs a *trained* model, so skip it now: lesson 5 sends you back.
That's where it gets fun: you stop asking what attention computes and start asking what your
model actually does with it. Spoiler: not what the diagrams say.

Start: [step 1, the embedding](1-embedding.md).
