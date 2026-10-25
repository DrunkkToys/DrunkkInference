# Lesson 4: the model

theOG is a **decoder-only transformer**: the same family as GPT and Qwen. This lesson reads
`theog/model.py` from the bottom up. It is about 130 lines.

## The shape of the whole thing

```
token ids ─► embedding ─► block ×10 ─► RMSNorm ─► × embedding table ─► one score per vocabulary token
```

1. **Embedding.** Every token id picks one row of a 16,000 × 512 table: a list of 512 numbers that
   the model learns. From here on each token is a vector.
2. **10 blocks.** Each block lets every token look at the tokens before it (attention) and then
   transform itself (the MLP). The vector of each token is updated, never replaced:
   `x = x + something(x)`. This is the **residual stream**.
3. **Output.** The last vector of each position is compared to every row of the embedding table.
   Each comparison is a score (a **logit**) for "this is the next token". The same table is used at
   the input and at the output (**tied embeddings**), which saves 8.2M parameters.

## One block

```python
x = x + self.attn(self.attn_norm(x), cos, sin)
x = x + self.mlp(self.mlp_norm(x))
```

**RMSNorm** rescales each vector to a standard size before it enters a layer, so numbers neither
explode nor vanish over 10 layers. Normalising *before* each layer ("pre-norm") is what makes deep
transformers train stably.

### Attention

Each token makes three vectors from its own vector: a **query** ("what am I looking for?"), a
**key** ("what do I contain?") and a **value** ("what do I pass on?"). The score of token *i*
looking at token *j* is query*ᵢ* · key*ⱼ*. Scores go through a softmax, and token *i* receives the
weighted sum of values.

- **Causal.** Token *i* may only look at tokens 0 to *i*. Otherwise, during training, it could read
  the next token instead of predicting it. `is_causal=True` does this. `test_04_model.py` checks it:
  changing a later token never changes an earlier output.
- **Heads.** theOG runs 4 attentions side by side ("heads") of 128 dimensions each, so one head can
  track brackets while another tracks variable names.
- **QK-norm.** `q_norm` and `k_norm` apply RMSNorm to every query and key, per head, so the
  attention scores can't grow without limit.
- **RoPE.** Attention by itself does not know word order. Rotary position embeddings rotate each
  query and key by an angle that grows with its position, so `query · key` depends on how far apart
  the two tokens are. `rotate()` pairs dimension *i* with *i* + 64 (the "NEOX" layout). The angles are
  computed once in `rope_tables()`.

### The MLP: SwiGLU

```python
return self.down(F.silu(self.gate(x)) * self.up(x))
```

Two projections from 512 to 2048. One goes through SiLU, a smooth "off below zero" curve, and acts
as a gate on the other. A third projection goes back down to 512. This is where most of the
parameters are, and where most of the facts are thought to be stored.

## Counting the parameters

| part | parameters |
|---|--:|
| embedding (shared with the output) | 16,000 × 512 = 8,192,000 |
| per block: q, k, v, o | 4 × 512 × 512 = 1,048,576 |
| per block: gate, up, down | 3 × 512 × 2048 = 3,145,728 |
| per block: 2 norms + q/k norms | 512 + 512 + 128 + 128 = 1,280 |
| 10 blocks | 41,955,840 |
| final norm | 512 |
| **total** | **50,148,352** |

```bash
uv run python -m theog.model
```

prints the same total, which is also theOG-50M v2's published count.

## Why this exact layout

Every choice above (pre-norm RMSNorm, QK-norm then RoPE, SwiGLU, tied embeddings, no biases) is what
Qwen3 uses. llama.cpp, Ollama, LM Studio and MLX already know how to run `qwen3`, so lesson 8 can
export theOG with no custom code. theOG is not Qwen: it has no Qwen weights, only the same
arrangement of layers.

theOG-50M v2's research model has two extra pieces that `qwen3` has no place for, and they are
removed at export (see the "Export notes" on the
[model card](https://huggingface.co/DrunkkToys/theOG-50M)). The course model is the exported one:
with the same weights loaded, it gives the same logits as theOG's own model with those pieces off.

## Check it

```bash
uv run pytest tests/test_04_model.py
```

One test checks that an untrained model's loss is close to ln(vocabulary size). A model that knows
nothing should give every token the same chance, and the loss of a uniform guess over *V* tokens is
ln *V*. If the first loss is much higher, the initialisation is wrong.

## Try this

- Build `TheOG(tiny())` and print every parameter's name and shape.
- Remove `is_causal=True` and run the causal test. Read the failure.
