# Step 1: the embedding

Nothing to write in this step. Read it, run it, and make sure you understand the shape, because
every other step is about that shape.

The tokenizer turns text into ids: whole numbers from 0 to 7,999 (laptop) or 15,999 (theOG). An id
is just a label. Token 5 is not "less" than token 17. The model needs something it can compute with.

The **embedding** is a table with one row per token: 8,000 rows of 256 numbers on the laptop
model. Token id 5 is replaced by row 5. The numbers start random and are learned in training.
Tokens used in similar ways (`list` and `dict`, say) end up with similar rows.

In `TheOG.__init__`:

```python
self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
nn.init.normal_(self.embed.weight, std=0.02)
```

`std=0.02` starts every number small. Step 8 explains why that matters: the same table is also used
to produce the output.

## Run it

```python
import torch
from theog.model import tiny
torch.manual_seed(0)
cfg = tiny()
embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden)
ids = torch.tensor([[5, 17, 5]])
x = embed(ids)
print(x.shape)
print(torch.equal(x[0, 0], x[0, 2]))
print(torch.equal(x[0, 0], embed.weight[5]))
```

```text
torch.Size([1, 3, 64])
True
True
```

- The shape is **(batch, tokens, hidden)**: 1 sequence, 3 tokens, 64 numbers each. Every layer from
  here on takes this shape in and gives the same shape out.
- Both 5s get the same vector. Nothing here knows *where* a token is. Step 4 fixes that.
- That vector is simply row 5 of the table. An embedding is a lookup, nothing more.

Sizes: laptop 8,000 × 256 = **2,048,000 parameters**, a quarter of the model. theOG-50M
16,000 × 512 = **8,192,000**, 16%.

Next: [step 2, RMSNorm](2-rmsnorm.md).
