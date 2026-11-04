# Step 1: the embedding

The tokenizer (lesson 1) turns text into ids: whole numbers from 0 to 15,999. An id is just a label.
Token 5 is not "less" than token 17. The model needs something it can compute with.

The **embedding** is a table with one row per token: 16,000 rows of 512 numbers each for theOG.
Token id 5 is replaced by row 5. The numbers start random and are learned in training. Tokens used
in similar ways (`list` and `dict`, say) end up with similar rows.

```python
self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
nn.init.normal_(self.embed.weight, std=0.02)
```

`std=0.02` starts every number small. Step 8 explains why that matters: the same table is also used
to produce the output.

## Run it

```python
import torch
from theog.model import TheOG, tiny
torch.manual_seed(0)
model = TheOG(tiny())
ids = torch.tensor([[5, 17, 5]])
x = model.embed(ids)
print(x.shape)
print(torch.equal(x[0, 0], x[0, 2]))
print(torch.equal(x[0, 0], model.embed.weight[5]))
```

```text
torch.Size([1, 3, 64])
True
True
```

- The shape is (batch, tokens, hidden): 1 sequence, 3 tokens, 64 numbers each.
- Both 5s get the same vector. Nothing here knows *where* a token is. Step 5 adds that.
- That vector is simply row 5 of the table.

theOG-50M: 16,000 × 512 = **8,192,000 parameters**, 16% of the model.

Next: [step 2, RMSNorm](2-rmsnorm.md).
