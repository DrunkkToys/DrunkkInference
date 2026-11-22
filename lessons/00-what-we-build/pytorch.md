# The PyTorch you need

**You learn:** the dozen tensor operations the whole course runs on. **Time:** about 15 minutes.

You don't need to know PyTorch to take this course. You need to know about a dozen things it does,
and this page is all of them. If you've used NumPy, most of it will feel familiar. If you haven't,
don't panic: a tensor is a list of lists of numbers that knows its own shape.

Paste each snippet into `uv run python` from the repo root. The output under it is what you should
see. Before you run each one, guess the output. You'll be wrong once or twice, and those are the
ones you'll remember.

## Shape: the first thing you check, always

```python
import torch
x = torch.arange(24.0).reshape(2, 3, 4)
print(x.shape, x.dtype)
print(x[0, 1])
```

```text
torch.Size([2, 3, 4]) torch.float32
tensor([4., 5., 6., 7.])
```

In this course the shape is almost always `(batch, tokens, hidden)`: so many sequences, so many
tokens in each, so many numbers per token. Half the bugs you'll hit in lesson 4 are a shape that
isn't what you thought. When something breaks, print the shape before you do anything else.

## `@`: matrix multiply

```python
import torch
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
v = torch.tensor([10.0, 1.0])
print(a @ v)
print(a @ a.T)
```

```text
tensor([12., 34.])
tensor([[ 5., 11.],
        [11., 25.]])
```

Every row of `a` dotted with `v`. That's what a linear layer does, and most of a transformer is
linear layers. `a @ a.T` is every row dotted with every row: one score per pair. Remember that one,
because attention is exactly that.

## `transpose`: swap two dimensions

```python
import torch
x = torch.arange(24.0).reshape(2, 3, 4)
print(x.transpose(1, 2).shape)
```

```text
torch.Size([2, 4, 3])
```

`.T` flips a 2-D matrix. For more dimensions you name the two to swap. Attention uses
`k.transpose(-2, -1)`: negative numbers count from the end, so it swaps the last two whatever comes
before them.

## `softmax`: scores into probabilities

```python
import torch
s = torch.tensor([2.0, 1.0, 0.0])
print(torch.softmax(s, dim=-1))
print(torch.softmax(s, dim=-1).sum())
```

```text
tensor([0.6652, 0.2447, 0.0900])
tensor(1.)
```

Any numbers in, positive numbers that sum to 1 out, the biggest one getting the most. `dim=-1`
says along which dimension they sum to 1. Get it wrong and nothing errors: the numbers are just
quietly wrong.

## `masked_fill`: hide what a token mustn't see

```python
import torch
scores = torch.zeros(3, 3)
mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
print(mask)
m = scores.masked_fill(mask, float("-inf"))
print(torch.softmax(m, dim=-1))
```

```text
tensor([[False,  True,  True],
        [False, False,  True],
        [False, False, False]])
tensor([[1.0000, 0.0000, 0.0000],
        [0.5000, 0.5000, 0.0000],
        [0.3333, 0.3333, 0.3333]])
```

`triu(1)` marks everything above the diagonal: for each token, the tokens that come after it.
`-inf` turns into exactly 0 after softmax. Token 0 sees only itself, token 2 sees everything before
it. That's the whole "causal" in causal attention, and it's lesson 4 step 3.

## `view` and `reshape`: same numbers, new shape

```python
import torch
h = torch.arange(12.0).reshape(1, 2, 6)
print(h.view(1, 2, 2, 3).shape)
print(h.view(1, 2, 2, 3)[0, 0])
```

```text
torch.Size([1, 2, 2, 3])
tensor([[0., 1., 2.],
        [3., 4., 5.]])
```

Nothing moves in memory: the 6 numbers of each token are now read as 2 groups of 3. That's how one
hidden vector becomes several attention heads in lesson 4 step 5. `view` fails loudly if the data
isn't laid out for it (after a `transpose`, say). `reshape` copies when it has to and always works.

## `cat`: glue tensors together

```python
import torch
print(torch.cat([torch.tensor([1, 2]), torch.tensor([3])]))
print(torch.cat([torch.zeros(2, 1), torch.ones(2, 2)], dim=-1))
```

```text
tensor([1, 2, 3])
tensor([[0., 1., 1.],
        [0., 1., 1.]])
```

The shapes must match everywhere except along `dim`. RoPE uses it to put two halves of a vector
back together.

## `arange` and `outer`: build a table

```python
import torch
print(torch.arange(4))
print(torch.outer(torch.arange(3.0), torch.tensor([1.0, 0.5])))
```

```text
tensor([0, 1, 2, 3])
tensor([[0.0000, 0.0000],
        [1.0000, 0.5000],
        [2.0000, 1.0000]])
```

`arange` counts. `outer` multiplies every item of one by every item of the other. Positions times
frequencies: that's the RoPE angle table in lesson 4 step 4, in one line.

## Broadcasting: shapes that stretch

```python
import torch
print(torch.ones(2, 3) * torch.tensor([1.0, 2.0, 3.0]))
```

```text
tensor([[1., 2., 3.],
        [1., 2., 3.]])
```

A `(3,)` tensor times a `(2, 3)` tensor: PyTorch repeats the small one along the missing
dimension. RMSNorm's learned weight is applied to every token this way. Handy, and also the reason
a wrong shape sometimes *doesn't* crash. It just broadcasts into nonsense.

## `topk` and `multinomial`: pick the next token

```python
import torch
p = torch.tensor([0.1, 0.6, 0.05, 0.25])
print(torch.topk(p, 2))
g = torch.Generator().manual_seed(0)
print(torch.multinomial(p, 10, replacement=True, generator=g))
```

```text
torch.return_types.topk(
values=tensor([0.6000, 0.2500]),
indices=tensor([1, 3]))
tensor([3, 2, 1, 3, 1, 3, 1, 1, 1, 1])
```

`topk` gives the biggest values *and where they are*. `multinomial` draws ids at random, weighted
by the probabilities: id 1 (0.6) comes up most, but not every time. That's sampling, lesson 5's
sampler, and the reason the same prompt can give different answers.

## Gradients

The one thing missing here is how a model learns: `loss.backward()` and the gradient. Lesson 5
explains it with one parameter and four steps, right before you need it.

Back to [lesson 0](README.md), or on to [lesson 1](../01-tokenizer/README.md).
