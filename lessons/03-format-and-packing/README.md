# Lesson 3: the chat format and packing

## What the model learns

A language model learns one thing: **given the tokens so far, predict the next one.** Writing a whole
answer is that step repeated: predict a token, append it, predict the next.

So each row has to become one token sequence where "the next token" is something worth learning.

## The format: `theog/format.py`

```
<|User|>
Write a function that reverses a string. /no_think
<|/User|><|Assistant|><|Think|><|/Think|><|Answer|>
```python
def reverse(s):
    return s[::-1]
```
<|/Answer|><|end|>
```

- `<|User|> … <|/User|>` is the task. It ends with a mode, `/no_think` or `/think` (lesson 6).
- `<|Assistant|>` means "the model speaks now".
- `<|Think|><|/Think|>` is the think span, empty here (lesson 6).
- `<|Answer|> … <|/Answer|>` is the code, in a ```` ```python ```` block.
- `<|end|>` ends the conversation. At generation time the model stops on `<|/Answer|>`.

## Labels: what counts as a mistake

`sequence()` returns two lists of the same length:

- `input_ids`: the tokens.
- `labels`: for every position, the token that should come next, or `-100` (`IGNORE`) where there is
  nothing to learn.

The prompt is given at generation time, never written by the model, so all its positions get `-100`.
From `<|Assistant|>` on, `labels[i] = input_ids[i + 1]`. The model is graded only on the think span and
the answer. The loss function in lesson 5 skips every `-100`.

## Packing: `theog/pack.py`

The GPU processes a batch of sequences of the same length, 2048 tokens for theOG. A typical row here
is a few hundred tokens, so padding each row to 2048 would mostly train on padding.

**Packing** fills each 2048-token window with as many whole rows as fit, then pads only the rest.
Two rules:

- **A row is never cut in half.** If it doesn't fit, it starts the next window.
- **A row longer than the window is dropped**, never truncated. A cut-off answer would teach the
  model code that never ends.

Rows go to `train` or `val` by a hash of their prompt (5% val). The same task always lands on the
same side, so the validation loss in lesson 5 is measured on tasks the model never trained on.

The windows are written as raw `int16` files, 2 bytes per token. `load()` opens them with
`np.memmap`, so a multi-gigabyte dataset is not loaded into RAM: each batch reads only its own windows
from disk.

```bash
uv run python -m theog.pack --rows data/rows.clean.jsonl --tokenizer data/tokenizer.json --out data/packed
```

It prints `meta.json`: rows, windows and tokens per split. On the 27,653-row sample that was 3,649
train windows holding 6.8M tokens.

## Check it

```bash
uv run pytest tests/test_03_format_pack.py
```

The tests check that the prompt has no labels, that every other label is the next token, that the
generation prompt is exactly the start of a training sequence, and that no row is split across two
windows.

## Try this

- Decode one window of `train_ids.bin` with the tokenizer and read it.
- What fraction of a window is padding on average? (Count the `0`s.)
