# Lesson 3: the chat format and packing

**You write:** the function that turns one row into one training sequence with its labels, and the
packer that fills fixed-size windows with them. **You get:** `data/packed/`, the files the training
loop reads. **You learn:** that a language model only ever learns "predict the next token", and
how the labels decide which of those predictions count. **Time:** about 45 minutes.

## What the model learns

A language model learns exactly one thing: **given the tokens so far, predict the next one.**
That's it. No understanding module, no reasoning engine. Writing a whole answer is that step
repeated: predict a token, append it, predict the next.

So each row has to become one token sequence where "the next token" is something worth learning,
plus a list that says, at every position, which token should come next. Get that list wrong by one
position and the model trains happily for hours on garbage. Ask me how I know.

## The format

```
<|User|>
Reverse a string. /no_think
<|/User|><|Assistant|><|Think|><|/Think|><|Answer|>
```python
def reverse(s):
    return s[::-1]
```
<|/Answer|><|end|>
```

- `<|User|> … <|/User|>` is the task. It ends with a mode, `/no_think` or `/think` (lesson 7).
- `<|Assistant|>` means "the model speaks now".
- `<|Think|><|/Think|>` is the think span, empty here (lesson 7).
- `<|Answer|> … <|/Answer|>` is the code, inside a ```` ```python ```` block.
- `<|end|>` ends the conversation. When generating, the model stops at `<|/Answer|>`.

## Labels: what counts as a mistake

The training sequence comes with a second list of the same length, `labels`. At position *i* it
holds the token the model should predict after reading positions 0 to *i*, which is simply
`input_ids[i + 1]`.

Except where there's nothing to learn. The prompt is given to the model at generation time; it
never has to write it. Why grade it on something it will never do? So every prompt position, up to and including `<|Assistant|>`, gets
`-100` (`IGNORE` in `theog/tokens.py`), and the loss in lesson 5 skips those. The model is graded
only on what it writes: the think span and the answer. The last position has no next token, so it
gets `-100` too.

Here is the whole thing for the row above, with the laptop tokenizer:

```text
pos  token          label
  0  <|User|>       -100
  1  Ċ              -100
  2  Re             -100
 ...
 14  <|Assistant|>  -100
 15  <|Think|>      5      the id of <|/Think|>: after <|Think|>, close the span
 16  <|/Think|>     6      the id of <|Answer|>
 17  <|Answer|>     207    the id of Ċ (newline)
 ...
 36  <|/Answer|>    8      the id of <|end|>
 37  <|end|>        -100   nothing comes after
```

38 tokens, 15 of them prompt. Notice `/no_think` costs 5 tokens with this tokenizer (`Ġ/`, `no`,
`_`, `th`, `ink`): it's rare in the training text, so BPE never merged it.

### Exercise 3.1: `sequence` in `theog/format.py`

Build `input_ids` from three parts:

1. **prompt:** `[USER]` + the tokens of `"\n{user.strip()} /no_think\n"` + `[USER_CLOSE, ASSISTANT]`.
   (`/think` instead when `think_text` is set: a row with text in its span is a `/think` row.)
2. **think span:** `[THINK]` + the tokens of `"\n{think_text}\n"` if there is think text + `[THINK_CLOSE]`.
3. **answer:** `[ANSWER]` + the tokens of `"\n```python\n{answer}\n```\n"` + `[ANSWER_CLOSE, END]`.

The exact strings are given because the format has to match to the byte: it's a spec, not a
puzzle. The labels are the part to think about. Which positions are graded, and what does each one
predict? Work it out on the 38-token table above before you write any code.

`tok.encode(text).ids` gives the token ids of a string.

<details><summary>Hint</summary>

Keep a parallel list of booleans, False for prompt positions and True for the rest. Then label *i*
is `input_ids[i + 1]` where the boolean is True, and `IGNORE` everywhere else, including the last
position.

</details>

```bash
uv run pytest tests/test_03_format_pack.py -k "sequence or think_text"
```

<details><summary>Solution</summary>

```python
def sequence(tok, user, answer, think_text=""):
    mode = "think" if think_text else "no_think"
    prompt = [USER] + tok.encode(f"\n{user.strip()} /{mode}\n").ids + [USER_CLOSE, ASSISTANT]
    think = [THINK] + (tok.encode(f"\n{think_text}\n").ids if think_text else []) + [THINK_CLOSE]
    body = [ANSWER] + tok.encode(f"\n```python\n{answer}\n```\n").ids + [ANSWER_CLOSE, END]
    input_ids = prompt + think + body
    supervised = [False] * len(prompt) + [True] * (len(think) + len(body))
    labels = [input_ids[i + 1] if supervised[i] else IGNORE for i in range(len(input_ids) - 1)] + [IGNORE]
    return {"input_ids": input_ids, "labels": labels}
```

</details>

## Packing

The GPU works on a batch of sequences that all have the same length: 512 tokens on the laptop
track, 2048 for theOG. Rows have all sorts of lengths. Padding every row to the window length would
mostly train on padding.

**Packing** fills each window with as many whole rows as fit, then pads only the rest. Think
Tetris, but lazy. Two rules:

- **A row is never cut in half.** If it doesn't fit, it starts the next window.
- **A row longer than the window is dropped**, never truncated. A cut-off answer teaches the model
  code that never ends.

### Exercise 3.2: `split_of` in `theog/pack.py`

Every row goes to `train` or `val`. Don't pick at random: hash the stripped prompt and send about
5% to `val`. Then the same task always lands on the same side, even if it appears twice, and the
validation loss in lesson 5 is measured on tasks the model has never trained on.

<details><summary>Solution</summary>

```python
def split_of(user):
    h = int(hashlib.sha256(user.strip().encode()).hexdigest()[:8], 16)
    return "val" if h % 100 < VAL_PCT else "train"
```

</details>

### Exercise 3.3 and 3.4: `Packer.add` and `Packer.flush`

`Packer` holds the window being filled in `self.ids` and `self.lbl`.

- `add(ids, labels)`: if the row doesn't fit in what's left of the window, `flush()` first. Then
  append the row and add its length to `self.tokens`.
- `flush()`: if the window is empty, do nothing. Otherwise pad it to `seq_len` (`PAD` in ids,
  `IGNORE` in labels, so padding is never learned), write both as `int16` to the two open files,
  count the window, and start a new empty one.

`np.asarray(values, dtype=np.int16).tofile(f)` writes raw numbers, 2 bytes each, with no header.

```bash
uv run pytest tests/test_03_format_pack.py
```

All 6 should pass.

<details><summary>Solution</summary>

```python
def add(self, ids, labels):
    if len(self.ids) + len(ids) > self.seq_len:
        self.flush()
    self.ids += ids
    self.lbl += labels
    self.tokens += len(ids)

def flush(self):
    if not self.ids:
        return
    pad = self.seq_len - len(self.ids)
    np.asarray(self.ids + [PAD] * pad, dtype=np.int16).tofile(self.ids_f)
    np.asarray(self.lbl + [IGNORE] * pad, dtype=np.int16).tofile(self.lbl_f)
    self.windows += 1
    self.ids, self.lbl = [], []
```

</details>

Why int16? Token ids go up to 7,999 here (15,999 on the full track), which fits in 2 bytes.
`load()` opens these files with `np.memmap`, so a multi-gigabyte dataset is never loaded into RAM:
each batch reads only its own windows from disk.

## Run it

Before you run it, guess: what share of each 512-token window will be padding? Write it down.
Question 1 at the end checks it.

```bash
uv run python -m theog.pack --rows data/rows.clean.jsonl --tokenizer data/tokenizer.json --out data/packed --seq-len 512
```

```text
{
  "seq_len": 512,
  "rows": 89818,
  "dropped_longer_than_seq_len": 5665,
  "tokenizer": "data/tokenizer.json",
  "vocab_size": 8000,
  "train_windows": 58052,
  "val_windows": 3140,
  "train_tokens": 21991017,
  "val_tokens": 1181570,
  "think": "none"
}
```

About 20 seconds. 22 million training tokens. 5,665 rows were longer than 512 tokens and were
dropped, about 6%.

`theog.pack` has one more flag, `--think`, which fills every row's think span with one of lesson 7's
designs. Leave it at its default, `none`: lesson 5 trains on the empty span.

Now look at what you made. Decode the first window:

```bash
uv run python -c "
import numpy as np, pathlib
from tokenizers import Tokenizer
from theog.pack import load
tok = Tokenizer.from_file('data/tokenizer.json')
ids, _ = load(pathlib.Path('data/packed'), 'train')
print(tok.decode(np.asarray(ids[0]).tolist(), skip_special_tokens=False)[:600])
"
```

You should see a `<|User|>` task, the think span, a fenced answer, `<|end|>`, maybe a second row,
then `<|pad|>` to the end. If the special tokens are missing or the answer is cut off, go back to
exercise 3.1.

**Full track:** leave out `--seq-len` (the default is 2048).

## Going further

1. What fraction of a window is padding on average? (Count the zeros in `train_ids.bin`.) How far
   off was your guess?
2. Your answer to 1 is probably uncomfortable. Why is it so high at 512, and what would you change?
3. `prompt_ids()` (lesson 5) builds the prompt at generation time. Why must it produce exactly the
   same ids as the start of `sequence()`, down to the newlines?

<details><summary>Answers</summary>

1. About 26% on the laptop track: 22.0M real tokens in 58,052 windows of 512 (29.7M slots). The
   first windows, OpenCodeInstruct's longer rows, are worse: `(np.asarray(ids[:2000]) == 0).mean()`
   gives 0.30.
2. An average row is about 245 tokens, but they range from 30 to 512. When the next row doesn't fit
   in what's left, the rest of the window becomes padding, even if a shorter row later in the file
   would have filled it. A quarter of the GPU's work goes into padding the loss then ignores.
   Options: longer windows (the waste at the end of each window matters less, but each step needs
   more memory); or let a row that doesn't fit look for space in earlier windows that are still
   open ("best-fit" packing). theOG's 2048 windows waste much less for the first reason.
3. The model learned what comes after one exact sequence of ids. A missing newline at generation
   time is a prompt it has never seen, and small models are bad at prompts they have never seen.
   This bug is easy to write and very hard to spot from the output, which is why lesson 5 has a
   test for it.

</details>

You have the data, the tokens and the windows. What you don't have yet is a model to feed them to.

Next: [lesson 4, the model](../04-model/README.md).
