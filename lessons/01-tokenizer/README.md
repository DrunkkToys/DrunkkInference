# Lesson 1: the tokenizer

**You write:** byte-pair encoding by hand in pure Python, then the real tokenizer with the
`tokenizers` library. **You get:** `data/tokenizer.json`, which every later lesson uses.
**You learn:** how text becomes the integers a model reads, and why `return` should cost one token,
not six. **Time:** about an hour.

A neural network can't read. It multiplies numbers, and that's all it does. So before it can see
`def add(a, b):`, the text has to become a list of integers, and the integers it writes have to
become text again. That's the tokenizer's job.

Boring? It's the one part of the model you can't change later. Every weight you train in lesson 5
is tied to these ids. Pick a bad tokenizer and you live with it.

## From bytes to tokens

The simplest tokenizer gives every byte a number. There are 256 possible bytes, so 256 tokens, and
any text at all can be written with them. The problem: `return` is six tokens, and the model spends
its effort re-reading the same six letters every time.

**Byte-pair encoding (BPE)** starts from those 256 bytes and repeats one step:

1. Find the pair of neighbouring tokens that appears most often.
2. Give that pair a new token id, and replace every occurrence of the pair with it.

After a few thousand rounds, frequent pieces such as `return`, `self` or four spaces of
indentation are each one token. Rare text still falls back to single bytes, so nothing is ever
"unknown".

You'll write that loop now, slowly, in plain Python. Then in part 2 you'll know exactly what the
fast library is doing, instead of trusting it.

## Part 1: BPE by hand

All three functions are in `theog/tokenizer.py`, under "BPE by hand".

### Exercise 1.1: `most_frequent_pair`

Given a list of ids, return the neighbouring pair that occurs most often:

```python
most_frequent_pair([1, 2, 3, 1, 2])   # (1, 2): it occurs twice
most_frequent_pair([5, 6, 7])         # (5, 6): a tie, so the first pair seen wins
most_frequent_pair([4])               # None: no pairs
```

`zip(ids, ids[1:])` walks every neighbouring pair. Python dicts keep insertion order, so "first seen
wins" comes for free if you count in a dict and take the max.

```bash
uv run pytest tests/test_01_tokenizer.py -k most_frequent
```

<details><summary>Solution</summary>

```python
def most_frequent_pair(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return max(counts, key=counts.get) if counts else None
```

</details>

### Exercise 1.2: `merge`

Replace every occurrence of a pair with a new id, left to right:

```python
merge([1, 2, 3, 1, 2], (1, 2), 99)    # [99, 3, 99]
merge([7, 7, 7], (7, 7), 50)          # [50, 7], not [7, 50]: leftmost first
```

The second example is the one people get wrong. Once `7, 7` at positions 0 and 1 is merged,
position 1 is used up and can't start another pair.

```bash
uv run pytest tests/test_01_tokenizer.py -k merge
```

<details><summary>Solution</summary>

```python
def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
```

</details>

### Exercise 1.3: `bpe_by_hand`

Put the two together. Start from `list(text.encode("utf-8"))`, then do `n_merges` rounds. Merge
number *k* gets id 256 + *k*. Keep a `vocab` dict that says what bytes each id stands for: ids 0 to
255 are single bytes, and a new id is the bytes of its two halves joined.

```bash
uv run pytest tests/test_01_tokenizer.py -k bpe_by_hand
```

<details><summary>Solution</summary>

```python
def bpe_by_hand(text, n_merges):
    ids = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    for new_id in range(256, 256 + n_merges):
        pair = most_frequent_pair(ids)
        if pair is None:
            break
        ids = merge(ids, pair, new_id)
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return ids, vocab
```

</details>

Now watch it learn. Before you run it, guess: 100 merges on a 940-byte Python file. How many
tokens are left? Write the number down. Then run it on that file:

```bash
uv run python -c "
from theog.tokenizer import bpe_by_hand
text = open('theog/tokens.py').read()
ids, vocab = bpe_by_hand(text, 100)
print(len(text.encode()), 'bytes ->', len(ids), 'tokens')
print([vocab[i] for i in range(256, 276)])
"
```

```text
940 bytes -> 410 tokens
[b'  ', b'    ', b' t', b'he', b', ', b'er', b'en', b'# ', b's ', b' the', b'on', b'at', b'\n    ', b'\n    "', b'\n    "<', b'\n    "<|', b'|>', b'|>"', b'|>",', b'in']
```

100 merges cut the length by more than half. Did you guess that low? The first things it learned
were two spaces, then
four (indentation). By merge 15 it had built `\n    "<|` one piece at a time, because that file is a
list of `"<|...|>"` strings, one per line. BPE learns whatever your text repeats. Train it on Python
and it learns Python.

## Part 2: the real tokenizer

Your version needs a full pass over the text for every merge. For 16,000 tokens over hundreds of
megabytes, that's hours. Nobody has hours. The Rust `tokenizers` library runs the same algorithm in
seconds, and adds the things real text needs.

**Byte-level.** It works on bytes, as yours did, so emoji, accents and any other Unicode
round-trip exactly. It shows bytes as printable characters: `Ġ` is a space and `Ċ` a newline.

**A split regex before merging.** First it cuts the text into words, numbers, punctuation and
spaces, the way GPT-2 does, and only merges inside each piece. Without this, BPE happily learns
tokens like `e(s` that glue the end of one word to the start of the next. This also matters in lesson
8: llama.cpp knows this exact split by the name `gpt-2`.

**Special tokens.** `theog/tokens.py` lists the markers of the chat format: `<|User|>`,
`<|Answer|>`, `<|end|>` and so on. Each must be one token that ordinary text can never produce, at a
fixed id (0 to 8) whatever the training text. Lesson 3 builds the format from them.

Exercises 1.4 and 1.5 are wiring, not thinking: you tell the library what you just built by hand.
The API names are given, because guessing them teaches you nothing.

### Exercise 1.4: `build_tokenizer`

Return an empty byte-level BPE: a `Tokenizer(models.BPE())` with a `ByteLevel` pre-tokenizer
(`add_prefix_space=False`, `use_regex=True`) and a `ByteLevel` decoder. The imports are already at
the top of the file.

<details><summary>Solution</summary>

```python
def build_tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    return tok
```

</details>

### Exercise 1.5: `train`

Train it. Make a `trainers.BpeTrainer` with the vocabulary size, `min_frequency`,
`special_tokens=SPECIAL_TOKENS` (listed first, so they get ids 0 to 8) and
`initial_alphabet=pre_tokenizers.ByteLevel.alphabet()` (all 256 bytes, so every byte has a token
even if your data never used it). Pass `show_progress=False`. Then call `train_from_iterator`.

```bash
uv run pytest tests/test_01_tokenizer.py
```

All 7 tests should pass now. They check that special tokens keep ids 0 to 8, that any text (accents,
emoji, tabs) decodes back to exactly itself, and that `<|Answer|>` is a single token.

<details><summary>Solution</summary>

```python
def train(texts, vocab_size=VOCAB_SIZE, min_frequency=2):
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok
```

</details>

## Run it on real data

The tokenizer trains on the rows lesson 2 builds. Do lesson 2 first (it's next), then come back and
run this on the laptop track:

```bash
uv run python -m theog.tokenizer --rows data/rows.clean.jsonl --out data/tokenizer.json --vocab-size 8000
```

```text
data/tokenizer.json: 8000 tokens
'def add(a, b):\n    return a + b\n' -> 13 tokens: ['def', 'Ġadd', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb', 'Ċ']
```

About 5 seconds on an M2. Look at `ĊĠĠĠ`: a newline plus three spaces is one token, and
`Ġreturn` carries the fourth space in front of the word. Indentation costs almost nothing.

**Full track:** leave out `--vocab-size`. The default, 16,000, is theOG's size. A bigger vocabulary
means shorter sequences but a bigger embedding table: 16,000 × 512 = 8.2M of theOG's 50M parameters
are that table. The laptop model can't afford that, so it uses 8,000.

## theOG's own tokenizer

Same kind: byte-level BPE with the GPT-2 split. Its special tokens sit partly at the start (0 to 3)
and partly at the end (15851 to 15857) of the vocabulary, because some were added to an existing
tokenizer later. Starting fresh, putting them all first is simpler. It also has `<|Focus|>` tokens
for theOG's research experiments, which the exported model doesn't need, so they're left out here.

## Going further

1. Encode four spaces and one tab with your trained tokenizer. Which is cheaper for the model?
2. Train with `--vocab-size 2000` and `--vocab-size 16000` (use another `--out`), then count the
   tokens of `theog/progress.py` with each. How much shorter is it?
3. Why must `<|Answer|>` never be produced by ordinary text? What would go wrong in lesson 5 if a
   training answer contained the literal string `<|/Answer|>`?

<details><summary>Answers</summary>

1. Alone, both are one token (`ĠĠĠĠ` and `ĉ`). In real code, spaces win: the data is almost all
   space-indented, so newline-plus-indent got merged into single tokens like `ĊĠĠĠĠĠĠĠ`. On the
   laptop tokenizer, `"    if x:\n        return 1"` is 7 tokens and the tab-indented version is 9.
2. Count with `len(tok.encode(open('theog/progress.py').read()).ids)`. On the laptop data that file
   is 636 tokens with a 2,000 vocabulary, 530 with 8,000 and 496 with 16,000: 22% shorter, and each
   extra merge helps less than the one before.
3. Generation stops on `<|/Answer|>` (lesson 5). If text could produce that token, the model would
   learn to stop in the middle of answers that happened to contain it. Added special tokens are
   matched before BPE runs, so the string becomes the single control token, not text, which is
   exactly why the training data must never contain them.

</details>

A tokenizer with nothing to learn from is an empty dictionary. Time to get it some data.

Next: [lesson 2, the data](../02-data/README.md).
