# Lesson 1: the tokenizer

A neural network only does arithmetic on numbers. Before it can read `def add(a, b):`, the text has to
become a list of integers, and the integers it writes have to become text again. That is the
tokenizer's job.

## From characters to tokens

The simplest tokenizer gives every byte a number (256 possible values). That works for any text, but
`return` becomes six numbers, and the model spends its effort re-reading the same letters.

**Byte-pair encoding (BPE)** starts from those 256 bytes and repeats one step: find the pair of
neighbouring tokens that appears most often in the training text, and add that pair as a new token.
After thousands of merges, frequent pieces such as `return`, `self` or four spaces of indentation are
each one token, and rare text still falls back to single bytes. Nothing is ever "unknown".

theOG uses 16,000 tokens. A bigger vocabulary means shorter sequences but a bigger embedding table:
16,000 × 512 = 8.2M of theOG's 50M parameters are that table.

## The code: `theog/tokenizer.py`

```python
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
tok.decoder = decoders.ByteLevel()
```

- `ByteLevel` works on bytes, so emoji, accents and any other Unicode round-trip exactly.
- `use_regex=True` first splits text the way GPT-2 does (words, numbers, punctuation, spaces), so a
  merge never glues the end of one word to the start of the next. This also matters in lesson 8:
  llama.cpp knows this exact split as `gpt-2`.

```python
trainer = trainers.BpeTrainer(vocab_size=16_000, min_frequency=2,
                              special_tokens=SPECIAL_TOKENS,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
```

`SPECIAL_TOKENS` (in `theog/tokens.py`) are the markers of the chat format: `<|User|>`, `<|Answer|>`,
`<|end|>`, and so on. They are listed first, so they always get ids 0 to 8, and each is one token that
ordinary text can never produce. Lesson 3 builds the format from them.

## Run it

The tokenizer is trained on the rows of lesson 2. Do lesson 2's `build` first, then:

```bash
uv run python -m theog.tokenizer --rows data/rows.clean.jsonl --out data/tokenizer.json
```

It prints how a small function splits into tokens. Look for `Ġ`: that is how byte-level BPE shows a
space, and `Ċ` is a newline.

## Check it

```bash
uv run pytest tests/test_01_tokenizer.py
```

The tests check that the special tokens keep ids 0 to 8, that any text (accents, emoji, tabs) decodes
back to exactly itself, and that `<|Answer|>` is one token.

## theOG's own tokenizer

theOG's tokenizer is the same kind (byte-level BPE, GPT-2 split). Its special tokens sit partly at the
start (0 to 3) and partly at the end (15851 to 15857) of the vocabulary, because some were added to an
existing tokenizer later. Starting fresh, putting them all first is simpler. It also has `<|Focus|>`
tokens that theOG v2 never trains on, so they are left out here.

## Try this

- Train with `--vocab-size 4000` and with `32000`, then count the tokens of the same file with each.
- Encode `    ` (four spaces) and `\t`. Which one is cheaper for the model?
