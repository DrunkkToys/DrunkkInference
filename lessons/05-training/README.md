# Lesson 5: training

Training repeats four steps millions of times:

1. Feed a batch of windows to the model and get its logits.
2. Measure how wrong it was: the **loss**.
3. Work out, for every parameter, which direction makes the loss smaller: the **gradient**
   (`loss.backward()`).
4. Move every parameter a small step in that direction: the **optimizer** (`opt.step()`).

## The loss

```python
F.cross_entropy(logits, labels, ignore_index=IGNORE)
```

Cross-entropy is −ln(the probability the model gave to the correct next token), averaged over every
position whose label is not `-100`. If the model gives the right token probability 1, the loss is 0.
A uniform guess over 16,000 tokens gives ln 16000 ≈ 9.68, the number you'll see on the first step.

## The recipe: `theog/train.py`

These are theOG-50M v3's settings (the same as v2's):

| setting | value | what it does |
|---|---|---|
| optimizer | AdamW, betas 0.9 / 0.95, eps 1e-8 | a per-parameter step size, from a running average of past gradients |
| peak learning rate | 3e-4 | how big the steps are |
| weight decay | 0.01 | pulls every weight slightly towards 0, a mild brake on memorising |
| warmup | first 5% of updates, 0 → 3e-4 | early gradients are noisy, so start with small steps |
| decay | cosine, from 3e-4 down to 3e-5 | smaller steps at the end, to settle into a minimum |
| gradient clipping | 1.0 | if one batch's gradient is huge, shrink it so one bad batch can't wreck the model |
| batch | 4 windows × 8 accumulation × 2048 tokens | 65,536 tokens per update |
| precision | float32 | slower than mixed precision, but nothing to tune |

**Gradient accumulation.** A GPU may not fit 32 windows at once. So the loop runs 8 micro-batches of
4, adds up their gradients (`(loss / accum).backward()`), and updates once. The result is the same
as one batch of 32.

**One epoch.** By default the run reads every training window once, in a shuffled order fixed by
`--seed`.

## Validation, checkpoints, resuming

Every 2,000 micro-batches the loop measures the loss on 400 **validation** windows: tasks it never
trains on (lesson 3). Training loss always goes down. Validation loss tells you whether the model
is learning things that carry over to new tasks.

- `best.pt`: the weights with the lowest validation loss so far. Lessons 7 and 8 use this one.
- `resume.pt`: weights plus optimizer state. Run the same command again and it continues from here,
  reading the same batches the interrupted run would have read.
- `final.pt`: the weights after the last step.

Each is about 200 MB.

## Run it

First the smoke test: a few steps, to check that everything fits on your GPU.

```bash
uv run python -m theog.train --packed data/packed --out runs/smoke --max-steps 20 --log-every 1
```

The first loss should be close to 9.7. Then the full run:

```bash
uv run python -m theog.train --packed data/packed --out runs/theog-50m
```

The first line prints the device (`cuda`, `mps` or `cpu`), the number of train windows and tokens, and
the number of steps. The log prints the elapsed seconds every 50 steps. Divide to get your GPU's
seconds per step, then multiply by the step count to get the length of the whole run. If it stops,
run the same command again to resume.

If the GPU runs out of memory, use `--batch-size 2 --accum 16`. That's the same 32 windows per
update, in smaller pieces.

## Try the model

```bash
uv run python -m theog.generate --ckpt runs/theog-50m/best.pt --tokenizer data/tokenizer.json \
    "Write a function that returns the n-th Fibonacci number."
```

`generate()` encodes the prompt, then repeatedly runs the model, takes the scores of the last
position, picks a token and appends it. It stops on `<|/Answer|>`, `<|end|>` or `<|User|>`.

**Sampling.** `--temperature 0` always picks the highest-scoring token (**greedy**). At 0.2 (theOG's
default), the scores are sharpened and one of the 40 best tokens (`--top-k`) is drawn at random,
mostly the top one.

## Check it

```bash
uv run pytest tests/test_05_train.py
```

The test trains the tiny model for 100 steps on toy rows on the CPU. It checks that the validation
loss falls below 2.5 (from about 6.2), that resuming continues the same run, and that the trained
model generates text.

## How this differs from theOG's own run

Same recipe and same model. The data differs (lesson 2): theOG-50M v3 trained on 2.49B tokens from
12 public Hugging Face sources, and the published checkpoint is step 330,000 of one epoch of 348,177
micro-batches. This course's two sources give fewer tokens, so one epoch is shorter. `data/packed/meta.json`
has your exact token count.

## Try this

- Train the tiny preset on your real data (`--preset tiny`) and watch how low it gets.
- Plot loss against step from the log. Where does it flatten?
