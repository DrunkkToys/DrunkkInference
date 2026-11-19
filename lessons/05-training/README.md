# Lesson 5: training

**You write:** the learning-rate schedule, the loss, the sampler, the prompt, and the generation
loop.
**You get:** a trained model, `runs/laptop/best.pt`, that writes Python. **Time:** about 30 minutes
of work and 35 minutes of training.

Training repeats four steps thousands of times:

1. Feed a batch of windows to the model and get its logits.
2. Measure how wrong it was: the **loss**.
3. Work out, for every parameter, which direction makes the loss smaller: the **gradient**
   (`loss.backward()`, PyTorch does this for you).
4. Move every parameter a small step in that direction: the **optimizer** (`opt.step()`).

The loop itself, `train()` in `theog/train.py`, is written for you: batches, gradient
accumulation, validation, checkpoints, resuming. Read it once. You write the pieces it calls.

## The loss

### Exercise 5.1: `loss_fn`

At each position the model gives a score to every token in the vocabulary. **Cross-entropy** turns
the scores into probabilities (softmax) and takes −ln(the probability of the correct next token).
Averaged over every position whose label isn't `-100`.

If the model gives the right token probability 1, the loss is 0. A uniform guess over 8,000 tokens
gives ln 8000 ≈ 8.99. That's the first number you'll see.

`F.cross_entropy(logits, labels, ignore_index=IGNORE)` does all of it, but it wants logits as
(N, vocab) and labels as (N,), so flatten the batch and time axes first. Call `.float()` on the
logits so the loss is computed in float32 even if the model runs in lower precision.

<details><summary>Solution</summary>

```python
def loss_fn(logits, labels):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=IGNORE)
```

</details>

The test gives the loss a uniform guess and checks ln 4, then puts a confident guess at an ignored
position and checks that nothing changes. That's lesson 3's labels at work: the prompt is never
graded.

## The learning rate

The **learning rate** is how big each step is. Too big and the loss jumps around or explodes; too
small and nothing happens in 35 minutes. And the best size changes during the run:

- **Warmup.** At the start the weights are random and the gradients point in noisy directions. A
  big step in a wrong direction is hard to undo. So start at 0 and grow in a straight line over
  the first 5% of updates.
- **Cosine decay.** Then shrink, along half a cosine, down to 10% of the peak at the last update.
  Big steps early cover ground; small steps late settle into a low point instead of bouncing
  around it.

### Exercise 5.2: `lr_factor`

Return the multiplier of the peak learning rate at update `update` of `total`:

- `update < warmup`: `update / warmup`.
- otherwise, with `progress = (update - warmup) / max(1, total - warmup)`:
  `0.1 + 0.9 * 0.5 * (1 + cos(pi * progress))`.

```python
lr_factor(0, 100, 5)     # 0.0
lr_factor(5, 100, 5)     # 1.0, the peak
lr_factor(100, 100, 5)   # 0.1
```

<details><summary>Solution</summary>

```python
def lr_factor(update, total, warmup):
    if update < warmup:
        return update / warmup
    progress = (update - warmup) / max(1, total - warmup)
    return 0.10 + 0.90 * 0.5 * (1 + math.cos(math.pi * progress))
```

</details>

## The rest of the recipe

These are theOG-50M v3's settings, already in `train()`:

| setting | value | what it does |
|---|---|---|
| optimizer | AdamW, betas 0.9 / 0.95, eps 1e-8 | a step size per parameter, from running averages of its past gradients |
| weight decay | 0.01 | pulls every weight slightly towards 0, a mild brake on memorising |
| gradient clipping | 1.0 | if one batch's gradient is huge, shrink it, so one bad batch can't wreck the model |
| full track | peak lr 3e-4, batch 4 windows × 8 accumulation × 2048 tokens | 65,536 tokens per update |
| laptop track | peak lr 1e-3, batch 16 × 512 tokens, no accumulation | 8,192 tokens per update |

The laptop model is smaller, so it tolerates a bigger learning rate, and with 35 minutes there's no
time for small steps.

**Gradient accumulation.** A GPU may not fit 32 windows of 2048 at once. So the loop runs 8
micro-batches of 4, adds up their gradients (`(loss / accum).backward()`), and updates once. Same
result as one batch of 32.

**Validation and checkpoints.** Every `--eval-every` steps the loop measures the loss on validation
windows, tasks it never trains on (lesson 3). Training loss always goes down. Validation loss tells
you whether what it learns carries over to new tasks.

- `best.pt`: the weights with the lowest validation loss so far. Lessons 7 and 8 use this one.
- `resume.pt`: weights plus optimizer state. Run the same command again and it continues from here,
  reading the same batches the interrupted run would have read.
- `final.pt`: the weights after the last step.

## Train it

Check first that your code trains at all. The test trains the tiny model on toy rows on the CPU:

```bash
uv run pytest tests/test_05_train.py -k learns
```

It checks the validation loss falls below 2.5 (from about 6.2), that a second run resumes instead of
restarting, and that the trained model writes the start of an answer. It needs exercises 5.3 to
5.5 too (below), so it goes green once you've done them.

Then the real run:

```bash
uv run python -m theog.train --packed data/packed --out runs/laptop --preset laptop \
    --batch-size 16 --accum 1 --lr 1e-3 --eval-every 500 --eval-windows 200 --log-every 250
```

```text
laptop: 8,343,552 params on mps | 44,907 train windows (17,574,904 tokens) | 2,807 steps = 2,807 updates
step       0/2,807 | loss 9.0207 | lr 7.14e-06 | 1s
step     250/2,807 | loss 2.4165 | lr 9.96e-04 | 173s
step     500/2,807 | loss 1.6854 | lr 9.60e-04 | 348s
eval step 500: val loss 1.5889 (best 1.5889) *best*
...
eval step 2,000: val loss 0.6181 (best 0.6181) *best*
eval step 2,500: val loss 0.5418 (best 0.5418) *best*
done: 2,807 steps in 2,164s | final val loss 0.5177
```

36 minutes on an M2 MacBook Air (`mps` is Apple's GPU). The first loss is 9.02, right where a
uniform guess over 8,000 tokens would be. In the first 250 steps it falls to 2.4: the model learns
the format, the indentation, the commonest words. After that every step buys less.

The training loss on single batches jumps around (one batch has easy rows, the next hard ones).
Watch the validation loss, measured on the same 200 windows every time: it falls at every
check.

Leave it running and do the next two exercises meanwhile. If you stop it, run the same command
again and it resumes.

**Full track:**

```bash
uv run python -m theog.train --packed data/packed --out runs/theog-50m
```

The log prints elapsed seconds; divide by the step to get your GPU's seconds per step, multiply by
the step count for the length of the whole run. If the GPU runs out of memory, use
`--batch-size 2 --accum 16`: the same 32 windows per update, in smaller pieces.

## Writing code with it

### Exercise 5.3: `sample` in `theog/generate.py`

The model gives a score to every token. Which one comes next?

- `temperature 0`: the highest-scoring one (**greedy**). `int(logits.argmax())`. Always the same
  answer for the same prompt.
- otherwise: keep only the `top_k` highest scores (`torch.topk`), divide them by the temperature,
  softmax, and draw one (`torch.multinomial(probs, 1)`). A temperature below 1 sharpens the
  distribution, so at 0.2 (theOG's default) it's mostly the top token, sometimes the second. Top-k
  makes sure a very unlikely token is never drawn.

`torch.topk` gives scores *and* their positions. Return the position, as an `int`.

<details><summary>Solution</summary>

```python
def sample(logits, temperature=0.2, top_k=40):
    if temperature <= 0:
        return int(logits.argmax())
    vals, idx = torch.topk(logits / temperature, min(top_k, logits.numel()))
    return int(idx[torch.multinomial(torch.softmax(vals, -1), 1)])
```

</details>

### Exercise 5.4: `prompt_ids` in `theog/format.py`

At generation time there is no answer yet: the model reads the prompt and writes the rest. The
prompt must be **exactly** the start of what `sequence()` built in lesson 3, down to the last
newline. A model this small has only ever seen those exact ids, and one extra space is a prompt it
has never seen (lesson 6 shows what that does).

- The same head as `sequence()`: `[USER]` + the tokens of `"\n{user.strip()} /{mode}\n"` +
  `[USER_CLOSE, ASSISTANT]`, with `mode` `"think"` or `"no_think"`.
- `/no_think` (`think=False`): add `[THINK, THINK_CLOSE, ANSWER]`. The span is closed for the
  model, so its first token starts the answer.
- `/think` (`think=True`): add only `[THINK]`. The model closes the span itself (lesson 6).

```bash
uv run pytest tests/test_05_train.py -k prompt
```

The last of those tests builds a training sequence and checks that your prompt is its exact start.

<details><summary>Solution</summary>

```python
def prompt_ids(tok, user, think=False):
    mode = "think" if think else "no_think"
    head = [USER] + tok.encode(f"\n{user.strip()} /{mode}\n").ids + [USER_CLOSE, ASSISTANT]
    return head + ([THINK] if think else [THINK, THINK_CLOSE, ANSWER])
```

</details>

### Exercise 5.5: `generate` in `theog/generate.py`

1. `ids = prompt_ids(tok, task, think)`.
2. Repeat up to `max_new` times: run the model on the last `model.cfg.max_seq_len` ids, take the
   logits of the **last** position, `sample` a token. If it's one of `STOP_IDS` (`<|/Answer|>`,
   `<|end|>`, `<|User|>`), stop. Otherwise append it.
3. Return `tok.decode(out, skip_special_tokens=False)`: only what the model wrote, special tokens
   visible.

The `@torch.no_grad()` above the function stops PyTorch from recording what it needs for
`backward()`, which generation never calls.

```bash
uv run pytest tests/test_05_train.py
```

All 7 should pass.

<details><summary>Solution</summary>

```python
@torch.no_grad()
def generate(model, tok, task, think=False, max_new=1200, temperature=0.2, top_k=40):
    device = next(model.parameters()).device
    ids = prompt_ids(tok, task, think)
    out = []
    for _ in range(max_new):
        window = (ids + out)[-model.cfg.max_seq_len:]
        logits = model(torch.tensor([window], device=device))[0, -1].float()
        t = sample(logits, temperature, top_k)
        if t in STOP_IDS:
            break
        out.append(t)
    return tok.decode(out, skip_special_tokens=False)
```

</details>

This runs the whole window through the model for every new token. Real runtimes keep the keys and
values of earlier positions (a **KV cache**) and only compute the new one. llama.cpp does that for
you in lesson 8.

## Try the model

```bash
uv run python -m theog.generate --ckpt runs/laptop/best.pt --tokenizer data/tokenizer.json \
    "Write a function that checks if a string is a palindrome."
```

````text

```python
def is_palindrome_string(s):
    s = s.replace(' ', '').lower()
    return s == s[::-1]
```
````

Correct, and it even ignores spaces and case. Now a harder one:

```bash
uv run python -m theog.generate --ckpt runs/laptop/best.pt --tokenizer data/tokenizer.json \
    "Write a function that returns the n-th Fibonacci number."
```

````text

```python
def fib_list(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib_list(n - 1) + fib_list(n - 2)
```
````

Correct too, but called `fib_list`, a name it picked up from some training row. A test calling
`fib(10)` would fail with "no function named fib". You'll see this in lesson 7.

Your output will differ: at temperature 0.2 the sampler draws. `--temperature 0` gives the same
answer every time.

## How this differs from theOG's own run

Same recipe and same model layout. theOG-50M v3 trained on 2.49B tokens from 12 public sources; the
published checkpoint is step 346,000 (model card). The laptop track sees 17.6M tokens, about 0.7%
of that, on a model a sixth the size.

## Going further

1. Plot loss against step from your log. Where does it flatten?
2. The run reads every window once. Why not `--epochs 3`?
3. Why is the first logged loss close to ln 8000 and not, say, 20?

<details><summary>Answers</summary>

1. On the laptop run it never really flattens. Validation loss went 1.59, 1.06, 0.77, 0.62, 0.54,
   0.52: still falling at the end, mostly slowed by the shrinking learning rate. That means the
   model would learn more from more data or more steps. On the full track the curve flattens much
   later, and that's where theOG's 2.49B tokens come in.
2. With 17.6M tokens, a second pass would probably still lower the validation loss: the model is
   small and hasn't memorised anything yet. Run `--epochs 2` into a new `--out` and compare the val
   loss and the lesson 7 score. The risk with many epochs is that the model starts memorising the
   training answers: the training loss keeps falling while the validation loss stops or rises.
   That gap is the sign.
3. The embedding table starts small (`std=0.02`, lesson 4 step 1), so the first logits are all
   near 0 and the softmax is close to uniform: −ln(1/8000) ≈ 8.99. With large random weights the
   model would start confidently wrong, with a loss of 20 or more, and the first steps would be
   spent undoing that.

</details>

Next: [lesson 6, the /think span](../06-think-span/README.md).
