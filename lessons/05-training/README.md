# Lesson 5: training

**You write:** the learning-rate schedule, the loss, the sampler, the prompt, and the generation
loop.
**You get:** a trained model, `runs/laptop/best.pt`, that writes Python. **Time:** about 30 minutes
of work and about 2½ hours of training.

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

```bash
uv run pytest tests/test_05_train.py -k loss
```

The test gives the loss a uniform guess and checks ln 4, then puts a confident guess at an ignored
position and checks that nothing changes. That's lesson 3's labels at work: the prompt is never
graded.

## The learning rate

The **learning rate** is how big each step is. Too big and the loss jumps around or explodes; too
small and nothing happens in the time you have. And the best size changes during the run:

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

```bash
uv run pytest tests/test_05_train.py -k lr_schedule
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

The laptop model is smaller, so it tolerates a bigger learning rate, and with a couple of hours
there's no time for small steps.

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

Your two pieces green? Lesson 4's tests too? Then start the real run. It takes 2½ hours, so you
write the sampler and the generation loop (exercises 5.3 to 5.5, below) while it trains. Why sit
and watch a loss curve?

The last test of the lesson, `-k learns`, waits for those three. It trains the tiny model on toy
rows on the CPU and checks that the validation loss falls below 2.5 (from about 6.2), that a second
run resumes instead of restarting, and that the trained model writes the start of an answer.

The real run:

```bash
uv run python -m theog.train --packed data/packed --out runs/laptop --preset laptop --epochs 3 \
    --batch-size 16 --accum 1 --lr 1e-3 --eval-every 1000 --eval-windows 200 --log-every 500
```

```text
laptop: 8,343,552 params on mps | 58,052 train windows (21,991,017 tokens) | 10,885 steps = 10,885 updates
step       0/10,885 | loss 9.0633 | lr 1.84e-06 | 3s
step     500/10,885 | loss 1.8265 | lr 9.21e-04 | 611s
...
eval step 1,000: val loss 1.2267 (best 1.2267) *best*
eval step 2,000: val loss 0.7374 (best 0.7374) *best*
eval step 3,000: val loss 0.5519 (best 0.5519) *best*
eval step 4,000: val loss 0.4657 (best 0.4657) *best*
...
eval step 9,000: val loss 0.2897 (best 0.2897) *best*
eval step 10,000: val loss 0.2776 (best 0.2776) *best*
done: 10,885 steps in 9,239s | final val loss 0.2696
```

2 hours 34 minutes on an M2 MacBook Air (`mps` is Apple's GPU). The first loss is 9.06, right
where a uniform guess over 8,000 tokens would be. In the first few hundred steps it falls fast: the
model learns the format, the indentation, the commonest words. After that every step buys less.

`--epochs 3` means the model reads every training window three times, in a new order each time.
The next section explains why three, and what one pass gave instead.

The training loss on single batches jumps around (one batch has easy rows, the next hard ones).
Watch the validation loss, measured on the same 200 windows every time: it falls at every
check. The final weights scored best of all, so `best.pt` is the model after the last step.

## Why three passes: Chinchilla and Phi

There are two well-known answers to "how much data does a model need?", and they disagree.

**Chinchilla** ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) asks: for a fixed
compute budget, what's the best split between model size and training tokens? Its answer is about
20 tokens per parameter, each token seen once. It treats all tokens as equal: web text is web text.
By that rule our 8.3M model wants about 170M tokens, and the laptop data has 22M.

**Phi** ([Gunasekar et al., 2023, "Textbooks Are All You Need"](https://arxiv.org/abs/2306.11644))
asks a different question: what if the tokens are not equal? phi-1 trained a 1.3B code model on
about 7B tokens of carefully chosen, textbook-quality code, about 8 passes over it, and it beat
much bigger models trained on far more web code. Clean data you can learn from, read several
times, does more than more data read once.

theOG follows Phi. Lesson 2's cleaning rules are the "textbook quality" half. Epochs are the other
half. We measured it on the laptop track, same data, same model, same settings, changing only the
number of passes:

| | 1 epoch | 3 epochs |
|---|:-:|:-:|
| training time on an M2 | 48 minutes | 2 h 34 min |
| final validation loss | 0.449 | 0.270 |
| MBPP-500 (lesson 7) | 19/500 | 49/500 |
| HumanEval-164 (lesson 7) | 2/164 | 8/164 |

Repeating data stops paying at some point. [Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264)
found that up to about 4 passes are worth almost as much as new data, and that the gain fades after
that. The sign to stop is in your own log: the training loss keeps falling while the validation loss
stops or rises. Here it fell at every check, to the last one.

The full track keeps one epoch: theOG's 2.49B tokens are about 50 tokens per parameter already.

Leave it running and do the next three exercises meanwhile. If you stop it, run the same command
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
    --temperature 0 "Write a function that checks if a string is a palindrome."
```

````text

```python
def is_palindrome(s):
    return s == s[::-1]

def is_palindrome(s):
    return is_palindrome(s)
```
````

The first function is right. Then the model didn't stop: it wrote a second one that calls itself
forever. In Python the last definition wins, so this answer fails. Knowing
when to stop is one of the last things a small model learns.

Now give it the task the way MBPP does, and the way exercise 2.4 taught it, with a test that
names the function:

```bash
uv run python -m theog.generate --ckpt runs/laptop/best.pt --tokenizer data/tokenizer.json --temperature 0 \
    "Write a function that returns the n-th Fibonacci number.
Your code should pass these tests:

assert fib(10) == 55"
```

````text

```python
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)
```
````

Correct, and called `fib`, the name in the test. Without the test line it wrote a function called
`fibonacci` that returns the string `'Input should be positive integer'` for almost every `n`: it
had no test to tell it what a right answer looks like.

Right name doesn't mean right answer:

````text
Write a function to reverse the words in a string.
Your code should pass these tests:

assert reverse_words("hello world") == "world hello"

```python
def reverse_words(s):
    words = s.split()
    reversed_words = words[::-1]
    reversed_string = ' '.join(reversed_words)
    return reversed_string
```
````

Right. After one epoch the same model reversed the letters of each word instead (`"olleh dlrow"`),
a different, common task. That kind of answer, code that runs but does the wrong thing, is still
the laptop model's commonest failure. Lesson 7 counts it.

`--temperature 0` gives the same answer every time. Leave it out (the default is 0.2) and the
sampler draws, so your output will vary.

## How this differs from theOG's own run

Same recipe and same model layout. theOG-50M v3 trained on 2.49B tokens from 12 public sources, one
epoch; the published checkpoint is step 346,000 (model card). The laptop track has 22M tokens,
about 0.9% of that, read three times, on a model a sixth the size.

## Going further

1. Plot loss against step from your log. Where does it flatten?
2. Why `--epochs 3` and not 1, or 10?
3. Why is the first logged loss close to ln 8000 and not, say, 20?

<details><summary>Answers</summary>

1. On the laptop run it never really flattens. Validation loss went 1.23, 0.74, 0.55, 0.47, 0.41,
   0.38, 0.33, 0.31, 0.29, 0.28, and 0.27 at the end: still falling, more slowly as the learning
   rate shrinks. The model would still learn from more data or more steps.
2. One epoch left the model undertrained: its validation loss was still falling when the run ended
   (0.45), and it scored 19/500 on MBPP. Three gave 0.27 and 49/500 (the table above). Ten would
   probably start to memorise: 8M parameters reading the same 95k answers ten times learns the
   answers, not the skill. The sign is in the log: the training loss keeps falling while the
   validation loss stops or rises. Try `--epochs 5` into a new `--out` and see where yours turns.
3. The embedding table starts small (`std=0.02`, lesson 4 step 1), so the first logits are all
   near 0 and the softmax is close to uniform: −ln(1/8000) ≈ 8.99. With large random weights the
   model would start confidently wrong, with a loss of 20 or more, and the first steps would be
   spent undoing that.

</details>

You have a trained model now. So what is it actually doing in there? Go back to lesson 4 and find
out: [step 9, reading the attention](../04-model/9-reading-attention.md). A couple of minutes, no
training. Or skip it and go straight to [lesson 6, the /think span](../06-think-span/README.md).
