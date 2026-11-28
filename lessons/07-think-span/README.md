# Lesson 7: the /think span, an experiment

**You write:** `grounded` in `theog/think.py` (exercise 7.1) and `nats` in `theog/cost.py`
(exercise 7.2). **Then you run:** one experiment, `psr`: a prediction in writing, a training run, and
the scoring. **You learn:** why a span can hurt, how to measure what it costs, and how to run an
experiment that can prove you wrong. **Time:** about an hour to read and do the exercises, plus
about 4 hours of machine time for your run.

| step | what you run | time on an M2 |
|---|---|---|
| 1 | your lesson 5 model, in `/think` mode | a minute |
| 4 | the cost of your lesson 5 model | a few minutes |
| [your experiment](#your-experiment-psr) | train and score `psr` | about 4 hours, almost all of it training |

Everything else we ran, twelve designs and two seeds, is at the end, in
[Do you want to fail or underperform?](#do-you-want-to-fail-or-underperform) For the few who want
more than one experiment.

Every score here comes from the lesson 6 harness. The experiment scores `final.pt`, the weights after the last step, so every
design is measured at the same budget. For the lesson 5 run that is the same model as `best.pt`.

This lesson is a real experiment, wrong predictions included. Spoiler: the theory loses a lot.

## What the span is

Every theOG sequence has a think span between the prompt and the answer:

```
<|Assistant|><|Think|><|/Think|><|Answer|>
```

and every prompt ends with a mode, `/no_think` or `/think`. You wrote both modes in `prompt_ids`
(exercise 5.4):

- `/no_think`: the prompt ends `<|Think|><|/Think|><|Answer|>`. The span is closed for the model,
  so it starts writing code at once.
- `/think`: the prompt ends `<|Think|>`. The model writes the rest: what goes in the span, and when
  to close it.

theOG's own design leaves the span **empty**: a pause before the answer, worked out independently by
its author. The pause-tokens paper ([Goyal et al., 2023](https://arxiv.org/abs/2310.02226)) arrived
at a similar idea separately: extra positions before the answer give the model more computation
steps. It is not chain of thought, which trains the model on reasoning written out in words.

`sequence(..., think_text=...)` (lesson 3) can put text inside the span. This lesson asks what that
text should be.

## Step 1: the empty span, in a mode the model never saw

Our lesson 5 model trained on `/no_think` rows only. Give it lesson 5's task in `/think` mode
(`--think` opens the span instead of closing it):

```bash
uv run python -m theog.generate --ckpt runs/laptop/final.pt --tokenizer data/tokenizer.json \
    --think --temperature 0 --max-new 12 "Write a function that checks if a string is a palindrome."
```

````text
<|/Think|><|Answer|>
```python stringstringstringstringstringstringstring
````

In `/no_think` the same model wrote a correct `is_palindrome` (lesson 5). Here it closes the span,
as it should, and then loses the thread. Score it on HumanEval in `/think` mode:

```bash
uv run python -m theog.evaluate --ckpt runs/laptop/final.pt --tokenizer data/tokenizer.json \
    --think --dataset humaneval --out results/
```

It scores **1/164**, against 8/164 in `/no_think`. Only 48 of the 164 answers continue the span
with `<|Answer|>`; the rest skip it and start the code straight after the span, some of it in the
middle of a line. The model never saw a prompt ending in `/think`, and three passes over the data
made it more sure of the only mode it knows.

So `/think` needs training rows of its own. And once you write those rows, you have to decide what
goes inside the span.

## Step 2: what can go in the span

A training row gives you two texts: the prompt P and the code C. Anything you put in the span has to
be built from them by a fixed rule, because at inference time the model writes the span itself.
Nobody writes it for each row, and no bigger model writes it either: that would be distillation,
another course.

So a span is a function S = f(P, C): somewhere between "nothing" and "the code itself". Five designs,
all in `theog/think.py`, on one MBPP-style row:

```text
prompt   Write a function count_char_freq that counts how often each character appears
         in the string s, ignoring case. Return a dict.
code     def count_char_freq(s):
             """Count each character."""
             freq = {}
             for char in s.lower():
                 if char in freq:
                     freq[char] += 1
                 else:
                     freq[char] = 1
             return freq
```

| design | the span | the idea |
|---|---|---|
| `none` | (empty) | theOG's pause |
| `sig` | `def count_char_freq(s):` | commit to the signature first |
| `shared` | `count_char_freq s` | the code's tokens the prompt already contains: what the model can copy |
| `skeleton` | `def count_char_freq s` / `  for in s` / `    if in` / `    else` / `  return` | `shared` plus keywords, one line per code line: the control flow |
| `plan` | `name: count_char_freq(s)` / `test: …` / `plan:` / `- freq = {}` / `- for char in s.lower(): branch, updates freq` / `- return freq` | a coarse plan of the code, after Dong & Lapata ([2018](https://arxiv.org/abs/1805.04793), coarse-to-fine) and Li et al. ([2023](https://arxiv.org/abs/2305.06599), structured chain of thought) |

`shared` and `skeleton` are built from one idea: a code token is **grounded** when the prompt
already contains it. Grounded tokens can be copied; everything else the model has to invent. That's
the first exercise.

### Exercise 7.1: `grounded` in `theog/think.py`

`grounded(t, prompt, words)` takes one Python token of the code (from the standard `tokenize`
module) and returns True when the prompt already contains it:

- a **name** that is one of the prompt's words, and not a Python keyword (`if` in the prompt
  doesn't make `if` in the code a copy: it's English);
- a **number** that is one of the prompt's words (`10` in "above 10" grounds `10`, but not the `1`
  inside it);
- a **string** literal longer than `''` whose exact text, quotes included, is in the prompt.

Everything else, operators included, is False. `words` is the set of the prompt's words, already
computed for you.

```bash
uv run pytest tests/test_07_think.py -k grounded
```

<details><summary>Solution</summary>

```python
def grounded(t, prompt, words):
    if t.type == tokenize.NAME:
        return t.string in words and not keyword.iskeyword(t.string)
    if t.type == tokenize.NUMBER:
        return t.string in words
    if t.type == tokenize.STRING:
        return len(t.string) > 2 and t.string in prompt
    return False
```

</details>

Then all the design tests pass:

```bash
uv run pytest tests/test_07_think.py -k "design or plan or think_text"
```

## Step 3: the first try, the plan

The plan came first: it looks the most like "thinking". We packed every row with it, trained the
lesson 5 recipe (3 epochs) and wrote down a prediction before scoring. You don't need to train this
one: it's 2½ hours to watch an idea lose. The commands are in
[the extras](#do-you-want-to-fail-or-underperform) if you want to watch anyway. Our prediction:

> HumanEval `/think` 4 to 10 of 164, MBPP-500 35 to 60 of 500. No collapse: the span is trained now.

The result:

| 3 epochs, same recipe | HumanEval-164 | MBPP-500 |
|---|:-:|:-:|
| `none`, `/no_think` | 8 | 49 |
| `plan`, `/think` | 1 | 34 |

No collapse this time: the span is closed in 475 of 500 MBPP answers and the `name:` line is
right in 426 of them. And yet 15 fewer MBPP tasks pass. Both predictions were wrong, and the run alone doesn't say
why. You need a measure that does.

## Step 4: the measure, nats per row

A span built from P and C adds no information: everything in it was already in the prompt or in the
code. It can only split "prompt → code" into two steps the model finds easier. So the fair measure
is the cost of the **whole** job, span and answer together:

```
cost = −log p(span | prompt) − log p(answer | prompt, span)
```

summed over the tokens of a row (in nats: natural log) and averaged over validation rows. With an
empty span it's just −log p(answer | prompt). A design helps only if its total is lower.

This is the minimum description length view of the problem: the best interpolation is the one that
makes the whole row cheapest to write. Notice what it rules out: a span that makes the answer easy
but is itself expensive to write (the model has to write it at inference time, with nobody to
check it), and a span that is cheap but doesn't help the answer.

You can't use the training loss for this: `train.py` reports the loss **per token**, and every design
adds a different number of tokens. A long span full of easy tokens lowers the average and raises the
total.

### Exercise 7.2: `nats` in `theog/cost.py`

`nats(model, ids, labels)` runs the model on one training sequence and returns two numbers: the
nats spent on the span and the nats spent on the answer. At each position i whose label isn't
`IGNORE`, the cost is −log p(labels[i]) under the model's softmax. Positions before the one holding
`<|Answer|>` predict the span (and the tokens that close it); that position and every later one
predict the answer.

```bash
uv run pytest tests/test_07_think.py -k "nats or costs"
```

<details><summary>Solution</summary>

```python
@torch.no_grad()
def nats(model, ids, labels):
    device = next(model.parameters()).device
    lp = torch.log_softmax(model(torch.tensor([ids], device=device))[0].float(), -1)
    a = ids.index(ANSWER)
    span = answer = 0.0
    for i, y in enumerate(labels):
        if y == IGNORE:
            continue
        if i < a:
            span -= float(lp[i, y])
        else:
            answer -= float(lp[i, y])
    return span, answer
```

</details>

Now score your lesson 5 model on 1,000 validation rows:

```bash
uv run python -m theog.cost --ckpt runs/laptop/final.pt --tokenizer data/tokenizer.json --think none \
    --fit none,sig,shared,skeleton,plan,pause16
```

`--fit` picks the rows: the first 1,000 that fit in 512 tokens in every design listed, so every
table in this lesson scores the same rows. Here's your number next to our plan run:

| 3 epochs | span | answer | total, nats per row |
|---|:-:|:-:|:-:|
| `none` | 0 | 38.5 | **38.5** |
| `plan` | 23.7 | 29.8 | 53.5 |

There's the reason. The plan does make the answer cheaper, by 8.7 nats. But writing the plan costs
23.7, almost three times what it saves. Most of the plan is the code with its details removed, so
writing it is about as hard as writing the code. The score in step 3 was this table, measured
another way.

## Your experiment: `psr`

That's the wall: if the model has to write the span, the span costs about what it saves. So build a
span the model never has to write. `psr` fills it with two things the harness can work out from the
prompt alone, before the model writes a token:

- **the signature**, read from the prompt: its first `assert`, or its `def` line. For "Write a
  function `count_pos`… `assert count_pos([1,-2,3]) == 2`" that's `def count_pos(1 arg) -> int`.
- **a neighbour**: `similar task:` and the code of the training task whose prompt shares the most
  rare words with this one (`theog/retrieve.py`). It searches the training split only, never
  returns the row itself, and on the benchmarks the neighbour is never the reference solution (0 of
  164, 0 of 500).

Neither uses the answer, so at inference the harness writes the span in (`theog.evaluate --design
psr`), and training puts no loss on it (`mask_prefill` in `theog/think.py`). The span costs the
model nothing. The only question is what it does to the answer.

**Write your prediction first.** Before you train anything, put three numbers and one rule in
`runs/think-psr/PREDICTION.md`: the answer cost against your `none`, the HumanEval score, the MBPP
score, and what result would make you call `psr` better. Now, not after the run. A prediction you
write after seeing the score is a caption.

Then build the index, pack and train:

```bash
uv run python -m theog.retrieve
uv run python -m theog.pack --rows data/rows.clean.jsonl --tokenizer data/tokenizer.json \
    --out data/packed_psr --seq-len 512 --think psr
uv run python -m theog.train --packed data/packed_psr --out runs/think-psr --preset laptop --epochs 3 \
    --batch-size 16 --accum 1 --lr 1e-3 --eval-every 1000 --eval-windows 200 --log-every 500
```

The index takes about 5 minutes. Training takes about 3½ hours, an hour more than lesson 5: the
span makes every row longer, so the same 3 passes are about 15,000 steps instead of 10,885.

Score it the way you scored your lesson 5 model, then compare the two task by task:

```bash
uv run python -m theog.cost --ckpt runs/think-psr/final.pt --tokenizer data/tokenizer.json --think psr \
    --fit none,sig,shared,skeleton,plan,pause16
uv run python -m theog.evaluate --ckpt runs/think-psr/final.pt --tokenizer data/tokenizer.json \
    --design psr --dataset humaneval --out results/
uv run python -m theog.evaluate --ckpt runs/think-psr/final.pt --tokenizer data/tokenizer.json \
    --design psr --dataset mbpp500 --out results/
uv run python -m theog.compare results/laptop_humaneval_no_think_t0.0_*.jsonl results/think-psr_humaneval_psr_t0.0_*.jsonl
uv run python -m theog.compare results/laptop_mbpp500_no_think_t0.0_*.jsonl results/think-psr_mbpp500_psr_t0.0_*.jsonl
```

Read your two benchmarks together. If one moves and the other goes the opposite way, don't shrug:
open the answers and find out why.

Got your numbers? Good. Only now read on: the next section has ours.

## What we found: twelve designs

One design is an anecdote. So we ran twelve. [The lab notebook](lab-notebook.md) has the full
write-up, in its steps 5 to 7. Here's what it found, in three moves:

- **Step 5, the screen.** Five spans built from the prompt and the code, half an epoch each. Only
  `sig` (the function's signature) came in under the empty span, by 2%.
- **Step 6, check the winner.** At 3 epochs `sig` lost, 41.7 nats against 38.5. And a second seed
  of the screen flipped the order: the screen can kill big losers, not rank close ones.
- **Step 7, spans the model doesn't pay for.** Seven more designs that dodge the wall: pause
  tokens, outline-then-fill, a signature or a neighbor's code written in for free by the harness,
  and latent vectors instead of words.

Every 3-epoch run, seed 42, same 1,000 rows:

| design | the span | **total, nats** | HumanEval-164 | MBPP-500 |
|---|---|:-:|:-:|:-:|
| `none` | empty | **38.5** | 8 | 49 |
| `pause4` | four fixed tokens | **38.7** | 8 | 49 |
| `retr` | the closest training task's code, for free | **38.8** | 12 | 46 |
| `psig` | the signature, read from the prompt, for free | **38.9** | 8 | 53 |
| `c2f1` | the outline, then the fills | **39.1** | 2 | 47 |
| `psr` | `psig`, then `retr` | **40.0** | 14 | 56 |
| `sig` | the signature, written by the model | **41.7** | 2 | 53 |
| `plan` | a plan, written by the model | **53.5** | 1 | 34 |
| `lat8` | 8 vectors trained on the answer | **94.4** | 0 | 0 |

`lat8p`, the leak-free version of `lat8`, was stopped at step 4,000: swap its vectors for another
prompt's and the cost moves by 0.1 nats. Vectors that carry nothing are pause tokens with extra
steps.

Nothing beats the empty span on cost. The four closest land within 0.6 nats of it, and a second
seed of `none` lands 0.7 away from the first. So: noise. One row looks different, though. `psr`
costs more and passes the most tasks on *both* benchmarks. Real, or lucky?

## The seed check: is `psr` real?

One seed of `psr` beat `none` on both benchmarks. That's the first time any design did. So before
writing a word about it, I trained both again with seed 7, and wrote the bar down before the run:
`psr` counts only if it reaches HumanEval 6 or more and MBPP 58 or more against `none` at seed 7.

| 3 epochs | cost, seed 42 | seed 7 | HumanEval, seed 42 | seed 7 | MBPP, seed 42 | seed 7 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `none` | 38.5 | 37.8 | 8 | 3 | 49 | 58 |
| `psr` | 40.0 | 39.9 | 14 | 16 | 56 | 57 |

Look at `none` first. Same data, same recipe, only the seed changes, and HumanEval goes from 8 to 3,
MBPP from 49 to 58. Tempting to call that "the seed noise" and use it as a ruler. Don't: run the
paired test from lesson 6 instead. Seed 42 against seed 7 splits 6 to 1 on HumanEval (p = 0.125)
and 26 to 35 on MBPP (p = 0.31). Neither is a real difference, which is what you'd hope for the same
recipe. The single-seed gaps earlier in this lesson get the same treatment: `psig`'s +4 on MBPP
against `none` is a 23 to 19 split (p = 0.64), `retr`'s +4 on HumanEval a 10 to 6 split (p = 0.45).
Not results. I'd have bet those were small hints. The test says they're coin flips.

Now `psr`, against `none` at the same seed:

| `none` → `psr` | HumanEval, only `none` / only `psr` | p | MBPP, only `none` / only `psr` | p |
|---|:-:|:-:|:-:|:-:|
| seed 42 | 3 / 9 | 0.146 | 24 / 31 | 0.42 |
| seed 7 | 3 / 16 | 0.004 | 31 / 30 | 1.0 |

- **Cost: worse at both seeds**, by 1.5 and 2.1 nats. That's real.
- **HumanEval: better at both seeds, and real at seed 7.** 16 to 3, and the two sets don't
  overlap at all: every task `none` solves at seed 7, `psr` misses, and the other way round 16
  times. At seed 42 it points the same way, 9 to 3, but not by enough on its own. And `psr` repeats
  itself: 7 HumanEval tasks pass at both of its seeds, against 2 for `none`.
- **MBPP: a tie.** +7 at seed 42, −1 at seed 7, and the test calls both coin flips. It missed my
  bar by one task.

So by the rule I wrote before the run, `psr` doesn't clear it. One task short, and I'm not moving the
bar after the fact. That's the whole point of writing it first. But the HumanEval gain held at two
seeds, passed the paired test at one of them, and nothing else in this lesson did either.

### Your run against ours

Put your numbers in the table: cost, HumanEval, MBPP, and your two paired tests against your own
`none`. Ours, at seed 42 and seed 7: cost 40.0 and 39.9, HumanEval 14 and 16, MBPP 56 and 57. And
our prediction, written before our first `psr` run: cost 38.6 to 39.0, HumanEval 9 to 13, MBPP 47
to 53. Wrong on all three. The cost came in worse and both benchmarks better.

Then answer three questions in your prediction file, under what you wrote before:

1. Which of your numbers fell outside your range, and which way?
2. Does your HumanEval gain pass the paired test? Does your MBPP one?
3. By your own rule, written before the run, is `psr` better? Not by the rule you'd like now.

Want to know if your win is real? Train `psr` again with `--seed 7` and a new `--out`, and run the
paired test once more. That's how ours got its second row.

## What we conclude

No span beats the empty span on cost on this model: not text built from the code (steps 3 to 6),
not a reordering, not free tokens from the prompt, not a neighbor's code, not latent vectors. The
empty span, theOG's pause, is still the best design we found at 8M parameters by that measure. One
design, `psr`, writes code that passes more HumanEval tests at two seeds while its cost is worse.
Cost measures how well the model predicts *the* reference solution; the benchmarks measure whether
it writes *a* working one. At 8M parameters, those two disagree. Every text design moves
work from the answer to the span, and the span costs more to write than it saves; the designs that
cost nothing to write save nothing either, beyond the noise. That's a result about this model size,
not about think spans: the question for a bigger model is whether its span cost falls faster than
its answer cost, and `theog.cost` measures exactly that.

## The method

Why train twelve designs when the theory already tells you which one should win? Because the
theory was wrong seven times out of seven in step 7 alone. The notebook has
[the scoreboard](lab-notebook.md#why-we-ran-them-all), argument against result. The arguments
weren't stupid. They were blind: none could see that only 9 of 95,483 training prompts look like
HumanEval, or that a small model ignores any input it can live without. Every one of those showed
up the same way: a number that didn't fit the story.

What keeps the numbers honest:

- The prediction goes in writing before the run. A theory that can't lose isn't a theory.
- Same rows, same seed, same scoring, every time.
- The benchmarks and the dataset don't change. Rewrite a benchmark prompt and you're measuring your
  edit, not the model.
- Every number gets checked against its log. Mine included: I once pinned two validation losses
  on the wrong runs, and only rereading each log one by one caught it.

Theory tells you what to try. The data tells you what's true.

## What it did for theOG

At 50M parameters and 2.49B tokens, theOG's empty span starts to matter. theOG-50M v3, from the
[model card](https://huggingface.co/DrunkkToys/theOG-50M), PyTorch checkpoint at step 346,000,
greedy:

| mode | graded-30 | HumanEval-164 | MBPP-500 |
|---|:-:|:-:|:-:|
| `/no_think` | 26/30 | 42/164 | 91/500 |
| `/think` | 28/30 | 41/164 | 93/500 |
| either mode (a task counts if it passes in one of them) | 28/30 | 42/164 | 96/500 |

The two modes solve partly different tasks, with nothing written in the span. Whether a span with
text in it beats that at 50M is the experiment this lesson's tools let you run.

## Do you want to fail or underperform?

Every variation we discarded on the way to `psr` is here, so you can run it yourself. Pick one and
it will do one of two things: fail, worse than the empty span, or underperform, a tie that costs you
3 hours to find out. That's the point. Knowing why a reasonable idea loses is worth more than one more
row in a table, and you only really know it once you've watched it lose on your own machine.

Most people will skip this, and that's fine. Each run has its prediction, its commands and its
result in [the lab notebook](lab-notebook.md).

Each row gives you a tip, not the result. Write your prediction, run it, find out why it loses.
Our numbers are folded under the table: open them after, not before. Failing on your own teaches
you more than reading that we did.

| variation | the idea | tip | time on an M2 | where |
|---|---|---|---|---|
| `plan` | write a plan of the code first | put the span's cost next to what it saves on the answer | about 2½ hours | below |
| `shared`, `skeleton` | write the tokens the prompt already has, then the control flow | count how much of the span is copied and how much the model has to invent | about 25 minutes each | [step 5](lab-notebook.md#step-5-the-screen) |
| `sig` | write the signature first | it wins at half an epoch. Does it still win at 3? On HumanEval, read the names it writes | about 2½ hours | [step 6](lab-notebook.md#step-6-check-the-winner) |
| `c2f1` | the outline, then the fills | add the two costs up and compare with writing the code top to bottom | about 2½ hours | [step 7](lab-notebook.md#step-7-spans-the-model-doesnt-pay-for) |
| `lat8` | 8 vectors instead of words | ask what the vectors are built from in training, and what's there at inference | about 3 hours | [step 7](lab-notebook.md#step-7-spans-the-model-doesnt-pay-for) |
| `pause4` | four fixed tokens to compute in | look at *which* tasks pass, not how many: `theog.compare --show` | about 3 hours | [step 7](lab-notebook.md#step-7-spans-the-model-doesnt-pay-for) |
| `psig` | the signature, for free | a HumanEval prompt has no `assert`. Where does the name come from? | about 3 hours | [step 7](lab-notebook.md#step-7-spans-the-model-doesnt-pay-for) |
| `retr` | a neighbour's code, for free | read the neighbour on the tasks it lost: does it help or mislead? | about 3½ hours | [step 7](lab-notebook.md#step-7-spans-the-model-doesnt-pay-for) |

<details><summary>Our results (cost in nats / HumanEval / MBPP; <code>none</code> is 38.5 / 8 / 49)</summary>

| variation | ours | verdict |
|---|---|---|
| `plan` | 53.5 / 1 / 34 | fail |
| `shared`, `skeleton` | lost the screen by 28 and 34 nats | fail |
| `sig` | 41.7 / 2 / 53 at 3 epochs | fail |
| `c2f1` | 39.1 / 2 / 47 | fail |
| `lat8` | 94.4 / 0 / 0 | fail |
| `pause4` | 38.7 / 8 / 49 | underperform: a tie |
| `psig` | 38.9 / 8 / 53 | underperform: a tie |
| `retr` | 38.8 / 12 / 46 | underperform: a tie |

`psig` and `retr` are the two halves of `psr`: neither won alone.

</details>

The same treatment for the attention, seven changes to the model itself, is
[lesson 4, step 10](../04-model/10-attention-experiments.md). The span and the attention together,
the first thing in this course to clear both benchmarks at three seeds, is
[step 12](../04-model/12-think-and-attention.md).

The `plan` run from step 3:

```bash
uv run python -m theog.pack --rows data/rows.clean.jsonl --tokenizer data/tokenizer.json \
    --out data/packed_plan --seq-len 512 --think plan
uv run python -m theog.train --packed data/packed_plan --out runs/think-plan --preset laptop --epochs 3 \
    --batch-size 16 --accum 1 --lr 1e-3 --eval-every 1000 --eval-windows 200 --log-every 500
uv run python -m theog.cost --ckpt runs/think-plan/final.pt --tokenizer data/tokenizer.json --think plan \
    --fit none,sig,shared,skeleton,plan,pause16
uv run python -m theog.evaluate --ckpt runs/think-plan/final.pt --tokenizer data/tokenizer.json \
    --think --dataset humaneval --out results/
uv run python -m theog.evaluate --ckpt runs/think-plan/final.pt --tokenizer data/tokenizer.json \
    --think --dataset mbpp500 --out results/
```

## Going further

1. Why can't the span be written by hand, or by a bigger model, for each training row?
2. `shared` drops a token when it repeats the one before it (`count_char_freq s`, not
   `count_char_freq s s`). What would the span cost if it didn't?
3. Design your own f(P, C) and add it to `DESIGNS`. Before you train it, write down what you
   expect its span and answer costs to be, then run [the screen](lab-notebook.md#step-5-the-screen). Ideas: `sig` plus the first
   assert of the prompt; the names of the variables the code introduces; `sig` on MBPP and `none`
   on everything else. `theog.cost` scores the rows that fit in the five lesson designs. If yours
   is longer, pass `--fit none,sig,shared,skeleton,plan,<yours>` to every cost command you
   compare, so they all score the same rows.
4. The screen ran half an epoch per design. What could change at 3 epochs, and what would you need
   to see to believe it?

<details><summary>Answers</summary>

1. The model writes the span itself at inference time, from the prompt alone, so it has to learn
   one rule that works on every row. A rule applied by a program is exactly that. Hand-written
   spans aren't consistent across 95k rows, and spans written by a bigger model teach the small one
   to imitate the bigger one (distillation): a different experiment with a different question.
2. More, almost certainly: every repeat is a token the model has to decide to write and to stop
   writing. The degenerate `count 1 count 1 …` answer is that decision going wrong.
3. Your own result. Keep the prediction file next to the run (`runs/<name>/PREDICTION.md`) and
   compare after.
4. At 3 epochs both span and answer get cheaper, and not by the same amount for every design. A
   design that loses at half an epoch can win at 3 if its span cost falls faster than `none`'s
   answer cost, and a winner can lose. To believe a win you need the full-cost table at 3 epochs, on
   the same rows, at more than one seed, and a benchmark difference the paired test believes
   (lesson 6, [Is B really better than A?](../06-evaluation/README.md#is-b-really-better-than-a)). The lab notebook's step 6 ran the 3-epoch check for `sig`: it lost.

</details>

You know how to test an idea now, and you've tested one. Next: [lesson 8, export](../08-export/README.md),
where your lesson 5 model becomes a file you run in Ollama.
