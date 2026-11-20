# Lesson 6: the /think span, an experiment

**You write:** `grounded` in `theog/think.py` (exercise 6.1) and `nats` in `theog/cost.py`
(exercise 6.2). **Then you run:** an experiment that asks what to write inside the think span, and
how to tell if it helps. **Time:** about 30 minutes for the code.

The exercises and their tests are all you need for lesson 7. Every run below is optional: the
lesson gives its results, so you can read it straight through. If you do run them:

| step | what it needs | time on an M2 |
|---|---|---|
| 1 | your lesson 5 model | a minute (the benchmark score needs lesson 7) |
| 3 | a 3-epoch run with the plan | about 2½ hours |
| 4 | your lesson 5 model and the step 3 model | a few minutes |
| 5 | one half-epoch run per design | about 25 minutes each |

Commands that run `theog.evaluate` need lesson 7's `extract_code` and `run_tests`. Come back to
them after lesson 7. The experiment scores `final.pt`, the weights after the last step, so every
design is measured at the same budget. For the lesson 5 run that is the same model as `best.pt`.

This lesson is a real experiment, written up in the order it happened, wrong predictions included.
You can read it straight through, or rerun any step on your own model.

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
as it should, and then loses the thread. After lesson 7, score it on HumanEval in `/think` mode:

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

### Exercise 6.1: `grounded` in `theog/think.py`

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
uv run pytest tests/test_06_think.py -k grounded
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
uv run pytest tests/test_06_think.py -k "design or plan or think_text"
```

## Step 3: the first try, the plan

The plan came first: it looks the most like "thinking". We packed every row with it, trained the
lesson 5 recipe (3 epochs) and wrote down a prediction before scoring:

```bash
uv run python -m theog.pack --rows data/rows.clean.jsonl --tokenizer data/tokenizer.json \
    --out data/packed_plan --seq-len 512 --think plan
uv run python -m theog.train --packed data/packed_plan --out runs/think-plan --preset laptop --epochs 3 \
    --batch-size 16 --accum 1 --lr 1e-3 --eval-every 1000 --eval-windows 200 --log-every 500
```

`--seq-len 512` matches the laptop model, as in lesson 5. The scores, after lesson 7:

```bash
uv run python -m theog.evaluate --ckpt runs/think-plan/final.pt --tokenizer data/tokenizer.json \
    --think --dataset humaneval --out runs/think-plan/he
uv run python -m theog.evaluate --ckpt runs/think-plan/final.pt --tokenizer data/tokenizer.json \
    --think --dataset mbpp500 --out runs/think-plan/mbpp
```

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

### Exercise 6.2: `nats` in `theog/cost.py`

`nats(model, ids, labels)` runs the model on one training sequence and returns two numbers: the
nats spent on the span and the nats spent on the answer. At each position i whose label isn't
`IGNORE`, the cost is −log p(labels[i]) under the model's softmax. Positions before the one holding
`<|Answer|>` predict the span (and the tokens that close it); that position and every later one
predict the answer.

```bash
uv run pytest tests/test_06_think.py -k "nats or costs"
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

Now score the two 3-epoch models on the same 1,000 validation rows:

```bash
uv run python -m theog.cost --ckpt runs/laptop/final.pt --tokenizer data/tokenizer.json --think none
uv run python -m theog.cost --ckpt runs/think-plan/final.pt --tokenizer data/tokenizer.json --think plan
```

| 3 epochs | span | answer | total, nats per row |
|---|:-:|:-:|:-:|
| `none` | 0 | 38.5 | **38.5** |
| `plan` | 23.7 | 29.8 | 53.5 |

There's the reason. The plan does make the answer cheaper, by 8.7 nats. But writing the plan costs
23.7, almost three times what it saves. Most of the plan is the code with its details removed, so
writing it is about as hard as writing the code. The score in step 3 was this table, measured
another way.

## Step 5: the screen

So the question becomes: which f(P, C) gives the lowest total? A 3-epoch run per design is 2½ hours,
so we screened all five at a smaller budget: half an epoch each (about 25 minutes), same seed, same
1,000 validation rows (only rows that fit 512 tokens in every design), then MBPP's first 100 tasks
greedy, in the design's own mode. Prediction, written before the first run:

> `plan` stays the most expensive. `shared` and `sig` cost under 3 nats of span (they're copied from
> the prompt) and save at least that on the answer: total at or below `none`. `skeleton` in between.

To train one design yourself:

```bash
uv run python -m theog.pack --rows data/rows.clean.jsonl --tokenizer data/tokenizer.json \
    --out data/packed_sig --seq-len 512 --think sig
uv run python -m theog.train --packed data/packed_sig --out runs/screen/sig --preset laptop --epochs 0.5 \
    --batch-size 16 --accum 1 --lr 1e-3 --eval-every 1000 --eval-windows 200 --log-every 500
uv run python -m theog.cost --ckpt runs/screen/sig/final.pt --tokenizer data/tokenizer.json --think sig
uv run python -m theog.evaluate --ckpt runs/screen/sig/final.pt --tokenizer data/tokenizer.json \
    --think --dataset mbpp500 --limit 100 --out runs/screen/sig_mbpp
```

(For `none`, pack and evaluate without `--think`. The last command needs lesson 7.)

What we got:

| 0.5 epoch | span | answer | **total** | val loss per token | MBPP, first 100 |
|---|:-:|:-:|:-:|:-:|:-:|
| `none` | 0 | 94.0 | **94.0** | 0.82 | 1 |
| `sig` | 10.3 | 81.9 | **92.1** | 0.67 | 3 |
| `shared` | 33.4 | 88.5 | **121.9** | 0.91 | 1 |
| `skeleton` | 47.3 | 81.2 | **128.4** | 0.79 | 1 |
| `plan` | 42.0 | 61.1 | **103.1** | 0.47 | 5 |

**`sig` is the only design cheaper than the empty span**, by 1.9 nats per row, 2%. Every other
design costs more in total than it saves on the answer. On MBPP all five land between 1 and 5 of
100: at this budget that's noise (lesson 7, question 4), so the cost column is the result.

`plan` is the interesting loser. Once its span is written, the answer is by far the cheapest (61
nats against 94), and it passes the most MBPP tasks of the five. But the span costs 42, more than it
saves. Compare with the 3-epoch table in step 4: there the plan's total was 15 nats above `none`'s,
here 9. More training made the plain answer cheaper faster than it made the plan cheaper.

What the prediction got right and wrong:

- **Right:** `sig` has the cheapest span, and it is the winner. **Wrong:** `plan` is not the most
  expensive; `shared` and `skeleton` cost more. And no span costs under 3 nats: `sig`'s costs 10.

- **Copying is not free for a small model.** `shared` is made only of tokens that are in the
  prompt, and it still costs 33 nats. The model has to pick which prompt words appear in the code,
  in which order, and when to stop, and an 8M model is bad at all three. On 5 of the 100 MBPP
  tasks it wrote things like `count_common count 1 count 1 count 1 …` until generation hit its
  limit, and never closed the span.
- **Per-token loss points the wrong way.** `plan` has by far the lowest validation loss of all the
  runs, 0.47 against `none`'s 0.82, and yet costs 9 nats more per row. Its span adds many easy
  tokens, so the average drops while the sum grows. That's why step 4 measures the sum.
- **The cheapest span is the one the prompt almost spells out.** `sig` costs 10 nats: on MBPP the
  name and the argument count are in the test line. It is the only design that saves more on the
  answer (12.1 nats) than its span costs (10.3).

## What we conclude

At 8M parameters and half an epoch, the best interpolation between prompt and code we found is the
thinnest one: the signature, which the prompt nearly gives away. It wins by 2%, too little to see on
100 MBPP tasks. Everything that tells the model more about the code costs more to write than it
saves. That's a result about this model size, not about think spans: the question for a bigger
model is whether its span cost falls faster than its answer cost, and `theog.cost` measures exactly
that. Going further, question 4, says what it would take to settle it at 3 epochs.

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

## Going further

1. Why can't the span be written by hand, or by a bigger model, for each training row?
2. `shared` drops a token when it repeats the one before it (`count_char_freq s`, not
   `count_char_freq s s`). What would the span cost if it didn't?
3. Design your own f(P, C) and add it to `DESIGNS`. Before you train it, write down what you
   expect its span and answer costs to be, then run the screen. Ideas: `sig` plus the first
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
   answer cost. To believe it you need the full-cost table at 3 epochs, on the same rows, and a
   benchmark difference of several tasks in the same direction (lesson 7, question 4).

</details>

Next: [lesson 7, evaluation](../07-evaluation/README.md).
