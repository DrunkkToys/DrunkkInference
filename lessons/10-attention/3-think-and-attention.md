# Step 3: think and attention, together

**After step 2.** Lesson 7 gave you the think span, `psr`. Steps 1 and 2 gave you the
gate. This step puts them together, and it's the first time in this course that something clears
both bars, HumanEval and MBPP, at three seeds out of three.

**You learn:** why one small model keeps trading HumanEval for MBPP, how to read that trade in
asserts instead of tasks, and how two models that each lose one benchmark beat both together.
**Time:** about 30 minutes to read. Redoing it is two trainings per seed: `psr` with the gate on
layer 0 (about 3½ hours) and `psr` at 6 epochs (about 7 hours). Plus about 40 minutes of
benchmarks for the pair.

## The bar

Lesson 7's `psr` at seed 42 scores 14 HumanEval and 56 MBPP. The bar for this step: more than 16
on HumanEval and more than 59 on MBPP, in the same run. Above every single run we had on one, and
not below on the other.

Step 2 ended with this line: *not one of these beats both of its parents on both benchmarks.* So
I kept going.

## The seesaw

Here are the runs that came closest, one model each:

| run | HumanEval | MBPP |
|---|:-:|:-:|
| `psr` + gate on layer 0 | **18** | 49 |
| `psr`, 6 epochs | 10 | **65** |
| `psr` + gate on layers 1 to 5 | 13 | 60 |
| `psr` | 14 | 56 |

Every time one goes up, the other goes down. You saw it in step 2. You'll see it in your own runs.
After a while it stops looking like luck.

Task counts hide it, though. A HumanEval task passes only if about 7 hidden asserts all pass. An
MBPP task needs 3. So count asserts instead: for each run, the hidden HumanEval asserts it passes
(of 1,073) and the MBPP asserts (of 1,500). Over the first 10 runs I scored on `psr`'s data:

- Spearman between the two: **−0.90** (p = 0.0003).
- The sum stays between **469 and 529**. Every one of the 10.

That's not a bug to fix. It's a budget. An 8M model has so much room, and every recipe spends it
somewhere else: more of the room on HumanEval's style, less on MBPP's, or the other way round.

So the question changes. Not "which recipe wins both?", but "how do you get more room?"

## One model, both changes

The obvious try first: the gate on layer 0 (HumanEval's side) and 6 epochs (MBPP's side) in one
model.

My prediction, written before the run: against `psr` at 6 epochs, HumanEval up, MBPP down.

| | HumanEval | MBPP | asserts (HE hidden + MBPP) |
|---|:-:|:-:|:-:|
| `psr`, 6 epochs | 10 | 65 | 498 |
| `psr` + gate on layer 0, 6 epochs | 13 | **76** | **605** |

Wrong on MBPP: it went up, not down. And 605 asserts is above the 529 ceiling. But HumanEval stays
at 13. It gets almost as many hidden asserts right as the HumanEval winner (293 against 295), only
spread over more tasks: 82 tasks where it passes some asserts and fails others. Closer. Not there.

## Two models, one answer

So don't pick. Take the HumanEval model and the MBPP model, and let them write each answer
together.

At every step, each model gives a log-probability to every token. Average the two, take the best
token, feed it to both, repeat:

```
score(t) = ( log p_gate(t) + log p_6ep(t) ) / 2
next     = the t with the highest score
```

That's a geometric mean of the two distributions. A token only scores high if *both* models think
it's plausible. A token one of them rules out is vetoed, however much the other likes it.

Same prompt, same prefilled span, same extraction, same grading as every other number in this
course. Only the next-token scores change. In the code it's a dozen lines: `ensemble` in
`theog/evaluate.py`, behind `--ensemble`.

**For a five-year-old:** two kids do the same homework. One is great at puzzles and bad at sums.
The other is great at sums and bad at puzzles. Alone, each gets one bad grade. So they sit
together, and for every word they write, they whisper and pick the word they both like. Together,
two good grades.

## Did it work?

| seed | gate on layer 0 | 6 epochs | **the two together** |
|:-:|:-:|:-:|:-:|
| 42 | 18 / 49 | 10 / 65 | **21 / 78** |
| 7 | 12 / 58 | 19 / 66 | **24 / 73** |
| 14 | 13 / 46 | 16 / 67 | **21 / 76** |

HumanEval / MBPP. Bar: more than 16 / more than 59.

Three seeds, three passes. And look at the parts: at seed 42 the gate model is the HumanEval one,
at seed 7 it's the 6-epoch model. The roles flip with the seed. The pair doesn't care.

In asserts, HumanEval hidden + MBPP:

| seed | gate on layer 0 | 6 epochs | the two together |
|:-:|:-:|:-:|:-:|
| 42 | 529 | 498 | **615** |
| 7 | 495 | 564 | **636** |
| 14 | 509 | 548 | **647** |

The first 10 single runs never got past 529. Training longer helps a bit: `psr` at 6 epochs
reaches 564 at seed 7 and 548 at seed 14. The pair is above every single model at every seed.

Now the honest test: against the *better* of its two models, on each benchmark, task by task
(lesson 6's paired test), the three seeds added up:

| | only the pair solves | only the better model solves | p |
|---|:-:|:-:|:-:|
| HumanEval | 24 | 11 | 0.041 |
| MBPP | 84 | 55 | 0.017 |

One seed at a time, none of those differences passes the test. Three seeds together, both do.

## What I didn't do right

I picked this pair after I saw the seed 42 scores. That's choosing the winner after the race. So
seed 42 proves nothing on its own. Seeds 7 and 14 are the real check: the recipe was fixed, and the
seeds were new. They held.

And I tried to squeeze the pair into one model: train one 8M model to copy the pair's
log-probabilities on the same rows (distillation, half the loss on the real token, half on the
pair). 15 HumanEval, 53 MBPP. It kept the HumanEval side and lost the MBPP side. That one lives in
my lab, not in the course code.

## Why it's still not something you ship

- **It's two models.** 16.7M parameters at inference, twice the compute per token. Two 8M models
  beat one 8M model. Whether they beat one 16M model is a run nobody's done yet.
- **Lesson 8 exports one model.** Ollama runs one GGUF. And the gate has no slot in llama.cpp's
  `qwen3` layout anyway (step 2).

It's a result about *room*, not a product. The seesaw isn't a law of small models. It's what
happens when one set of weights has to cover two styles of task.

## Run it

Pack `psr` first if you haven't ([lesson 7](../07-think-span/README.md#your-experiment-psr)). Then
the two models, at the seed of your choice:

```bash
uv run python -m theog.train --packed data/packed_psr --out runs/psr-gate1 --preset laptop --epochs 3 \
    --batch-size 16 --accum 1 --lr 1e-3 --eval-every 1000 --eval-windows 200 --log-every 500 \
    --attn gate=1,gatemask=1
uv run python -m theog.train --packed data/packed_psr --out runs/psr-6ep --preset laptop --epochs 6 \
    --batch-size 16 --accum 1 --lr 1e-3 --eval-every 1000 --eval-windows 200 --log-every 500
```

`gatemask=1` keeps the gate on layer 0 only (one bit per layer). Add `--seed 7` to both for a
second seed. Then score each model with `--design psr` as in lesson 7, and the pair:

```bash
uv run python -m theog.evaluate --ckpt runs/psr-gate1/final.pt --ensemble runs/psr-6ep/final.pt \
    --tokenizer data/tokenizer.json --design psr --dataset humaneval --out results/
uv run python -m theog.evaluate --ckpt runs/psr-gate1/final.pt --ensemble runs/psr-6ep/final.pt \
    --tokenizer data/tokenizer.json --design psr --dataset mbpp500 --out results/
uv run python -m theog.compare results/psr-6ep_mbpp500_psr_t0.0_*.jsonl \
    results/psr-gate1+psr-6ep_mbpp500_psr_t0.0_*.jsonl
```

Write your prediction before the pair runs. Then compare the pair with each of its two models, on
both benchmarks.

## Try it

1. What does `--ensemble` give you if both checkpoints are the same file?
2. Why the mean of log-probabilities and not the mean of probabilities? What can one confident
   model do in each case?
3. Pair two `psr` models that differ only in seed. Write down what you expect before you run it.
4. The distilled model kept HumanEval and lost MBPP. Which number would you look at first to find
   out why?

<details><summary>Answers</summary>

1. The same model, token for token. The mean of a number with itself is the number. The test
   `test_ensemble_is_the_mean_of_log_probabilities` checks exactly that.
2. With log-probabilities, a token one model gives almost zero gets a very negative score, and the
   other model can't rescue it: both have to agree. With probabilities, one model at 0.9 beats
   anything the other says, and you mostly get the more confident model back. I ran only the
   log version. Run the other and tell me.
3. Your own result. Two seeds of the same recipe disagree less than two different recipes, so my
   guess is a smaller gain. A guess, until you run it.
4. Its MBPP asserts against the 6-epoch teacher's, then which MBPP tasks the pair solves and the
   student doesn't. Read those answers before you change anything.

</details>

That's the course. What's still open, including one model that does what this pair does, is in
[lesson 9's last section](../09-publish/README.md#where-to-go-from-here). Back to the
[start](../../README.md).
