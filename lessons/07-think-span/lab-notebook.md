# Lesson 7 lab notebook: steps 5 to 7

This is the long version of [lesson 7](README.md): the eleven designs after the plan, with every
command, every prediction written before the run, and every table. You don't need it to move on.
Read it when you want to see what an experiment looks like from the inside, dead ends included.

Steps 1 to 4, the two exercises and the cost measure are in the lesson. Everything here uses them.

| step | what it needs | time on an M2 |
|---|---|---|
| 5 | one half-epoch run per design | about 25 minutes each |
| 6 | one 3-epoch run and two half-epoch runs | about 2½ hours, plus 25 minutes each |
| 7 | one 3-epoch run per design | 2½ to 4 hours each (`lat8p` about 6) |

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

(For `none`, pack and evaluate without `--think`.)

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
100: even a 4 to 0 split can't pass lesson 6's paired test (p = 0.125), so the cost column is the
result.

`plan` is the interesting loser. Once its span is written, the answer is by far the cheapest (61
nats against 94), and it passes the most MBPP tasks of the five. But the span costs 42, more than it
saves. Compare with the 3-epoch table in the lesson's step 4: there the plan's total was 15 nats above `none`'s,
here 9. More training made the plain answer cheaper faster than it made the plan cheaper.

What the prediction got right and wrong:

- **Right:** `sig` has the cheapest span, and it wins the screen (step 6 checks that win). **Wrong:** `plan` is not the most
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

## Step 6: check the winner

A 2% lead from one run at one seed is a claim, not a result. Two checks: the 3-epoch run that
the lesson's question 4 asks for, and the screen again at a second seed. Prediction for the 3-epoch run,
written before training:

> `sig` costs 36 to 40 nats per row, at or below `none`'s 38.5. HumanEval 5 to 12 of 164, MBPP-500
> 40 to 58 of 500. `sig` wins only if its cost is below 38.5 and it passes several more tasks than
> `none` on both benchmarks.

```bash
uv run python -m theog.train --packed data/packed_sig --out runs/think-sig --preset laptop --epochs 3 \
    --batch-size 16 --accum 1 --lr 1e-3 --eval-every 1000 --eval-windows 200 --log-every 500
uv run python -m theog.cost --ckpt runs/think-sig/final.pt --tokenizer data/tokenizer.json --think sig
uv run python -m theog.evaluate --ckpt runs/think-sig/final.pt --tokenizer data/tokenizer.json \
    --think --dataset humaneval --out runs/think-sig/he
uv run python -m theog.evaluate --ckpt runs/think-sig/final.pt --tokenizer data/tokenizer.json \
    --think --dataset mbpp500 --out runs/think-sig/mbpp
```

| 3 epochs, same recipe | span | answer | **total** | HumanEval-164 | MBPP-500 |
|---|:-:|:-:|:-:|:-:|:-:|
| `none`, `/no_think` | 0 | 38.5 | **38.5** | 8 | 49 |
| `sig`, `/think` | 4.8 | 36.9 | **41.7** | 2 | 53 |
| `plan`, `/think` | 23.7 | 29.8 | **53.5** | 1 | 34 |

**At 3 epochs `sig` loses to the empty span**, by 3.2 nats per row (8%). Its span got cheap (4.8
nats, from 10.3), but it now saves only 1.6 on the answer. It is the same pattern as `plan`: more
training made the plain answer cheaper faster than it made span plus answer cheaper. MBPP is 4
tasks better, a 30 to 26 split (p = 0.69): a coin flip. HumanEval is 6 worse, 7 to 1 (p = 0.07), for a reason you can read in the answers: the
span is closed in 151 of 164, but its `def` line has the right name in only 58. HumanEval prompts
are code, a `def` and a docstring, while almost every training prompt names the function in
English, so the model copies the name badly. On MBPP the name is right in 424 of 500.

The second check reruns the screen for `none` and `sig` with `--seed 7` on every `theog.train`
command, and nothing else changed:

| 0.5 epoch, total nats per row | seed 42 (step 5) | seed 7 |
|---|:-:|:-:|
| `none` | 94.0 | 93.0 |
| `sig` | 92.1 | 94.1 |

The order flips. We expected two seeds of the same design to land about 1 nat apart; they land 1
to 2 apart, as much as the gap the screen was measuring. So the screen can rule out the big losers
(`shared` and `skeleton` were 28 to 34 nats behind) but it can't rank designs a few nats apart. For
that you need several seeds, or longer runs.

## Step 7: spans the model doesn't pay for

Steps 3 to 6 all hit the same wall. If the span is a function of P and C, writing it can't be
cheaper than writing C straight away: at best the work is the same, split in two. So we tried
seven designs that get around the wall in different ways. Every run is the lesson 5 recipe, 3
epochs, seed 42, scored on `final.pt`, on the same 1,000 rows as every table in the lesson. Before each
run we wrote the prediction down (`runs/<name>/PREDICTION.md`).

| design | the span | the idea |
|---|---|---|
| `pause4` | `~~~~` | four fixed tokens: positions to compute in, nothing to say ([Goyal et al., 2023](https://arxiv.org/abs/2310.02226)) |
| `c2f1` | the code with every block deeper than one indent replaced by `...` | coarse to fine: the outline first, then the fills, nothing written twice |
| `psig` | `def count_pos(1 arg) -> int` | the signature, built from the **prompt** alone (its first `assert`, or its `def` line) |
| `retr` | `similar task:` and the code of the closest **training** task | retrieval: knowledge from outside the weights ([RETRO](https://arxiv.org/abs/2112.04426), [kNN-LM](https://arxiv.org/abs/1911.00172)) |
| `lat8`, `lat8p` | 8 vectors, not tokens | latent interpolation: points on the line from the prompt to the code ([Coconut](https://arxiv.org/abs/2412.06769)) |
| `psr` | `psig`'s signature, then `retr`'s neighbor | the two half-winners in one span |

`psig`, `retr` and `psr` are built without the answer, so at inference the harness writes them into the
span for free: training puts no loss on them (`mask_prefill` in `theog/think.py`), and
`theog.evaluate --design` prefills them. `retr` searches only the training split, never returns the
row itself, and on the benchmarks its neighbor is never the reference solution (0 of 164, 0 of
500).

`lat8` puts the vector (1 − a)·mean(embed(prompt)) + a·mean(embed(code)) at slot k, with a = k/9,
centered on a running mean and scaled to a token embedding's norm. A small head learns to predict
each vector from the hidden state before it (`theog/latent.py`). At scoring and benchmark time the
model writes the 8 vectors itself: nothing from the answer goes in.

To run one (retrieval needs its index first):

```bash
uv run python -m theog.retrieve
uv run python -m theog.pack --rows data/rows.clean.jsonl --tokenizer data/tokenizer.json \
    --out data/packed_retr --seq-len 512 --think retr
uv run python -m theog.train --packed data/packed_retr --out runs/long/retr_s42 --preset laptop --epochs 3 \
    --batch-size 16 --accum 1 --lr 1e-3 --eval-every 1000 --eval-windows 200 --log-every 500 --seed 42
uv run python -m theog.cost --ckpt runs/long/retr_s42/final.pt --tokenizer data/tokenizer.json --think retr \
    --fit none,sig,shared,skeleton,plan,pause16
uv run python -m theog.evaluate --ckpt runs/long/retr_s42/final.pt --tokenizer data/tokenizer.json \
    --design retr --dataset mbpp500 --out runs/long/retr_s42/bench
```

The `--fit` list keeps the same 1,000 rows as the tables above. What we got:

| 3 epochs, seed 42 | span | answer | **total** | HumanEval-164 | MBPP-500 |
|---|:-:|:-:|:-:|:-:|:-:|
| `none` | 0 | 38.5 | **38.5** | 8 | 49 |
| `pause4` | 0 | 38.7 | **38.7** | 8 | 49 |
| `c2f1` | 25.6 | 13.5 | **39.1** | 2 | 47 |
| `psig` | 0 | 38.9 | **38.9** | 8 | 53 |
| `retr` | 0 | 38.8 | **38.8** | 12 | 46 |
| `lat8` | 0 | 94.4 | **94.4** | 0 | 0 |
| `psr` | 0 | 40.0 | **40.0** | 14 | 56 |

None of them beats the empty span on cost. `pause4`, `c2f1`, `psig` and `retr` land within 0.6 nats of
`none`, inside the seed noise: a second seed of `none` scores 37.8, of `pause4` 38.3. What each one taught us:

- **`pause4` changes which tasks pass, not how many.** Same score as `none` on both benchmarks, but
  only 4 HumanEval and 21 MBPP tasks pass in both runs; 28 MBPP tasks pass only with `none` and 28
  only with `pause4`, and the code is identical on 2 of 164 and 13 of 500. We looked for a pattern
  (task length, nesting, loops in the reference solution) and found none: near-twins like the volume
  of a cube and of a cuboid fall on opposite sides. Two runs of the same size land on different
  tasks; that's the shakiness lesson 6's paired test is built for, seen task by task.
- **`c2f1` reorders the work and saves nothing.** The outline costs 25.6 nats and the fills 13.5:
  writing the code in a different order costs about the same as writing it top to bottom.
- **`psig` is the only design ahead on one benchmark and level on the other**: +4 tasks on MBPP,
  equal on HumanEval. Its first HumanEval score was 1 of 164: HumanEval prompts have no `assert`,
  so the span was `def ?` on all 164 tasks, and the model, trained to trust the span, invented a
  name (`find_elements` where the test calls `has_close_elements`). Reading the prompt's `def` line
  instead gives the right name on 163 of 164, and the score comes back to 8. Four tasks over 500, a
  23 to 19 split: a coin flip (p = 0.64).
- **`retr` is the most interesting loser.** It is the only design that brings in something the
  model doesn't have, and the prediction was 30 to 36 nats. It's 38.8. On HumanEval it passes 12,
  the best score in the table, and on MBPP 46, so +1 task over 664. The neighbor helps on some
  tasks and misleads on others.
- **`psr` loses on cost and wins on both benchmarks.** Wait, what? Its answers cost 1.5 nats more
  than `none`'s, outside the noise, and it's the best HumanEval and the best MBPP in the table. My guess:
  the answer it's scored on is the reference solution, and the neighbor pulls the model away from that
  exact text. The benchmarks don't care about the text: they run the tests. A different solution
  that passes is a pass. So which number do you believe? Neither, yet. See [the seed check](README.md#the-seed-check-is-psr-real) in the lesson.
- **`lat8` collapses, and the reason is in the training.** The model always trained on vectors
  built from the answer, so it learned to read the answer out of them. On 20 rows the answer costs
  14.9 nats with those vectors (a leak, a reference only), 19.4 with no vectors (`none`), and 65.6
  with the vectors the model writes itself, though they're close to the targets (cosine 0.76). The
  model never saw its own vectors in training. It's exposure bias, in one number.

`lat8p` is the fix: it never trains on the targets. A first pass reads the prompt and 8 placeholder
slots and predicts all 8 vectors at once; a second pass reads those vectors and writes the code. The
token loss is on the second pass and flows back through the vectors; the targets only enter as a
cosine loss. Inference runs the same two passes, so training and inference see the same vectors
(`test_lat8p_trains_on_the_vectors_inference_writes`). Prediction, written before training: 37.5 to
41 nats; no answer leaks in, so at best the vectors are 8 positions of extra compute plus a summary
of the prompt.

The leak is gone, and so is everything else the vectors could carry. The test is to feed the model,
on the same 200 rows, either its own vectors or the vectors it wrote for a different prompt (`theog.cost --swap`), and
compare the answer cost:

| `lat8p` checkpoint | own vectors | another prompt's vectors | gap |
|---|:-:|:-:|:-:|
| step 1,000 | 112.2 | 112.7 | 0.5 |
| step 2,000 | 66.7 | 66.9 | 0.2 |
| step 4,000 | 41.8 | 41.9 | 0.1 |

The gap shrinks as training goes on: the model learns to ignore its vectors. At step 4,000 its
validation loss is 0.475, behind `none` (0.466, seed 7) and `pause4` (0.457). We stopped the run
there, since 7,000 more steps can't make vectors that carry nothing beat the pause. Swap your vectors for someone else's before you trust them: if the
cost doesn't move, they're pause tokens.

## Why we ran them all

Why train seven designs when the theory already tells you which one should win? Because the
theory was wrong seven times out of seven. I wrote every prediction down before the run, so I
couldn't quietly bend it afterwards. Here's the scoreboard:

| design | what the argument said | what the run said |
|---|---|---|
| `retr` | the closest training solution brings knowledge the model doesn't have: 30 to 36 nats | 38.8. Worse than the empty span |
| `c2f1` | outline first, fills after, nothing written twice: cheaper | 25.6 + 13.5 = 39.1. Same work, new order |
| `psig` | a signature read from the prompt is free and correct | 1 of 164 on HumanEval. No `assert` there, so the span said `def ?` on every task |
| `lat8` | vectors on the road from the prompt to the code. At step 1,000 its val loss beats `none`'s | 94.4 nats, 0 and 0. The val loss was a lie: the vectors carried the answer |
| `lat8p` | no leak, so the vectors carry a summary of the prompt | swap them for another prompt's: 41.8 vs 41.9. They carry nothing |
| `pause4` | extra positions to think in | same score as `none`, on different tasks: 28 MBPP tasks pass only with one, 28 only with the other |
| `psr` | the two half-winners together: cost 38.6 to 39.0, HumanEval 9 to 13, MBPP 47 to 53 | 40.0, 14 and 56. Wrong on all three, and the two benchmark misses went the good way |

The arguments weren't stupid. They were blind. None of them could see that only 9 of 95,483
training prompts look like HumanEval, that HumanEval prompts have no `assert`, that the target
vectors held the answer, or that a small model ignores any input it can live without. Every one of
those showed up the same way: a number that didn't fit the story.

One run proves nothing. Two runs of the same design with different seeds land 0.7 nats apart on
cost, and on the benchmarks they share far fewer passes than their totals suggest: 61 MBPP tasks
pass for only one of `none`'s two seeds. A design scored alone is an
anecdote; seven designs on the same 1,000 rows, same seed, same scoring, are an experiment, and a
second seed is what turns a lucky number into a result. On cost, none of them clears the noise.
That's a result too, just not the one anybody wanted.

Every dead design takes a whole family of ideas with it: text built from the code, reordering the
code, free tokens from the prompt, vectors instead of words. Measured, not argued. That's how you
get closer to the right answer: not by being clever, by being wrong faster. And sometimes two
half-results point somewhere. `psig` nudged MBPP, `retr` nudged HumanEval. Neither passed the paired test
on its own. Put together as `psr`, they gave the only HumanEval win that survived a second
seed. You only find that if you ran both, and then ran it twice.

What keeps the numbers honest:

- The prediction goes in writing before the run. A theory that can't lose isn't a theory.
- Same rows, same seed, same scoring, every time.
- The benchmarks and the dataset don't change. Rewrite a benchmark prompt and you're measuring your
  edit, not the model.
- Every number gets checked against its log. Mine included: I once pinned two validation losses
  on the wrong runs, and only rereading each log one by one caught it.

Theory tells you what to try. The data tells you what's true. At 8M parameters the empty span wins
on cost, `psr` wins on HumanEval, and not one of our arguments said either would.

Back to [lesson 7](README.md#the-seed-check-is-psr-real), the seed check.
