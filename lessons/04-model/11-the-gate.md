# Step 11: the gate

**An extra, after step 10.** No new code and no new training: you read the one change in step 10
that beat the baseline and find out what it actually did. If you trained `gate` yourself in step
10, run the commands on yours. If not, read mine.

**You learn:** what an output gate on attention computes, how to look inside it, and why a result
that points the right way on both benchmarks still isn't proven. **Time:** about 30 minutes.

## Why it gets its own step

Every other change in step 10 is about size: fewer key/value heads, more heads, smaller heads.
They move parameters around. The gate doesn't. It adds almost nothing and changes *what flows*:
how much of each head reaches the rest of the model, decided token by token.

It's also the only change in step 10 that beat the baseline. One that wins deserves a closer look
than a row in a table.

## Why a small model should care

A big model can afford waste. It has dozens of layers, a head that does nothing is a rounding
error, and if one head carries too much, another learns to back it up.

Yours has 6 layers and 24 heads. Step 9 showed what that means: one head in layer 0 held the
whole model up, and some layer ended up nearly spare. No room for waste, and no backup.

The gate is cheap capacity. 0.07% more parameters, and the model gets a way to turn a head down
when it isn't helping, instead of letting it add noise to every token. If that's worth 2 nats
anywhere, it's worth it on the model that can't afford anything else.

That's the argument. Below is what the run says.

## What the gate is

Two lines in `theog/attnlab.py`. Each block gets one small linear layer that reads the block's
input and gives every head a number per token:

```python
self.gate = nn.Linear(cfg.hidden, cfg.heads, bias=True)
...
out = out * torch.sigmoid(self.gate(x)).transpose(1, 2)[..., None]
```

The sigmoid puts that number between 0 and 1. At 1 the head's output goes through untouched. At 0
it's shut. The model learns, token by token, how much of each head to let through.

The cost: `hidden + 1` weights per head, 257 × 4 heads × 6 blocks = 6,168. That's 0.07% of the
model. Remember that number when you read the results.

## What it did

Same data, same seed, same recipe as the lesson 5 baseline. One change.

| | cost | HumanEval | MBPP |
|---|:-:|:-:|:-:|
| `none` (lesson 5) | 38.5 | 8 | 49 |
| `gate` | 36.5 | 12 | 59 |

Cost first: 2.0 nats better, and the seed noise is 0.7. That's real at one seed.

The benchmarks, task by task with the paired test (lesson 6): HumanEval 9 tasks only `gate`
solves, 5 only `none` solves, p = 0.42. MBPP 29 to 19, p = 0.19. Neither is significant.

But read them together. Both point the same way as the cost. Nothing breaks the pattern. That's
a candidate, not noise and not a result. My prediction was "a tie, nothing to fix". Wrong, the
good way.

## What it did inside

Here's the part the paper chart doesn't show you. Run the mean gate value over the answer
tokens of the same 100 rows step 9 uses:

| layer | h0 | h1 | h2 | h3 |
|---|:-:|:-:|:-:|:-:|
| 0 | 0.10 | 0.06 | 0.10 | 0.09 |
| 1 | 0.32 | 0.40 | 0.59 | 0.31 |
| 2 | 0.30 | 0.72 | 0.47 | 0.28 |
| 3 | 0.63 | 0.52 | 0.60 | 0.53 |
| 4 | 0.52 | 0.68 | 0.50 | 0.56 |
| 5 | 0.51 | 0.47 | 0.50 | 0.65 |

Layer 0 is nearly shut: about 0.1 on every head, and below 0.1 on more than half of the answer
tokens. Every other layer sits around half open.

Now remember step 9. In the baseline, layer 0 *was* the model: switch off head 3 and the cost went
up by +138 nats. So did the gate turn off the most important layer? Run the head table on `gate`:

| layer | h0 | h1 | h2 | h3 |
|---|:-:|:-:|:-:|:-:|
| 0 | +8.27 | +1.41 | +13.49 | +27.67 |
| 1 | +0.98 | +2.48 | +5.92 | +0.30 |
| 2 | +14.13 | +26.14 | +1.04 | +0.66 |
| 3 | +6.46 | +0.74 | +1.20 | +1.05 |
| 4 | +1.69 | +1.86 | +1.62 | +1.76 |
| 5 | +5.73 | +1.94 | +0.82 | +10.38 |

Layer 0 still matters: +27.67 for h3. But it's no longer one head holding the whole model up. The
biggest head went from +138 to +28, and layer 2 now carries as much as layer 0 does. The work got
spread out.

So "the gate is 0.1" doesn't mean "layer 0 is 10% on". The `o` projection after it can learn to
be bigger and make up the difference. What the gate gives the model is a per-token volume knob:
turn layer 0 down most of the time, up when the token needs it. That's my reading. It fits both
tables. It's one seed.

Two more things from step 9's `mass` table, `none` against `gate`:

- **The sink.** Position 0 gets 1.0% in `none` and 0.3% in `gate`. My prediction said the gate had
  nothing to fix because the sink was already small. It shrank anyway, but 0.7% of attention
  can't be where 2 nats came from. Right fact, wrong conclusion.
- **Layer 1 reads the prompt more.** 10.9% of its attention on the prompt in `none`, 21.3% in
  `gate`. The "what did I just write" layer got less local.

```bash
uv run python -m theog.attnlab mass --ckpt runs/attn_gate/final.pt
uv run python -m theog.attnlab heads --ckpt runs/attn_gate/final.pt
```

## The gate and the think span

Why would the gate and lesson 7's span go together? `psr`'s span carries a neighbour: the code of
a similar training row. Lesson 4 step 9 showed the answer reads it (14% of its attention in
`retr`), and lesson 7 showed it helps about as often as it misleads. The gate can turn a head down
per token. So my bet: with a gate, the model learns to shut the neighbour off when it misleads.

The prediction, written before the run: cost 39.0–40.0, HumanEval 12–18, MBPP 50–58.

| | cost | HumanEval | MBPP |
|---|:-:|:-:|:-:|
| `psr` | 40.0 | 14 | 56 |
| `psr` + gate | 38.7 | 8 | 48 |

Cost: 1.3 nats better, beyond my range. Benchmarks: both below it. HumanEval 10 to 4 against the
gate (p = 0.18), MBPP 26 to 18 against it (p = 0.29). Not significant, but it breaks the pattern:
better cost, worse on both benchmarks, against what both parents said. Not seed noise until
something shows it is.

So did the gate learn to filter the neighbour? Test it directly. On 200 rows, score the answer
three ways: with its own span, with another row's span, and with an empty one.

| model | own span | another row's span | empty span |
|---|:-:|:-:|:-:|
| `psr` | 25.53 | 30.12 | 27.97 |
| `psr` + gate | 24.57 | 28.77 | 26.80 |

A wrong neighbour costs `psr` 4.6 nats, and the gated model 4.2. Almost the same. The gate's gain,
about 1 nat, is there with the right neighbour, a wrong one, or none at all. It has nothing to do
with the neighbour.

And the gate itself? At the answer tokens, layer 0 sits at 0.07–0.09 on every head, the other
layers 0.3–0.8. The same shape as the gate on `none`. It learned the same per-layer damping, not
a neighbour filter. My bet was wrong: the mechanism I wanted isn't the one the model learned.

What's still open:

- **Is it the seed?** One seed can't say whether the gate cancels `psr`'s HumanEval gain, or seed
  42 was unlucky. `psr` + gate at seed 7, against `psr` at seed 7 (16 / 57), would.
- **Which layers?** Gate only layers 1 to 5 on `psr`, leaving layer 0 plain. If the layer 0 damping
  is what cost `psr` its HumanEval, this keeps it. My prediction: 38.9–39.6 / 11–16 / 50–58. And
  the opposite, the gate on layer 0 only: 38.8–39.6 / 8–14 / 46–56.

Neither has run. When they do, the results land here, next to those predictions.

## Why it's a candidate, not a result

- **One seed.** Lesson 7's seed check flipped `psr`'s MBPP lead. Nothing here has had that check.
- **The benchmarks don't pass the paired test.** Pointing right is not proven.
- **It breaks on `psr`.** A change that really helps shouldn't hurt both benchmarks on the next
  model over.
- **You can't ship it.** llama.cpp's `qwen3` layout has no slot for the gate, so lesson 8's export
  refuses it.

Four reasons to keep it in the drawer. One reason to keep looking: it's the only change in this
course that beat the baseline on cost twice.

## Try it

1. Count the 6,168 yourself. Then for theOG-50M (10 layers, hidden 512, 4 heads).
2. Layer 0's gate is 0.1, and switching off its head 3 still costs +27.67. How can both be true?
3. Write down what you predict for `gate` at seed 7: cost, HumanEval, MBPP. Then decide whether
   that prediction is worth 2½ hours of your laptop.

<details><summary>Answers</summary>

1. Each head has `hidden` weights and a bias: 257 × 4 × 6 = 6,168. theOG: 513 × 4 × 10 = 20,520.
2. The gate scales the head's output; it doesn't decide how much that output matters. `o` can
   learn bigger weights for layer 0, so 0.1 of a big number is still a big number. And 0.1 is the
   average: on some tokens the gate opens, and switching the head off loses those too.
3. There's no right answer. The point is to write it before you run, so the run can tell you
   you're wrong.

</details>

Next: [lesson 8, export](../08-export/README.md). The best result in step 10 is the one the
export refuses. Find out why.
