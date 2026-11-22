# Step 9: reading the attention

**Come back here after lesson 5.** This step needs a trained model, and you don't have one yet.

You wrote attention in steps 3 to 5. You know what it computes. Do you know what your model
*does* with it? Me neither, until I looked. Everybody draws attention as arrows from a word to the
word it "means". Nice slides. Then you look at a real 8M model and most of those arrows point at
the last few tokens.

Two ways to look, both in `theog/attnlab.py`. No training, a couple of minutes each.

## Where does the answer look?

Take 100 rows. For every answer token, in every layer, add up its attention weights by region:
position 0, the prompt, the /think span, and the answer written so far. Average over heads,
tokens and rows.

```bash
uv run python -m theog.attnlab mass --ckpt runs/laptop/final.pt
```

My laptop model, trained in lesson 5:

```
layer   first  prompt  span  answer
  0     0.008  0.514  0.016  0.462
  1     0.005  0.109  0.022  0.864
  2     0.008  0.167  0.041  0.784
  3     0.015  0.382  0.033  0.570
  4     0.016  0.452  0.033  0.499
  5     0.007  0.389  0.035  0.569
  all   0.010  0.335  0.030  0.625
```

Read it like this:

- **About 60% of the attention goes to the answer itself.** Writing code is mostly reading the
  code you just wrote. Indentation, the open bracket, the variable from two lines up.
- **A third goes to the prompt.** That's the "understand the task" part, and layers 0, 4 and 5
  do most of it.
- **Layer 1 is local.** 86% on the answer so far. It's the "what did I just write" layer. Every
  model I checked has one, including the same model trained with another seed (83%).
- **Position 0 gets 1%.** Big models are famous for an *attention sink*: heads that dump half
  their weight on the first token because the softmax has to put it somewhere. QK-norm (step 5)
  keeps the scores small, and the sink mostly goes away. Remember this one, it matters in step 10.

Come back to this table after lesson 6, when the span designs mean something to you. Run it on
the lesson 6 models with `--think <design>` (or just read mine). Same tool, a different question:
does the answer actually read the span?

| model | first | prompt | span | answer |
|---|:-:|:-:|:-:|:-:|
| none | 0.010 | 0.335 | 0.030 | 0.625 |
| pause4 | 0.014 | 0.338 | 0.060 | 0.588 |
| retr | 0.009 | 0.284 | 0.144 | 0.563 |
| lat8p (stopped at step 4,000) | 0.012 | 0.361 | 0.037 | 0.590 |

`retr` puts 14% of its attention on the neighbour's code, mostly taken from the prompt. So it
didn't lose because the model ignored the neighbour. The model read it, and it helped about as
often as it misled.

`lat8p`'s 8 latent vectors get 3.7%. That's what `none`'s two empty span markers get. The swap
test in lesson 6 said the vectors carry nothing. The attention says the same thing from the other
side: the answer barely looks at them. Two tools, one verdict. That's when I start believing it.

## Which heads do the work?

Switch off one head (zero its slice of `o`'s input), score the answers again, put it back. Do it
for all 24 heads. The cost that goes up is the work that head was doing.

```bash
uv run python -m theog.attnlab heads --ckpt runs/laptop/final.pt
```

Mine, base cost 19.79 nats per row:

| layer | h0 | h1 | h2 | h3 |
|---|:-:|:-:|:-:|:-:|
| 0 | +57.57 | +38.36 | +16.17 | +138.10 |
| 1 | +7.41 | +4.40 | +0.91 | +13.00 |
| 2 | +2.26 | +1.77 | +4.80 | +2.46 |
| 3 | +0.28 | +0.55 | +0.85 | +0.94 |
| 4 | +1.92 | +0.59 | +1.72 | +1.20 |
| 5 | +3.63 | +3.48 | +0.43 | +1.13 |

Layer 0 head 3 alone: switch it off and the cost is 8 times higher. Layer 0 is the model's
foundation, and every one of its 4 heads matters. Layer 3 is the opposite: each head is under 5%
of the cost. It's close to spare.

So should you delete layer 3? I almost wrote that down. Then I ran the same table on the same
model trained with seed 7 instead of 42. Same data, same recipe, base cost 19.66:

| layer | h0 | h1 | h2 | h3 |
|---|:-:|:-:|:-:|:-:|
| 0 | +112.56 | +25.97 | +14.03 | +12.89 |
| 1 | +9.76 | +9.14 | +0.97 | +3.30 |
| 2 | +2.17 | +0.75 | +11.98 | +2.76 |
| 3 | +12.56 | +0.72 | +0.57 | +1.57 |
| 4 | +1.84 | +6.19 | +1.93 | +7.77 |
| 5 | +0.34 | +1.14 | +0.41 | +0.64 |

Layer 0 still carries the model. That part is real. But the big head is h0 now, not h3. And
layer 3 has a head worth +12.56: not spare at all. The near-spare layer moved to layer 5. One
run told me "layer 3 is useless". Two runs told me "some layer ends up mostly spare, and the seed
picks which". Very different sentences. Only one of them is true.

And even that's not a licence to cut. Zeroing one head at a time misses heads that back each
other up, and a model trained without them would arrange itself differently. This table tells you
where to *look*, not what to cut. The only way to know is to train the changed model and score it.
That is step 10.

## Try it

```bash
uv run pytest tests/test_04_attnlab.py
```

1. Run `mass` on your own checkpoint. Is your local layer also layer 1?
2. Which head of yours is the "layer 0 head 3"? Is it the same index as mine?
3. `attention_maps` returns the full (heads, T, T) weights. Plot one row of layer 1 for an answer
   token. How many tokens back does it look?

<details><summary>Answers</summary>

1. Probably. Both of my seeds put it in layer 1 (86% and 83%). A shallow layer going local is the
   cheapest win there is: most next tokens follow from the last few.
2. Probably not. Heads start from random weights, so which slot does what is luck: seed 42's big
   head is h3, seed 7's is h0. What survives the seed is the shape: one head in layer 0 that
   costs 5 to 7 times the whole model when you switch it off.
3. Mine, averaged over every answer token: 73% of layer 1's weight is on the last 10 tokens, and
   the biggest single share, 23%, is on the token right before. A bigram reader with a bit of
   memory.

</details>

Next: [lesson 6, the /think span](../06-think-span/README.md). Step 10 waits until after lesson 7:
it scores every change on the benchmarks, and lesson 7 is where you build the harness.
