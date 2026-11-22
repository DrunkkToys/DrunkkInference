# Step 10: changing the attention

**After lesson 5, like step 9.** Each run here is a full training, about 2.5 hours on a laptop.

Every new model paper changes the attention. Fewer key/value heads. More heads. A gate. A bigger
RoPE base. Each comes with a chart that goes up and to the right, measured on a model 1,000 times
bigger than yours.

Does any of it help at 8M parameters? I don't know. Neither do you. Neither does the paper. So we
do what lesson 6 taught us: write the prediction down first, change **one** thing, train with the
same data, seed and recipe, and let the log decide.

## The tool

`theog/attnlab.py` has `LabAttention`: your step 5 attention, plus three switches. It swaps into
the model after it's built, so `theog/model.py` stays exactly as you wrote it.

| switch | what it does | exports to Ollama (lesson 8)? |
|---|---|:-:|
| `--attn kv=2` | **GQA**: 2 key/value heads, each shared by 2 query heads. `kv=1` is **MQA** | yes |
| `--attn gate=1` | a sigmoid gate per head on the attention output, read from the block input ([Qiu et al., 2025](https://arxiv.org/abs/2505.06708)) | no |
| `--attn qknorm=0` | no QK-norm | no |
| `--set heads=8,head_dim=32` | more, smaller heads (same width) | yes |
| `--set rope_theta=1e6` | Qwen3's RoPE base instead of ours | yes |

"No" means llama.cpp's `qwen3` layout has no slot for it, so the export refuses it instead of
writing a model that runs wrong. A trick you can't ship is a research result, not a product.

```bash
uv run python -m theog.train --packed data/packed --out runs/attn_kv2 --preset laptop --epochs 3 \
  --batch-size 16 --accum 1 --lr 1e-3 --eval-every 1000 --eval-windows 200 --log-every 500 \
  --seed 42 --attn kv=2
```

Then score it like any lesson 6 design: `theog.cost` on the same 1,000 rows, HumanEval and MBPP in
no-think mode, and the two step 9 diagnostics. The baseline is the lesson 5 model: cost 38.5 nats
per row, HumanEval 8/164, MBPP 49/500. Seed noise on the cost is 0.7 nats. On the benchmarks it's
much worse: the same model with seed 7 scores 3 and 58 (lesson 6, the seed check). So a benchmark gap
under 5 HumanEval or 9 MBPP tasks is a tie too.

## Why GQA saves anything

Look at step 5. Every head has its own k and v. At generation time you keep every past token's k
and v around: the **KV cache**. Share each k/v head between 2 query heads and the cache halves.
Each block also drops 4 of the 8 `hidden × head_dim` slices of k and v: 65,536 parameters per block,
393,216 in all. Big models do this because the KV cache, not the weights, is what runs out of
memory on long contexts.

The code is one line: before the attention, `k.repeat_interleave(heads // kv, dim=1)`, same for
v. Query heads 0 and 1 read k/v head 0, heads 2 and 3 read head 1.

## The predictions (written before any run)

Seven runs, one change each, seed 42. Step 9 told us a few things first: the sink is 1% (so the
gate has little to fix), layer 0 uses all 4 of its heads (so 2 heads should hurt), and in both
seeds some layer ends up mostly spare (so there's room to cut).

| run | change | parameters | my prediction: cost / HumanEval / MBPP |
|---|---|:-:|---|
| `kv2` | GQA, 2 k/v heads | 7,950,336 (−4.7%) | 38.3–39.2 / 6–10 / 45–53. A tie, with a cache half the size |
| `h8x32` | 8 heads of 32 | 8,343,168 | 38.0–39.0 / 6–10 / 45–53. Layer 0 wants heads, maybe a small gain |
| `gate` | output gate per head | +6,168 | 38.2–39.0 / 6–10 / 45–53. Nothing to fix, so a tie |
| `kv1` | MQA, 1 k/v head | 7,753,728 (−7.1%) | 38.8–39.8 / 5–9 / 43–51. A bit worse |
| `h2x128` | 2 heads of 128 | 8,344,320 | 38.8–40.0 / 4–8 / 42–50. Worse: layer 0 uses all 4 |
| `theta1e6` | RoPE base 1e6 | 8,343,552 | 38.3–38.8 / 6–10 / 45–53. No change at 512 tokens |
| `noqk` | no QK-norm | 8,342,784 | 38.8–41 / 4–8 / 40–50, or it blows up at lr 1e-3 |

Success means beating the baseline by more than the seed noise on cost, or on both benchmarks. A
tie with fewer parameters (`kv2`, `kv1`) is a practical win: same model, cheaper to run. Not a
research one.

Look at that table. Five ties, two losses, one maybe. Am I that pessimistic? No. At this size
almost every clever idea has been a tie or worse. Lesson 6 went six for six against my
arguments. If one of these surprises me, good. That's the run worth having.

## The results

| run | cost | HumanEval | MBPP | verdict |
|---|:-:|:-:|:-:|---|
| `none` (lesson 5) | 38.5 | 8 | 49 | the baseline |
| `kv2` | 38.6 | 3 | 48 | a tie, 4.7% fewer parameters. My HumanEval range (6–10) was wrong; 3 is what `none` scores at seed 7 |
| `h8x32` | 39.9 | 5 | 43 | worse: 1.4 nats behind, outside the noise. I said maybe a small gain. Wrong way |
| `gate` | | | | running |
| `kv1` | | | | queued |
| `h2x128` | | | | queued |
| `theta1e6` | | | | queued |
| `noqk` | | | | queued |

The predictions above were committed before the first run started. Git has the date. When the
numbers land here, they land next to what I said they'd be, wrong ones included.

## Try it

```bash
uv run pytest tests/test_04_attnlab.py
```

1. `kv=2` on the laptop preset: work out the 393,216 yourself. Then do it for theOG (10 layers,
   hidden 512, 4 heads of 128).
2. Why must `kv` divide `heads`? What would `kv=3` with 4 heads even mean?
3. Pick one change that isn't in the table. Write your prediction down. Then run it.

<details><summary>Answers</summary>

1. Laptop: k and v each go from 256 × 256 to 256 × 128, so 2 × 256 × 128 = 65,536 per block, × 6
   blocks = 393,216. theOG: 2 × 512 × 256 = 262,144 per block, × 10 = 2,621,440, about 5% of 50.1M.
2. Each k/v head serves a fixed group of `heads // kv` query heads. With 3 for 4, one k/v head
   would serve two query heads and the others one each. It could be built, but nobody does, and
   `repeat_interleave` can't express it. `LabAttention` refuses it.
3. That's your experiment. If you didn't write the prediction first, you'll remember it as
   "about what I expected". Everybody does.

</details>

Next: [lesson 5, training](../05-training/README.md), if you came here early. Otherwise
[lesson 6, the /think span](../06-think-span/README.md).
