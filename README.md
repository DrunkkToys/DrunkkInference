# DrunkkInference

Build your own Python code model from scratch, then run it offline in Ollama. You write the
important parts yourself: the BPE tokenizer, the data filters, the label shift, attention and RoPE,
the training loss, the sampler, the evaluation harness and the GGUF export. Each one has a test that
fails until your code is right.

The course is for Python developers who are new to machine learning. No pretrained weights, no
borrowed tokenizer, no framework beyond PyTorch.

**Two tracks.** The laptop track trains an 8.3M-parameter model in about 2½ hours on an M2
MacBook Air. The full track runs the same commands at theOG-50M's size, on a GPU, for days. Start
with the laptop track.

The full track rebuilds theOG-50M, whose goal was to beat CodeGen-350M-mono with under 350M
parameters. theOG-50M v3 does it on all three benchmarks: `/no_think`, greedy, it scores 26/30 on graded-30 (CodeGen 19/30), 42/164 on
HumanEval (CodeGen 22/164) and 91/500 on MBPP-500 (CodeGen 87/500). Source:
[theOG-50M model card](https://huggingface.co/DrunkkToys/theOG-50M).

## Lessons

| # | lesson | you write |
|---|---|---|
| 0 | [What we're building](lessons/00-what-we-build/README.md) | — |
| 1 | [The tokenizer](lessons/01-tokenizer/README.md) | BPE by hand, then the real tokenizer |
| 2 | [The data](lessons/02-data/README.md) | cleaning rules, dedup, naming the function, the contamination check |
| 3 | [The chat format and packing](lessons/03-format-and-packing/README.md) | one training sequence with its labels, the packer |
| 4 | [The model](lessons/04-model/README.md), in 8 steps, plus 2 you come back for (step 9 after lesson 5, step 10 after lesson 7) | RMSNorm, attention, RoPE, heads, SwiGLU, block, whole model; reading and changing the attention |
| 5 | [Training](lessons/05-training/README.md) | the LR schedule, the loss, the sampler, the prompt, generation |
| 6 | [Evaluation](lessons/06-evaluation/README.md) | answer extraction, the sandboxed test runner |
| 7 | [The /think span](lessons/07-think-span/README.md) | grounded tokens, the cost of a span in nats, and an experiment with twelve span designs |
| 8 | [Export to GGUF and Ollama](lessons/08-export/README.md) | the weight mapping, the Modelfile |
| 9 | [Publishing](lessons/09-publish/README.md) | your model card |

## Start

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/DrunkkToys/DrunkkInference
cd DrunkkInference
uv sync --extra dev --extra export
uv run python -m theog.progress
```

Every lesson shows "to do". Open [lesson 0](lessons/00-what-we-build/README.md). The full working
code is on the [`solutions`](https://github.com/DrunkkToys/DrunkkInference/tree/solutions) branch,
for when you're stuck.

## The whole pipeline, laptop track

Once your code passes the tests, these commands are the course. About 200 MB of disk and an Apple
silicon Mac or an NVIDIA GPU (a CPU works, slowly).

```bash
uv run python -m theog.data download --raw data/raw --parts 1
uv run python -m theog.data build --raw data/raw --out data/rows.jsonl
uv run python -m theog.contamination --rows data/rows.jsonl --out data/rows.clean.jsonl
uv run python -m theog.tokenizer --rows data/rows.clean.jsonl --out data/tokenizer.json --vocab-size 8000
uv run python -m theog.pack --rows data/rows.clean.jsonl --tokenizer data/tokenizer.json --out data/packed --seq-len 512
uv run python -m theog.train --packed data/packed --out runs/laptop --preset laptop --epochs 3 --batch-size 16 --accum 1 --lr 1e-3 --eval-every 1000 --eval-windows 200 --log-every 500
uv run python -m theog.evaluate --ckpt runs/laptop/best.pt --tokenizer data/tokenizer.json --dataset humaneval --out results/
uv run python -m theog.evaluate --ckpt runs/laptop/best.pt --tokenizer data/tokenizer.json --dataset mbpp500 --out results/
uv run --extra export python -m theog.export --ckpt runs/laptop/best.pt --tokenizer data/tokenizer.json --out release/laptop-Q8_0.gguf --quant Q8_0
ollama create mymodel -f release/Modelfile.no_think
```

**Full track:** drop `--parts`, `--vocab-size` and `--seq-len`, train with only `--packed` and
`--out` (the defaults are theOG-50M's), and expect about 7 GB of disk and days of GPU time.

## Data

The course trains on two public datasets, downloaded at pinned revisions:
[nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) (CC BY 4.0) and
[OpenCoder-LLM/opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) (MIT).
theOG-50M v3 trained on 12 public Hugging Face sources (2.49B tokens), these two among them, so a
model trained here will not score the same.

## Licence

Code: [MIT](LICENSE). Lesson text: [CC BY 4.0](LICENSE-docs).
