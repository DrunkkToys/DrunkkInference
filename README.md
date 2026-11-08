# DrunkkInference

Build **theOG**, a 50M-parameter Python code model, from scratch on your own GPU. You write the
tokenizer, the data pipeline, the model, the training loop, the evaluation harness and the export,
and end up with a `.gguf` file that runs offline in Ollama.

The course is for Python developers who are new to machine learning. Each lesson explains one step
from zero, points at the module that does it (each one is short enough to read in one sitting), and
gives the command to run and the test that checks it.

**Goal of theOG:** beat CodeGen-350M-mono with under 350M parameters. theOG-50M v3 does it on all
three benchmarks: `/no_think`, greedy, it scores 25/30 on graded-30 (CodeGen 19/30), 39/164 on
HumanEval (CodeGen 22/164) and 94/500 on MBPP-500 (CodeGen 87/500). Source:
[theOG-50M model card](https://huggingface.co/DrunkkToys/theOG-50M).

## Lessons

| # | lesson | code |
|---|---|---|
| 0 | [What we're building](lessons/00-what-we-build/README.md) | — |
| 1 | [The tokenizer](lessons/01-tokenizer/README.md) | `theog/tokenizer.py`, `theog/tokens.py` |
| 2 | [The data](lessons/02-data/README.md) | `theog/data.py`, `theog/contamination.py` |
| 3 | [The chat format and packing](lessons/03-format-and-packing/README.md) | `theog/format.py`, `theog/pack.py` |
| 4 | [The model](lessons/04-model/README.md), in 8 steps: embedding, RMSNorm, attention, heads + QK-norm, RoPE, SwiGLU, block, whole model | `theog/model.py` |
| 5 | [Training](lessons/05-training/README.md) | `theog/train.py`, `theog/generate.py` |
| 6 | [The /think span](lessons/06-think-span/README.md) | `theog/format.py` |
| 7 | [Evaluation](lessons/07-evaluation/README.md) | `theog/evaluate.py` |
| 8 | [Export to GGUF and Ollama](lessons/08-export/README.md) | `theog/export.py` |
| 9 | [Publishing](lessons/09-publish/README.md) | — |

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --extra export
```

```bash
uv run pytest
```

The tests use a tiny model and toy data and run on a CPU.

## The whole pipeline

Lesson 2 needs about 7 GB of disk for the download. Lesson 5 needs an NVIDIA (CUDA) or Apple silicon
(MPS) GPU.

```bash
uv run python -m theog.data download --raw data/raw
uv run python -m theog.data build --raw data/raw --out data/rows.jsonl
uv run python -m theog.contamination --rows data/rows.jsonl --out data/rows.clean.jsonl
uv run python -m theog.tokenizer --rows data/rows.clean.jsonl --out data/tokenizer.json
uv run python -m theog.pack --rows data/rows.clean.jsonl --tokenizer data/tokenizer.json --out data/packed
uv run python -m theog.train --packed data/packed --out runs/theog-50m
uv run python -m theog.evaluate --ckpt runs/theog-50m/best.pt --tokenizer data/tokenizer.json --dataset humaneval --out results/ --temperature 0.2
uv run python -m theog.evaluate --ckpt runs/theog-50m/best.pt --tokenizer data/tokenizer.json --dataset mbpp500 --out results/ --temperature 0.2
uv run --extra export python -m theog.export --ckpt runs/theog-50m/best.pt --tokenizer data/tokenizer.json --out release/theog-50m-Q8_0.gguf --quant Q8_0
ollama create theog:no_think -f release/Modelfile.no_think
```

## Data

The course trains on two public datasets, downloaded at pinned revisions:
[nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) (CC BY 4.0) and
[OpenCoder-LLM/opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) (MIT).
theOG-50M v3 trained on 12 public Hugging Face sources (2.49B tokens), these two among them, so a
model trained here will not score the same.

## Licence

Code: [MIT](LICENSE). Lesson text: [CC BY 4.0](LICENSE-docs).
