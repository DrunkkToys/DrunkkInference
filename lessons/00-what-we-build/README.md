# Lesson 0: what we're building

By the end of this course you will have trained **theOG**, a 50-million-parameter language model that
writes Python, starting from nothing: no pretrained weights and no borrowed tokenizer. You'll export it
to a single `.gguf` file that runs offline in Ollama, LM Studio or llama.cpp.

## The model you are rebuilding

theOG-50M v2 is public: [huggingface.co/DrunkkToys/theOG-50M](https://huggingface.co/DrunkkToys/theOG-50M).
Try it before you build your own:

```bash
ollama run hf.co/DrunkkToys/theOG-50M:Q8_0
```

Ask it for a function in plain English, for example *"Write a function that checks if a string is a palindrome."*

## The goal, and where it stands

**Goal:** beat Salesforce CodeGen-350M-mono, a model 7× larger, with under 350M parameters.

Scores of the shipped file (Q8_0, `/no_think`, temperature 0.2, top-k 40) against CodeGen-350M on the
same harness. The source is the theOG-50M model card, which lists the result file for every cell:

| benchmark | theOG-50M v2 | CodeGen-350M-mono |
|---|:-:|:-:|
| graded-30 (30 everyday function tasks) | 21/30 | 19/30 |
| HumanEval-164 | 13/164 | 22/164 |
| MBPP-500 | 31/500 | 87/500 |

theOG leads on graded-30 and trails on HumanEval and MBPP. The goal is not reached yet. This course
gives you the whole pipeline, so you can try to close the gap yourself.

## The pipeline

```
Hugging Face datasets ──► rows.jsonl ──► tokenizer.json ──► packed windows ──► model ──► best.pt ──► .gguf ──► Ollama
     lesson 2              lesson 2        lesson 1           lesson 3         lesson 4/5   lesson 7     lesson 8
```

| lesson | you build | file |
|---|---|---|
| 1 | a tokenizer: text to numbers and back | `theog/tokenizer.py` |
| 2 | the training data, cleaned and deduplicated | `theog/data.py`, `theog/contamination.py` |
| 3 | the chat format, and packing rows into windows | `theog/format.py`, `theog/pack.py` |
| 4 | the transformer itself | `theog/model.py` |
| 5 | the training loop | `theog/train.py`, `theog/generate.py` |
| 6 | the /think span | `theog/format.py` |
| 7 | evaluation on HumanEval and MBPP | `theog/evaluate.py` |
| 8 | export to GGUF and Ollama | `theog/export.py` |
| 9 | publishing on Hugging Face | — |

## Setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/DrunkkToys/DrunkkInference
cd DrunkkInference
uv sync --extra dev --extra export
uv run pytest
```

Every test runs on a laptop CPU in seconds. For the real training run in lesson 5 you need your own
GPU: NVIDIA (CUDA) or Apple silicon (MPS).

## How this course relates to theOG

The code here is a clean rewrite of theOG's own pipeline, written to be read. Three things differ, and
each lesson says where:

- **Model.** Lesson 4's model computes the same function as theOG-50M v2's exported model. The test is
  zero difference in the logits once the same weights are loaded. theOG's research code also carries
  experiments that the shipped file does not use, and this course leaves those out.
- **Data.** theOG v2 trained on a larger mix that includes scraped tutorial sites. This course uses
  only two public datasets that can be downloaded and redistributed, so your scores will differ.
- **Tokenizer.** Same kind and size (byte-level BPE, 16,000 tokens), trained on your data, so the
  token ids differ from theOG's.
