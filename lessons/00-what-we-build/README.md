# Lesson 0: what we're building

By the end of this course you will have written, trained and run your own language model that writes
Python. You start from nothing: no pretrained weights and no borrowed tokenizer. At the end it is a
`.gguf` file you run offline with `ollama run`.

You don't watch someone else's code run. You write the important parts yourself: the BPE merge
loop, the data filters, the label shift, attention, RoPE, the training loss, the sampler, the
evaluation harness and the export. The repo gives you the plumbing (argument parsing, file
formats, downloads) and a test for every function you write.

## Two tracks

| | laptop track | full track |
|---|---|---|
| model | 8.3M parameters (`--preset laptop`) | 50M parameters, theOG-50M's exact size |
| data | 1 file per source, 178 MB, 95k rows | every file, 6.9 GB |
| hardware | any Apple silicon Mac or NVIDIA GPU; a CPU works, slowly | a GPU you can leave busy for days |
| training time | about 2½ hours on an M2 MacBook Air (3 passes over the data) | days |
| what you get | a model that writes Python-shaped answers and solves the easy ones | the same pipeline as theOG-50M |

**Do the laptop track first.** Every lesson gives its commands and the output we got on an M2, so
you can compare. The full track is the same commands with bigger numbers. Each lesson has a short
"full track" note.

Set your expectations for the laptop model now, so it doesn't disappoint you later: 8M parameters
trained for 2½ hours writes code that *looks* right much more often than it *is* right. Lesson 7
shows you exactly how often it is right. That gap, and what closes it, is most of what there is to
learn about small models.

## The model you are rebuilding

theOG-50M v3 is public: [huggingface.co/DrunkkToys/theOG-50M](https://huggingface.co/DrunkkToys/theOG-50M).
Try it before you build your own:

```bash
ollama run hf.co/DrunkkToys/theOG-50M:Q8_0 "Write a function that checks if a string is a palindrome."
```

Its goal was to beat Salesforce CodeGen-350M-mono, a model 7× larger. The model card has the
scores on HumanEval, MBPP-500 and graded-30, measured on the same harness as CodeGen.

## The pipeline

```
Hugging Face datasets ─► rows.jsonl ─► tokenizer.json ─► packed windows ─► model ─► best.pt ─► scores ─► .gguf ─► Ollama
      lesson 2            lesson 2       lesson 1          lesson 3        lesson 4  lesson 5   lesson 7  lesson 8
```

| lesson | you write | file |
|---|---|---|
| 1 | BPE by hand, then the real tokenizer | `theog/tokenizer.py` |
| 2 | the cleaning rules, dedup, naming the function, the contamination check | `theog/data.py`, `theog/contamination.py` |
| 3 | one training sequence with its labels, and the packer | `theog/format.py`, `theog/pack.py` |
| 4 | the transformer, in 8 steps | `theog/model.py` |
| 5 | the learning-rate schedule, the loss, the sampler, the prompt, generation | `theog/train.py`, `theog/format.py`, `theog/generate.py` |
| 6 | nothing new: what the /think span does, and an experiment | — |
| 7 | answer extraction and the sandboxed test runner | `theog/generate.py`, `theog/evaluate.py` |
| 8 | the weight mapping for GGUF, the Ollama Modelfile | `theog/export.py` |
| 9 | your model card and release | — |

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/DrunkkToys/DrunkkInference
cd DrunkkInference
uv sync --extra dev --extra export
```

Then see where you are:

```bash
uv run python -m theog.progress
```

```text
lesson                       tests
1  tokenizer                 to do  3 failed, 4 errors in 0.04s
2  data                      to do  8 failed, 3 passed in 0.05s
3  format + packing          to do  2 failed, 4 errors in 0.69s
4  model                     to do  14 failed, 1 passed in 12.73s
5  training + generation     to do  3 failed, 4 errors in 2.04s
7  evaluation                to do  3 failed in 1.91s
8  export                    to do  3 failed, 1 error in 2.20s
```

Everything is "to do". That's the course. Lessons 6 and 9 have no code to write, so they have no row.

## How an exercise works

Open the file the lesson names. The function is already there, with a docstring that says exactly
what it must do and a hint:

```python
def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair`, left to right, with `new_id`.

    merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    """
    # EXERCISE 1.2: lessons/01-tokenizer/README.md
    # Hint: Walk with an index i. ...
    raise NotImplementedError("exercise 1.2: write merge")
```

Replace the `raise` with your code, then run the test the lesson gives you. Red means not yet;
green means done. Each lesson also has the answer in a folded **Solution** block. Try first. Reading
the answer before you have been stuck for a while teaches you about as much as reading this
sentence did.

## When you're stuck

The full working code is on the `solutions` branch. To take one finished file and move on:

```bash
git fetch origin solutions
git checkout origin/solutions -- theog/tokenizer.py
```

That replaces the whole file, including the exercises in it you already did. Commit your own work
first if you want to keep it.

Later lessons run the code of earlier ones (lesson 5 trains the model you wrote in lesson 4), so a
bug in lesson 4 shows up in lesson 5. The tests catch most of those before they get that far.

## How this course relates to theOG

The code is a clean rewrite of theOG's pipeline, written to be read. Three things differ, and each
lesson says where:

- **Model.** Lesson 4's model computes the same function as theOG-50M v3's exported model: with the
  same weights loaded, the logits are identical. theOG's research code also has experiments the
  shipped file doesn't use, and the course leaves those out.
- **Data.** theOG-50M v3 trained on 12 public Hugging Face sources, 4.96M rows, 2.49B tokens. The
  course uses two of them, so your scores will differ.
- **Tokenizer.** Same kind (byte-level BPE), trained on your data, so the token ids differ.

Next: [lesson 1, the tokenizer](../01-tokenizer/README.md).
