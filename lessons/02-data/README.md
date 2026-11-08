# Lesson 2: the data

A model learns only what its data shows it. A small model is even more sensitive to this: it has no
spare capacity for broken examples, duplicates or tasks whose answer is already written in the
question. Most of theOG's work went into this step.

## What one row looks like

```json
{"source": "opencodeinstruct",
 "user": "Write a function to find the subarray with the largest sum in a given array.",
 "answer": "def find_largest_sum(nums):\n    max_so_far = max_ending_here = nums[0]\n    ...",
 "entry_point": "", "tests": "..."}
```

One task in plain English, one Python solution. That's all the model will ever see.

## The sources

| source | licence | why |
|---|---|---|
| [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | CC BY 4.0 | every answer comes with unit tests the dataset authors ran; we keep only rows where all of them passed |
| [OpenCoder-LLM/opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2), `educational_instruct` | MIT | short, textbook-style tasks |

Both are downloaded at a **pinned revision** (a commit hash in `theog/data.py`), so the same command
gives the same rows next year.

```bash
uv run python -m theog.data download --raw data/raw
```

This is about 6.9 GB, almost all of it OpenCodeInstruct's 50 files.

## Cleaning, rule by rule

`theog/data.py` reads every row and drops it for the first rule it breaks. The build prints how many
rows each rule dropped.

1. **The tests must have passed** (OpenCodeInstruct): an answer that fails its own tests teaches bugs.
2. **Exactly one solution block.** The answer must hold exactly one fenced Python block that defines a
   function or class. Zero means no code. Several means we can't tell which one is the answer.
3. **Solution only.** `solution_only()` parses the code and keeps imports, functions, classes and
   constants. Usage examples (`print(f(3))`, `if __name__ == "__main__":`), docstrings and comments
   are dropped. The task asked for a function, and the model should learn to write just that.
   `ast.unparse` writes the code back in one consistent style.
4. **The answer must not be in the prompt.** If a real line of the solution is already in the task
   text, the model learns to copy, not to solve.
5. **Dedup.** Same prompt twice: keep the first. Same answer twice: keep the first. Then
   `skeleton()` renames every variable and function to `_`, so `def add(a, b): return a + b` and
   `def plus(x, y): return x + y` have the same shape. At most 3 rows per shape are kept: a few
   renamed variants are useful variety, but hundreds would crowd out other tasks.

```bash
uv run python -m theog.data build --raw data/raw --out data/rows.jsonl
```

Add `--limit 20000` for a quick sample first. On one file of each source, that kept 27,733 rows.

## Contamination: don't train on the exam

Lesson 7 scores the model on HumanEval and MBPP. If a training row contains one of those tasks, the
model can answer it from memory, and the score measures memory, not skill. Public datasets often
contain benchmark tasks.

`theog/contamination.py` splits text into words and builds every run of 13 consecutive words
(**13-grams**) of every benchmark prompt and reference solution. A training row is dropped when it
contains at least half of the 13-grams of any one of them.

```bash
uv run python -m theog.contamination --rows data/rows.jsonl --out data/rows.clean.jsonl
```

It prints which benchmark tasks were found and how many rows matched each. On the 27,733-row sample
it dropped 80 rows. Solutions shorter than 13 words have no 13-gram and are not checked.

## Check it

```bash
uv run pytest tests/test_02_data.py
```

## theOG's own pipeline

theOG-50M v3 applied more rules of this kind to a larger mix: 12 public Hugging Face sources,
4.96M rows and 2.49B tokens (the source list is on the
[model card](https://huggingface.co/DrunkkToys/theOG-50M)). This course keeps two of those sources
and the rules above, the ones that drop the most rows.

## Try this

- Run the build with `--limit 20000` and read 20 random rows of `rows.jsonl`. Would you keep them all?
- Change `MAX_PER_SKELETON` to 1 and to 10: how many rows does each keep?
