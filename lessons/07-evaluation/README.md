# Lesson 7: evaluation

A lower validation loss doesn't tell you whether the code works. The only real test is to run the
code the model writes against tests it has never seen.

## The benchmarks

| benchmark | tasks | what the model sees |
|---|:-:|---|
| [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | 164 | a function signature and docstring, written by OpenAI |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp), test split | 500 | one sentence and three `assert` lines, e.g. *"Write a function to find the shared elements from the given two lists."* |

Both are loaded at a pinned revision. A task **passes** when the model's code plus the task's own
tests run without error. The score is the number of tasks passed: HumanEval 39/164 means 39 tasks
passed.

## How a task is run: `theog/evaluate.py`

1. The model gets the task in its chat format (lesson 3) and writes an answer.
2. `extract_code()` takes the code out of the ```` ```python ```` block.
3. For HumanEval, the prompt's own imports and helper functions are put back in front, because the
   model was only asked for the one function.
4. `run_tests()` starts a **fresh Python process** (`python -I`, which ignores your environment and
   current folder). It runs the code, then the task's setup, then each test, with a 10-second limit.
   A fresh process means the model's code can't change the harness. The time limit catches infinite
   loops.
5. Each task prints PASS or FAIL with the reason: `assert failed: …`, `code does not run: NameError: …`,
   `timeout (10s)`, `no function named …`.

The results file keeps the raw output and the code for every task, so any score can be checked by
reading what the model wrote.

Generated code is untrusted. The fresh process isolates the harness, not your machine: run
evaluations in a container or VM if that matters to you.

## First: test the harness, not the model

If a score is 0 or makes no sense, assume the harness is broken before you assume the model is.
Two checks:

**1. Known answers pass.** Each task ships with a reference solution. Run every reference through
`run_tests()`:

```bash
uv run python -c "
from theog.evaluate import load_tasks, run_tests
for d in ('humaneval', 'mbpp500'):
    ts = load_tasks(d)
    fails = [t['task_id'] for t in ts
             if not run_tests((t['prompt'] + t['solution']) if d == 'humaneval' else t['solution'],
                              t['entry_point'], t['tests'], t['setup'])[0]]
    print(d, len(ts) - len(fails), '/', len(ts), 'references pass; fail:', fails)
"
```

This gives 164/164 for HumanEval and 498/500 for MBPP. Mbpp/180 and Mbpp/493 compare floats with `==`,
and their own reference solutions fail those exact float comparisons, so a model can't be expected to pass them.

**2. A known model gets its known score.** Run a public model through the same harness:

```bash
uv run --extra sanity python -m theog.evaluate --hf Salesforce/codegen-350M-mono --dataset humaneval --out results/
```

CodeGen-350M-mono is a plain completion model, so `HFSolver` gives it the raw prompt and decodes
greedily. On theOG's own harness it scored 22/164 HumanEval and 87/500 MBPP (model card). A result far
from that points to a harness bug.

## Score your model

```bash
uv run python -m theog.evaluate --ckpt runs/theog-50m/best.pt --tokenizer data/tokenizer.json \
    --dataset humaneval --out results/ --temperature 0.2
```

```bash
uv run python -m theog.evaluate --ckpt runs/theog-50m/best.pt --tokenizer data/tokenizer.json \
    --dataset mbpp500 --out results/ --temperature 0.2
```

Add `--think` for the `/think` mode. Add `--limit 5` for a quick smoke run: that checks the
pipeline, it is not a score.

**Compare on both benchmarks, always.** A change that helps HumanEval can hurt MBPP. At temperature
0.2 the score also moves by a few tasks from run to run, so a gain of 1 or 2 tasks is not evidence.

## Differences from theOG's own harness

- **MBPP setup order.** A few MBPP tasks have setup code that uses the solution's classes, e.g.
  Mbpp/367 builds a tree from the solution's `Node` class. This harness runs the solution first,
  then the setup. theOG's harness runs the setup first, so those tasks always fail there.
- **graded-30** is theOG's own set of 30 everyday tasks. It is not public, so it's not in this course.

## Check it

```bash
uv run pytest tests/test_07_evaluate.py
```

## Try this

- Open a results `.jsonl` and read five FAIL rows. Is the model wrong, or nearly right?
- Count the failure reasons. Which one is most common for your model?
