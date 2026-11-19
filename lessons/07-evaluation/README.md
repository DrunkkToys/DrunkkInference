# Lesson 7: evaluation

**You write:** the code that pulls the answer out of the model's text, and the sandboxed test
runner. **You get:** your model's real score on HumanEval and MBPP, and a file showing every answer
it wrote. **Time:** about 30 minutes, plus the runs.

A lower validation loss doesn't tell you whether the code works. The only real test is to run the
code the model writes against tests it has never seen.

## The benchmarks

| benchmark | tasks | what the model sees |
|---|:-:|---|
| [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | 164 | a function signature and docstring, written by OpenAI |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp), test split | 500 | one sentence and three `assert` lines, e.g. *"Write a function to find the shared elements from the given two lists."* |

Both are loaded at a pinned revision. A task **passes** when the model's code plus the task's own
tests run without error. The score is the number of tasks passed.

## How a task is run: `theog/evaluate.py`

1. The model gets the task in its chat format and writes an answer (`generate`, lesson 5).
2. `extract_code()` takes the code out of the ```` ```python ```` block. **Exercise 7.1.**
3. For HumanEval, the prompt's own imports and helper functions are put back in front, because the
   model was only asked for the one function.
4. `run_tests()` runs the code and the tests in a fresh Python process with a time limit.
   **Exercise 7.2.**
5. Each task prints PASS or FAIL with the reason, and the results file keeps the raw output and the
   code, so any score can be checked by reading what the model wrote.

### Exercise 7.1: `extract_code` in `theog/generate.py`

The model writes something like `"\n```python\ndef f(x):\n    return x\n```\n"`. Return what's
inside the fence, stripped. Two cases to get right:

- The model ran out of tokens before closing the fence: keep what's there. It may still be a
  complete function.
- No fence at all: return the whole text, stripped. Let the tests decide.

The regex `_CODE` above the function already matches both fence cases.

<details><summary>Solution</summary>

```python
def extract_code(text):
    m = _CODE.search(text)
    return (m.group(1) if m else text).strip()
```

</details>

### Exercise 7.2: `run_tests` in `theog/evaluate.py`

The model's code is untrusted. It can loop forever, call `sys.exit()`, or overwrite the function the
tests call. Running it with `exec` inside the harness would let it break the harness, so each task
gets its own Python process.

`_RUNNER` (written for you) is the program that runs inside it: it reads a JSON job on stdin, runs
the code, then the setup, then each test, and prints one line `__RESULT__{"ok": ..., "why": ...}`.
You write the side that starts it:

1. `job = json.dumps({"solution": ..., "entry": ..., "tests": ..., "setup": ...})`.
2. `subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=job, capture_output=True,
   text=True, timeout=timeout)`. `-I` is isolated mode: no environment variables, no current folder
   on the path, so a file named `json.py` next to the code can't hijack the import.
3. `subprocess.TimeoutExpired` means `(False, f"timeout ({timeout}s)")`.
4. Find `__RESULT__(\{.*\})` in stdout and return `(ok, why)` from its JSON. If it isn't there the
   process died before printing (a segfault, `os._exit`): return
   `(False, "crashed: " + the last line of stderr)`.

```bash
uv run pytest tests/test_07_evaluate.py
```

All 3 should pass. The second test checks every failure reason, including a real infinite loop.

<details><summary>Solution</summary>

```python
def run_tests(solution, entry, tests, setup="", timeout=10):
    job = json.dumps({"solution": solution, "entry": entry, "tests": tests, "setup": setup})
    try:
        r = subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=job,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout ({timeout}s)"
    m = re.search(r"__RESULT__(\{.*\})", r.stdout)
    if not m:
        return False, "crashed: " + ((r.stderr.strip().splitlines() or ["no output"])[-1])
    res = json.loads(m.group(1))
    return res["ok"], res["why"]
```

</details>

The fresh process isolates the harness, not your machine. The code can still delete your files.
Run evaluations in a container or VM if that matters to you.

## First: test the harness, not the model

If a score is 0 or makes no sense, assume the harness is broken before you assume the model is.

**Known answers pass.** Each task ships with a reference solution. Run every one through your
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

```text
humaneval 164 / 164 references pass; fail: []
mbpp500 498 / 500 references pass; fail: ['Mbpp/180', 'Mbpp/493']
```

Mbpp/180 and Mbpp/493 compare floats with `==`, and their own reference solutions fail those exact
comparisons, so no model can be expected to pass them.

**A known model gets its known score** (full track, optional). Run a public model through the same
harness:

```bash
uv run --extra sanity python -m theog.evaluate --hf Salesforce/codegen-350M-mono --dataset humaneval --out results/
```

On theOG's harness CodeGen-350M-mono scored 22/164 HumanEval and 87/500 MBPP (model card). A result
far from that points to a harness bug.

## Score your model

```bash
uv run python -m theog.evaluate --ckpt runs/laptop/best.pt --tokenizer data/tokenizer.json --dataset humaneval --out results/
uv run python -m theog.evaluate --ckpt runs/laptop/best.pt --tokenizer data/tokenizer.json --dataset mbpp500 --out results/
```

Greedy (temperature 0, the default here), so a rerun gives the same score. HumanEval takes about
5 minutes on an M2, MBPP about 15. Add `--think` for the other mode, `--limit 10` for a quick
check that everything runs.

Each task prints one line as it goes:

```text
[  1/500] Mbpp/11          FAIL  no function named remove_Occ
[  2/500] Mbpp/12          FAIL  assert failed: assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]
[  7/500] Mbpp/17          FAIL  TypeError: square_perimeter() missing 1 required positional argument: 'y'
...
[ 52/500] Mbpp/62          PASS  all tests pass
...
mbpp500: 3/500 passed  ->  results/laptop_mbpp500_no_think_t0.0_20261119T113601.jsonl
```

**These are the scores of this course's first laptop run**, trained before exercise 2.4 existed:
**1/164** on HumanEval (HumanEval/51) and **3/500** on MBPP (Mbpp/62, Mbpp/99, Mbpp/297). Your
model, trained with exercise 2.4, should do better. Don't read the score first: read *why* it
failed. Count the reasons:

```bash
uv run python -c "
import json, collections, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(collections.Counter(r['why'].split(':')[0].split(' named ')[0] for r in rows).most_common(6))
" results/laptop_mbpp500_no_think_t0.0_*.jsonl
```

For that first run:

| reason | HumanEval | MBPP |
|---|:-:|:-:|
| assert failed: the code runs, the answer is wrong | 118 | 29 |
| no function named …: right idea, wrong name | 13 | 343 |
| code does not run: it doesn't even parse | 17 | 74 |
| TypeError, IndexError, other errors at run time | 15 | 51 |
| passed | 1 | 3 |

Two different models are hiding in that table. On HumanEval, where the prompt *is* the function
signature, the name is never the problem: the code runs and gives wrong answers. On MBPP, where the
name only appears inside the asserts, 343 answers never got as far as being wrong. That second
number is what lesson 2's exercise 2.4 fixes, and it was found exactly this way: by reading the
reasons, not the score.

Open the results file to see any answer in full:

```json
{"task_id": "Mbpp/11", "passed": false, "why": "no function named remove_Occ",
 "code": "def remove_in_char(s):\n    return s.replace(' ', '').lower()", "raw": "..."}
```

## Differences from theOG's own harness

- **MBPP setup order.** A few MBPP tasks have setup code that uses the solution's classes, e.g.
  Mbpp/367 builds a tree from the solution's `Node` class. This harness runs the solution first,
  then the setup. theOG's harness runs the setup first, so those tasks always fail there.
- **graded-30** is theOG's own set of 30 everyday tasks. It is not public, so it's not in this course.

## Going further

1. Compare your table with the first run's. Which reason went down, and which went up? (A wrong
   name hides every other kind of failure. Fix the name, and some answers move to "assert failed".)
2. Run MBPP with `--think`. Which tasks pass in only one of the two modes?
3. Your model writes code that *looks* right. Pick five "assert failed" answers and find the bug by
   hand. Is it the algorithm, an edge case, or did it answer a different question?
4. Why is a one-task difference between two runs not a result?

<details><summary>Answers</summary>

1. On our runs, "no function named" fell the most, as it should. Some of those tasks then failed
   their asserts instead, so "assert failed" can go *up* even though the model got better. That's
   why you read the whole table, not one row.
2. Run it and compare the two results files by `task_id`. At this size expect a handful of tasks
   each way: the two modes are not just noise, they write different code.
3. On the first run, many wrong answers solved a nearby, more common task. Mbpp/11 asks to remove
   the first and last occurrence of a given character; the model wrote a function that removes
   spaces. A small model that has seen little data writes the function it has seen most often with
   a similar prompt. More data (the full track) is what fixes that.
4. Greedy decoding makes each run repeatable, but the model's choice at a near-tie can flip with
   any small change: a retrain, a different seed, a slightly different prompt. One task out of
   500 is 0.2%. Before believing a change, look for a difference of several tasks on both
   benchmarks.

</details>

Next: [lesson 8, export](../08-export/README.md).
