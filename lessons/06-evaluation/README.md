# Lesson 6: evaluation

**You write:** the code that pulls the answer out of the model's text, and the sandboxed test
runner. **You get:** your model's real score on HumanEval and MBPP, and a file showing every answer
it wrote. **You learn:** how to measure whether a code model is right, not just confident, and
how to read a score without fooling yourself. **Time:** about 30 minutes, plus the runs.

Your validation loss went down. Congratulations. Does the code work? The loss has no idea. It
measures how well the model predicts one reference answer, token by token, and a program can be
one token away from the reference and completely broken.

The only real test is to run the code the model writes against tests it has never seen. That's
this lesson. It's also where most small-model claims on the internet quietly fall apart.

## The benchmarks

| benchmark | tasks | what the model sees |
|---|:-:|---|
| [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) | 164 | a function signature and docstring, written by OpenAI |
| [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp), test split | 500 | one sentence and three `assert` lines, e.g. *"Write a function to find the shared elements from the given two lists."* |

Both are loaded at a pinned revision. A task **passes** when the model's code plus the task's own
tests run without error. The score is the number of tasks passed.

## How a task is run: `theog/evaluate.py`

1. The model gets the task in its chat format and writes an answer (`generate`, lesson 5).
2. `extract_code()` takes the code out of the ```` ```python ```` block. **Exercise 6.1.**
3. For HumanEval, the prompt's own imports and helper functions are put back in front, because the
   model was only asked for the one function.
4. `run_tests()` runs the code and the tests in a fresh Python process with a time limit.
   **Exercise 6.2.**
5. Each task prints PASS or FAIL with the reason, and the results file keeps the raw output and the
   code, so any score can be checked by reading what the model wrote.

### Exercise 6.1: `extract_code` in `theog/generate.py`

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

### Exercise 6.2: `run_tests` in `theog/evaluate.py`

The model's code is untrusted. It can loop forever, call `sys.exit()`, or overwrite the function the
tests call. Not out of malice: an 8M model doesn't plot. It's just wrong in creative ways. Running
it with `exec` inside the harness would let it break the harness, so each task gets its own Python
process.

`_RUNNER` (written for you) is the program that runs inside it: it reads a JSON job on stdin, runs
the code, then the setup, then each test, and prints one line `__RESULT__{"ok": ..., "why": ...}`.
You write the side that starts it:

1. `job = json.dumps({"solution": ..., "entry": ..., "tests": ..., "setup": ...})`.
2. Start a fresh Python in isolated mode (`-I`: no environment variables, no current folder on the
   path, so a file named `json.py` next to the code can't hijack the import), give it `_RUNNER` as
   its program and the job on stdin, capture its output, and stop it after `timeout` seconds.
3. A timeout means `(False, f"timeout ({timeout}s)")`.
4. Find `__RESULT__(\{.*\})` in stdout and return `(ok, why)` from its JSON. If it isn't there the
   process died before printing (a segfault, `os._exit`): return
   `(False, "crashed: " + the last line of stderr)`.

<details><summary>Hint: the call for step 2</summary>

`subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=job, capture_output=True, text=True,
timeout=timeout)`. Too slow raises `subprocess.TimeoutExpired`.

</details>

```bash
uv run pytest tests/test_06_evaluate.py
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

Before you run it, write down a guess: how many of the 500 MBPP tasks does your laptop model pass?
You've seen it write `is_palindrome` and `fib` correctly in lesson 5. Be honest.

```bash
uv run python -m theog.evaluate --ckpt runs/laptop/best.pt --tokenizer data/tokenizer.json --dataset humaneval --out results/
uv run python -m theog.evaluate --ckpt runs/laptop/best.pt --tokenizer data/tokenizer.json --dataset mbpp500 --out results/
```

Greedy (temperature 0, the default here), so a rerun gives the same score. HumanEval takes about
5 minutes on an M2, MBPP about 15. Add `--think` for the other mode, `--limit 10` for a quick
check that everything runs.

Each task prints one line as it goes:

```text
[  1/500] Mbpp/11          FAIL  assert failed: assert remove_Occ("hello","l") == "heo"
[  2/500] Mbpp/12          FAIL  assert failed: assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]
[  3/500] Mbpp/13          FAIL  assert failed: assert count_common(['red','green','black','pink', ...
[  4/500] Mbpp/14          FAIL  code does not run: SyntaxError: expected '(' (<string>, line 1)
...
[  9/500] Mbpp/19          PASS  all tests pass
...
mbpp500: 49/500 passed  ->  results/laptop_mbpp500_no_think_t0.0_20261119T183447.jsonl
```

**These are the scores of this course's laptop run, 3 epochs:** **8/164** on HumanEval and
**49/500** on MBPP. The same data read once scored 2/164 and 19/500 (lesson 5 explains the
difference). An earlier one-epoch run, trained before exercise 2.4 existed, scored 1/164 and 3/500.
Yours should land within a few tasks of ours. How far off was your guess? Most people guess far
too high. Code that *looks* right is cheap.

Now the important part. Don't stare at the score: read *why* it failed. Count the reasons:

```bash
uv run python -c "
import json, collections, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(collections.Counter(r['why'].split(':')[0].split(' named ')[0] for r in rows).most_common(6))
" results/laptop_mbpp500_no_think_t0.0_*.jsonl
```

Here are all three runs side by side:

| reason | HumanEval, before 2.4 | HumanEval, 1 epoch | HumanEval, 3 epochs | MBPP, before 2.4 | MBPP, 1 epoch | MBPP, 3 epochs |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| assert failed: the code runs, the answer is wrong | 118 | 114 | 109 | 29 | 227 | 247 |
| no function named …: right idea, wrong name | 13 | 3 | 2 | 343 | 81 | 26 |
| code does not run: it doesn't even parse | 17 | 29 | 30 | 74 | 67 | 82 |
| TypeError, IndexError, other errors at run time | 15 | 16 | 15 | 51 | 106 | 96 |
| passed | 1 | 2 | 8 | 3 | 19 | 49 |

Read the MBPP columns first. Before exercise 2.4, 343 answers never got as far as being wrong: the
name only appears inside MBPP's asserts, and the model had learned to pick its own. That number was
found exactly this way, by reading the reasons instead of the score, and exercise 2.4 is the fix.
After it, 262 fewer answers have the wrong name. Most of them still fail, but now they fail for real
reasons: "assert failed" went from 29 to 227 and run-time errors from 51 to 106. A wrong name hides
every other kind of failure. 16 of those answers were right.

On HumanEval the table hides the name problem. The prompt *is* the function stub, and the harness
puts the whole prompt in front of the model's code, so the function always exists: when the model
picks another name, the stub with only a docstring is what runs, returns `None`, and fails as
"assert failed". On the 3-epoch run the model defines the function under the right name in only 62
of 164 answers; of the other 102, 71 count as "assert failed", 29 don't parse and 2 fail as "no function named". Exercise 2.4 can't
help here, because HumanEval's prompt already names the function. Counting the names is the same
habit as counting the reasons: read what the model wrote, not only what the harness says.

The third run changed no data at all, only the number of passes over it. Wrong names fell again,
from 81 to 26: the model reads the test line better. And 30 more MBPP tasks pass. Most answers still
fail on "assert failed": three passes taught the model more of the format and the common patterns,
not yet the arithmetic of each task.

Open the results file to see any answer in full:

```json
{"task_id": "Mbpp/11", "passed": false, "why": "assert failed: assert remove_Occ(\"hello\",\"l\") == \"heo\"",
 "code": "def remove_Occ(s: str, char: str) -> str:\n    result = ''\n    for char in s:\n        if char != char:\n            result += char\n    return result", "raw": "..."}
```


## Differences from theOG's own harness

- **MBPP setup order.** A few MBPP tasks have setup code that uses the solution's classes, e.g.
  Mbpp/367 builds a tree from the solution's `Node` class. This harness runs the solution first,
  then the setup. theOG's harness runs the setup first, so those tasks always fail there.
- **graded-30** is theOG's own set of 30 everyday tasks. It is not public, so it's not in this course.

## Going further

1. In the table, which reasons went down after exercise 2.4, and which went up? Why did "assert
   failed" go *up* by 198 when the model got better?
2. Run HumanEval with `--think`. Which tasks pass in only one of the two modes? Which row of the
   table changes the most?
3. Your model writes code that *looks* right. Pick five "assert failed" answers and find the bug by
   hand. Is it the algorithm, an edge case, or did it answer a different question?
4. HumanEval went from 1 to 2 between the first two runs. Why is that not a result, when MBPP's 3 to 19 is?

<details><summary>Answers</summary>

1. "No function named" fell by 262, as it should. Those answers didn't all pass: most moved to
   "assert failed" and to run-time errors, which is why both went up. The model didn't get worse at
   those tasks; you can finally see how it fails them. That's why you read the whole table, not one
   row.
2. On our 3-epoch run, `--think` scored 1/164 (HumanEval/26), against 8/164 without it; only
   HumanEval/26 passes in both modes. Three epochs helped `/no_think` and did nothing for `/think`.
   The reason is in lesson 3's `sequence()`: every training row is written in `/no_think` mode, so
   the model has never seen a prompt ending in `/think`. The more it trains on one mode, the more
   the other is a prompt it doesn't know. Lesson 7, step 1, shows what it writes instead, and the
   rest of lesson 7 is the experiment of training that mode.
3. On our run most bugs are in the algorithm. Mbpp/17 asks for the perimeter of a square and gets
   `n * (n + 1) * (n + 1) // 2`. Mbpp/11 has the right name and the right loop, then reuses `char`
   as the loop variable, so `char != char` is never true and it returns `''`. The shape is right
   and the details aren't. An 8M model trained on 22M tokens has seen too little code to know which
   arithmetic goes with which words; more data and a bigger model (the full track) is what fixes
   that.
4. Greedy decoding makes each run repeatable, but the model's choice at a near-tie can flip with any
   small change: a retrain, a different seed, a slightly different prompt. One task out of 164 is
   noise. Sixteen out of 500, together with a 262-answer drop in the exact failure the fix targets,
   is not. Before believing a change, look for a difference of several tasks and a reason in the
   failure table.

</details>


You can score a model now. So let's ask a harder question: what if the model thinks before it
answers? Next: [lesson 7, the /think span](../07-think-span/README.md).
