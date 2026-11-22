# Lesson 2: the data

**You write:** the rules that strip an answer down to its solution, catch answers copied from the
prompt, remove duplicates, tell the model what to call its function, and keep benchmark tasks out
of the training set. **You get:**
`data/rows.clean.jsonl`, about 95,000 clean tasks on the laptop track. **You learn:** why most
of a small model's quality is decided before training starts, and how to keep the exam out of the
training set. **Time:** about an hour, plus a few minutes of downloading.

Everybody wants to talk about the architecture. Nobody wants to talk about the data. Guess which
one decides how good your model is.

A model learns only what its data shows it. A small model is even more sensitive: it has no spare
capacity for broken examples, duplicates, or tasks whose answer is already written in the question.
Most of my work on theOG went into this step, not into lesson 4.

This is the Phi approach ([Gunasekar et al., 2023, "Textbooks Are All You Need"](https://arxiv.org/abs/2306.11644)):
a small set of clean tasks a model can learn from beats a much bigger pile of scraped code. The
other well-known rule, Chinchilla's 20 tokens per parameter, assumes every token is worth the same,
and these rules exist because they aren't. Lesson 5 shows the second half of the Phi approach:
reading that clean set more than once.

## What one row looks like

```json
{"source": "opencodeinstruct",
 "user": "Write a function to find the subarray with the largest sum in a given array.",
 "answer": "def find_largest_sum(nums):\n    max_so_far = max_ending_here = nums[0]\n    ...",
 "entry_point": "", "tests": "..."}
```

One task in plain English, one Python solution. That's all the model will ever see. So every
row better be worth seeing.

## Download

| source | licence | why |
|---|---|---|
| [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | CC BY 4.0 | every answer comes with unit tests the dataset authors ran; we keep only rows where all of them passed |
| [OpenCoder-LLM/opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2), `educational_instruct` | MIT | short, textbook-style tasks |

Both are downloaded at a **pinned revision** (a commit hash in `theog/data.py`), so the same command
gives the same rows next year. Laptop track, one file per source:

```bash
uv run python -m theog.data download --raw data/raw --parts 1
```

```text
nvidia/OpenCodeInstruct @ 8f3ba5ba (CC BY 4.0): 1 file(s)
OpenCoder-LLM/opc-sft-stage2 @ 7d28f40d (MIT): 1 file(s)
```

178 MB. **Full track:** leave out `--parts`. That's 6.9 GB, almost all of it OpenCodeInstruct's
50 files.

## The cleaning rules

`build()` in `theog/data.py` reads every row and drops it at the first rule it breaks, counting
why. These are already written for you:

- **The tests must have passed** (OpenCodeInstruct only). An answer that fails its own tests teaches
  bugs.
- **Exactly one solution block.** `single_solution()` wants exactly one fenced Python block that
  defines a function or class. Zero means no code; several means we can't tell which one is the
  answer.

You write the next four. Each one exists because a model taught me the hard way that it was
missing.

### Exercise 2.1: `solution_only`

Answers in these datasets come with extras: a docstring, comments, a usage example, a
`if __name__ == "__main__":` demo. The task asked for a function, and the model should learn to
write just that. Given this:

```python
import math

def area(r):
    """Circle area."""
    # pi r squared
    return math.pi * r ** 2

print(area(2))
if __name__ == "__main__":
    print(area(3))
```

return exactly this:

```python
import math

def area(r):
    return math.pi * r ** 2
```

Tempted to do this with string tricks and a couple of regexes? Don't. Parse the code with `parse()` (it returns an `ast.Module`, or
None for code that isn't Python 3) and work on the tree:

- At the top level, keep only node types in `_KEEP` (imports, functions, classes, assignments).
  Prints, calls, loops and `if` blocks go.
- Remove the docstring from the module and from every function and class. `_drop_docstring(body)`
  does one body; `ast.walk(tree)` visits every node.
- If nothing defines a function or class any more (`_defines(tree)`), return None.
- `ast.unparse(tree)` writes the code back. Comments aren't in the tree, so they're gone for free,
  and every answer comes out in the same formatting style, which helps a small model too.

```bash
uv run pytest tests/test_02_data.py -k solution_only
```

<details><summary>Solution</summary>

```python
def solution_only(code):
    tree = parse(code)
    if tree is None:
        return None
    tree.body = [n for n in _drop_docstring(tree.body) if isinstance(n, _KEEP)]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.body = _drop_docstring(node.body)
    if not _defines(tree):
        return None
    return ast.unparse(tree)
```

</details>

### Exercise 2.2: `answer_in_prompt`

Some prompts are "fix this code" tasks with most of the answer pasted in. Train on those and the
model learns to copy, not to solve. Return True when a real line of the answer appears in the
prompt. "Real" means at least `min_len` characters and not a `def`, `class`, `import`, `from` or
`@decorator` line, because those repeat the task's own signature legitimately. Collapse runs of
whitespace on both sides first (`_WS.sub(" ", ...)`) so indentation differences don't hide a copy.

```bash
uv run pytest tests/test_02_data.py -k answer_in_prompt
```

<details><summary>Solution</summary>

```python
def answer_in_prompt(user, answer, min_len=20):
    prompt = _WS.sub(" ", user)
    for line in answer.splitlines():
        s = _WS.sub(" ", line).strip()
        if len(s) >= min_len and not s.startswith(("def ", "class ", "import ", "from ", "@")) and s in prompt:
            return True
    return False
```

</details>

### Exercise 2.3: `Dedup.keep`

Three checks, in this order:

1. Same prompt as a row already kept: drop it, count `"duplicate prompt"`.
2. Same answer: drop it, count `"duplicate answer"`.
3. Same **shape**: `skeleton()` renames every name to `_`, so `def add(a, b): return a + b` and
   `def plus(x, y): return x + y` have the same skeleton. At most `self.max` rows per skeleton
   (3 by default). A few renamed variants are useful variety; hundreds would crowd out other tasks.

If it passes all three, remember its prompt hash, answer hash and skeleton, and return True. Use
`sha()` for the hashes, so you store 64 characters per row instead of the whole text.

```bash
uv run pytest tests/test_02_data.py -k dedup
```

<details><summary>Solution</summary>

```python
def keep(self, row, drops):
    p, a, s = sha(row["user"]), sha(row["answer"]), skeleton(row["answer"])
    if p in self.prompts:
        drops["duplicate prompt"] += 1
        return False
    if a in self.answers:
        drops["duplicate answer"] += 1
        return False
    if self.shapes[s] >= self.max:
        drops[f"more than {self.max} rows with the same solution shape"] += 1
        return False
    self.prompts.add(p)
    self.answers.add(a)
    self.shapes[s] += 1
    return True
```

</details>

### Exercise 2.4: `with_tests`

Read this opc_educational row:

```text
user:   Write a function to check if a given string is a palindrome. ...
answer: def is_palindrome(s): ...
tests:  assert is_palindrome("No lemon, no melon") == True
        assert is_palindrome("hello") == False
```

The prompt never says the function is called `is_palindrome`. Train on 68,000 rows like this and
the model learns that it picks the name. Then lesson 6 asks it for `remove_vowels`, it writes a
perfect `remove_vowel_chars`, and the test fails with "no function named remove_vowels".

This isn't a guess. The first version of this course trained without this rule, and the laptop
model passed **1 of 164** HumanEval tasks and **3 of 500** MBPP tasks. In 343 of the 500 MBPP
answers, the code used a different name from the one the tests call, even though MBPP's prompts
show the name in their asserts. The model had never learned to look for it.

The fix uses what the row already has: its tests. When the name isn't in the prompt, append up to
three of the row's `assert` lines that call it, with the exact wording MBPP uses:

```text
Write a function to check if a given string is a palindrome. ...
Your code should pass these tests:

assert is_palindrome("No lemon, no melon") == True
assert is_palindrome("A man a plan a canal Panama") == True
assert is_palindrome("hello") == False
```

`function_name(answer, entry_point)` (written for you) gives the name. Keep only lines that start
with `assert` and contain the name. If the name is already in the prompt, or there's no such line,
return the prompt unchanged.

`build()` calls this *after* dedup, so two copies of a task with different test lists still count
as duplicates.

```bash
uv run pytest tests/test_02_data.py -k with_tests
```

<details><summary>Solution</summary>

```python
def with_tests(user, answer, entry_point, tests, n=3):
    name = function_name(answer, entry_point)
    if not name or name in user:
        return user
    asserts = [t.strip() for t in tests.splitlines() if t.strip().startswith("assert") and name in t][:n]
    if not asserts:
        return user
    return user + "\nYour code should pass these tests:\n\n" + "\n".join(asserts)
```

</details>

On the laptop data this adds tests to 74,324 of the 95,483 rows.

### Run the build

Before you run it, guess: of the rows you downloaded, what share survives the rules? Write it
down.

```bash
uv run python -m theog.data build --raw data/raw --out data/rows.jsonl
```

```text
kept    27,303  opencodeinstruct
kept    68,462  opc_educational
drop    65,267  a unit test of the source did not pass
drop    29,983  more than 3 rows with the same solution shape
drop    19,221  duplicate prompt
drop     7,018  duplicate answer
drop     1,022  answer code is inside the prompt
drop         2  answer is not Python 3 with a def/class
```

About a minute on an M2. Read those numbers: more than half the rows we read were thrown away.
Was your guess that brutal?
The skeleton cap alone removed 30,000 near-copies. If your counts differ from these, one of
exercises 2.1 to 2.3 behaves differently from the reference. The tests cover the main cases, but
the counts on real data are the real check.

## Contamination: don't train on the exam

Lesson 6 scores the model on HumanEval and MBPP. If a training row contains one of those tasks, the
model answers it from memory, and the score measures memory, not skill. Studying with the answer
key is not studying. And public code datasets are full of benchmark tasks, copied in by someone,
somewhere.

`theog/contamination.py` splits text into words and builds every run of 13 consecutive words
(**13-grams**) of every benchmark prompt and reference solution. A training row is dropped when it
contains at least half of the 13-grams of any one of them.

### Exercise 2.5: `grams`

```python
grams("a b c d", n=2)   # {("a", "b"), ("b", "c"), ("c", "d")}
grams("f(x)", n=2)      # {("f", "("), ("(", "x"), ("x", ")")}: punctuation is a word too
```

Words are what the regex `_WORD` finds.

<details><summary>Solution</summary>

```python
def grams(text, n=N):
    words = _WORD.findall(text)
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
```

</details>

### Exercise 2.6: `Index.hits`

The index is built for you: `self.where[gram]` is the set of benchmark texts that contain that
gram, and `self.size[key]` is how many grams that benchmark text has. For a training row, count
how many of its grams land in each benchmark text, and return the keys where
`count / size >= THRESHOLD`.

Checking each row against each of the 1,328 benchmark texts one by one would be slow. With the
index, each gram is one dict lookup.

```bash
uv run pytest tests/test_02_data.py
```

All 11 should pass now.

<details><summary>Solution</summary>

```python
def hits(self, text):
    count = Counter()
    for x in grams(text):
        for key in self.where.get(x, ()):
            count[key] += 1
    return [k for k, c in count.items() if c / self.size[k] >= THRESHOLD]
```

</details>

### Run it

```bash
uv run python -m theog.contamination --rows data/rows.jsonl --out data/rows.clean.jsonl
```

```text
kept 95,483 rows; dropped rows matching 90 benchmark tasks
      60  Mbpp/101
      31  HumanEval/19
      26  Mbpp/504
       8  Mbpp/316
       8  Mbpp/401
...
```

The first run downloads HumanEval and MBPP (a few MB). 282 rows matched 90 benchmark tasks: one MBPP
task alone had 60 near-copies in the training data. Without this step, your lesson 6 score would
include those tasks answered from memory.

Before exercise 2.4, the same check matched only 58 tasks. The test lines it added made some rows
look much more like MBPP, which is the point of the contamination check: run it *last*, on exactly
the text you will train on.

Solutions shorter than 13 words have no 13-gram and aren't checked. That's a known hole. I'd
rather tell you than pretend it isn't there.

Now go back to [lesson 1](../01-tokenizer/README.md#run-it-on-real-data) and train the tokenizer
on `rows.clean.jsonl`.

## theOG's own pipeline

theOG-50M v3 applied more rules of this kind to a larger mix: 12 public Hugging Face sources,
4.96M rows, 2.49B tokens (the list is on the
[model card](https://huggingface.co/DrunkkToys/theOG-50M)). The course keeps two of those sources and
the rules that drop the most rows.

## Going further

1. Read 20 random rows: `uv run python -c "import json,random; rows=open('data/rows.clean.jsonl').readlines(); [print(json.loads(r)['user'][:200], '\n---') for r in random.sample(rows, 20)]"`.
   Would you keep them all? What rule would you add?
2. Set `MAX_PER_SKELETON` to 1 and to 10 and rebuild. How many rows does each keep?
3. Why are the tested OpenCodeInstruct rows read before the opc_educational rows?

<details><summary>Answers</summary>

1. There's no right answer. Common findings: tasks that are really "explain" questions, very long
   class hierarchies, near-identical "write a function to add two numbers" variants. Any new rule
   goes in `clean()`, with a drop reason so the build counts it.
2. On the laptop data, before the contamination check: 1 per shape keeps 82,306 rows, 3 keeps
   95,765, 10 keeps 107,169. Going from 3 to 1 throws away 13,000 rows that differ from a kept row
   only in names; going to 10 lets 11,000 of them back in. Whether the model gets better or worse is
   a question for lesson 6, not for intuition: retrain and compare the scores.
3. Dedup keeps the first copy it sees. Reading the source with passing unit tests first means that
   when the same task is in both, the verified answer wins.

</details>

Next: back to [lesson 1's real-data run](../01-tokenizer/README.md#run-it-on-real-data) if you
haven't done it, then [lesson 3, the chat format and packing](../03-format-and-packing/README.md).
