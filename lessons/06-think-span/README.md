# Lesson 6: the /think span

**You write:** no new code: you already wrote both modes in `prompt_ids` (exercise 5.4). This
lesson is about what they do, and an experiment that breaks the prompt on purpose. **Time:** about
20 minutes.

Every theOG sequence has a think span between the prompt and the answer:

```
<|Assistant|><|Think|><|/Think|><|Answer|>
```

and every prompt ends with a mode, `/no_think` or `/think`.

## What the span is

The think span is theOG's own design, worked out independently by its author: **a pause before the
answer.** In `/think` mode the model gets `<|Think|>` and not `<|/Think|>`, so it decides itself
when to close the span and start the answer.

It is **not chain of thought.** Chain of thought means the model writes out its reasoning in words,
and it has to be trained on reasoning text. Here the span is empty in every training row. The
pause-tokens paper ([Goyal et al., 2023](https://arxiv.org/abs/2310.02226)) arrived at a similar idea
separately: extra tokens before the answer that give the model more computation steps.

`sequence(..., think_text=...)` (lesson 3) can put text inside the span. This course never does. The
slot is there for your own experiments.

## The two modes in `prompt_ids`

You wrote this in exercise 5.4:

- `/no_think`: the prompt ends `<|Think|><|/Think|><|Answer|>`. The span is closed for the model,
  so it starts writing code at once.
- `/think`: the prompt ends `<|Think|>`. The model writes the rest: it chooses when to close the
  span.

In training, `sequence()` always writes the span and labels its tokens, so the model has learned
`<|Think|>` → `<|/Think|>` → `<|Answer|>`.

## Try both modes

```bash
uv run python -m theog.generate --ckpt runs/laptop/best.pt --tokenizer data/tokenizer.json \
    --temperature 0 --think "Write a function that checks if a number is prime."
```

````text
<|/Think|><|Answer|>
```python
def is_prime(num):
    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    for i in range(3, int(num ** 0.5) + 1, 2):
        if num % i == 0:
            return False
    return True

def is_prime(n):
    ...                      # a second correct version
def is_prime(n):
    ...                      # a third, the classic 6k ± 1 test
def is_prime(n):
    if n <= 1:
        return n
    if n == 2:
        return 2
    if n % 2 == 0:
        return is_prime(n - 1)
    else:
        return is_prime(n - 1)
```
````

(Shortened here: the second and third functions are complete in the real output.)

The output starts with what the model wrote after `<|Think|>`, special tokens shown. It closed the
span and opened the answer by itself, as it was trained to.

Then look at the answer. The first `is_prime` is correct, and so are the next two, written three
different ways. Then the model kept going and wrote a fourth that returns numbers instead of
True/False. In Python the last definition wins, so this answer **fails**. Run the same task without
`--think` and the answer ends with the same broken function.

That's what an 8M-parameter model trained for 48 minutes does: it has learned what correct code
looks like, but not reliably when to stop. Lesson 7 measures how often that costs it the task.

## What it did for theOG

At 50M parameters and 2.49B tokens the span starts to matter. theOG-50M v3, from the
[model card](https://huggingface.co/DrunkkToys/theOG-50M), PyTorch checkpoint at step 346,000,
greedy:

| mode | graded-30 | HumanEval-164 | MBPP-500 |
|---|:-:|:-:|:-:|
| `/no_think` | 26/30 | 42/164 | 91/500 |
| `/think` | 28/30 | 41/164 | 93/500 |
| either mode (a task counts if it passes in one of them) | 28/30 | 42/164 | 96/500 |

The two modes solve partly different tasks. Lesson 7 can score both (`--think`), so you can measure
the same for your model.

## Going further

1. In `/think` mode, what does your model write right after `<|Think|>`? Why that?
2. Break your `prompt_ids` on purpose: put a space before the newline (`" /no_think \n"`) and
   generate a few answers at `--temperature 0`. What changes? (Put it back afterwards; the lesson 5
   test catches it.)
3. theOG never trained on text inside the span. How could a span with no text still change the
   answer?

<details><summary>Answers</summary>

1. `<|/Think|><|Answer|>`, immediately. Every training row had an empty span, so after `<|Think|>`
   the only token it has ever seen is `<|/Think|>`. The two modes differ only in that the model
   wrote two tokens itself instead of being given them.
2. On the laptop model, at temperature 0, two of the four tasks we tried changed. `is_prime`
   renamed its argument from `num` to `n`. `largest` went from a loop to
   `max_num = max(nums); return max(nums)`. Not reliably worse, just different. That's the
   problem: your lesson 7 score would measure the model on prompts it never saw in training, and
   you'd never know why it moved.
3. Each extra position is one more pass through every block, with its own attention over the
   prompt. The answer tokens can attend to those positions. The model can learn to use them as
   scratch space even without being told to, and the effect shows up in the scores, not in any text
   you could read.

</details>

Next: [lesson 7, evaluation](../07-evaluation/README.md).
