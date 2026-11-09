# Lesson 6: the /think span

Every theOG sequence has a think span between the prompt and the answer:

```
<|Assistant|><|Think|><|/Think|><|Answer|>
```

and every prompt ends with a mode, `/no_think` or `/think`.

## What it is

The think span is theOG's own design, worked out independently by its author: **a pause before the
answer.** In `/think` mode the model is given `<|Think|>` and not `<|/Think|>`, so it decides itself
when to close the span and start the answer. It writes no reasoning text there.

It is **not chain of thought.** Chain of thought means the model writes out its reasoning in words,
and it has to be trained on reasoning text. The think span carries no text.

The pause-tokens paper ([Goyal et al., 2023](https://arxiv.org/abs/2310.02226)) arrived at a similar
idea separately: extra tokens before the answer that give the model more computation steps.

## In the code

`format.py` builds both modes from the same tokens:

```python
def prompt_ids(tok, user, think=False):
    mode = "think" if think else "no_think"
    head = [USER] + tok.encode(f"\n{user.strip()} /{mode}\n").ids + [USER_CLOSE, ASSISTANT]
    return head + ([THINK] if think else [THINK, THINK_CLOSE, ANSWER])
```

- `/no_think`: the prompt already closes the span and opens the answer. The model goes straight to
  code.
- `/think`: the prompt stops after `<|Think|>`. The model writes the rest.

In training, `sequence()` always writes the span, and labels its tokens. So the model learns
`<|Think|>` → `<|/Think|>` → `<|Answer|>`, and in `/think` mode it can close the span on its own.

`sequence(..., think_text=...)` can put text inside the span. This course never does: every
training row has an empty span and `/no_think`. The text slot is there for your own experiments.

## Try both modes

```bash
uv run python -m theog.generate --ckpt runs/theog-50m/best.pt --tokenizer data/tokenizer.json \
    "Write a function that checks if a number is prime."
```

```bash
uv run python -m theog.generate --ckpt runs/theog-50m/best.pt --tokenizer data/tokenizer.json \
    --think "Write a function that checks if a number is prime."
```

The `/think` output starts with whatever the model wrote after `<|Think|>`, special tokens shown.

## What it did for theOG

theOG-50M v3, from the [model card](https://huggingface.co/DrunkkToys/theOG-50M), PyTorch
checkpoint at step 346,000, greedy:

| mode | graded-30 | HumanEval-164 | MBPP-500 |
|---|:-:|:-:|:-:|
| `/no_think` | 26/30 | 42/164 | 91/500 |
| `/think` | 28/30 | 41/164 | 93/500 |
| either mode (a task counts if it passes in one of them) | 28/30 | 42/164 | 96/500 |

The card notes that the two modes solve partly different tasks. Lesson 7 can score both
(`--think`), so you can measure the same for your model.

## Try this

- Score your model in both modes on HumanEval. Which tasks pass in only one of them?
