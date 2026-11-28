# Lesson 10: changing the attention

**The extra lesson, after the course.** You've built, trained, scored, exported and published a
model. Now you change the part everybody changes: the attention. **You need:** your lesson 4 model,
a trained run (lesson 5), lesson 6's harness and paired test, and lesson 7's method and its `psr`
span. **You learn:** how to test a change a paper swears by on a model 1,000 times smaller, how to
look inside the one that wins, and why one small model keeps trading HumanEval for MBPP.
**Time:** about an hour and a half to read all three steps. Redoing them is days of laptop.

Every number in this lesson is in the steps, with the commands that made it. So you choose:

- **Read it.** Take my runs, follow the reasoning, do the "Try it" questions. That's the lesson.
- **Redo it.** Train the runs yourself, at a seed of your own, and check me. That's the
  experiment. Write your prediction before every run.

| step | what | you do | time to redo |
|---|---|---|---|
| [1](1-attention-experiments.md) | seven changes to the attention block: GQA, heads, a gate, RoPE base | your prediction, then the runs | about 2½ hours a run |
| [2](2-the-gate.md) | the gate inside, with the think span, and the mistake you can see | read it, run the diagnostics | about 16 hours |
| [3](3-think-and-attention.md) | the think span and the gate together: the seesaw, and two models as one | your prediction | about 10½ hours a seed |

Tests: `uv run pytest tests/test_04_attnlab.py`. The code is `theog/attnlab.py`. It swaps the
attention after the model is built, so your `theog/model.py` stays exactly as you wrote it.

One word you'll meet here first: **Spearman**. It's a correlation on ranks. Sort the runs by one
number, sort them by another, and ask how well the two orders agree: +1 is the same order, −1 is
the exact opposite, 0 is no relation. It cares about "higher or lower", not by how much. That's
what you want when you have ten runs and a hunch.

The whole lesson is lesson 7's method, pointed at the model instead of the span. One change at a
time, both benchmarks every time, a second seed before you believe anything. Spoiler: the change
that wins isn't the one you'd bet on, and what it does inside isn't what the paper says.

Start: [step 1, changing the attention block](1-attention-experiments.md).
