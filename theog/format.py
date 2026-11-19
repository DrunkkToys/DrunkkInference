"""Lesson 3: turn one row into one training sequence.

    <|User|>\n{task} /no_think\n<|/User|><|Assistant|>
    <|Think|><|/Think|>
    <|Answer|>\n```python\n{code}\n```\n<|/Answer|><|end|>

input_ids is that sequence. labels[i] is the token the model must predict
after reading input_ids[..i], or IGNORE where there is nothing to learn: the
prompt is given, never predicted.
"""
from __future__ import annotations

from tokenizers import Tokenizer

from theog.tokens import (ANSWER, ANSWER_CLOSE, ASSISTANT, END, IGNORE, THINK, THINK_CLOSE,
                          USER, USER_CLOSE)


def prompt_ids(tok: Tokenizer, user: str, think: bool = False) -> list[int]:
    """What the model reads before it writes: exactly the start of the
    training sequence of lesson 3. /no_think: the prompt ends with
    [THINK, THINK_CLOSE, ANSWER], so the model starts on the answer. /think:
    it ends with [THINK], so the model writes inside the span first.
    """
    # EXERCISE 5.4: lessons/05-training/README.md
    # Hint: Same head as in sequence(): [USER] + encode('\n{task} /mode\n') + [USER_CLOSE, ASSISTANT].
    raise NotImplementedError("exercise 5.4: write prompt_ids")


def sequence(tok: Tokenizer, user: str, answer: str, think_text: str = "") -> dict:
    """One training sequence for one row, as {"input_ids", "labels"}.

    input_ids: prompt + think span + answer, laid out as in the module
    docstring. The prompt is [USER] + "\n{user stripped} /no_think\n"
    (or /think when think_text is set) + [USER_CLOSE, ASSISTANT].
    The think span is [THINK] + "\n{think_text}\n" (only if set) + [THINK_CLOSE].
    The answer is [ANSWER] + "\n```python\n{answer}\n```\n" + [ANSWER_CLOSE, END].

    labels[i] = input_ids[i + 1] at every position of the think span and the
    answer; IGNORE at every prompt position (up to and including <|Assistant|>)
    and at the very last position, which has no next token.
    """
    # EXERCISE 3.1: lessons/03-format-and-packing/README.md
    # Hint: Build three lists, join them, and keep a parallel list of booleans
    #       (False for prompt positions). Then shift by one.
    raise NotImplementedError("exercise 3.1: write sequence")
