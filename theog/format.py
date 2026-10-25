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
    """What the model reads before it writes. /no_think closes the think span
    for the model; /think leaves it open (lesson 6)."""
    mode = "think" if think else "no_think"
    head = [USER] + tok.encode(f"\n{user.strip()} /{mode}\n").ids + [USER_CLOSE, ASSISTANT]
    return head + ([THINK] if think else [THINK, THINK_CLOSE, ANSWER])


def sequence(tok: Tokenizer, user: str, answer: str, think_text: str = "") -> dict:
    mode = "think" if think_text else "no_think"
    prompt = [USER] + tok.encode(f"\n{user.strip()} /{mode}\n").ids + [USER_CLOSE, ASSISTANT]
    think = [THINK] + (tok.encode(f"\n{think_text}\n").ids if think_text else []) + [THINK_CLOSE]
    body = [ANSWER] + tok.encode(f"\n```python\n{answer}\n```\n").ids + [ANSWER_CLOSE, END]
    input_ids = prompt + think + body
    supervised = [False] * len(prompt) + [True] * (len(think) + len(body))
    # Shift by one: position i learns the NEXT token. The last position has
    # no next token, so it learns nothing.
    labels = [input_ids[i + 1] if supervised[i] else IGNORE for i in range(len(input_ids) - 1)] + [IGNORE]
    return {"input_ids": input_ids, "labels": labels}
