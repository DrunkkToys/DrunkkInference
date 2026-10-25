"""The special tokens of the chat format (lesson 1, lesson 3).

They get the first ids of the vocabulary, in this order, so every id below is
fixed no matter what text the tokenizer is trained on.
"""

SPECIAL_TOKENS = [
    "<|pad|>",        # 0  fills the end of a training window
    "<|User|>",       # 1  opens the user turn
    "<|/User|>",      # 2  closes it
    "<|Assistant|>",  # 3  the model's turn starts
    "<|Think|>",      # 4  opens the think span (lesson 6)
    "<|/Think|>",     # 5  closes it
    "<|Answer|>",     # 6  opens the answer
    "<|/Answer|>",    # 7  closes it: generation stops here
    "<|end|>",        # 8  end of the conversation
]

PAD, USER, USER_CLOSE, ASSISTANT, THINK, THINK_CLOSE, ANSWER, ANSWER_CLOSE, END = range(len(SPECIAL_TOKENS))

# Generation stops at any of these.
STOP_IDS = (ANSWER_CLOSE, END, USER)

# Label value that cross-entropy ignores: "no loss on this position".
IGNORE = -100
