"""Lesson 1. Part 1: BPE by hand. Part 2: the real tokenizer."""
from tokenizers import Tokenizer

from theog.tokenizer import bpe_by_hand, merge, most_frequent_pair
from theog.tokens import SPECIAL_TOKENS


# ── part 1: BPE by hand ───────────────────────────────────────────────────

CODE = """def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def total(items):
    result = 0
    for item in items:
        result = add(result, item)
    return result


def average(items):
    if not items:
        return 0
    return total(items) / len(items)
"""

def test_most_frequent_pair():
    assert most_frequent_pair([1, 2, 3, 1, 2]) == (1, 2)
    assert most_frequent_pair([5, 6, 7]) == (5, 6)      # a tie: the first pair seen wins
    assert most_frequent_pair([4]) is None


def test_merge_replaces_left_to_right():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    assert merge([7, 7, 7], (7, 7), 50) == [50, 7]     # overlapping: left first
    assert merge([1, 3], (1, 2), 99) == [1, 3]


def test_bpe_by_hand_shortens_text_and_round_trips():
    text = CODE
    ids, vocab = bpe_by_hand(text, n_merges=60)
    assert len(ids) < 0.6 * len(text.encode())
    assert b"".join(vocab[i] for i in ids).decode() == text      # nothing lost
    assert b"    " in vocab.values()                             # indentation became one token


# ── part 2: the real tokenizer ────────────────────────────────────────────

def test_special_tokens_have_fixed_ids(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    assert [tok.token_to_id(t) for t in SPECIAL_TOKENS] == list(range(len(SPECIAL_TOKENS)))


def test_any_text_round_trips(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    for text in ["def f(x):\n    return x + 1\n", "naïve café → ünïcode 🐍", "\t\ttabs  and   spaces"]:
        assert tok.decode(tok.encode(text).ids) == text


def test_special_token_is_one_id(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    assert tok.encode("<|Answer|>").ids == [SPECIAL_TOKENS.index("<|Answer|>")]


def test_real_tokenizer_learned_return(tokenizer_file):
    tok = Tokenizer.from_file(str(tokenizer_file))
    assert "Ġreturn" in tok.get_vocab()                # ` return` became one token
