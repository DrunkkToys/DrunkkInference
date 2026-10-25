from tokenizers import Tokenizer

from theog.tokens import SPECIAL_TOKENS


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
