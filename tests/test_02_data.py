from collections import Counter

from theog.contamination import Index, grams
from theog.data import Dedup, answer_in_prompt, clean, single_solution, solution_only, with_tests


def test_single_solution_picks_the_def_block_and_skips_tests():
    text = "Here:\n```python\ndef f(x):\n    return x\n```\nTests:\n```python\nassert f(1) == 1\n```\n"
    assert single_solution(text) == ("def f(x):\n    return x", "")


def test_two_solutions_are_ambiguous():
    text = "```python\ndef f():\n    pass\n```\n```python\ndef g():\n    pass\n```"
    assert single_solution(text)[0] is None


def test_solution_only_drops_driver_code_docstrings_and_comments():
    code = ('import math\n\ndef area(r):\n    """Circle area."""\n    # pi r squared\n    return math.pi * r ** 2\n\n'
            'print(area(2))\nif __name__ == "__main__":\n    print(area(3))\n')
    assert solution_only(code) == "import math\n\ndef area(r):\n    return math.pi * r ** 2"


def test_solution_only_rejects_non_python_and_code_without_def():
    assert solution_only("function f() { return 1; }") is None
    assert solution_only("print('hi')") is None


def test_with_tests_names_the_function_only_when_the_prompt_does_not():
    answer = "def is_pal(s):\n    return s == s[::-1]"
    tests = 'assert is_pal("aba")\nassert not is_pal("ab")\nprint("ok")\nassert is_pal("")\nassert is_pal("x")'
    out = with_tests("Check if a string is a palindrome.", answer, "", tests)
    assert out == ('Check if a string is a palindrome.\nYour code should pass these tests:\n\n'
                   'assert is_pal("aba")\nassert not is_pal("ab")\nassert is_pal("")')
    assert with_tests("Write `is_pal`.", answer, "", tests) == "Write `is_pal`."   # already named
    assert with_tests("Check it.", answer, "", "") == "Check it."                 # no tests to show


def test_answer_in_prompt():
    answer = "def f(xs):\n    return sorted(xs, key=lambda x: -x)"
    assert answer_in_prompt("Fix: return sorted(xs, key=lambda x: -x)", answer)
    assert not answer_in_prompt("Sort a list in descending order.", answer)


def test_dedup_exact_and_skeleton_cap():
    drops, d = Counter(), Dedup(max_per_skeleton=2)
    rows = [{"user": f"task {i}", "answer": f"def f{i}(x):\n    return x + 1"} for i in range(4)]
    kept = [d.keep(r, drops) for r in rows]
    assert kept == [True, True, False, False]  # same shape, renamed: capped at 2
    assert not d.keep({"user": "task 0", "answer": "def g(): pass"}, drops)  # same prompt


def test_clean_keeps_a_good_row():
    row = {"source": "s", "user": "Add one.", "answer": "def f(x):\n    return x + 1\nprint(f(1))"}
    assert clean(row, Counter())["answer"] == "def f(x):\n    return x + 1"


def test_contamination_index():
    task = "def has_close_elements(numbers, threshold): " + " ".join(f"word{i}" for i in range(30))
    idx = Index({"HumanEval/0:prompt": task})
    assert idx.hits("some prefix " + task) == ["HumanEval/0:prompt"]
    assert idx.hits("an unrelated row about sorting lists in python with a key function") == []


def test_grams():
    assert grams("a b c d", n=2) == {("a", "b"), ("b", "c"), ("c", "d")}
    assert grams("f(x)", n=2) == {("f", "("), ("(", "x"), ("x", ")")}   # punctuation is its own word
    assert grams("too short", n=13) == set()


def test_skeleton_ignores_names():
    from theog.data import skeleton
    assert skeleton("def f(a):\n    return a + 1") == skeleton("def g(x):\n    return x + 1")
    assert skeleton("def f(a):\n    return a + 1") != skeleton("def f(a):\n    return a - 1")
