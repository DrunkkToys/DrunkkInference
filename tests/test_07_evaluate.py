from theog.evaluate import run_tests
from theog.generate import extract_code


def test_passing_solution():
    assert run_tests("def add(a, b):\n    return a + b", "add", ["assert add(1, 2) == 3"]) == (True, "all tests pass")


def test_failure_reasons_are_specific():
    assert run_tests("def add(a, b):\n    return a - b", "add", ["assert add(1, 2) == 3"])[1].startswith("assert failed")
    assert run_tests("def plus(a, b):\n    return a + b", "add", ["assert add(1, 2) == 3"])[1] == "no function named add"
    assert run_tests("def add(a, b) return", "add", [])[1].startswith("code does not run")
    assert run_tests("def add(a, b):\n    while True: pass", "add", ["add(1, 2)"], timeout=2)[1].startswith("timeout")


def test_extract_code():
    assert extract_code("\n```python\ndef f():\n    pass\n```\n") == "def f():\n    pass"
    assert extract_code("```python\ndef f():\n    pa") == "def f():\n    pa"  # cut off: kept as is
