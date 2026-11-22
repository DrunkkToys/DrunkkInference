"""Lesson 6, comparing two models. Plumbing, not an exercise: this passes from the start."""
import json

import pytest

from theog.compare import compare, load, mcnemar_p


def test_p_values():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(3, 3) == 1.0
    assert mcnemar_p(0, 5) == pytest.approx(2 / 32)      # 5 coin flips all one way
    assert mcnemar_p(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_p(2, 9) == pytest.approx(0.0654, abs=1e-4)
    assert mcnemar_p(9, 2) == mcnemar_p(2, 9)


def test_only_discordant_tasks_count():
    a = {"t1": True, "t2": True, "t3": False, "t4": False}
    b = {"t1": True, "t2": False, "t3": True, "t4": False}
    r = compare(a, b)
    assert (r["a"], r["b"], r["both"]) == (2, 2, 1)
    assert r["only_a"] == ["t2"] and r["only_b"] == ["t3"]
    assert r["p"] == 1.0


def test_load_and_mismatch(tmp_path):
    f = tmp_path / "r.jsonl"
    f.write_text("\n".join(json.dumps({"task_id": t, "passed": p}) for t, p in [("x", True), ("y", False)]))
    assert load(f) == {"x": True, "y": False}
    with pytest.raises(ValueError):
        compare({"x": True}, {"y": True})
