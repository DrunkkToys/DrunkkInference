"""Lesson 6: score a model on HumanEval (164 tasks) and MBPP-500.

  uv run python -m theog.evaluate --ckpt runs/theog-50m/best.pt --tokenizer data/tokenizer.json \
      --dataset humaneval --out results/
  uv run python -m theog.evaluate --ckpt runs/a/final.pt --ensemble runs/b/final.pt --tokenizer data/tokenizer.json \
      --design psr --dataset humaneval --out results/      # two models as one (lesson 4, step 12)
  uv run --extra sanity python -m theog.evaluate --hf Salesforce/codegen-350M-mono --dataset humaneval --out results/

A task passes when the generated code and the task's own tests run clean in a
fresh Python process within 10 seconds. Every task prints PASS/FAIL and the
reason; the results file keeps the code the model wrote, so any score can be
checked by reading it.

Run the --hf sanity gate first. If a known model scores far from its known
number, the harness is broken, not the model.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# (repo id, revision, file)
HUMANEVAL = ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
             "openai_humaneval/test-00000-of-00001.parquet")
MBPP = ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "full/test-00000-of-00001.parquet")

# Direct completion stops here: the function is finished.
HF_STOPS = ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert ", "\nif __name__", "\nprint(", "\n#"]


def _parquet(spec) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo, rev, f = spec
    return pq.read_table(hf_hub_download(repo, f, repo_type="dataset", revision=rev)).to_pylist()


def load_tasks(dataset: str) -> list[dict]:
    """Every task as {task_id, prompt, entry_point, tests, setup, solution}."""
    if dataset == "humaneval":
        return [{"task_id": r["task_id"], "prompt": r["prompt"], "entry_point": r["entry_point"],
                 "tests": [f"{r['test']}\ncheck({r['entry_point']})"], "setup": "",
                 "solution": r["canonical_solution"]} for r in _parquet(HUMANEVAL)]
    tasks = []
    for r in _parquet(MBPP):  # the original test split: task ids 11-510
        m = re.match(r"assert\s+\(?\s*(?:not\s+)?\(?\s*([A-Za-z_]\w*)\s*\(", r["test_list"][0])
        tasks.append({"task_id": f"Mbpp/{r['task_id']}", "entry_point": m.group(1) if m else "",
                      "prompt": r["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(r["test_list"]),
                      "tests": r["test_list"], "setup": r["test_setup_code"] or "", "solution": r["code"]})
    return tasks


_RUNNER = r'''
import json, sys
job = json.loads(sys.stdin.read())
ns = {"__name__": "__solution__"}  # the code's own `if __name__ == "__main__":` demo does not run
def result(ok, why):
    print("\n__RESULT__" + json.dumps({"ok": ok, "why": why}))
    sys.exit(0)
try:
    exec(job["solution"], ns)
    exec(job["setup"], ns)  # after the code: MBPP setup may build objects of the code's classes
except BaseException as e:
    result(False, f"code does not run: {type(e).__name__}: {e}"[:300])
if job["entry"] and job["entry"] not in ns:
    result(False, f"no function named {job['entry']}")
for t in job["tests"]:
    try:
        exec(t, ns)
    except AssertionError:
        result(False, "assert failed: " + t.strip().splitlines()[-1][:200])
    except BaseException as e:
        result(False, f"{type(e).__name__}: {e}"[:300])
result(True, "all tests pass")
'''


def run_tests(solution: str, entry: str, tests: list[str], setup: str = "", timeout: int = 10):
    """(passed, why), in a fresh isolated interpreter."""
    job = json.dumps({"solution": solution, "entry": entry, "tests": tests, "setup": setup})
    try:
        r = subprocess.run([sys.executable, "-I", "-c", _RUNNER], input=job,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timeout ({timeout}s)"
    m = re.search(r"__RESULT__(\{.*\})", r.stdout)
    if not m:
        return False, "crashed: " + ((r.stderr.strip().splitlines() or ["no output"])[-1])
    res = json.loads(m.group(1))
    return res["ok"], res["why"]


def prefilled(model, tok, task, prefill, think, temperature, top_k, max_new):
    """generate() with `prefill` written into the model's turn after <|Think|> (a prompt-only span)."""
    import torch
    from theog.format import prompt_ids
    from theog.generate import sample
    from theog.tokens import STOP_IDS
    device = next(model.parameters()).device
    ids, out = prompt_ids(tok, task, think) + prefill, []
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(torch.tensor([(ids + out)[-model.cfg.max_seq_len:]], device=device))[0, -1].float()
        t = sample(logits, temperature, top_k)
        if t in STOP_IDS:
            break
        out.append(t)
    return tok.decode(out, skip_special_tokens=False)


def ensemble(a, b):
    """Two models as one (lesson 4, step 12). At every step the next-token scores are the mean of
    the two models' log-probabilities: a geometric mean of their distributions. Greedy decoding
    picks the token both models like best together. Same tokenizer, same window."""
    import torch

    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.cfg = a, b, a.cfg

        def forward(self, ids):
            return 0.5 * (torch.log_softmax(self.a(ids).float(), -1) + torch.log_softmax(self.b(ids).float(), -1))

    return Ensemble().eval()


class TheOGSolver:
    """theOG in its chat format: the task goes in the user turn."""

    def __init__(self, ckpt, tokenizer, think, temperature, top_k, max_new, design=None):
        from tokenizers import Tokenizer
        from theog.train import load_model, pick_device
        self.model = load_model(Path(ckpt), pick_device())
        self.tok = Tokenizer.from_file(str(tokenizer))
        self.design = design
        if design == "lat8p":
            self.model.latent.own = True
        self.kw = dict(think=think or bool(design), temperature=temperature, top_k=top_k, max_new=max_new)

    def solve(self, task, dataset):
        from theog.generate import extract_code, generate
        from theog.think import PREFILL, REORDER, RETRIEVED, fine_into_coarse
        pre = PREFILL[self.design]({"user": task["prompt"]}) if self.design in PREFILL else None
        if pre and self.design in RETRIEVED:  # leave room for the answer in the 512-token window
            from theog.format import prompt_ids
            from theog.think import shrink
            head = len(prompt_ids(self.tok, task["prompt"].strip(), True))
            pre = shrink(pre, lambda t: head + len(self.tok.encode(f"\n{t}\n").ids) <= 352)
        ids = self.tok.encode(f"\n{pre}\n").ids if pre else None
        if self.design in ("lat8", "lat8p"):
            from theog import latent
            raw = latent.generate(self.model, self.tok, task["prompt"].strip(), self.kw["max_new"])
        else:
            raw = prefilled(self.model, self.tok, task["prompt"].strip(), ids, **self.kw) if ids \
                else generate(self.model, self.tok, task["prompt"].strip(), **self.kw)
        if pre:
            raw = f"\n{pre}\n" + raw
        code = extract_code(raw.split("<|Answer|>", 1)[-1])
        if self.design in REORDER and "<|/Think|>" in raw:
            outline = raw.split("<|/Think|>", 1)[0].strip("\n")
            if pre:  # psig_c2f1: the first line is the prefilled signature, not code
                outline = outline.split("\n", 1)[1] if "\n" in outline else ""
            m = re.search(r"```(?:python)?\n(.*?)(?:```|$)", raw.split("<|Answer|>", 1)[-1], re.DOTALL)
            fills = (m.group(1) if m else "").strip("\n")  # keep the first fill's indent
            code = fine_into_coarse(outline, fills)
        if dataset == "humaneval":
            entry = f"def {task['entry_point']}"
            # keep what the prompt defines before the function (imports, helpers)
            code = (task["prompt"][: task["prompt"].find(entry)] + code) if entry in code else task["prompt"] + code
        return code, raw


class HFSolver:
    """A Hugging Face base model as plain completion (the sanity gate)."""

    def __init__(self, name, max_new):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from theog.train import pick_device
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(self.device).eval()
        self.max_new = max_new

    def solve(self, task, dataset):
        import torch
        prompt = (task["prompt"].strip() + "\n") if dataset == "humaneval" else f'"""\n{task["prompt"].strip()}\n"""\n'
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        n_pos = getattr(self.model.config, "n_positions", None) or self.model.config.max_position_embeddings
        ids = ids[:, -(n_pos - 1):]
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=min(self.max_new, n_pos - ids.shape[1]), do_sample=False,
                                      pad_token_id=self.tok.eos_token_id, stop_strings=HF_STOPS, tokenizer=self.tok)
        completion = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        completion = completion[:min([completion.find(s) for s in HF_STOPS if s in completion] or [len(completion)])]
        return prompt + completion, completion


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--ckpt", type=Path)
    who.add_argument("--hf", help="a Hugging Face model id, for the sanity gate")
    ap.add_argument("--ensemble", type=Path, help="a second checkpoint: decode from both models' mean log-probabilities")
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp500"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--design", help="lesson 7 design the model was trained with: prefills a prompt-only span "
                    "(psig) and rebuilds coarse-to-fine code (c2f0, c2f1, psig_c2f1); implies --think")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only (a smoke test, not a score)")
    a = ap.parse_args()
    if a.ckpt and not a.tokenizer:
        ap.error("--ckpt needs --tokenizer")
    if a.ensemble and not a.ckpt:
        ap.error("--ensemble needs --ckpt")
    if a.hf and (a.think or a.temperature):
        ap.error("--hf scores greedy plain completion: --think and --temperature don't apply to it")

    tasks = load_tasks(a.dataset)[: a.limit or None]
    if a.ckpt:
        solver, name = TheOGSolver(a.ckpt, a.tokenizer, a.think, a.temperature, a.top_k, a.max_new, a.design), a.ckpt.parent.name
        if a.ensemble:
            from theog.train import load_model, pick_device
            solver.model = ensemble(solver.model, load_model(a.ensemble, pick_device()))
            name = f"{name}+{a.ensemble.parent.name}"
    else:
        solver, name = HFSolver(a.hf, a.max_new), a.hf.replace("/", "__")
    mode = (a.design or ("think" if a.think else "no_think"))
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{name}_{a.dataset}_{mode}_t{a.temperature}_{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    passed = 0
    with open(path, "w") as f:
        for i, task in enumerate(tasks, 1):
            code, raw = solver.solve(task, a.dataset)
            ok, why = run_tests(code, task["entry_point"], task["tests"], task["setup"])
            passed += ok
            print(f"[{i:>3}/{len(tasks)}] {task['task_id']:<16} {'PASS' if ok else 'FAIL'}  {why}", flush=True)
            f.write(json.dumps({"task_id": task["task_id"], "passed": ok, "why": why, "code": code, "raw": raw}) + "\n")
    print(f"\n{a.dataset}: {passed}/{len(tasks)} passed  ->  {path}")


if __name__ == "__main__":
    main()
