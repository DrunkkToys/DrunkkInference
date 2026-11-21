"""Lesson 6: what goes inside the think span, <|Think|> ... <|/Think|>.

Every design is a fixed rule (row) -> text, computed from the prompt and the
code of a training row. No model writes it. Pack with one design:

  uv run python -m theog.pack ... --think sig

and lesson 6 compares them with theog.cost.

  none      empty span (theOG's own design)
  sig       the code's def line(s): commit to the signature first
  shared    the code's tokens that the prompt already contains, in code order
  skeleton  shared plus Python keywords, one line per code line
  plan      name, one test, and the code's steps without their details
            (Dong & Lapata 2018, coarse-to-fine; Li et al. 2023, structured CoT)
"""
from __future__ import annotations

import ast
import io
import keyword
import re
import tokenize
import warnings

from theog.data import function_name

MAX_LINES = 10      # longer plans are cut with a final "..."
MAX_TEST = 120      # characters; longer asserts are left out
_WORD = re.compile(r"\w+")


# --- sig ---------------------------------------------------------------------

def sig(row: dict) -> str:
    """Every def or class line of the code, stripped, in order."""
    return "\n".join(line.strip() for line in row["answer"].splitlines()
                     if line.lstrip().startswith(("def ", "class ", "async def ")))


# --- shared and skeleton: what the prompt already contains -------------------

def code_tokens(code: str) -> list[tokenize.TokenInfo]:
    """The code's Python tokens (names, numbers, strings, operators, line and
    indent markers), or [] if it doesn't tokenize."""
    keep = (tokenize.NAME, tokenize.NUMBER, tokenize.STRING, tokenize.OP,
            tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
    try:
        return [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type in keep]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []


def grounded(t: tokenize.TokenInfo, prompt: str, words: set[str]) -> bool:
    """True when the prompt already contains this code token, so the model can
    copy it instead of inventing it:

    - a name (not a Python keyword) that is one of the prompt's words
    - a number that is one of the prompt's words
    - a string literal longer than '' whose exact text is in the prompt

    `words` is the set of the prompt's words, set(_WORD.findall(prompt)).
    """
    if t.type == tokenize.NAME:
        return t.string in words and not keyword.iskeyword(t.string)
    if t.type == tokenize.NUMBER:
        return t.string in words
    if t.type == tokenize.STRING:
        return len(t.string) > 2 and t.string in prompt
    return False


def shared(row: dict) -> str:
    """The grounded tokens of the code, in code order, a repeat in a row once."""
    prompt = row["user"]
    words = set(_WORD.findall(prompt))
    out: list[str] = []
    for t in code_tokens(row["answer"]):
        if grounded(t, prompt, words) and (not out or out[-1] != t.string):
            out.append(t.string)
    return " ".join(out)


def skeleton(row: dict) -> str:
    """One line per line of code, indented like it: its keywords and its
    grounded tokens, everything invented dropped."""
    prompt = row["user"]
    words = set(_WORD.findall(prompt))
    lines, cur, depth = [], [], 0
    for t in code_tokens(row["answer"]):
        if t.type == tokenize.INDENT:
            depth += 1
        elif t.type == tokenize.DEDENT:
            depth -= 1
        elif t.type == tokenize.NEWLINE:
            if cur:
                lines.append("  " * depth + " ".join(cur))
            cur = []
        elif (t.type == tokenize.NAME and keyword.iskeyword(t.string)) or grounded(t, prompt, words):
            cur.append(t.string)
    return "\n".join(lines)


# --- plan --------------------------------------------------------------------

def _short(node: ast.AST) -> str:
    """An expression kept only when it carries no detail to fill in: a name, a
    constant, an empty container, or a simple attribute."""
    if isinstance(node, (ast.Name, ast.Constant)):
        s = ast.unparse(node)
        return s if len(s) <= 20 else "..."
    if isinstance(node, (ast.List, ast.Dict, ast.Set, ast.Tuple)) and not ast.unparse(node)[1:-1].strip():
        return ast.unparse(node)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return ast.unparse(node)
    if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
        f = node.func
        if isinstance(f, ast.Name) or isinstance(f.value, ast.Name):
            return ast.unparse(f) + ("(...)" if node.args or node.keywords else "()")
    return "..."


def _iter(node: ast.AST) -> str:
    """Loop iterables keep one more level: `range(len(nums))` is the plan."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("range", "enumerate", "zip", "reversed", "sorted"):
        s = ast.unparse(node)
        if len(s) <= 30:
            return s
    return _short(node)


def _targets(node: ast.AST) -> str:
    s = ast.unparse(node)
    return s if len(s) <= 25 else "..."


def _cond(node: ast.AST) -> str:
    """A condition is kept when it is short: `num < to_base` is the plan."""
    s = ast.unparse(node)
    return s if len(s) <= 25 else "..."


def _ret(n: ast.Return, name: str) -> str:
    if n.value is None:
        return "return"
    calls_self = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == name
                     for c in ast.walk(n.value))
    short = _short(n.value)
    if short != "...":
        return f"return {short}" + (" (recursion)" if calls_self else "")
    return "return ..." + (f" using {_inside([ast.Expr(n.value)], name)}" if _inside([ast.Expr(n.value)], name) != "..." else "")


def _inside(body: list, name: str) -> str:
    """What happens inside a block, in order, without the details: nested
    branches and loops, the calls it makes, what it updates, and whether it
    returns from the function."""
    seen: list[str] = []

    def add(x: str) -> None:
        if x not in seen:
            seen.append(x)

    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.If):
                add("branch")
            elif isinstance(n, (ast.For, ast.While, ast.comprehension)):
                add("loop")
            elif isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    add("recursion" if f.id == name else f"{f.id}()")
                elif isinstance(f, ast.Attribute):
                    add(f".{f.attr}()")
            elif isinstance(n, (ast.Assign, ast.AugAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                        add(f"updates {t.value.id}")
                    elif isinstance(n, ast.AugAssign) and isinstance(t, ast.Name):
                        add(f"{t.id} {_OPS.get(type(n.op), '?')}= ...")
            elif isinstance(n, ast.Return):
                add("returns")
            elif isinstance(n, ast.Break):
                add("break")
    if len(body) == 1 and isinstance(body[0], ast.Return):
        return _ret(body[0], name)
    return ", ".join(seen[:6]) or "..."


def _steps(body: list, name: str) -> list[str]:
    """One line per top-level statement of the function (Li et al.'s sequence,
    branch and loop), each block summed up by `_inside`."""
    out = []
    for n in body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            continue  # docstring
        if isinstance(n, (ast.For, ast.AsyncFor)):
            out.append(f"- for {_targets(n.target)} in {_iter(n.iter)}: {_inside(n.body, name)}")
        elif isinstance(n, ast.While):
            out.append(f"- while {_cond(n.test)}: {_inside(n.body, name)}")
        elif isinstance(n, ast.If):
            out.append(f"- if {_cond(n.test)}: {_inside(n.body, name)}")
            orelse = n.orelse
            while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                out.append(f"- elif {_cond(orelse[0].test)}: {_inside(orelse[0].body, name)}")
                orelse = orelse[0].orelse
            if orelse:
                out.append(f"- else: {_inside(orelse, name)}")
        elif isinstance(n, ast.Try):
            out.append(f"- try: {_inside(n.body, name)}")
            for h in n.handlers:
                exc = ast.unparse(h.type) if h.type is not None and len(ast.unparse(h.type)) <= 25 else ""
                out.append(f"- except {exc}: {_inside(h.body, name)}".replace(" :", ":"))
        elif isinstance(n, ast.With):
            out.append(f"- with ...: {_inside(n.body, name)}")
        elif isinstance(n, ast.Return):
            out.append("- " + _ret(n, name))
        elif isinstance(n, ast.Assign):
            out.append(f"- {' = '.join(_targets(t) for t in n.targets)} = {_short(n.value)}")
        elif isinstance(n, ast.AugAssign):
            out.append(f"- {_targets(n.target)} {_OPS.get(type(n.op), '?')}= ...")
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            out.append(f"- {_targets(n.target)} = {_short(n.value)}")
        elif isinstance(n, ast.Expr):
            out.append(f"- {_short(n.value)}")
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"- def {n.name}({_args(n)}): {_inside(n.body, n.name)}")
        elif isinstance(n, ast.Raise):
            out.append("- raise " + (_short(n.exc.func) if isinstance(n.exc, ast.Call) else "..."))
    return out


_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
        ast.Pow: "**", ast.BitOr: "|", ast.BitAnd: "&", ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>"}


def _args(fn: ast.FunctionDef) -> str:
    return ", ".join(a.arg for a in fn.args.args)


def plan(code: str, name: str) -> list[str]:
    """The steps of `name`, then one line for each other top-level def or
    class the answer needs. A class lists its methods instead."""
    tree = ast.parse(code)
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            lines += _steps(node.body, name)
        elif isinstance(node, ast.ClassDef) and node.name == name:
            lines += [f"- def {m.name}({_args(m)}): {_inside(m.body, m.name)}" for m in node.body
                      if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name != name:
            lines.append(f"- also: {'class ' if isinstance(n, ast.ClassDef) else 'def '}{n.name}")
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES - 1] + ["- ..."]
    return lines


def _signature(code: str, name: str) -> str:
    for node in ast.parse(code).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return f"{name}({_args(node)})"
        if isinstance(node, ast.ClassDef) and node.name == name:
            return f"class {name}"
    return ""


def think_text(answer: str, name: str, tests: str) -> str:
    """The think span for one row, or "" when the answer has no `name` to plan."""
    sig = _signature(answer, name)
    if not sig:
        return ""
    out = [f"name: {sig}"]
    test = next((t.strip()[len("assert"):].strip() for t in tests.splitlines()
                 if t.strip().startswith("assert") and name in t), "")
    if test and len(test) <= MAX_TEST:
        out.append(f"test: {test}")
    out.append("plan:")
    out += plan(answer, name)
    return "\n".join(out)


def plan_for(row: dict) -> str:
    """The plan design for one row ("" when the code doesn't parse)."""
    try:
        return think_text(row["answer"], function_name(row["answer"], row.get("entry_point", "")), row.get("tests", ""))
    except SyntaxError:
        return ""


DESIGNS = {"none": lambda row: "", "sig": sig, "shared": shared, "skeleton": skeleton, "plan": plan_for}

# Pause spans (Goyal et al. 2023): N copies of "~", one token each, no text.
PAUSE = "~"
DESIGNS.update({f"pause{n}": (lambda row, n=n: PAUSE * n) for n in (4, 16)})

# Goyal et al. train with no loss on the pause positions: the pauses are fixed, so at
# inference they are written into the prompt, not generated. pause4nl = pause4 with that mask.
DESIGNS["pause4nl"] = lambda row: PAUSE * 4
NO_LOSS = {"pause4nl"}


def mask_pauses(seq: dict, pause_id: int) -> dict:
    """Set to IGNORE every label that predicts a pause token inside the think span."""
    from theog.tokens import IGNORE, THINK_CLOSE
    ids, labels = seq["input_ids"], seq["labels"]
    close = ids.index(THINK_CLOSE)
    for i in range(close):
        if labels[i] == pause_id and ids[i + 1] == pause_id:
            labels[i] = IGNORE
    return seq


# Coarse to fine, with nothing written twice. The span is the code with every block deeper than
# `depth` indent levels replaced by one "..." line; the answer is those blocks, in order, separated
# by "..." lines. Span and answer together hold exactly the code's lines, so the total cost
# compares directly with `none`: only the order in which the model writes the code changes.
HOLE = "..."


def _unit(lines: list[str]) -> int:
    ind = [len(l) - len(l.lstrip(" ")) for l in lines if l.strip()]
    ind = [i for i in ind if i > 0]
    return min(ind) if ind else 4


def coarse_fine(code: str, depth: int) -> tuple[str, str]:
    lines = code.split("\n")
    unit, span, blocks, cur, d_prev = _unit(lines), [], [], None, 0
    for l in lines:
        d = (len(l) - len(l.lstrip(" "))) // unit if l.strip() else d_prev
        d_prev = d
        if d > depth:
            if cur is None:
                cur = []
                blocks.append(cur)
                span.append(" " * (len(l) - len(l.lstrip(" "))) + HOLE)
            cur.append(l)
        else:
            cur = None
            span.append(l)
    return "\n".join(span), f"\n{HOLE}\n".join("\n".join(b) for b in blocks)


def fine_into_coarse(span: str, answer: str) -> str:
    """Put the answer's blocks back into the span's holes: the inverse of coarse_fine."""
    blocks = answer.split(f"\n{HOLE}\n") if answer else []
    out, k = [], 0
    for l in span.split("\n"):
        if l.strip() == HOLE and k < len(blocks):
            out.append(blocks[k])
            k += 1
        else:
            out.append(l)
    return "\n".join(out)


# REORDER designs return (span, answer) instead of a span: the answer is no longer the whole code.
REORDER = {f"c2f{d}": (lambda row, d=d: coarse_fine(row["answer"], d)) for d in (0, 1)}
DESIGNS.update({k: (lambda row, k=k: REORDER[k](row)[0]) for k in REORDER})


def span_answer(design: str, row: dict) -> tuple[str, str]:
    """(think text, answer text) of a row under a design."""
    if design in REORDER:
        return REORDER[design](row)
    return DESIGNS[design](row), row["answer"]


# Prompt-only spans. psig is computed from the prompt alone (its first assert), so at inference
# the harness writes it into the prompt: the model never pays for it. Its labels are IGNORE in
# training and in theog.cost, and any saving on the answer is a pure gain.
_KINDS = {"list": "list", "str": "str", "int": "int", "float": "float", "bool": "bool",
          "tuple": "tuple", "dict": "dict", "set": "set", "none": "NoneType"}


def _def_sig(user: str) -> str | None:
    """The signature from the last `def` the prompt writes (HumanEval prompts have no assert)."""
    defs = re.findall(r"^def\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*(?:->\s*(.+?))?\s*:\s*$", user, re.M)
    if not defs:
        return None
    name, args, ret = defs[-1]
    try:
        n = len(ast.parse(f"def f({args}): pass").body[0].args.args)
    except SyntaxError:
        return None
    head = re.match(r"\w+", (ret or "").strip())
    kind = _KINDS.get(head.group(0).lower(), "?") if head else "?"
    return f"def {name}({n} {'arg' if n == 1 else 'args'}) -> {kind}"


def psig(row: dict) -> str:
    m = re.search(r"^\s*assert\s+(?:not\s+)?([A-Za-z_]\w*\(.*)$", row["user"], re.M)
    if not m:
        d = _def_sig(row["user"])
        if d:
            return d
        e = re.search(r"(?:function|method)\s+(?:named\s+|called\s+)?`([A-Za-z_]\w*)`", row["user"])
        return f"def {e.group(1)}(?) -> ?" if e else "def ?"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node = ast.parse(m.group(1).strip()).body[0].value
    except SyntaxError:
        return "def ?"
    call, out = (node.left, node.comparators[0]) if isinstance(node, ast.Compare) else (node, None)
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return "def ?"
    kind = "?"
    if out is not None:
        try:
            kind = type(ast.literal_eval(out)).__name__
        except (ValueError, SyntaxError, TypeError):
            pass
    n = len(call.args) + len(call.keywords)
    return f"def {call.func.id}({n} {'arg' if n == 1 else 'args'}) -> {kind}"


PREFILL = {"psig": psig, "psig_c2f1": psig}
DESIGNS["psig"] = psig
REORDER["psig_c2f1"] = lambda row: (psig(row) + "\n" + coarse_fine(row["answer"], 1)[0],
                                    coarse_fine(row["answer"], 1)[1])
DESIGNS["psig_c2f1"] = lambda row: REORDER["psig_c2f1"](row)[0]


def mask_prefill(seq: dict, tok, text: str) -> bool:
    """IGNORE the labels of the first `text` of the span: the part the harness writes at
    inference (prompt-only or retrieved), so the model never pays for it. False if the tokens
    don't line up."""
    from theog.tokens import IGNORE, THINK
    ids, labels = seq["input_ids"], seq["labels"]
    pre = tok.encode(f"\n{text}\n").ids
    t = ids.index(THINK)
    if ids[t + 1:t + 1 + len(pre)] != pre:
        pre = pre[:-1]  # the trailing newline may merge with what follows
        if ids[t + 1:t + 1 + len(pre)] != pre:
            return False
    for i in range(t, t + len(pre)):
        labels[i] = IGNORE
    return True


# Retrieval span (theog/retrieve.py): the nearest training solution, found from the prompt alone.
from theog.retrieve import retr  # noqa: E402

PREFILL["retr"] = retr
DESIGNS["retr"] = retr
# psr: the prompt-only signature, then the nearest training solution. Both prefilled.
PREFILL["psr"] = DESIGNS["psr"] = lambda row: psig(row) + "\n" + retr(row)
RETRIEVED = ("retr", "psr")  # spans shortened to fit the window


def shrink(text: str, fits) -> str:
    """Drop the span's last lines until fits(text) is true (a retrieved span may be long)."""
    lines = text.split("\n")
    while len(lines) > 1 and not fits("\n".join(lines)):
        lines = lines[:-2] + ["..."] if lines[-1] == "..." else lines[:-1] + ["..."]
        if lines[-2:] == ["...", "..."]:
            lines = lines[:-1]
    return "\n".join(lines)


def build(tok, design: str, row: dict, seq_len: int = 512) -> dict:
    """The training sequence of a row under a design, with every mask applied.
    A retrieved span is shortened until the row fits seq_len."""
    from theog.format import sequence
    if design in ("lat8", "lat8p"):
        from theog import latent
        return latent.sequence(tok, row["user"], row["answer"])
    span, answer = span_answer(design, row)
    if design in RETRIEVED:
        span = shrink(span, lambda s: len(sequence(tok, row["user"], answer, s)["input_ids"]) <= seq_len)
    seq = sequence(tok, row["user"], answer, span)
    if design in NO_LOSS:
        mask_pauses(seq, tok.token_to_id(PAUSE))
    if design in PREFILL:
        mask_prefill(seq, tok, span if design in RETRIEVED else PREFILL[design](row))
    return seq

DESIGNS["lat8"] = lambda row: ""  # continuous: theog/latent.py builds its sequence
DESIGNS["lat8p"] = DESIGNS["lat8"]
