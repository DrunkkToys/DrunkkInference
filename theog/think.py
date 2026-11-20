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
    # EXERCISE 6.1: lessons/06-think-span/README.md
    # Hint: three ifs on t.type (tokenize.NAME, tokenize.NUMBER, tokenize.STRING);
    #       keyword.iskeyword(t.string) for the first; everything else is False.
    raise NotImplementedError("exercise 6.1: write grounded")


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
