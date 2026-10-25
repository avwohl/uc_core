"""Level-3 copy propagation must not read a variable as its old source.

``int a = v; int b = a++; return a;`` returned ``v``: the cache of
copies (``a`` is ``v``) ignored the side effects inside a declaration
initialiser.  The tests below check the optimised AST (a read of the
changed variable must stay) and, when a uc80 tool chain is present,
run the programs.  Expected run output comes from a host compiler.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from uc_core import ast
from uc_core.ast_optimizer import ASTOptimizer
from uc_core.frontend import parse


def _t(x):
    return x.text if hasattr(x, "text") else str(x)


def _c(n) -> str:
    """The expression or statement as a compact C-like string."""
    if n is None:
        return ""
    if isinstance(n, ast.Identifier):
        return n.name.text
    if isinstance(n, ast.IntLiteral):
        return _t(n.value)
    k = type(n).__name__
    if k == "BinaryOp":
        return f"({_c(n.left)} {_t(n.op)} {_c(n.right)})"
    if k == "PostfixOp":
        return f"{_c(n.operand)}{_t(n.op)}"
    if k == "UnaryOp":
        return f"{_t(n.op)}{_c(n.operand)}"
    if k == "TernaryOp":
        return f"({_c(n.condition)} ? {_c(n.true_expr)} : {_c(n.false_expr)})"
    if k == "ReturnStmtValue":
        return f"return {_c(n.value)};"
    if k == "ExpressionStmt":
        return _c(n.expr) + ";"
    if k == "Cast":
        return f"(T){_c(n.expr)}"
    if k == "Index":
        return f"{_c(n.array)}[{_c(n.index)}]"
    if k == "Declaration":
        parts = []
        for d in n.declarators:
            dd = d.declarator
            while not hasattr(dd, "name"):
                dd = getattr(dd, "declarator", None) or dd.inner
            init = getattr(d, "init", None)
            parts.append(_t(dd.name) + (f" = {_c(init)}" if init is not None else ""))
        return "decl " + ", ".join(parts) + ";"
    if k in ("IfStmt", "IfStmtElse"):
        t = f"if {_c(n.condition)} {_c(n.then_branch)}"
        if k == "IfStmtElse":
            t += f" else {_c(n.else_branch)}"
        return t
    if k == "CompoundStmt":
        return "{ " + " ".join(_c(i) for i in n.items) + " }"
    return k


def _body(src: str, name: str = "f", level: int = 3) -> list[str]:
    tu = ASTOptimizer(level).optimize(parse(src, "t.c"))
    for item in tu.items:
        if (isinstance(item, ast.FunctionDef)
                and item.declarator is not None
                and name in _c_decl_name(item.declarator)):
            return [_c(i) for i in item.body.items]
    raise AssertionError("no function " + name)


def _c_decl_name(d) -> str:
    out = []

    def walk(x):
        if isinstance(x, ast.Declarator):
            out.append(_t(x.name))
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            for f in dataclasses.fields(x):
                v = getattr(x, f.name, None)
                if isinstance(v, (list, tuple)):
                    for y in v:
                        walk(y)
                elif dataclasses.is_dataclass(v):
                    walk(v)
    walk(d)
    return " ".join(out)


def _ret(src: str, name: str = "f") -> str:
    """The last statement of ``f`` after optimisation."""
    return _body(src, name)[-1]


# ---- AST level: the read of a changed variable stays

@pytest.mark.parametrize("src", [
    "int f(int v){ int a = v; int b = a++; return a; }",
    "int f(int v){ int a = v; int b = a--; return a; }",
    "int f(int v){ int a = v; int b = ++a; return a; }",
    "int f(int v){ int a = v; int b = --a; return a; }",
    "int f(int v){ int a = v; int b = (a += 2); return a; }",
    "int f(int v){ int a = v; int b = (a <<= 1); return a; }",
    "int f(int v){ int a = v; int b = (a = 5); return a; }",
    "int f(int v){ int a = v; int b = (int)(a++); return a; }",
    "int f(int v, int *t){ int a = v; int b = t[a++]; return a; }",
    "int f(int v){ int a = v; int b = (a++, 1); return a; }",
    "int f(int v){ int a = v; int b = v ? a++ : 0; return a; }",
    "int f(int v){ int a = v; int b = v && a++; return a; }",
    "int f(int v){ int a = v; int b[2] = { a++, 1 }; return a; }",
    "int g(int); int f(int v){ int a = v; int b = g(a++); return a; }",
    "int f(int v){ int a = v; int b = ({ a++; 1; }); return a; }",
    "int f(int v){ int a = v, b = a++; return a; }",
    "int f(int v){ int a = v, b = a++, c = 1; return a; }",
])
def test_declaration_initialiser_effect_keeps_the_read(src):
    assert _ret(src) == "return a;"


def test_later_declarator_sees_earlier_effect():
    # ``c = a`` must not become ``c = v`` after ``b = a++``.
    body = _body("int f(int v){ int a = v, b = a++, c = a; return c; }")
    assert body[0] == "decl a = v, b = a++, c = a;"


def test_later_declarator_still_uses_earlier_copy():
    # no effect in between: the chain a, b, c still folds to v
    body = _body("int f(int v){ int a = v, b = a, c = b; return c; }")
    assert body[-1] == "return v;"


@pytest.mark.parametrize("src", [
    "int f(int v){ int a = v; a++; return a; }",
    "int f(int v){ int a = v; ++a; return a; }",
    "int f(int v){ int a = v; a += 3; return a; }",
    "int f(int v){ int a = v; a = a + 1; return a; }",
    "int f(int v){ int a = v; a <<= 1; return a; }",
])
def test_statement_effect_keeps_the_read(src):
    assert _ret(src) == "return a;"


@pytest.mark.parametrize("src", [
    "int f(int v){ int a = v; return (a++ > 0) && a; }",
    "int f(int v){ int a = v; return (a++ > 0) || a; }",
    "int f(int v){ int a = v; return (a++ > 0) ? a : 0; }",
    "int f(int v){ int a = v; return (a += 1) ? a : 0; }",
])
def test_sequence_point_inside_an_expression(src):
    # the read after the sequence point is ``a``, not ``v``
    assert "v" not in _ret(src).replace("return", "")


def test_if_condition_effect_reaches_the_branch():
    body = _body("int f(int v){ int a = v; if (a++ > 0) return a; return 0; }")
    assert body[1] == "if (a++ > 0) return a;"


def test_conditional_assignment_makes_no_copy():
    for e in ("c && (x = y)", "c || (x = y)", "c ? (x = y) : 0"):
        body = _body(f"int f(int c, int v){{ int x = 0, y = v; {e}; return x; }}")
        assert body[-1] == "return x;", e


def test_branch_copy_does_not_reach_else_branch():
    body = _body("int f(int c, int v){ int x = 0, y = v; "
                 "if (c) { x = y; } else { return x; } return 0; }")
    assert "return x;" in body[1]


def test_case_label_clears_copies():
    text = _find_switch_text(
        "int f(int c, int v){ int x = 0, y = v; "
        "switch (c) { case 1: x = y; break; case 2: return x; } return 0; }")
    assert "return x;" in text


def _find_switch_text(src):
    tu = ASTOptimizer(3).optimize(parse(src, "t.c"))
    fn = [i for i in tu.items if isinstance(i, ast.FunctionDef)][0]
    sw = [i for i in fn.body.items if isinstance(i, ast.SwitchStmt)][0]
    return " ".join(_c(i) for i in _flatten(sw.body))


def _flatten(n):
    if isinstance(n, ast.CompoundStmt):
        for i in n.items:
            yield from _flatten(i)
    elif isinstance(n, (ast.CaseStmt, ast.DefaultStmt)):
        yield from _flatten(n.stmt)
    else:
        yield n


def test_copy_to_a_closed_block_name_is_dropped():
    # ``return i`` would name a variable that is out of scope
    assert _ret("int f(int v){ int o = 0; { int i = v + 1; o = i; } return o; }") \
        == "return o;"


def test_shadowing_declaration_kills_the_copy():
    body = _body("int f(int v){ int a = v; { int v = 9; return a; } }")
    assert "return a;" in body[-1]


def test_uninitialised_inner_declaration_kills_the_copy():
    body = _body("int f(int v){ int a = v; { int v; v = 4; return a; } }")
    assert "return a;" in body[-1]


def test_static_local_initialiser_is_no_copy():
    body = _body("int g = 3; int f(void){ static int s = g; s++; return s; }")
    assert body[-1] == "return s;"


def test_plain_copy_still_propagates():
    assert _ret("int f(int v){ int a = v; int b = a; return b; }") == "return v;"
    assert _ret("int f(int v){ int a = v; return a + 1; }") == "return (v + 1);"


def test_copy_in_loop_body_not_used_after_loop():
    src = "int f(int v){ int a = v; int i = 0; while (i < 3) { int b = a++; i++; } return a; }"
    assert _ret(src) == "return a;"
    src = ("int f(int v){ int a = v; int i; "
           "for (i = 0; i < 3; i++) { int b = a; a++; } return a; }")
    assert _ret(src) == "return a;"


# ---- run time (uc80 tool chain, when present)

def _uc80_tools():
    try:
        import uc80  # noqa: F401
    except ImportError:
        return None
    cpmemu = Path(os.environ.get("CPMEMU", Path.home() / "src/cpmemu/src/cpmemu"))
    if not (shutil.which("um80") and shutil.which("ul80") and cpmemu.exists()):
        return None
    r = subprocess.run([sys.executable, "-m", "uc80.main", "--print-lib-dir"],
                       capture_output=True, text=True)
    lib = Path(r.stdout.strip()) if r.returncode == 0 else None
    if lib is None or not (lib / "libc.lib").exists():
        return None
    return cpmemu, lib


TOOLS = _uc80_tools()
needs_tools = pytest.mark.skipif(TOOLS is None, reason="no uc80 tool chain")


def _run(tmp_path, source, name):
    cpmemu, lib = TOOLS
    c = tmp_path / (name + ".c")
    c.write_text("#include <stdio.h>\n" + source)
    mac, rel, com = (tmp_path / (name + e) for e in (".mac", ".rel", ".com"))
    r = subprocess.run([sys.executable, "-m", "uc80.main", str(c), "-o", str(mac)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    subprocess.run(["um80", str(mac), "-o", str(rel)], check=True,
                   capture_output=True)
    subprocess.run(["ul80", str(rel), str(lib / "libc.lib"),
                    str(lib / "runtime.lib"), "-o", str(com)],
                   check=True, capture_output=True)
    out = subprocess.run([str(cpmemu), str(com)], capture_output=True, timeout=120)
    return out.stdout.decode("latin-1").replace("\r", "").split()


@needs_tools
def test_run_increment_in_initialiser(tmp_path):
    src = """
    int p1(int v) { int a = v; int b = a++; return a * 10 + b; }
    int p2(int v) { int a = v; int b = ++a; return a * 10 + b; }
    int p3(int v) { int a = v; int b = a--; return a * 10 + b; }
    int p4(int v) { int a = v, b = a++, c = a; return c * 10 + b; }
    int main(void) {
        volatile int z = 3;
        printf("%d %d %d %d\\n", p1(z), p2(z), p3(z), p4(z));
        return 0;
    }
    """
    assert _run(tmp_path, src, "inc") == ["43", "44", "23", "43"]


@needs_tools
def test_run_assignment_forms_in_initialiser(tmp_path):
    src = """
    int q1(int v) { int a = v; int b = (a += 2); return a * 10 + b; }
    int q2(int v) { int a = v; int b = (a = 7); return a * 10 + b; }
    int q3(int v) { int a = v; int b = (a <<= 1); return a * 10 + b; }
    int q4(int v, int *t) { int a = v; int b = t[a++]; return a * 10 + b; }
    int main(void) {
        volatile int z = 1;
        int t[4] = {5, 6, 7, 8};
        printf("%d %d %d %d\\n", q1(z), q2(z), q3(z), q4(z, t));
        return 0;
    }
    """
    assert _run(tmp_path, src, "asg") == ["33", "77", "22", "26"]


@needs_tools
def test_run_sequence_points(tmp_path):
    src = """
    int s1(int v) { int a = v; return (a++ > 0) && a; }
    int s2(int v) { int a = v; return (a++ > 5) || a; }
    int s4(int v, int c) { int x = 0, y = v; c && (x = y); return x; }
    int s5(int v, int c) { int x = 0, y = v; c ? (x = y) : 0; return x; }
    int main(void) {
        volatile int z = 4;
        printf("%d %d %d %d\\n", s1(z), s2(z), s4(z, 0), s5(z, 0));
        printf("%d %d\\n", s4(z, 1), s5(z, 1));
        return 0;
    }
    """
    assert _run(tmp_path, src, "seq") == ["1", "1", "0", "0", "4", "4"]


def _uc80_has_ternary_hook():
    try:
        from uc80.ast_opt import Uc80ASTOptimizer
    except ImportError:
        return False
    # uc80 used to walk the three operands of ``?:`` itself, which
    # skipped the sequence point after the condition.
    return "_optimize_expr" not in vars(Uc80ASTOptimizer)


@needs_tools
@pytest.mark.skipif(not _uc80_has_ternary_hook(),
                    reason="uc80 still has its own ?: walk")
def test_run_ternary_after_increment(tmp_path):
    src = """
    int t1(int v) { int a = v; return (a++ > 0) ? a : 99; }
    int t2(int v) { int a = v; int r = (a++ > 0) ? a : 99; return r + a; }
    int main(void) {
        volatile int z = 4;
        printf("%d %d\\n", t1(z), t2(z));
        return 0;
    }
    """
    assert _run(tmp_path, src, "tern") == ["5", "10"]


@needs_tools
def test_run_scope_and_branches(tmp_path):
    src = """
    int b1(int v) { int o = 0; { int i = v + 1; o = i; } return o; }
    int b2(int v) { int a = v; { int v = 9; return a + v; } }
    int b3(int c, int v) { int x = 0, y = v;
        if (c) { x = y; } else { return x + 100; } return x; }
    int b4(int c, int v) { int x = 0, y = v;
        switch (c) { case 1: x = y; break; case 2: return x + 100; } return x; }
    int main(void) {
        volatile int z = 4;
        printf("%d %d %d %d %d %d\\n", b1(z), b2(z), b3(1, z), b3(0, z),
               b4(1, z), b4(2, z));
        return 0;
    }
    """
    assert _run(tmp_path, src, "scope") == ["5", "13", "4", "100", "4", "100"]


@needs_tools
def test_run_loops_and_statements(tmp_path):
    src = """
    int l1(int v) { int a = v; int i = 0; while (i < 3) { int b = a++; i++; } return a; }
    int l2(int v) { int a = v; int i; for (i = 0; i < 3; i++) { int b = a; a += 2; } return a; }
    int l3(int v) { int a = v; do { int b = a--; } while (a > 0); return a; }
    int l4(int v) { int a = v; if (a++ > 0) return a; return -1; }
    int main(void) {
        volatile int z = 2;
        printf("%d %d %d %d\\n", l1(z), l2(z), l3(z), l4(z));
        return 0;
    }
    """
    assert _run(tmp_path, src, "loops") == ["5", "8", "0", "3"]
