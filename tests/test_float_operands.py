"""Integer-only rewrites do not fire on a floating operand.

``x * 8`` -> ``x << 3`` is wrong when x is a float: uc386 turned the
float into an integer, shifted, and went back.  The optimizer knows no
types, so ``FloatInfo`` (floatness.py) collects the names declared with
a floating type, and the rewrites ask ``_maybe_float`` first.  Each test
checks that the float stays (the operator survives) and that the same
expression on an integer is still reduced.
"""

from __future__ import annotations

import dataclasses

import pytest

from uc_core import ast
from uc_core.ast_optimizer import ASTOptimizer
from uc_core.frontend import parse


def _ops(src: str) -> set[str]:
    unit = ASTOptimizer(3).optimize(parse(src, "t.c"))
    out: set[str] = set()

    def walk(n):
        if isinstance(n, list):
            for x in n:
                walk(x)
        elif dataclasses.is_dataclass(n):
            if isinstance(n, ast.BinaryOp):
                out.add(n.op.text if hasattr(n.op, "text") else n.op)
            for f in dataclasses.fields(n):
                walk(getattr(n, f.name, None))
    walk(unit)
    return out


@pytest.mark.parametrize("decl", [
    "float x;", "double x;", "long double x;", "const float x;",
    "typedef double real; real x;", "float x[4];",
])
@pytest.mark.parametrize("expr", [
    "x * 8", "8 * x", "x * 2", "x / 8u", "x / 4u", "(x * 3) * 5",
    "(x + 1) + 2", "x - x",
])
def test_float_object_keeps_its_operator(decl, expr):
    kind = expr
    if "[" in decl:
        expr = expr.replace("x", "x[1]")
    src = f"{decl} double f(void) {{ return {expr}; }}"
    ops = _ops(src)
    assert "<<" not in ops and ">>" not in ops, (src, ops)
    if kind.startswith("(") or kind == "x - x":
        assert ops & {"+", "-", "*"}, (src, ops)     # not folded away
    elif "/" in kind:
        assert "/" in ops, (src, ops)
    else:
        assert "*" in ops, (src, ops)


@pytest.mark.parametrize("src", [
    "typedef float F; F fn(void); double f(void) { return fn() * 8; }",
    "struct S { float m; }; double f(struct S *s) { return s->m * 8; }",
    "double f(float p) { return p * 8; }",
    "double f(void) { return (float)3 * 8; }",
    "double f(int i) { return (double)i * 8; }",
    "double f(int i) { typeof(1.5) t = 2; return t * 8; }",
])
def test_float_reached_by_other_paths(src):
    ops = _ops(src)
    assert "<<" not in ops, (src, ops)


def test_integer_operands_are_still_reduced():
    ops = _ops("int f(int x) { return x * 8; }")
    assert "<<" in ops
    ops = _ops("float g; int f(int x) { return x * 8 + (int)g; }")
    assert "<<" in ops
    ops = _ops("float g(void); int f(int x) { return x * 8 + g(); }")
    assert "<<" in ops
    ops = _ops("unsigned f(unsigned x) { return x / 8u; }")
    assert ">>" in ops
    ops = _ops("struct S { float m; int n; }; int f(struct S *s) "
               "{ return s->n * 8; }")
    assert "<<" in ops or "+" in ops          # n is not a float member


def test_float_equal_operands_are_not_folded():
    assert "-" in _ops("float x; float f(void) { return x - x; }")
    assert "==" in _ops("double x; int f(void) { return x == x; }")
    assert "-" not in _ops("int x; int f(void) { return x - x; }")
