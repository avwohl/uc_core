"""Phase-0 regression corpus for the ASTOptimizer auto-AST migration.

Context
-------
``src/uc_core/ast_optimizer.py`` is only partially ported to the
Phase-6 uplox-v3 auto-AST. Today it is a silent, non-crashing **no-op
on function bodies**: ``_optimize_stmt``'s ``CompoundStmt`` walker
gates recursion on ``isinstance(item, ast.Statement)``, but
``ast.Statement`` resolves to the ``_Removed`` tombstone, so no
statement is ever visited; and the surviving rewrite code still builds
legacy-shaped nodes (``IntLiteral(value=int, location=...)``). uc386 /
uc80 still pass validation only because a no-op optimizer yields
*correct* (just unoptimized) code, and the asm-level peephole pass is
independent.

What this file is
-----------------
The safety net to land *before* the behavioural migration (Phases 1-3):

* The **invariants** (no crash; a program with no optimization
  opportunity is returned structurally unchanged) must pass *now* and
  must keep passing through every phase — they catch a phase that
  starts mangling untouchable code once statement dispatch is enabled.

* Each **transform** test asserts the optimizer's *intended* behaviour
  and is marked ``xfail(strict=True)`` because the transform is
  currently dead. ``strict=True`` means: when a migration phase makes
  the transform work, the test XPASSes, pytest reports that as a
  failure, and whoever did the phase must delete the ``xfail`` marker.
  That is the migration progress tracker — do not loosen it.

So a green run here = invariants hold and no transform has silently
regressed/advanced without the marker being updated.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import pytest

from uc_core import ast
from uc_core._const import int_value
from uc_core.ast_optimizer import ASTOptimizer
from uc_core.frontend import parse

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _optimize(src: str, level: int = 3):
    """Parse a C translation unit and run the optimizer at ``level``."""
    return ASTOptimizer(level).optimize(parse(src, "t.c"))


def _walk(node):
    """Yield ``node`` and every dataclass descendant (pre-order)."""
    yield node
    if dataclasses.is_dataclass(node):
        for f in dataclasses.fields(node):
            value = getattr(node, f.name)
            for item in value if isinstance(value, list) else [value]:
                if dataclasses.is_dataclass(item):
                    yield from _walk(item)


def _kinds(node) -> Counter:
    """Multiset of node class names in the subtree (excludes Token/_Pos)."""
    return Counter(
        type(n).__name__
        for n in _walk(node)
        if type(n).__name__ not in ("Token", "_Pos")
    )


def _find(node, kind: str) -> list:
    """All nodes whose class name is exactly ``kind``."""
    return [n for n in _walk(node) if type(n).__name__ == kind]


def _int_literals(node) -> list[int]:
    """Decoded values of every IntLiteral in the subtree."""
    return [int_value(n) for n in _find(node, "IntLiteral")]


def _only_function_body(unit):
    """The single FunctionDef's body in a one-function translation unit."""
    fns = [d for d in unit.items if isinstance(d, ast.FunctionDef)]
    assert len(fns) == 1, f"expected exactly one function, got {len(fns)}"
    return fns[0].body


# Phases 0-3 are done: node construction, operator/identifier accessors,
# statement dispatch, and the ReturnStmt/ReturnStmtValue split are all
# migrated. The three transforms below stay xfail(strict) because each
# is blocked by a *distinct, non-mechanical* gap (a follow-up unit of
# work, not auto-AST migration). strict=True still flags any change.
_XFAIL_COPY = (
    "blocked: copy-prop's _types_compatible_for_copy needs resolved "
    "type categories, but the Phase-6 auto-AST carries no resolved "
    "types (_var_types is None), so the guard always refuses. Needs a "
    "declarator-shaped type-category derivation (separate work)."
)


# A spread of real-ish C the optimizer must survive without raising.
_NO_CRASH_SOURCES = [
    "int a = 2 + 3 * 4;",
    "int f(void){ return 7 * 6; }",
    "int g(int x){ if (x) return 1; else return 0; }",
    "int h(int p){ int a = p; int b = a + a; return b; }",
    "int m(int x){ return x * 8; }",
    "int s(int n){ int t = 0; for (int i = 0; i < n; i++){ if (i & 1) continue; t += i * 2; } return t; }",
    "int fib(int n){ if (n < 2) return n; return fib(n-1) + fib(n-2); }",
    "int sw(int x){ switch (x){ case 0: return 1; default: return x; } }",
    "void v(void){ int i = 0; while (i < 10){ i++; } }",
    "int d(void){ int x = 0; do { x++; } while (x < 3); return x; }",
]


# --------------------------------------------------------------------------
# Invariants — MUST pass now and through every migration phase
# --------------------------------------------------------------------------


@pytest.mark.parametrize("src", _NO_CRASH_SOURCES)
@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_optimizer_never_raises(src, level):
    """The optimizer must never crash on valid C, at any -O level."""
    _optimize(src, level)


def test_program_without_opportunities_is_structurally_unchanged():
    """Code with nothing to optimize must come back with an identical
    node-kind multiset — guards against a phase mangling untouchable
    code once statement dispatch is switched on."""
    src = "int keep(int a, int b){ int c = a + b; return c; }"
    before = _kinds(parse(src, "t.c"))
    after = _kinds(_optimize(src, 3))
    assert before == after


def test_optimize_returns_a_translation_unit():
    """optimize() returns the (same) TranslationUnit, never None."""
    unit = parse("int x = 1;", "t.c")
    out = ASTOptimizer(3).optimize(unit)
    assert isinstance(out, ast.TranslationUnit)


@pytest.mark.parametrize("src,kind,op", [
    ("int g(int i){ return (i++) * 2; }", "PostfixOp", "++"),
    ("int g(int i){ return (i--) * 2; }", "PostfixOp", "--"),
    ("int g(int i){ return (++i) * 2; }", "UnaryOp", "++"),
])
def test_side_effecting_operand_not_duplicated(src, kind, op):
    """Correctness: strength reduction must not duplicate an operand
    with side effects (`(i++) * 2` must NOT become `(i++) + (i++)`).
    Regression for PostfixOp being invisible to _expr_has_side_effects."""
    unit = _optimize(src)
    incs = [n for n in _find(unit, kind) if getattr(n.op, "text", n.op) == op]
    assert len(incs) == 1, f"{op} duplicated: {len(incs)} occurrences"


@pytest.mark.parametrize("src,callkind", [
    # No-arg call: CallNoArgs node-split, was invisible to side-effect
    # analysis -> strength reduction duplicated the call (caught by
    # uc386 validation, Phase 4).
    ("int side(void); int g(void){ return side() * 2; }", "CallNoArgs"),
    ("int side(int); int g(void){ return side(1) * 2; }", "Call"),
    # Side-effecting object of a member access must not be duplicated.
    ("struct S{int x;}; struct S mk(void); int g(void){ return mk().x * 2; }", "CallNoArgs"),
])
def test_call_not_duplicated_by_strength_reduction(src, callkind):
    """Correctness: a call (with or without args) has side effects and
    must never be duplicated by `* 2` -> `+`."""
    unit = _optimize(src)
    assert len(_find(unit, callkind)) == 1, f"{callkind} duplicated"


# --------------------------------------------------------------------------
# Constant folding
# --------------------------------------------------------------------------


def test_fold_constant_initializer():
    # Enabled by Phase 1: Declaration initializers are on the reachable
    # path (_optimize_decl -> _optimize_expr), so constructor/operator
    # accessor migration makes this fold work. Function-body folds stay
    # xfail until statement dispatch is enabled (Phase 2).
    unit = _optimize("int a = 2 + 3 * 4;")
    init = _find(unit, "InitDeclaratorWithInit")[0].init
    assert isinstance(init, ast.IntLiteral) and int_value(init) == 14


def test_fold_constant_return_value():
    unit = _optimize("int f(void){ return 7 * 6; }")
    val = _find(unit, "ReturnStmtValue")[0].value
    assert isinstance(val, ast.IntLiteral) and int_value(val) == 42


def test_fold_constant_comparison():
    unit = _optimize("int f(void){ return 5 > 3; }")
    val = _find(unit, "ReturnStmtValue")[0].value
    assert isinstance(val, ast.IntLiteral) and int_value(val) == 1


def test_fold_shift():
    unit = _optimize("int f(void){ return 1 << 4; }")
    val = _find(unit, "ReturnStmtValue")[0].value
    assert isinstance(val, ast.IntLiteral) and int_value(val) == 16


# --------------------------------------------------------------------------
# Dead-code elimination on a constant condition
# --------------------------------------------------------------------------


def test_dce_if_true_drops_else():
    # if (1) x = 10; else x = 99;  ->  the 99 store must be gone.
    unit = _optimize("int g(void){ int x = 0; if (1) x = 10; else x = 99; return x; }")
    lits = _int_literals(_only_function_body(unit))
    assert 99 not in lits and 10 in lits


def test_dce_if_false_keeps_else():
    unit = _optimize("int g(void){ int x = 0; if (0) x = 10; else x = 99; return x; }")
    lits = _int_literals(_only_function_body(unit))
    assert 10 not in lits and 99 in lits


def test_dce_if_false_no_else_is_removed():
    unit = _optimize("int g(void){ int x = 0; if (0) x = 10; return x; }")
    assert 10 not in _int_literals(_only_function_body(unit))


def test_unreachable_after_return_removed():
    # Enabled by Phase 3: the terminator check now recognizes
    # ReturnStmtValue (the value-bearing return node-split variant).
    unit = _optimize("int f(void){ int x = 0; return 1; x = 2; return x; }")
    body = _only_function_body(unit)
    assert 2 not in _int_literals(body)
    assert len(_find(body, "ReturnStmtValue")) == 1


# --------------------------------------------------------------------------
# Level-3 transforms: copy propagation / dead stores / loop unrolling
# --------------------------------------------------------------------------


@pytest.mark.xfail(reason=_XFAIL_COPY, strict=True)
def test_copy_propagation():
    # int a = p; int b = a + a;  ->  b computed from p, `a` propagated away.
    unit = _optimize("int h(int p){ int a = p; int b = a + a; return b; }")
    idents = {t.name.text for t in _find(unit, "Identifier")}
    assert "a" not in idents


def test_dead_store_elimination():
    # x = 1; is dead (overwritten before any use); the 1 must disappear.
    # Enabled by the dead-store follow-up: a single-init declaration
    # now counts as a store; the declarator is kept (var stays
    # declared), only the dead initializer is dropped.
    unit = _optimize("int f(void){ int x = 1; x = 2; return x; }")
    assert 1 not in _int_literals(_only_function_body(unit))


def test_dead_store_keeps_variable_declared():
    """Stripping a dead initializer must keep the declaration so the
    variable stays declared and scoped."""
    unit = _optimize("int f(void){ int x = 1; x = 2; return x; }")
    body = _only_function_body(unit)
    decls = _find(body, "InitDeclarator") + _find(body, "InitDeclaratorWithInit")
    names = {_d.declarator.name.text for _d in decls
             if type(_d.declarator).__name__ == "Declarator"}
    assert "x" in names, "declaration of x was lost"


def test_dead_store_not_applied_to_volatile():
    """A volatile initializer must NOT be elided (observable write)."""
    unit = _optimize("int f(void){ volatile int x = 1; x = 2; return x; }")
    assert 1 in _int_literals(_only_function_body(unit))


def test_dead_store_keeps_side_effecting_initializer():
    """`int x = f(); x = 2;` must keep the call (f() has side effects)."""
    unit = _optimize("int side(void); int f(void){ int x = side(); x = 2; return x; }")
    assert len(_find(unit, "CallNoArgs")) == 1


def test_small_constant_loop_is_unrolled():
    # Enabled by the decl-form follow-up: _try_loop_unroll now also
    # accepts `for(int i=0;..)` (Declaration init), re-declaring the
    # loop variable in the unrolled block to preserve scoping.
    unit = _optimize("int f(void){ int s = 0; for (int i = 0; i < 3; i++) s += i; return s; }")
    assert _find(_only_function_body(unit), "ForStmt") == []


# --------------------------------------------------------------------------
# Strength reduction / algebraic identities
# --------------------------------------------------------------------------


def test_strength_reduce_mul_by_power_of_two():
    unit = _optimize("int m(int x){ return x * 8; }")
    ops = {getattr(b.op, "text", b.op) for b in _find(unit, "BinaryOp")}
    assert "<<" in ops and "*" not in ops


def test_algebraic_add_zero():
    unit = _optimize("int f(int x){ return x + 0; }")
    val = _find(unit, "ReturnStmtValue")[0].value
    assert type(val).__name__ == "Identifier" and val.name.text == "x"


def test_algebraic_mul_zero():
    unit = _optimize("int f(int x){ return x * 0; }")
    val = _find(unit, "ReturnStmtValue")[0].value
    assert isinstance(val, ast.IntLiteral) and int_value(val) == 0
