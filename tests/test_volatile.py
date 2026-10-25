"""The AST optimizer must not elide, merge, duplicate or drop an access to
a volatile object (C23 5.1.2.3 and 6.7.3).

Volatility comes from the declared type, so each case declares the object
a different way: plain, static, extern, through a typedef, as a pointer
qualifier, as a pointer's target, as an array element, as a struct member,
as a local, as a parameter.  A twin without `volatile` shows that the
optimization still happens where it is allowed.
"""

from __future__ import annotations

import pytest

from uc_core import ast
from uc_core.ast_optimizer import ASTOptimizer
from uc_core.frontend import parse


def _optimize(src: str, level: int = 3):
    return ASTOptimizer(level).optimize(parse(src, "t.c"))


def _func(tu, name):
    for item in tu.items:
        if isinstance(item, ast.FunctionDef):
            d = item.declarator
            while not isinstance(d, ast.Declarator):
                d = d.inner
            if d.name.text == name:
                return item
    raise KeyError(name)


def _walk(node):
    yield node
    if isinstance(node, (list, tuple)):
        for x in node:
            yield from _walk(x)
        return
    if hasattr(node, "__dataclass_fields__"):
        for f in node.__dataclass_fields__:
            if f != "pos":
                yield from _walk(getattr(node, f))


def _items(tu, name):
    return _func(tu, name).body.items


def _count(tu, name, cls, op=None):
    n = 0
    for x in _walk(_func(tu, name).body):
        if isinstance(x, cls):
            if op is None or getattr(x.op, "text", x.op) == op:
                n += 1
    return n


def _assigns(tu, name):
    return _count(tu, name, ast.BinaryOp, "=")


PRELUDE = """
typedef volatile unsigned char vb;
typedef vb vb2;
volatile unsigned char g;
static volatile unsigned char sg;
extern volatile unsigned char eg;
vb tg;
vb2 tg2;
unsigned char plain, plain2;
unsigned char *pn;
volatile unsigned char *vp;
unsigned char * volatile pv;
volatile int gi, gj;
int r;
volatile unsigned char arr[4];
struct S { volatile int a; int b; } st;
struct T { int a; } nv;
"""


@pytest.mark.parametrize("target", ["g", "sg", "eg", "tg", "tg2", "gi"])
def test_adjacent_stores_to_a_volatile_global_are_kept(target):
    tu = _optimize(PRELUDE + f"void f(void){{ {target}=1; {target}=2; {target}=3; }}")
    assert _assigns(tu, "f") == 3


def test_adjacent_stores_to_a_plain_global_are_still_merged():
    tu = _optimize(PRELUDE + "void f(void){ plain=1; plain=2; plain=3; }")
    assert _assigns(tu, "f") == 1


@pytest.mark.parametrize("decl", [
    "volatile unsigned char l;", "vb l;", "vb2 l;",
    "unsigned char * volatile l;", "volatile int l;"])
def test_adjacent_stores_to_a_volatile_local_are_kept(decl):
    tu = _optimize(PRELUDE + f"void f(void){{ {decl} l=0; l=0; l=0; }}")
    assert _assigns(tu, "f") == 3


def test_adjacent_stores_to_a_volatile_parameter_are_kept():
    tu = _optimize(PRELUDE + "void f(volatile int a){ a=1; a=2; }")
    assert _assigns(tu, "f") == 2


@pytest.mark.parametrize("decl", ["vb l = 1;", "vb2 l = 1;",
                                  "volatile unsigned char l = 1;"])
def test_volatile_initializer_is_not_dropped(decl):
    tu = _optimize(PRELUDE + f"void f(void){{ {decl} l = 2; }}")
    inits = [d for d in _walk(_func(tu, "f").body)
             if isinstance(d, ast.InitDeclaratorWithInit)]
    assert len(inits) == 1


def test_volatile_pointer_initializer_is_not_dropped():
    tu = _optimize(PRELUDE
                   + "void f(void){ unsigned char * volatile l = pn; l = pn; }")
    inits = [d for d in _walk(_func(tu, "f").body)
             if isinstance(d, ast.InitDeclaratorWithInit)]
    assert len(inits) == 1 and _assigns(tu, "f") == 1


def test_plain_initializer_is_still_dropped():
    tu = _optimize(PRELUDE + "void f(void){ unsigned char l = 1; l = 2; }")
    inits = [d for d in _walk(_func(tu, "f").body)
             if isinstance(d, ast.InitDeclaratorWithInit)]
    assert len(inits) == 0


@pytest.mark.parametrize("body", [
    "r = gi; r = 1;",
    "r = gi + 1; r = 2;",
    "r = arr[1]; r = 2;",
    "r = st.a; r = 2;",
    "r = *vp; r = 2;",
    "r = (int)g; r = 2;",
    "r = *(volatile int *)0x1234; r = 2;",
    "r = tg; r = 2;",
])
def test_dead_store_does_not_drop_a_volatile_read(body):
    tu = _optimize(PRELUDE + f"void f(void){{ {body} }}")
    assert _assigns(tu, "f") == 2


def test_dead_store_of_a_plain_read_is_still_dropped():
    tu = _optimize(PRELUDE + "void f(void){ r = plain; r = 1; }")
    assert _assigns(tu, "f") == 1


def test_local_initialized_from_a_volatile_read_keeps_the_read():
    tu = _optimize(PRELUDE + "void f(void){ int a = gi; a = 2; r = a; }")
    assert any(isinstance(d, ast.InitDeclaratorWithInit)
               for d in _walk(_func(tu, "f").body))


@pytest.mark.parametrize("op", ["==", "!=", "<", ">", "<=", ">=", "|", "&"])
def test_self_comparison_and_idempotence_keep_both_reads(op):
    tu = _optimize(PRELUDE + f"int f(void){{ return gi {op} gi; }}")
    assert _count(tu, "f", ast.BinaryOp, op) == 1
    assert sum(1 for x in _walk(_func(tu, "f").body)
               if isinstance(x, ast.Identifier)) == 2


@pytest.mark.parametrize("op", ["==", "|", "&"])
def test_self_comparison_of_a_plain_object_is_still_folded(op):
    tu = _optimize(PRELUDE + f"int f(void){{ return plain {op} plain; }}")
    assert _count(tu, "f", ast.BinaryOp, op) == 0


@pytest.mark.parametrize("op", ["|", "&"])
def test_nested_idempotent_form_keeps_the_second_read(op):
    tu = _optimize(PRELUDE + f"int f(void){{ return (gi {op} gj) {op} gj; }}")
    assert _count(tu, "f", ast.BinaryOp, op) == 2


def test_modulo_one_keeps_the_read():
    tu = _optimize(PRELUDE + "void f(void){ r = gi % 1; }")
    assert any(getattr(x, "name", None) is not None
               and getattr(x.name, "text", "") == "gi"
               for x in _walk(_func(tu, "f").body) if isinstance(x, ast.Identifier))


def test_modulo_one_of_a_plain_object_is_still_folded():
    tu = _optimize(PRELUDE + "void f(void){ r = plain % 1; }")
    assert not any(isinstance(x, ast.Identifier) and x.name.text == "plain"
                   for x in _walk(_func(tu, "f").body))


def test_times_two_does_not_read_twice():
    tu = _optimize(PRELUDE + "int f(void){ return gi * 2; }")
    assert sum(1 for x in _walk(_func(tu, "f").body)
               if isinstance(x, ast.Identifier)) == 1


def test_times_two_of_a_volatile_object_stays_a_multiply():
    # x << 1 would be wrong for a volatile float.
    tu = _optimize(PRELUDE + "int f(void){ return gi * 2; }")
    assert _count(tu, "f", ast.BinaryOp, "*") == 1
    assert _count(tu, "f", ast.BinaryOp, "<<") == 0


def test_times_two_of_a_plain_object_becomes_an_add():
    tu = _optimize(PRELUDE + "int f(void){ return r * 2; }")
    assert _count(tu, "f", ast.BinaryOp, "+") == 1


@pytest.mark.parametrize("decl", ["volatile int i = 0", "vb i = 0"])
def test_loop_over_a_volatile_variable_is_not_unrolled(decl):
    tu = _optimize(PRELUDE + f"void f(void){{ for ({decl}; i < 3; i++) r += 1; }}")
    assert _count(tu, "f", ast.ForStmt) == 1


def test_loop_over_a_plain_variable_is_still_unrolled():
    tu = _optimize(PRELUDE + "void f(void){ for (int i = 0; i < 3; i++) r += 1; }")
    assert _count(tu, "f", ast.ForStmt) == 0


def test_copy_propagation_does_not_replace_a_volatile_object():
    tu = _optimize(PRELUDE
                   + "int f(void){ vb a = tg; vb b = a; return b + b; }")
    names = [x.name.text for x in _walk(_func(tu, "f").body)
             if isinstance(x, ast.Identifier)]
    assert "b" in names


def test_common_subexpression_does_not_merge_volatile_reads():
    tu = _optimize(PRELUDE + "void f(void){ r = gi + 1; r = gi + 1; }")
    assert sum(1 for x in _walk(_func(tu, "f").body)
               if isinstance(x, ast.Identifier) and x.name.text == "gi") == 2


def test_no_volatile_in_the_unit_changes_nothing():
    src = ("int r, s;\nvoid f(void){ r = 1; r = 2; s = r * 2; "
           "for (int i = 0; i < 3; i++) s += 1; }")
    tu = _optimize(src)
    # the dead store is gone and the loop unrolled
    assert _assigns(tu, "f") == 1 + 1 + 3   # r = 2; s = ...; i = 0, 1, 2
    assert _count(tu, "f", ast.ForStmt) == 0


def test_adjacent_string_literals_and_other_nodes_do_not_crash_the_analysis():
    tu = _optimize(PRELUDE + 'void f(void){ const char *t = "x" "y"; r = sizeof("a" "b");'
                   ' gi = (int)(long)t; gi = 2; }')
    assert _assigns(tu, "f") == 3
