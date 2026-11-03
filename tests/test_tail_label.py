"""C23 6.8.2: a label may be the last thing in a compound statement, and a
label may come before a declaration."""

import pytest

from uc_core import ast
from uc_core.frontend import parse


def _body(src):
    unit = parse(src, "t.c")
    func = [d for d in unit.items if isinstance(d, ast.FunctionDef)][0]
    return func.body.items


def test_case_label_at_end_of_switch_body():
    items = _body('int f(int x){ switch (x) { case 0: x = 1; break; case 1: }'
                  ' return x; }')
    sw = items[0]
    assert isinstance(sw, ast.SwitchStmt)
    last = sw.body.items[-1]
    assert isinstance(last, ast.CaseStmt)
    assert isinstance(last.stmt, ast.EmptyStmt)


def test_default_label_at_end_of_switch_body():
    items = _body('int f(int x){ switch (x) { case 0: x = 1; default: } '
                  'return x; }')
    last = items[0].body.items[-1]
    assert isinstance(last, ast.DefaultStmt)
    assert isinstance(last.stmt, ast.EmptyStmt)


def test_label_alone_in_block():
    items = _body('void f(void){ L: }')
    assert isinstance(items[0], ast.LabelStmt)
    assert isinstance(items[0].stmt, ast.EmptyStmt)


def test_two_labels_at_end_of_block():
    items = _body('void f(void){ goto M; L: M: }')
    assert [type(i).__name__ for i in items] == ["GotoStmt", "LabelStmt"]
    assert isinstance(items[1].stmt, ast.LabelStmt)
    assert isinstance(items[1].stmt.stmt, ast.EmptyStmt)


def test_label_at_end_of_nested_block_without_space():
    items = _body('void f(int n){ while (n) { n--; if (n == 3) goto E;E:} }')
    assert isinstance(items[0], ast.WhileStmt)


def test_label_before_declaration():
    items = _body('void f(void){ goto L; L: int y = 1; (void) y; }')
    lab = items[1]
    assert isinstance(lab, ast.LabelStmt)
    assert isinstance(lab.stmt, ast.Declaration)


def test_ternary_and_bit_field_colons_are_not_touched():
    items = _body('int f(int x){ struct { int a : 3; } s; s.a = 1;'
                  ' return x ? s.a : 2; }')
    assert len(items) == 3


def test_line_numbers_stay():
    unit = parse('void f(void)\n{\n  L:\n}\nint g;\n', "t.c")
    g = [d for d in unit.items if not isinstance(d, ast.FunctionDef)][0]
    assert g.pos.start_line == 5


def test_a_label_with_a_statement_still_parses():
    items = _body('int f(int x){ L: x++; if (x < 3) goto L; return x; }')
    assert isinstance(items[0].stmt, ast.ExpressionStmt)


def _parses(src):
    parse(src, "t.c")


def test_case_label_before_declaration():
    _parses('void f(int x){ switch(x){ case 1: int y=2; (void)y; } }')


def test_default_label_before_declaration():
    _parses('void f(int x){ switch(x){ default: static int y=2; (void)y; } }')


def test_case_with_ternary_before_declaration():
    _parses('void f(int x){ switch(x){ case 1 ? 2 : 3: long y=2; (void)y; } }')


def test_case_label_before_typedef_name_declaration():
    _parses('typedef int T; void f(int x){ switch(x){ case 1: T y=2; (void)y; } }')


def test_label_before_static_assert():
    _parses('void f(void){ L: static_assert(1, "a"); }')
    _parses('void f(int x){ switch(x){ case 1: static_assert(1, "a"); } }')


def test_case_before_statement_is_unchanged():
    items = _body('int f(int x){ switch(x){ case 1: x = 2; } return x; }')
    case = items[0].body.items[0]
    assert not isinstance(case.stmt, ast.EmptyStmt)
