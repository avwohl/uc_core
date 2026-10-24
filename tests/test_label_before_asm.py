"""C23 6.8.2: a label may precede any statement.  GCC's `asm(...)`
statement was not accepted after a label (`L: asm("nop");`), so a timer
macro that opens with a label and then an asm statement was a parse error."""

from uc_core import ast
from uc_core.frontend import parse


def _body_items(src):
    unit = parse(src, "t.c")
    func = [d for d in unit.items if isinstance(d, ast.FunctionDef)][0]
    return func.body.items


def test_label_then_asm_statement_parses():
    items = _body_items('int f(void) { LCONT: asm("nop"); return 0; }')
    labels = [i for i in items if isinstance(i, ast.LabelStmt)]
    assert len(labels) == 1
    assert isinstance(labels[0].stmt, ast.AsmDeclaration)


def test_label_then_extended_asm_statement_parses():
    items = _body_items(
        'int f(void) { L: __asm__ volatile ("" ::: "memory"); return 0; }')
    assert isinstance(items[0], ast.LabelStmt)
    assert isinstance(items[0].stmt, ast.AsmDeclaration)
