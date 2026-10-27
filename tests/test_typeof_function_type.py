"""typeof(function type): ``typeof(int (void))`` parses.

The grammar takes typeof(expr) only; the frontend rewrites a type name to
``typeof(*(type-name *)0)``.  A function type name (a parameter list right
after the specifiers) used to stay unrewritten and stopped the parse.
"""

import pytest

from uc_core.frontend import parse, _typeof_typename_fix


@pytest.mark.parametrize("src", [
    "typeof(int (void)) f;",
    "typeof(int (int, char *)) f;",
    "typeof(char *(void)) f;",
    "typeof(int ()) f;",
    "typeof(int [3]) a;",
])
def test_typeof_type_names_parse(src):
    unit = parse(src, "t.c")
    assert unit is not None


def test_rewrite_of_a_function_type():
    class Tok:
        offset = 0
    src = "typeof(int (void)) f;"
    Tok.offset = src.index("int")
    assert _typeof_typename_fix(src, Tok) == \
        "typeof(*(int (*)(void))0) f;"
