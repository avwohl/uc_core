"""DOS-era #pragma tolerance.

Period Watcom / MS-C / Borland headers use pragmas that have no meaning under
flat-32 or Z80. The preprocessor must silently drop them without affecting
the token stream. `#pragma pack` and `#pragma printf/scanf` remain honored.
"""

import pytest

from uc_core.lexer import Lexer
from uc_core.parser import Parser
from uc_core.preprocessor import Preprocessor


DROP_CASES = [
    ("hdrstop",            "#pragma hdrstop"),
    ("hdrfile",            '#pragma hdrfile "foo.pch"'),
    ("warn disable",       "#pragma warn -par"),
    ("warning disable",    "#pragma warning(disable: 4996)"),
    ("intrinsic",          "#pragma intrinsic(memcpy, strlen)"),
    ("function",           "#pragma function(memcpy)"),
    ("check_stack off",    "#pragma check_stack(off)"),
    ("code_seg",           '#pragma code_seg("MYCODE")'),
    ("data_seg",           '#pragma data_seg("MYDATA")'),
    ("alloc_text",         "#pragma alloc_text(FOO, func)"),
    ("disable_message",    "#pragma disable_message(202)"),
    ("argsused",           "#pragma argsused"),
    ("inline",             "#pragma inline"),
    ("library",            '#pragma library("foo.lib")'),
    ("startup",            "#pragma startup myfunc 64"),
    ("exit",               "#pragma exit myfunc"),
    # Watcom #pragma aux — the big one
    ("aux convention",     "#pragma aux f parm [eax] [edx] value [eax] modify [ecx];"),
    ("aux inline asm",     '#pragma aux f = "add eax, edx" parm [eax] [edx] value [eax];'),
    # Multi-line via backslash continuation
    ("aux multiline",      "#pragma aux f parm [eax] [edx] \\\n    value [eax] \\\n    modify [ecx];"),
]


@pytest.mark.parametrize("label,pragma", DROP_CASES, ids=[c[0] for c in DROP_CASES])
def test_pragma_drops_silently(label, pragma):
    src = f"{pragma}\nint x;\n"
    pp = Preprocessor()
    out = pp.preprocess(src, "t.c")
    # Parser must see only the real declaration
    tokens = list(Lexer(out, "t.c").tokenize())
    unit = Parser(tokens).parse()
    assert len(unit.declarations) == 1
    assert unit.declarations[0].name == "x"


def test_pragma_pack_still_honored():
    """#pragma pack is the one DOS-era pragma we must actually process."""
    src = "#pragma pack(1)\nint x;\n"
    pp = Preprocessor()
    out = pp.preprocess(src, "t.c")
    # It shouldn't error, and the declaration should still parse
    tokens = list(Lexer(out, "t.c").tokenize())
    unit = Parser(tokens).parse()
    assert len(unit.declarations) == 1


def test_pragma_printf_still_honored():
    """#pragma printf must still set printf_features (backend-visible state)."""
    pp = Preprocessor()
    pp.preprocess("#pragma printf int long\nint x;\n", "t.c")
    assert pp.printf_features is not None
    assert "int" in pp.printf_features
    assert "long" in pp.printf_features
