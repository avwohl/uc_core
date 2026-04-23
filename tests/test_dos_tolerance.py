"""DOS-era C syntactic tolerance.

uc_core must parse (and ignore) the non-standard keywords/qualifiers used by
MS-C / Borland / Watcom headers from the DOS period. Under flat-32 these are
all no-ops; under Z80 they're meaningless. Either way, the parser just needs
to not choke on them.
"""

import pytest

from uc_core.lexer import Lexer
from uc_core.parser import Parser
from uc_core import ast


def _parse(src: str) -> ast.TranslationUnit:
    tokens = list(Lexer(src, "t.c").tokenize())
    return Parser(tokens).parse()


# Each case is a complete translation unit that must parse without error.
DOS_CASES = [
    # 16-bit pointer qualifiers
    ("near on declarator",      "int near x;"),
    ("far on declarator",       "int far y;"),
    ("huge on declarator",      "int huge z;"),
    ("__near on declarator",    "int __near a;"),
    ("__far on declarator",     "int __far b;"),
    ("__huge on declarator",    "int __huge c;"),
    ("char far pointer",        "char far *p;"),
    ("char huge pointer",       "char huge *p;"),
    ("__far pointer",           "char __far *p;"),
    # Segment registers
    ("_cs qualifier",           "int _cs code_var;"),
    ("_ds qualifier",           "int _ds data_var;"),
    ("__seg qualifier",         "int __seg seg_var;"),
    # __based() — Microsoft-style based pointers
    ("__based paren",           "int __based(__segname(\"_DATA\")) *p;"),
    ("_based paren",            "int _based(seg) x;"),
    # Calling conventions on function declarations
    ("__cdecl decl",            "void __cdecl f(void);"),
    ("__pascal decl",           "void __pascal f(void);"),
    ("__stdcall decl",          "void __stdcall f(void);"),
    ("__fastcall decl",         "int __fastcall f(int a);"),
    ("__watcall decl",          "int __watcall f(int a, int b);"),
    ("__syscall decl",          "void __syscall f(void);"),
    ("bare cdecl",              "void cdecl f(void);"),
    ("_pascal decl",            "void _pascal f(void);"),
    ("fortran decl",            "void fortran f(void);"),
    # Function attributes
    ("__interrupt handler",     "void __interrupt handler(void);"),
    ("interrupt handler",       "void interrupt handler(void);"),
    ("__loadds",                "void __loadds f(void);"),
    ("__saveregs",              "void __saveregs f(void);"),
    ("__export",                "void __export f(void);"),
    # Qualifier on pointer after *
    ("far after star",          "char * far p;"),
    # Interleaved with storage class
    ("static __cdecl",          "static void __cdecl f(void) { }"),
    ("extern __pascal",         "extern void __pascal f(void);"),
    # Function definition (not just declaration)
    ("__cdecl def",             "int __cdecl main(void) { return 0; }"),
    ("far ptr param",           "void f(char far *p);"),
    # Combos
    ("cdecl + far + ptr",       "int __cdecl far *f(void);"),
    ("near int variable",       "near int n;"),
]


@pytest.mark.parametrize("label,src", DOS_CASES, ids=[c[0] for c in DOS_CASES])
def test_dos_qualifiers_parse(label, src):
    unit = _parse(src)
    assert unit is not None
    # Each snippet declares exactly one top-level entity
    assert len(unit.declarations) == 1


def test_dos_qualifiers_are_ignored_not_typedefs():
    """'far', 'near' etc. must not be treated as typedef names or identifiers
    that end up in the AST — they should be silently absorbed."""
    unit = _parse("int far x;")
    decl = unit.declarations[0]
    assert isinstance(decl, ast.VarDecl)
    assert decl.name == "x"


def test_watcall_function_parses_like_cdecl():
    """Phase 1: all calling conventions compile to the same ABI. Verify the
    AST shape matches a plain function declaration."""
    plain = _parse("int f(int a, int b);").declarations[0]
    watcall = _parse("int __watcall f(int a, int b);").declarations[0]
    assert plain.name == watcall.name
    assert isinstance(plain.var_type, ast.FunctionType)
    assert isinstance(watcall.var_type, ast.FunctionType)
    assert len(plain.var_type.param_types) == len(watcall.var_type.param_types)
