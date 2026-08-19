"""DOS-era C syntactic tolerance.

uc_core must parse (and ignore) the non-standard keywords/qualifiers used by
MS-C / Borland / Watcom headers from the DOS period. Under flat-32 these are
all no-ops; under Z80 they're meaningless. Either way, the parser just needs
to not choke on them.
"""

import pytest

from uc_core.frontend import parse as _frontend_parse
from uc_core.preprocessor import Preprocessor
from uc_core import ast


def _parse(src: str) -> ast.TranslationUnit:
    pre = Preprocessor().preprocess(src, "t.c")
    return _frontend_parse(pre, "t.c")


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
    assert len(unit.items) == 1


def test_dos_qualifiers_are_ignored_not_typedefs():
    """'far', 'near' etc. must not be treated as typedef names or identifiers
    that end up in the AST — they should be silently absorbed."""
    unit = _parse("int far x;")
    decl = unit.items[0]
    assert isinstance(decl, ast.Declaration)
    # The single init_declarator carries the variable's name.
    from uc_core.ast_optimizer import _declarator_ident
    assert len(decl.declarators) == 1
    inner = decl.declarators[0].declarator
    assert _declarator_ident(inner) == "x"


def test_watcall_function_parses_like_cdecl():
    """Phase 1: all calling conventions compile to the same ABI. Verify the
    AST shape matches a plain function declaration."""
    from uc_core.ast_optimizer import _declarator_ident, _outermost_fn_declarator
    plain = _parse("int f(int a, int b);").items[0]
    watcall = _parse("int __watcall f(int a, int b);").items[0]
    plain_decl = plain.declarators[0].declarator
    watcall_decl = watcall.declarators[0].declarator
    assert _declarator_ident(plain_decl) == _declarator_ident(watcall_decl) == "f"
    plain_fn = _outermost_fn_declarator(plain_decl)
    watcall_fn = _outermost_fn_declarator(watcall_decl)
    assert plain_fn is not None and watcall_fn is not None
    assert isinstance(plain_fn, ast.FnDeclarator)
    assert isinstance(watcall_fn, ast.FnDeclarator)
    # Same parameter count.
    plain_params = plain_fn.params.params if isinstance(plain_fn.params, ast.VariadicParams) else plain_fn.params
    watcall_params = watcall_fn.params.params if isinstance(watcall_fn.params, ast.VariadicParams) else watcall_fn.params
    assert len(plain_params) == len(watcall_params)


# The unprefixed spellings are ordinary identifiers in C -- C23 has no
# near/far/huge -- so absorbing one has to stop at the places a program can
# legitimately use the word.  A tag is the case the qualifier rule could not
# see: `struct huge mk(void)` is the word followed by another word, exactly
# like `near int n`, and erasing it left `struct mk(void)`, which does not
# parse.  The error then named the line, not the identifier, so it read as a
# compiler fault rather than a reserved name.
TAG_NAME_CASES = [
    ("struct tag",              "struct huge { int a; };"),
    ("struct tag in decl",      "struct huge { int a; }; struct huge v;"),
    ("struct tag as return",    "struct far { int a; }; struct far f(void);"),
    ("union tag",               "union near { int a; char b; };"),
    ("enum tag",                "enum huge { E1, E2 };"),
    ("tag through a pointer",   "struct huge { int a; }; struct huge *p;"),
    ("tag as a parameter",      "struct far { int a; }; void f(struct far *p);"),
    ("cdecl as a tag",          "struct cdecl { int a; }; struct cdecl v;"),
    ("interrupt as a tag",      "struct interrupt { int a; }; struct interrupt v;"),
]


@pytest.mark.parametrize("label,src", TAG_NAME_CASES,
                         ids=[c[0] for c in TAG_NAME_CASES])
def test_qualifier_words_survive_in_tag_position(label, src):
    unit = _parse(src)
    assert unit is not None
    assert unit.items


def test_tag_keeps_its_name():
    """Parsing is not enough -- the tag has to still be called `huge`, or a
    later `struct huge` reference resolves to nothing."""
    unit = _parse("struct huge { int a; };")
    decl = unit.items[0]
    assert "huge" in str(decl), decl


@pytest.mark.parametrize("word", ["huge", "far", "near", "cdecl", "interrupt"])
def test_a_variable_of_such_a_type_is_still_a_variable(word):
    """The quiet half of the bug, and the worse one.  Erasing the tag out of
    `struct huge v;` leaves `struct v;` -- a forward declaration of a struct
    called v.  That parses, so nothing was reported; the variable simply
    stopped existing, and the error landed later on whatever used it.  A
    declaration that declares something has a declarator; the wreckage has
    an empty list."""
    unit = _parse("struct %s { int a; }; struct %s v;" % (word, word))
    decl = unit.items[1]
    assert decl.declarators, (
        "the declarator was erased along with the tag: %r" % (decl,))
    assert "v" in str(decl.declarators[0])


def test_qualifier_after_a_tag_is_still_absorbed():
    """Only the word directly after struct/union/enum is a tag.  In
    `struct foo far *p` the far is a real qualifier and must still go, or
    DOS headers stop parsing."""
    unit = _parse("struct foo { int a; }; struct foo far *p;")
    assert unit is not None
    assert len(unit.items) == 2


def test_a_tag_named_far_can_still_take_a_far_pointer():
    """Both at once: the first far names the struct, the second qualifies
    the pointer."""
    unit = _parse("struct far { int a; }; struct far far *p;")
    assert unit is not None
    assert len(unit.items) == 2
