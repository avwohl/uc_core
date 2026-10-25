"""Opt-in strict mode: interrupt / fastcall markers are refused, not erased."""
import pytest

from uc_core.frontend import parse, UnsupportedConstructError

BAD = [
    ("__interrupt", "void __interrupt h(void) { }", "__interrupt"),
    ("_interrupt", "void _interrupt h(void) { }", "_interrupt"),
    ("bare interrupt", "void interrupt h(void) { }", "interrupt"),
    ("__interrupt with args", "void __interrupt(5) h(void) { }", "__interrupt"),
    ("gnu attr", "void h(void) __attribute__((interrupt)); ",
     "__attribute__((interrupt))"),
    ("gnu attr args", 'void h(void) __attribute__((interrupt("IRQ")));',
     "__attribute__((interrupt))"),
    ("gnu attr dunder", "void h(void) __attribute__((__interrupt__));",
     "__attribute__((interrupt))"),
    ("c23 attr", "[[gnu::interrupt]] void h(void) { }", "[[interrupt]]"),
    ("__fastcall", "int __fastcall f(int a);", "__fastcall"),
    ("_fastcall", "int _fastcall f(int a);", "_fastcall"),
    ("bare fastcall", "int fastcall f(int a);", "fastcall"),
]


@pytest.mark.parametrize("label,src,name", BAD, ids=[c[0] for c in BAD])
def test_refused(label, src, name):
    with pytest.raises(UnsupportedConstructError) as ei:
        parse("\n\n" + src + "\n", "t.c", strict_unsupported=True)
    assert ei.value.line == 3
    assert ei.value.name == name
    assert str(ei.value).startswith("t.c:3: ")


@pytest.mark.parametrize("label,src,name", BAD, ids=[c[0] for c in BAD])
def test_tolerant_by_default(label, src, name):
    if "args" in label and "__interrupt" in src:
        pytest.skip("tolerant mode cannot parse __interrupt(5) either")
    assert parse(src, "t.c") is not None


def test_custom_message():
    with pytest.raises(UnsupportedConstructError) as ei:
        parse("void __interrupt h(void) { }", "a.c", strict_unsupported=True,
              unsupported_message="uc80 does not support '{name}' yet; "
                                  "see docs/rom_target.md")
    assert str(ei.value) == ("a.c:1: uc80 does not support '__interrupt' yet; "
                             "see docs/rom_target.md")


OK = [
    "int interrupt_count; int f(void) { return interrupt_count; }",
    "int fastcall_hits; void g(void) { fastcall_hits = 1; }",
    "struct interrupt { int a; }; struct interrupt v;",
    "int interrupt; void g(void) { interrupt = 1; }",
    'const char *s = "__interrupt fastcall [[interrupt]]";',
    "char c = '_'; int my_interrupt_flag;",
    "void __attribute__((noreturn)) f(void);",
    "[[deprecated]] int x;",
    "int __cdecl f(void);",
]


@pytest.mark.parametrize("src", OK)
def test_unrelated_names_pass(src):
    assert parse(src, "t.c", strict_unsupported=True) is not None
