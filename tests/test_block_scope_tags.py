"""The front end renames a tag that a block declares (uc_core.frontend)."""

from uc_core.frontend import _rename_block_scope_tags


def _r(src):
    return _rename_block_scope_tags(src, "t.c")


def test_file_scope_tags_and_plain_uses_keep_their_names():
    src = "struct S { int a; };\nvoid f(void) { struct S x; (void)x; }\n"
    assert _r(src) == src


def test_a_block_definition_gets_its_own_name_and_its_uses_follow():
    out = _r("struct S { int a; };\n"
             "void f(void) { struct S { char c; }; struct S x; (void)x; }\n"
             "void g(void) { struct S y; (void)y; }\n")
    lines = out.split("\n")
    assert lines[0] == "struct S { int a; };"
    new = lines[1].split("struct ")[1].split(" ")[0]
    assert new.startswith("S__b") and lines[1].count(new) == 2
    assert lines[2] == "void g(void) { struct S y; (void)y; }"


def test_the_tag_of_a_struct_declared_in_a_tag_body_is_the_blocks():
    out = _r("void f(void) { struct O { struct I { int q; } i; } o; "
             "struct I k; }")
    inner = out.split("struct ")[3].split(" ")[0]
    assert inner.startswith("I__b") and out.count(inner) == 2


def test_a_block_scope_is_popped_at_its_closing_brace():
    out = _r("void f(void) { { struct T { int a; } t; } "
             "struct T { char c; } u; }")
    names = [w for w in out.replace("{", " ").split() if w.startswith("T__b")]
    assert len(set(names)) == 2


def test_an_anonymous_enum_with_values_gets_a_tag_and_one_without_does_not():
    out = _r("enum { A = 70000, B } x;\nenum { C, D } y;\n")
    first, second = out.split("\n")[:2]
    assert first.startswith("enum __anon_enum_") and "{ A = 70000, B } x;" in first
    assert second == "enum { C, D } y;"
