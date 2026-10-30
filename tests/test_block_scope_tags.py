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


def test_enum_constants_of_a_block_enum_get_their_own_names():
    out = _r("enum E { X1 = 5 };\n"
             "void f(void) { int a = X1; { enum F { X1 = 77, Y = X1 + 1 }; "
             "a = X1 + Y; } a = X1; }\n")
    first, second = out.split("\n")[:2]
    assert first == "enum E { X1 = 5 };"
    assert second.startswith("void f(void) { int a = X1; ")
    assert second.endswith(" a = X1; }")
    new = second.split("{ ")[3].split(" =")[0]
    assert new.startswith("X1__c") and second.count(new) == 3
    assert second.count("Y__c") == 2


def test_a_member_after_dot_or_arrow_is_not_renamed():
    out = _r("void f(void) { struct S { int X; } s; { enum { X = 3 }; "
             "s.X = X; } }")
    assert "s.X = X__c" in out and "{ int X; } s;" in out


def test_a_variable_that_hides_an_enum_constant_gets_its_own_name():
    out = _r("enum { A = 5 };\nint f(void) { { int A = 3; return A; } }\n"
             "int g(void) { return A; }\n")
    lines = out.split("\n")
    assert "{ int A__c" in lines[1] and lines[1].count("A__c") == 2
    assert lines[2] == "int g(void) { return A; }"


# ---- block-scope typedefs and a typedef name as a label ---------------------

from uc_core.frontend import _rename_block_scope_typedefs, parse


def test_a_block_typedef_of_a_file_scope_name_is_renamed_in_its_block():
    out = _rename_block_scope_typedefs(
        "typedef int T;\nint f(void) { T a; { typedef long T; T v; } "
        "return sizeof(T); }\n", "t.c")
    inner = out.split("typedef long ")[1].split(";")[0]
    assert inner.startswith("T__t") and out.count(inner) == 2
    assert out.endswith("return sizeof(T); }\n")


def test_a_block_typedef_of_a_new_name_is_not_renamed():
    src = "typedef int T;\nint f(void) { typedef long U; U v; return 0; }\n"
    assert _rename_block_scope_typedefs(src, "t.c") == src


def test_a_variable_hides_the_block_typedef_for_the_rest_of_the_block():
    out = _rename_block_scope_typedefs(
        "typedef int T;\nint f(void) { typedef long T; { int T = 3; "
        "return T; } }\n", "t.c")
    assert out.count("T__t") == 1       # only the declarator


def test_the_redeclared_typedef_parses():
    unit = parse("typedef int T;\nint main(void) { { typedef long T; "
                 "T v = 100000L; return (int) v; } }\n", "t.c")
    assert unit is not None


def test_goto_and_a_label_with_a_typedef_name_parse():
    src = ("typedef struct s s;\nstruct s { int a; } s2;\n"
           "int main(void) { goto s; struct s s; return s.a; s: return 0; }\n")
    assert parse(src, "t.c") is not None
