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


def test_a_block_typedef_does_not_rename_a_tag_of_the_same_name():
    from uc_core import frontend
    src = ("typedef int T;\nstruct T { int a; };\n"
           "int f(void) { { typedef long T; struct T s; return sizeof(T); } }\n")
    out = frontend._rename_block_scope_typedefs(src, "x.c")
    assert "struct T s;" in out and "struct T { int a; }" in out
    assert "typedef long T__t" in out


def test_a_typedef_redeclared_in_a_nested_block_is_renamed():
    from uc_core import frontend
    src = ("int f(void) { typedef long long L; { typedef char L; L c; }\n"
           "  return sizeof(L); }\n")
    out = frontend._rename_block_scope_typedefs(src, "x.c")
    assert "typedef char L__t" in out
    assert "return sizeof(L);" in out


# ---- the scope of an enumerator starts after its enumerator-definition ------


def test_a_block_enumerator_in_its_own_initializer_names_the_outer_one():
    out = _r("enum { A = 5 };\nint main(void) { { enum { A = A * 2 }; "
             "int x = A; } return A; }\n")
    body = out.split("\n")[1]
    new = body.split(" { ")[2].split(" =")[0]
    assert new.startswith("A__c")
    assert "= A * 2" in body              # the initializer: the outer A
    assert "int x = " + new + ";" in body  # after the definition: the new A
    assert body.endswith("return A; }")


def test_an_enumerator_is_visible_in_the_next_initializer_only():
    out = _r("enum { B = 1 };\nvoid f(void) { enum { B = 3, C = B + 1, "
             "D = g(B, C) }; }\n")
    body = out.split("\n")[1]
    new = body.split(" { ")[2].split(" =")[0]
    assert "C__c" in body and "D__c" in body
    assert body.count(new) == 3     # B = 3, C = B + 1, g(B, C)
    assert "= g(" + new + ", C__c" in body


def test_a_comma_in_parentheses_does_not_define_an_enumerator():
    out = _r("void f(void) { enum { E = h(1, K), K = 2 }; }\n")
    assert "h(1, K)" in out         # K is not yet declared there
    assert out.count("__c") == 2    # the definitions of E and K


def test_a_block_typedef_is_seen_in_an_enumerator_initializer():
    out = _rename_block_scope_typedefs(
        "typedef int T;\nstatic int h(void) { typedef char T; "
        "enum { T3 = sizeof(T) }; return T3; }\n", "t.c")
    new = out.split("typedef char ")[1].split(";")[0]
    assert new.startswith("T__t") and out.count(new) == 2
    assert "sizeof(" + new + ")" in out


def test_an_enumerator_named_like_a_typedef_is_not_renamed():
    src = ("typedef int T;\nint f(void) { typedef char T; "
           "enum { T2 = 1 }; return T2; }\n")
    out = _rename_block_scope_typedefs(src, "t.c")
    assert "T2 = 1" in out and "return T2" in out


# ---- typedef of a tag before the block definition of the tag ---------------


def test_a_tag_that_a_block_uses_first_belongs_to_the_block():
    out = _r("int main(void) { typedef struct T T; struct T { int a; }; "
             "return sizeof(T); }")
    names = [w for w in out.replace(";", " ").split() if w.startswith("T__b")]
    assert len(names) == 2 and names[0] == names[1]


def test_a_tag_of_the_file_is_not_renamed_by_a_use_in_a_block():
    src = "struct T { int a; };\nint f(void) { struct T x; return x.a; }\n"
    assert _r(src) == src


# ---- a typedef name in a struct or union body ------------------------------


def test_a_block_typedef_in_a_member_array_bound_is_renamed():
    out = _rename_block_scope_typedefs(
        "typedef int T;\nint f(void) { typedef char T; "
        "struct S { T a[sizeof(T) + 2]; } s; return sizeof s.a; }\n", "t.c")
    inner = out.split("typedef char ")[1].split(";")[0]
    assert inner.startswith("T__t")
    assert out.count(inner) == 3          # the declarator and both uses


def test_a_member_with_the_name_of_a_typedef_is_not_renamed():
    out = _rename_block_scope_typedefs(
        "typedef int T;\nint f(void) { typedef char T; "
        "struct S { int T; } s; return s.T; }\n", "t.c")
    assert "int T;" in out and "s.T" in out


# ---- a parameter hides a file-scope enumerator ------------------------------


def test_a_parameter_hides_an_enumerator_in_the_body():
    out = _r("enum { A = 2 };\nstatic int f(int A) { enum { B = 3 }; "
             "return A + B; }\nint g(void) { return A; }\n")
    lines = out.split("\n")
    new = lines[1].split("(int ")[1].split(")")[0]
    assert new.startswith("A__c")
    assert "return " + new + " + " in lines[1]
    assert lines[2] == "int g(void) { return A; }"


def test_a_pointer_and_a_function_pointer_parameter_hide_enumerators():
    out = _r("enum { A = 2, C = 5 };\nint f(char *A, int (*C)(int)) "
             "{ return A[0] + C(1); }\n")
    assert out.count("A__c") == 2 and out.count("C__c") == 2


def test_a_prototype_and_a_call_keep_the_enumerator():
    src = ("enum { A = 2 };\nint f(int A);\nint g(void) { return f(A); }\n")
    out = _r(src)
    assert "int f(int A);" in out and "return f(A);" in out


def test_a_parameter_named_like_an_enumerator_is_the_parameter_in_the_program():
    src = ("enum { A = 2 };\nint g(int A) { return A + 3; }\n"
           "int main(void) { return g(10); }\n")
    out = _r(src)
    assert "return A + 3" not in out and "g(10)" in out
