"""``enum e : T { ... }`` and ``enum e : T;`` (C23 6.7.3.3): the front end
takes the ``: T`` out of the text and keeps the type in
``unit._enum_fixed_types``."""

from uc_core.frontend import _strip_enum_fixed_types, parse


def test_the_type_is_taken_out_and_kept():
    out, table = _strip_enum_fixed_types(
        "enum e : unsigned char { A, B };\nenum g : short;\n"
        "enum g : short { G };\nenum h : uint8_t { H };\n", "t.c")
    assert out == ("enum e  { A, B };\nenum g ;\nenum g  { G };\n"
                   "enum h  { H };\n")
    assert table == {"e": ["unsigned char"], "g": ["short", "short"],
                     "h": ["uint8_t"]}


def test_an_anonymous_enumeration_gets_a_tag():
    out, table = _strip_enum_fixed_types("enum : long { Z };\n", "t.c")
    (tag, types), = table.items()
    assert tag.startswith("__enum_fixed_") and types == ["long"]
    assert out == f"enum  {tag} {{ Z }};\n"


def test_a_bit_field_of_an_enumeration_type_stays():
    src = "struct s { enum e : 3; enum e v : 2; int x; };\n"
    assert _strip_enum_fixed_types(src, "t.c") == (src, {})


def test_a_plain_enumeration_is_not_touched():
    src = "enum e { A, B };\nint f(enum e x) { return x ? 1 : 2; }\n"
    assert _strip_enum_fixed_types(src, "t.c") == (src, {})


def test_the_unit_carries_the_table():
    unit = parse("enum e : unsigned char { A = 255 };\nenum f { B };\n", "t.c")
    assert unit._enum_fixed_types == {"e": ["unsigned char"]}
    assert len(unit.items) == 2
