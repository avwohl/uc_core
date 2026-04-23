"""TypeConfig: configurable integer type sizes (Phase 1).

Verifies that size/width/limits are derived from config, that the
predefined-macro bundle is well-formed, and that sizeof(int) constant-folds
according to config (not hardcoded 16-bit).
"""

import pytest

from uc_core.type_config import TypeConfig, Z80_CPM, WATCOM_FLAT32
from uc_core.lexer import Lexer
from uc_core.parser import Parser
from uc_core.preprocessor import Preprocessor
from uc_core.ast_optimizer import ASTOptimizer
from uc_core import ast


def test_z80_preset():
    tc = Z80_CPM
    assert tc.int_size == 2 and tc.long_size == 4
    assert tc.long_long_size == 8 and tc.ptr_size == 2
    assert tc.int_max == 0x7FFF and tc.uint_max == 0xFFFF
    assert tc.long_max == 0x7FFFFFFF


def test_watcom_flat32_preset():
    tc = WATCOM_FLAT32
    assert tc.int_size == 4 and tc.long_size == 4
    assert tc.long_long_size == 8 and tc.ptr_size == 4
    assert tc.int_max == 0x7FFFFFFF
    assert tc.long_max == 0x7FFFFFFF


def test_constraint_char_le_short_le_int_le_long():
    with pytest.raises(ValueError):
        TypeConfig(int_size=1)  # int < short violates
    with pytest.raises(ValueError):
        TypeConfig(int_size=4, long_size=2)  # long < int violates
    with pytest.raises(ValueError):
        TypeConfig(long_long_size=2)  # long long < long violates


def test_sizeof_basic_covers_signed_unsigned_variants():
    tc = Z80_CPM
    assert tc.sizeof_basic("int") == 2
    assert tc.sizeof_basic("unsigned int") == 2
    assert tc.sizeof_basic("signed int") == 2
    assert tc.sizeof_basic("long") == 4
    assert tc.sizeof_basic("unsigned long long") == 8
    assert tc.sizeof_basic("char") == 1
    assert tc.sizeof_basic("not a type") is None


def test_predefined_macros_bundle():
    tc = WATCOM_FLAT32
    m = tc.predefined_macros()
    assert m["__SIZEOF_INT__"] == "4"
    assert m["__SIZEOF_LONG__"] == "4"
    assert m["__SIZEOF_POINTER__"] == "4"
    assert m["__INT_WIDTH__"] == "32"
    assert m["__LONG_WIDTH__"] == "32"
    assert m["__CHAR_BIT__"] == "8"
    assert m["__INT_MAX__"] == "2147483647"
    assert m["__LONG_MAX__"] == "2147483647L"


def _compile_and_fold(src: str, tc: TypeConfig) -> ast.TranslationUnit:
    pp = Preprocessor(target_predefines=tc.predefined_macros())
    pre = pp.preprocess(src, "t.c")
    tokens = list(Lexer(pre, "t.c").tokenize())
    unit = Parser(tokens).parse()
    return ASTOptimizer(3, type_config=tc).optimize(unit)


def _first_initializer_value(unit: ast.TranslationUnit):
    for decl in unit.declarations:
        if isinstance(decl, ast.VarDecl) and decl.init is not None:
            return decl.init
    return None


def test_sizeof_int_folds_to_config_width_z80():
    unit = _compile_and_fold("int a = sizeof(int);\n", Z80_CPM)
    init = _first_initializer_value(unit)
    assert isinstance(init, ast.IntLiteral)
    assert init.value == 2


def test_sizeof_int_folds_to_config_width_flat32():
    unit = _compile_and_fold("int a = sizeof(int);\n", WATCOM_FLAT32)
    init = _first_initializer_value(unit)
    assert isinstance(init, ast.IntLiteral)
    assert init.value == 4


def test_sizeof_long_long_folds_to_eight():
    unit = _compile_and_fold("int a = sizeof(long long);\n", Z80_CPM)
    init = _first_initializer_value(unit)
    assert init.value == 8


def test_sizeof_pointer_follows_config():
    unit = _compile_and_fold("int a = sizeof(int *);\n", WATCOM_FLAT32)
    assert _first_initializer_value(unit).value == 4
    unit = _compile_and_fold("int a = sizeof(int *);\n", Z80_CPM)
    assert _first_initializer_value(unit).value == 2


def test_predefined_sizeof_macro_visible_to_source():
    src = """
    #if __SIZEOF_INT__ == 4
    int is_wide = 1;
    #else
    int is_wide = 0;
    #endif
    """
    unit = _compile_and_fold(src, WATCOM_FLAT32)
    assert _first_initializer_value(unit).value == 1
    unit = _compile_and_fold(src, Z80_CPM)
    assert _first_initializer_value(unit).value == 0


def test_literal_mask_uses_config_width():
    """0x8000 is unsigned-int on 16-bit (exceeds INT_MAX=32767) but plain int
    on 32-bit. Verify _literal_mask picks the right width per config."""
    from uc_core.ast import IntLiteral
    lit = IntLiteral(value=0x8000, is_hex=True)
    opt16 = ASTOptimizer(3, type_config=Z80_CPM)
    opt32 = ASTOptimizer(3, type_config=WATCOM_FLAT32)
    # On 16-bit int, 0x8000 fits only as 16-bit unsigned int — mask = 0xFFFF
    assert opt16._literal_mask(lit) == 0xFFFF
    # On 32-bit int, 0x8000 fits in signed int — mask = 0xFFFFFFFF
    assert opt32._literal_mask(lit) == 0xFFFFFFFF
