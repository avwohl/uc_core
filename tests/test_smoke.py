"""Smoke tests: verify the uc_core pipeline is wired up and target-neutral."""

from uc_core.lexer import Lexer
from uc_core.parser import Parser
from uc_core.preprocessor import Preprocessor
from uc_core.ast_optimizer import ASTOptimizer
from uc_core import ast
from uc_core.backend import CodeGenerator


SRC = """
#ifdef __MYTARGET__
int answer(void) { return 42; }
#endif
"""


def test_pipeline_end_to_end():
    pp = Preprocessor(target_predefines={"__MYTARGET__": "1"})
    preprocessed = pp.preprocess(SRC, "t.c")
    tokens = list(Lexer(preprocessed, "t.c").tokenize())
    unit = Parser(tokens).parse()
    unit = ASTOptimizer(3).optimize(unit)
    funcs = [d for d in unit.declarations if isinstance(d, ast.FunctionDecl)]
    assert len(funcs) == 1
    assert funcs[0].name == "answer"


def test_target_predefines_are_caller_supplied():
    """uc_core is target-neutral: no built-in target macros."""
    pp = Preprocessor()  # no target_predefines
    for name in ("__Z80__", "__CPM__", "__UC80__", "__I386__", "__MSDOS__"):
        assert name not in pp.macros, f"{name} leaked into target-neutral preprocessor"


def test_target_predefines_flow_through():
    pp = Preprocessor(target_predefines={"__FOO__": "42", "__BAR__": "1"})
    assert pp.macros["__FOO__"].body == "42"
    assert pp.macros["__BAR__"].body == "1"


def test_backend_protocol_is_structural():
    """Any class with generate(TranslationUnit) -> str satisfies the Protocol."""

    class Stub:
        def generate(self, unit: ast.TranslationUnit) -> str:
            return ""

    assert isinstance(Stub(), CodeGenerator)


def test_macro_param_substitution_is_simultaneous():
    """C99 §6.10.3.1: parameter substitution happens simultaneously,
    not iteratively. When one arg's text contains a token that matches
    another param's name, that token must NOT be re-substituted.

    Regression test for a real-world case caught in MicroPython's
    `mp_seq_replace_slice_grow_inplace(... beg, end, slice, ...)`
    macro: callers pass `slice.start` for `beg` (referencing a local
    `mp_bound_slice_t slice;` variable). The body's `beg` substitutes
    to `slice.start` — the `slice` token in that result is the user's
    identifier and must stay put. Iterative substitution would
    rewrite it via the `slice` parameter, turning `slice.start` into
    whatever was passed for the `slice` param (e.g. `value_items`)
    and producing nonsense like `value_items.start`."""
    pp = Preprocessor()
    pp.preprocess(
        "#define M(beg, slice) beg + slice\n",
        "t.c",
    )
    # `slice.start` is the `beg` arg; `value_items` is the `slice` arg.
    # The `slice` token inside `slice.start` must NOT be rewritten.
    out = pp.preprocess("M(slice.start, value_items)\n", "t.c")
    assert "value_items.start" not in out, (
        f"iterative substitution leaked: {out!r}"
    )
    assert "slice.start + value_items" in out, (
        f"expected simultaneous substitution result, got: {out!r}"
    )


def test_macro_param_prefix_collision():
    """One parameter name being a prefix of another (`x` and `xy`)
    must not cause the shorter name to mis-match. Word-boundary
    regex semantics handle this — but only if the alternation
    doesn't short-circuit on the prefix. Pin it."""
    pp = Preprocessor()
    pp.preprocess("#define M(x, xy) x + xy\n", "t.c")
    out = pp.preprocess("M(1, 2)\n", "t.c")
    assert "1 + 2" in out, f"prefix-collision substitution broke: {out!r}"


def test_macro_param_repeated_substitution():
    """A param mentioned multiple times in the body must be
    substituted at every occurrence (re.sub's default replaces all
    matches, but since we changed the substitution mechanism this
    is worth pinning)."""
    pp = Preprocessor()
    pp.preprocess("#define M(p) p+p+p\n", "t.c")
    out = pp.preprocess("M(7)\n", "t.c")
    assert "7+7+7" in out, f"repeated substitution failed: {out!r}"
