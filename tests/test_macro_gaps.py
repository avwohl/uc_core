"""Macro expander cases that the host `cc -E` settles: the next token of
a function-like macro name, an object-like alias of one, `#__VA_OPT__`,
`__COUNTER__`, and the `#pragma` lines of -E output.

Each case compares the token list with the host `cc -E` when a host cc is
present; the fixed text is the check when it is not."""
import re
import shutil
import subprocess

import pytest

from uc_core.preprocessor import Preprocessor, preprocess

TOK = re.compile(
    r"""(?:u8|u|U|L)?"(?:\\.|[^"\\\n])*"|(?:u8|u|U|L)?'(?:\\.|[^'\\\n])*'"""
    r"""|\.?[0-9](?:[eEpP][+-]|[A-Za-z0-9_.])*|[A-Za-z_]\w*|\#\#|\S""")

HOST = shutil.which("cc")


def toks(text):
    return TOK.findall(text)


def host_toks(src, tmp_path, std="c2x"):
    f = tmp_path / "h.c"
    f.write_text(src)
    r = subprocess.run([HOST, "-E", "-P", f"-std={std}", str(f)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return toks(r.stdout)


CASES = [
    # (source, expected tokens as text)
    ("#define EMPTY\n#define f(x) x\nf EMPTY (1)", "f (1)"),
    ("#define LP (\n#define f(x) x\nf LP 1 )", "f ( 1 )"),
    ("#define EMPTY\n#define f(x) x\nf EMPTY EMPTY (1) f (2)", "f (1) 2"),
    ("#define f(x) <x>\n#define obj f\nobj\n(3)", "<3>"),
    ("#define f(x) <x>\n#define obj f\n#define obj2 obj\nobj2\n(3)", "<3>"),
    ("#define f(x) <x>\n#define E\n#define obj f\nobj E (4)", "f (4)"),
    ("#define f(x) <x>\n#define ID(x) x\nID(f)(1)", "<1>"),
    ("#define f(x) <x>\n#define E\n#define ID(x) x\nID(f E)(1)", "<1>"),
    ("#define f(x) <x>\n#define E\n#define ID(x) x E\nID(f)(1)", "f (1)"),
    ("#define EMPTY()\n#define DEFER(id) id EMPTY()\n#define A() 123\n"
     "#define EXPAND(...) __VA_ARGS__\nDEFER(A)() EXPAND(DEFER(A)())",
     "A () 123"),
    ("#define F(...) f(0 __VA_OPT__(,) __VA_ARGS__)\nF(a,b,c) F() F(a)",
     "f(0, a,b,c) f(0) f(0, a)"),
    ('#define H3(X, ...) #__VA_OPT__(X##X X##X)\nH3(, 0)', '""'),
    ('#define S1(x, ...) #__VA_OPT__(x, __VA_ARGS__)\nS1(a, b, c) S1(a)',
     '"a, b, c" ""'),
    ('#define S2(...) #__VA_OPT__(  p   q  ) "|" #__VA_ARGS__\nS2(1) S2()',
     '"p q" "|" "1" "" "|" ""'),
    ('#define S3(a, ...) #__VA_OPT__(a "x\\n")\nS3(1, 2)',
     '"1 \\"x\\\\n\\""'),
    ("__COUNTER__ __COUNTER__ __COUNTER__", "0 1 2"),
    ("#define S(x) #x\n#define C(a,b) a##b\n#define T(x) x x\n"
     "S(__COUNTER__) __COUNTER__ C(n,__COUNTER__) T(__COUNTER__)",
     '"__COUNTER__" 0 n__COUNTER__ 1 1'),
    ("#define NEXT __COUNTER__\nNEXT NEXT\n#ifdef __COUNTER__\nyes\n#endif",
     "0 1 yes"),
]


@pytest.mark.parametrize("src,want", CASES)
def test_fixed_text(src, want):
    assert toks(preprocess(src, "t.c")) == toks(want)


@pytest.mark.skipif(HOST is None, reason="no host cc")
@pytest.mark.parametrize("src,want", CASES)
def test_same_tokens_as_host(src, want, tmp_path):
    assert toks(preprocess(src, "t.c")) == host_toks(src, tmp_path)


def test_c23_va_opt_examples():
    src = """#define F(...) f(0 __VA_OPT__(,) __VA_ARGS__)
#define G(X, ...) f(0, X __VA_OPT__(,) __VA_ARGS__)
#define SDEF(sname, ...) S sname __VA_OPT__(= { __VA_ARGS__ })
#define EMP
F(a,b,c)
F()
F(EMP)
G(a,b,c)
G(a,)
G(a)
SDEF(foo);
SDEF(bar, 1, 2);
#define H2(X, Y, ...) __VA_OPT__(X ## Y,) __VA_ARGS__
H2(a, b, c, d)
#define H3(X, ...) #__VA_OPT__(X##X X##X)
H3(, 0)
#define H4(X, ...) __VA_OPT__(a X ## X) ## b
H4(, 1)
#define H5A(...) __VA_OPT__()/**/__VA_OPT__()
#define H5B(X) a ## X ## b
#define H5C(X) H5B(X)
H5C(H5A())
"""
    got = [toks(ln) for ln in preprocess(src, "t.c").split("\n") if ln.strip()]
    assert got == [toks(x) for x in [
        "f(0, a, b, c)", "f(0)", "f(0)", "f(0, a, b, c)", "f(0, a)",
        "f(0, a)", "S foo;", "S bar = { 1, 2 };", "ab, c, d", '""', "a b",
        "ab"]]


def test_counter_is_per_preprocessor():
    assert toks(preprocess("__COUNTER__", "t.c")) == ["0"]
    assert toks(preprocess("__COUNTER__", "t.c")) == ["0"]


def _e_text(src, keep):
    pp = Preprocessor()
    pp.keep_pragmas = keep
    return pp.preprocess(src, "t.c")


PRAGMA_SRC = """#pragma pack(push, 1)
#pragma once
#define P(x) _Pragma(#x) int y_##x;
int a; _Pragma("omp parallel") int b;
P(foo)
const char *s = "_Pragma(\\"no\\")";
_Pragma("say \\"hi\\" \\\\ x")
#if 0
#pragma inactive
#endif
#pragma uc80 keep f
"""


def test_pragma_lines_stay_out_of_the_compile_text():
    out = _e_text(PRAGMA_SRC, False)
    assert "#pragma" not in out
    assert "_Pragma" in out          # the compiler gets the text as before


def test_pragma_lines_in_e_text():
    lines = [ln.strip() for ln in _e_text(PRAGMA_SRC, True).split("\n")
             if ln.strip()]
    assert lines == [
        "#pragma pack(push, 1)",
        "int a;",
        "#pragma omp parallel",
        "int b;",
        "#pragma foo",
        "int y_foo;",
        'const char *s = "_Pragma(\\"no\\")";',
        '#pragma say "hi" \\ x',
        "#pragma uc80 keep f",
    ]


def test_pragma_recording_does_not_change_with_keep_pragmas():
    for keep in (False, True):
        pp = Preprocessor()
        pp.keep_pragmas = keep
        pp.preprocess("#pragma uc80 keep f\n#pragma printf long\n", "t.c")
        assert pp.target_pragmas == [("keep", ["f"], "t.c", 1)]
        assert pp.printf_features == {"long"}


@pytest.mark.skipif(HOST is None, reason="no host cc")
def test_e_text_pragma_lines_as_host(tmp_path):
    src = PRAGMA_SRC.replace("#pragma uc80 keep f", "#pragma weak foo")
    f = tmp_path / "h.c"
    f.write_text(src)
    r = subprocess.run([HOST, "-E", "-P", "-std=c2x", "-Wno-everything",
                        str(f)], capture_output=True, text=True)
    want = [ln.strip() for ln in r.stdout.split("\n") if ln.strip()]
    got = [ln.strip() for ln in _e_text(src, True).split("\n")
           if ln.strip()]
    assert [toks(x) for x in got] == [toks(x) for x in want]


def test_true_is_one_in_an_if_expression():
    src = "#if true\nA\n#endif\n#if false\nB\n#endif\n#if !true\nC\n#endif\n"
    assert toks(preprocess(src, "t.c")) == ["A"]
    src2 = "#define true 0\n#if true\nA\n#else\nB\n#endif\n"
    assert toks(preprocess(src2, "t.c")) == ["B"]
