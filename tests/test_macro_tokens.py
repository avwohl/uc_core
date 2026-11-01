"""A macro parameter is replaced only in an identifier token.

C23 6.10.5: a string literal or a character constant in a replacement
list is one preprocessing token, so a parameter name inside it stays.
Fixed cases check the text; a generated set compares the expansion with
the host `cc -E` (token by token, white space ignored).
"""
import random
import re
import shutil
import subprocess

import pytest

from uc_core.preprocessor import preprocess

TOK = re.compile(
    r"""(?:u8|u|U|L)?"(?:\\.|[^"\\\n])*"|(?:u8|u|U|L)?'(?:\\.|[^'\\\n])*'"""
    r"""|\.?[0-9](?:[eEpP][+-]|[A-Za-z0-9_.])*|[A-Za-z_]\w*|\#\#|\S""")


def toks(text):
    return TOK.findall(text)


def pp(src):
    return preprocess(src, "t.c").strip()


@pytest.mark.parametrize("src,want", [
    ('#define OUT(s) printf("%s %d\\n", s, 1)\nOUT("x")',
     'printf("%s %d\\n", "x", 1)'),
    ("#define C(s) 's' s\nC(7)", "'s' 7"),
    ('#define C(s) "a\\"s" s\nC(7)', '"a\\"s" 7'),
    ('#define C(s) u8"s" u"s" U"s" L"s" L\'s\' u\'s\' s\nC(7)',
     'u8"s" u"s" U"s" L"s" L\'s\' u\'s\' 7'),
    ('#define S(x) #x\nS("a\\n" \'b\'   c)', '"\\"a\\\\n\\" \'b\' c"'),
    ('#define L(s) s ## "t"\nL(L)', 'L"t"'),
    ('#define V(...) f(__VA_ARGS__ "__VA_ARGS__")\nV(1,2)',
     'f(1,2 "__VA_ARGS__")'),
    ('#define O(a,...) g(a __VA_OPT__(,) __VA_ARGS__ "__VA_OPT__(a)")\nO(1)\nO(1,2)',
     'g(1 "__VA_OPT__(a)")\ng(1 , 2 "__VA_OPT__(a)")'),
    ('#define O a ## b\nO', 'ab'),
    ('#define M(s) foo(s, \\\n   "s")\nM(1)', 'foo(1, "s")'),
    ('#define N(s) /* "s" */ s // s\nN(3) end', '3 end'),
    ('#define E(f, ...) p(f, ## __VA_ARGS__)\nE("a") E("a", 1)',
     'p("a") p("a",1)'),
    ('#define Q(A,B) A ## B+\n60 Q(+,)3', '60 + + 3'),
    ('#define NEG(x) -x\nNEG(-1) NEG(1)', '- -1 -1'),
    ('#define I(x) x++ + ++x\nI(a)', 'a++ + ++a'),
    ('#define Q(s) s\n#define P(x) Q("s" x)\nP(1)', '"s" 1'),
    # a macro expansion and the text after it are separate tokens
    ('#define T(t) t\nT(int)x=3;', 'int x=3;'),
    ('#define M(a) a\nM(a)x', 'a x'),
    ('#define ID(x) x\nint r = a ID(-)-1;', 'int r = a - -1;'),
    ('#define E\nx = +E+ -E-1;', 'x = + + - -1;'),
    ('#define E\nL E"x"', 'L "x"'),
    # an object-like macro that names itself expands once
    ('#define b (b+1)\nb', '(b+1)'),
    ('#define var A+var\nvar', 'A+var'),
    ('#define AA BB+1\n#define BB AA*2\nAA', 'AA*2+1'),
    ('#define BB AA*2\n#define AA BB+1\nBB', 'BB+1*2'),
    # white space in __VA_ARGS__ stays one space in a string
    ('#define SV(...) #__VA_ARGS__\nSV(1, 2, 3) SV(a ,  b,c) SV()',
     '"1, 2, 3" "a , b,c" ""'),
    ('#define SV(...) #__VA_ARGS__\nSV(The first, second, and third items.)',
     '"The first, second, and third items."'),
    # __VA_OPT__ looks at the argument after its expansion
    ('#define E\n#define O(a,...) a __VA_OPT__(X) b\nO(1,E) O(1,2) O(1)',
     '1 b 1 X b 1 b'),
])
def test_fixed(src, want):
    assert toks(pp(src)) == toks(want)


CC = shutil.which("cc")


def gen_case(rng, n, defined):
    """One macro definition and its use.  `defined` is a list of
    (name, definition text) of earlier macros that the body may call.
    Returns (definition text, call text, name)."""
    name = f"m{n}"
    variadic = rng.random() < 0.2
    nparam = rng.randint(0 if variadic else 1, 3)
    params = [f"p{n}_{i}" for i in range(nparam)]
    use_paste = rng.random() < 0.3
    allp = params + (["__VA_ARGS__"] if variadic else [])
    pieces = []
    for _ in range(rng.randint(1, 7)):
        r = rng.random()
        if allp and r < 0.35:
            pieces.append(rng.choice(allp))
        elif r < 0.5:
            pieces.append(rng.choice(["x", "foo", "1", "0x1f", "2u"]))
        elif r < 0.65:
            w = rng.choice(allp + ["q", "s"])
            pieces.append(rng.choice([
                '"%s"', '"a %s b"', '"\\"%s"', "'%s'", 'L"%s"', 'u8"%s"',
                "L'%s'", '"%s\\\\"']) % w)
        elif r < 0.72 and params:
            pieces.append("#" + rng.choice(params))
        elif r < 0.78 and variadic:
            pieces.append("__VA_OPT__(" + rng.choice(
                ["x", ",", "y " + rng.choice(allp)]) + ")")
        elif r < 0.85 and defined:
            d = rng.choice(defined)
            pieces.append(d[0] + "(" + (rng.choice(allp) if allp else "1") + ")")
        else:
            pieces.append("+")
    if use_paste and allp:
        pieces.insert(rng.randint(0, len(pieces)),
                      rng.choice(allp) + " ## " + rng.choice(allp + ["x", "7"]))
    sig = ", ".join(params + (["..."] if variadic else []))
    text = f"#define {name}({sig}) {' '.join(pieces)}\n"
    nargs = nparam + (rng.randint(0, 3) if variadic else 0)
    safe = ["x", "foo", "1", "0x2", "", "y"]
    rich = safe + ['"s"', "'s'", 'L"p"', "(a, b)", "x + y", '"a\\"b"']
    pool = safe if use_paste else rich
    args = [rng.choice(pool) for _ in range(nargs)]
    return text, f"{name}({', '.join(args)})", name


def make_cases(seed, count):
    """Self-contained sources: the macros a case calls come first."""
    rng = random.Random(seed)
    defined = []
    cases = []
    for n in range(count):
        deps = defined[-3:]
        text, call, name = gen_case(rng, n, deps)
        # a body may call any of `deps` only, so include just those
        # (and, in turn, what they call: keep all earlier text, simple)
        cases.append("".join(d[1] for d in defined) + text + "@@\n" + call + "\n")
        defined.append((name, text))
    return cases


def expansion(text):
    return toks(text.split("@@", 1)[1])


@pytest.mark.skipif(CC is None, reason="no host cc")
@pytest.mark.parametrize("seed", range(4))
def test_against_host_cc(seed, tmp_path):
    ok = 0
    f = tmp_path / "t.c"
    for src in make_cases(seed, 50):
        # an invalid paste or a bad call is rejected by the host: skip
        f.write_text(src)
        r = subprocess.run([CC, "-E", "-P", "-std=c2x", "-x", "c", str(f)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            continue
        want = expansion(r.stdout)
        got = expansion(pp(src))
        assert got == want, src
        ok += 1
    assert ok >= 25


@pytest.mark.parametrize("src,want", [
    # A self-named object-like macro that passes through an argument
    # expands once (C23 6.10.5.4 example with z[0]).
    ("#define b (b+1)\n#define f(x) x\nf(b)", "(b+1)"),
    ("#define z z[0]\n#define f(x) x\nf(z)", "z[0]"),
    ("#define x 2\n#define f(a) f(x * (a))\n#define z z[0]\nf(z)",
     "f(2 * (z[0]))"),
    # A function-like macro reached through an object-like alias reads
    # its arguments from the source text, so a comma in the expansion
    # of an argument stays inside the argument.
    ("#define w 0,1\n#define f(a) a\n#define g f\ng(w)", "0,1"),
    ("#define w 0,1\n#define f(a) a\n#define g f\ng(1-w)", "1-0,1"),
    ("#define w 0,1\n#define f(a) a\n#define g f\nf(g(w))", "0,1"),
    ("#define x 2\n#define w 0,1\n#define f(a) f(x * (a))\n#define g f\n"
     "g(2+(3,4)-w)", "f(2 * (2+(3,4)-0,1))"),
])
def test_self_name_and_alias_call(src, want):
    assert toks(pp(src)) == toks(want)


def test_latin1_source_file(tmp_path):
    from uc_core.preprocessor import preprocess_file
    p = tmp_path / "l.c"
    p.write_bytes(b"/* caf\xe9 */\nint a;\n")
    assert "int a;" in preprocess_file(str(p))
