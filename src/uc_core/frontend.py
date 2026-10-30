"""uplox-driven C23 front-end for uc_core.

Phase 6: returns the typed auto-AST directly from
``uc_core.c23_parser.parse``. No more CST + manual lowering — the
grammar's ``%ast=`` annotations build the typed tree in the LR driver.

The per-compiler preprocessor (uc_core, uc80, uc386 each instantiate
their own ``Preprocessor`` with target-specific predefines) still runs
upstream.

* ``_strip_attributes`` — strip C23 ``[[...]]`` and GCC
  ``__attribute__((...))`` from the source text. The c23 grammar
  doesn't model attributes.
* ``c23_parser.parse`` — uplox-runtime LR + auto-AST builder.
  Returns a :class:`uc_core.ast.TranslationUnit` whose ``items`` list
  is composed of ``FunctionDef`` / ``Declaration`` / ``StaticAssert`` /
  ``AsmDeclaration`` / ``EmptyDecl`` nodes per the grammar.
* :class:`TypedefTracker` runs as a token filter / declaration hook so
  identifier references to typedef'd names lex as ``TYPEDEF_NAME`` in
  cast / type-spec position.

The JSON bundle at ``uc_core/data/c23.json`` is the source for the
embedded scanner + LR tables; ``c23_parser.py`` is regenerated from it
via ``uplox emit-python``.
"""

from __future__ import annotations

import re
import sys

from . import ast
from . import c23_parser
from .c23_parser import HookRegistry, K, Token


# ---- C23 attributes are stripped at the source level -----------------------


_ATTRIBUTE_RE = re.compile(
    r"__attribute(?:__)?\s*\(\(\s*"
    r"[^()]*"
    r"(?:\([^()]*\)[^()]*)*"
    r"\)\)"
)

_C23_ATTR_RE = re.compile(r"\[\[[^\]]*\]\]")

# Borland / Watcom / Microsoft DOS-era qualifier keywords that the
# uplox c23 grammar doesn't model. Period code (Z80 / 8086 / 286 /
# 386 headers) sprays them everywhere; they have no meaning under
# flat-32 / Z80 so we erase them at the source level. See
# tests/test_dos_tolerance.py for the canonical case list.
# The unprefixed spellings are ordinary identifiers in C -- C23 has no such
# keywords -- so erasing one wherever it appears takes the user's name with
# it.  The lookahead below keeps most uses safe (a name followed by ``;``,
# ``=`` or ``(`` is never a qualifier), but it cannot tell ``struct huge mk``
# from ``near int n``: both are the word followed by another word.  Erasing
# the first leaves ``struct mk(void)``, which does not parse, and the error
# points at the line rather than the name.  A tag position is matched first
# and put back verbatim.
_DOS_BARE_QUALIFIERS = (
    "near|far|huge|cdecl|pascal|stdcall|fastcall|syscall|watcall"
    "|fortran|interrupt"
)

_DOS_QUALIFIER_RE = re.compile(
    r"(?P<keep>\b(?:struct|union|enum)\s+(?:" + _DOS_BARE_QUALIFIERS + r")\b)"
    r"|"
    r"\b(?:"
    r"(?:__|_)(?:near|far|huge)"
    r"|(?:__|_)(?:cs|ds|es|ss|seg)"
    r"|(?:__|_)(?:cdecl|pascal|stdcall|fastcall|syscall|watcall|fortran)"
    r"|(?:__|_)(?:interrupt|loadds|saveregs|export)"
    r"|__extension__"
    r")\b"
    r"|"
    r"\b(?:" + _DOS_BARE_QUALIFIERS + r")\b(?=\s+[A-Za-z_*])"
)


def _drop_dos_qualifier(m: "re.Match") -> str:
    """Erase the qualifier, except where it was a struct/union/enum tag."""
    return m.group("keep") or ""

# ``__based(seg)`` / ``__Seg16(...)`` — keyword + parenthesised
# argument list to discard together. Caller counts parens by hand
# since the body may nest.
_DOS_PAREN_QUALIFIER_RE = re.compile(
    r"\b(?:__|_)?(?:based|Seg16)\s*\("
)


_BUILTIN_ELEM_SIZES = {
    "char": 1, "signed char": 1, "unsigned char": 1,
    "short": 2, "unsigned short": 2, "signed short": 2,
    "int": 4, "unsigned int": 4, "signed int": 4, "unsigned": 4, "signed": 4,
    "long": 4, "unsigned long": 4, "signed long": 4,
    "long long": 8, "unsigned long long": 8, "signed long long": 8,
    "float": 4, "double": 8, "long double": 8,
}

_VECTOR_SIZE_RE = re.compile(
    r"\b((?:typedef\s+)?"
    r"(?:(?:unsigned|signed)\s+)?"
    r"(?:char|short|int|long(?:\s+long)?|float|double))"
    r"\s+([A-Za-z_]\w*)"
    r"\s*__attribute(?:__)?\s*\(\s*\(\s*(?:__)?vector_size(?:__)?"
    r"\s*\(\s*(\d+)\s*\)\s*\)\s*\)"
)


# `__attribute__((packed))` on struct definitions. The grammar
# doesn't model attributes, so we strip them at the source level;
# capture the affected names beforehand and re-attach `is_packed`
# to the parsed AST.
#
# Non-greedy body `\{[^{}]*\}` doesn't handle nested struct
# definitions — lwIP's packed structs are flat, which is the only
# real consumer today. Extend if a nested packed struct shows up.
_PACKED_NAMED_STRUCT_RE = re.compile(
    r"\bstruct\s+([A-Za-z_]\w*)\s*\{[^{}]*\}\s*"
    r"__attribute(?:__)?\s*\(\s*\(\s*(?:__)?packed(?:__)?\s*\)\s*\)"
)
_PACKED_TYPEDEF_STRUCT_RE = re.compile(
    r"\btypedef\s+struct\s*(?:[A-Za-z_]\w*\s*)?\{[^{}]*\}\s*"
    r"__attribute(?:__)?\s*\(\s*\(\s*(?:__)?packed(?:__)?\s*\)\s*\)\s*"
    r"([A-Za-z_]\w*)"
)


def _rewrite_vector_size(source: str, vector_names: set | None = None) -> str:
    """Rewrite ``T name __attribute__((vector_size(N)))`` into
    ``T name[N/sizeof(T)]`` so the resulting array shape is preserved
    after attribute stripping. uc386 codegen reads the array storage
    layout from the resulting ArrayType; if ``vector_names`` is
    supplied, names that get rewritten are added to it so downstream
    consumers can flip the GCC-specific ``is_vector`` flag on the
    resolved type."""
    def repl(m: re.Match) -> str:
        type_part = m.group(1)
        name = m.group(2)
        n_bytes = int(m.group(3))
        elem_type = type_part
        if elem_type.startswith("typedef "):
            elem_type = elem_type[len("typedef ") :]
        elem_type = " ".join(elem_type.split())
        elem_size = _BUILTIN_ELEM_SIZES.get(elem_type, 4)
        count = n_bytes // elem_size if elem_size else 0
        if count <= 0:
            return f"{type_part} {name}"
        if vector_names is not None:
            vector_names.add(name)
        return f"{type_part} {name}[{count}]"

    return _VECTOR_SIZE_RE.sub(repl, source)


# ---- strict mode: refuse to erase constructs the target cannot honour -------
#
# Tolerant mode (the default, used by uc386) erases ``interrupt`` and
# ``fastcall`` markers because they mean nothing on flat-32.  A target
# for which they DO mean something (a Z80 ROM: interrupt handlers need
# RETI/EI and register saves) must not lose them silently.  With
# ``strict_unsupported=True`` the scan below raises
# :class:`UnsupportedConstructError` instead.

_DEFAULT_UNSUPPORTED_MESSAGE = "'{name}' is not supported"

_STRICT_ATTR_WORD_RE = re.compile(r"\b(?:__)?interrupt(?:__)?\b")
_STRICT_TOKEN_RE = re.compile(
    r"(?P<skip>\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*')"
    r"|(?P<gnu>__attribute(?:__)?\s*\(\(\s*[^()]*(?:\([^()]*\)[^()]*)*\)\))"
    r"|(?P<c23>\[\[[^\]]*\]\])"
    r"|(?P<keep>\b(?:struct|union|enum)\s+(?:" + _DOS_BARE_QUALIFIERS + r")\b)"
    r"|(?P<prefixed>\b(?:__|_)(?:interrupt|fastcall)\b)"
    r"|(?P<bare>\b(?:interrupt|fastcall)\b(?=\s+[A-Za-z_*]))"
)


class UnsupportedConstructError(Exception):
    """A construct tolerant mode would erase is refused in strict mode.

    ``str(exc)`` is ``file:line: message``.  ``line`` counts lines of the
    preprocessed text handed to :func:`parse` (the preprocessor emits no
    line markers), which equals the source line when no ``#include`` or
    multi-line macro comes before the construct.
    """

    def __init__(self, filename: str, line: int, name: str, message: str):
        self.filename = filename
        self.line = line
        self.name = name
        self.message = message
        super().__init__(f"{filename}:{line}: {message}")


def _check_strict_unsupported(source: str, filename: str, template: str,
                              allow: frozenset = frozenset()) -> None:
    """Raise UnsupportedConstructError for the first interrupt / fastcall
    marker that tolerant mode would erase.  Identifiers that merely
    contain the words (interrupt_count) and struct/union/enum tags are
    not markers.  String and character literals are skipped.

    ``allow`` holds kinds ("interrupt", "fastcall") a host handles
    itself; those markers are skipped here and erased later as in
    tolerant mode.  A bare ``interrupt``/``fastcall`` before ``*`` counts
    as a marker only at the start of a declaration (start of text or
    after ``;``, ``{``, ``}``); elsewhere ``*`` is the multiply operator.
    Known limit: the statement ``interrupt * x;`` is still taken as a
    declaration."""
    for m in _STRICT_TOKEN_RE.finditer(source):
        kind = m.lastgroup
        name = None
        if kind in ("gnu", "c23"):
            if _STRICT_ATTR_WORD_RE.search(m.group(0)):
                name = ("__attribute__((interrupt))" if kind == "gnu"
                        else "[[interrupt]]")
        elif kind in ("prefixed", "bare"):
            name = m.group(0)
            if kind == "bare":
                rest = source[m.end():].lstrip()
                if rest.startswith("*"):
                    prev = source[:m.start()].rstrip()[-1:]
                    if prev not in ("", ";", "{", "}"):
                        continue
        if name is None:
            continue
        akind = "interrupt" if "interrupt" in name else "fastcall"
        if akind in allow:
            continue
        line = source.count("\n", 0, m.start()) + 1
        raise UnsupportedConstructError(
            filename, line, name, template.format(name=name))


def _strip_attributes(source: str, vector_names: set | None = None) -> str:
    """Strip GCC ``__attribute__`` and C23 ``[[...]]`` attributes, plus
    period DOS-era qualifier keywords (near/far/huge/__cdecl/__pascal/
    etc.). None of these are modelled by the c23 grammar; we erase
    them at the source level so the uplox parser sees clean C23.

    ``__attribute__((vector_size(N)))`` is rewritten first into an
    equivalent array shape so the storage layout survives the strip;
    names rewritten this way are recorded in ``vector_names`` (if
    supplied) so consumers can flag them later.
    """
    source = _rewrite_vector_size(source, vector_names)
    # __attribute__((...)) — fixed-point loop to handle nested parens
    # since a single regex pass leaves outer ))'s visible.
    prev = None
    while prev != source:
        prev = source
        source = _ATTRIBUTE_RE.sub(" ", source)
    source = _C23_ATTR_RE.sub(" ", source)
    source = _strip_dos_paren_qualifiers(source)
    source = _DOS_QUALIFIER_RE.sub(_drop_dos_qualifier, source)
    return source


def _strip_dos_paren_qualifiers(source: str) -> str:
    """Erase ``__based(seg)`` / ``__Seg16(...)`` style qualifiers,
    keyword and argument list together. The argument list nests
    parens, so count them by hand."""
    out: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        m = _DOS_PAREN_QUALIFIER_RE.match(source, i)
        if not m:
            out.append(source[i])
            i += 1
            continue
        depth = 1
        j = m.end()
        while j < n and depth > 0:
            c = source[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        i = j
    return "".join(out)


# ---- K&R / implicit-int pre-pass (opt-in) ----------------------------------
#
# The uplox c23 grammar models only ANSI/C23 function definitions: every
# external declaration must open with a declaration-specifier. Pre-ANSI
# sources — pervasive in the gcc-c-torture corpus — use implicit-int
# returns (`main() { }`) and K&R old-style parameter lists
# (`foo(a, b) int a; char *b; { }`). Rather than grow the LR grammar
# (conflict-prone, balloons the state table), we rewrite those two
# shapes into equivalent ANSI at the source level — but ONLY when the
# host explicitly asks (`kr_prepass=True`). Modern code never reaches
# this scanner, so it pays exactly zero cost.

_KR_TYPE_SPEC = frozenset({
    "void", "char", "short", "int", "long", "float", "double",
    "signed", "unsigned", "_Bool", "bool", "_Complex", "_Imaginary",
    "struct", "union", "enum", "_BitInt", "__int128",
    "_Decimal32", "_Decimal64", "_Decimal128",
    "typeof", "typeof_unqual", "__typeof", "__typeof__",
})
# Specifier keywords that may precede the return type / function name
# without themselves being a type (so `static foo() {}` is still
# implicit-int).
_KR_NONTYPE_SPEC = frozenset({
    "static", "extern", "inline", "register", "auto", "const",
    "volatile", "restrict", "_Noreturn", "_Thread_local", "_Atomic",
    "__inline", "__inline__", "__forceinline",
})
_KR_IDENT = re.compile(r"[A-Za-z_]\w*")


def _kr_tokens(s: str) -> list[tuple[str, int, int]]:
    """Lightweight tokenizer: (text, start, end), skipping whitespace,
    string/char literals and any stray comments. Identifiers and numbers
    come back whole; every other non-space char is a 1-char punctuation
    token. Good enough to find function-definition heads at file scope."""
    out: list[tuple[str, int, int]] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t\r\n\f\v":
            i += 1
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            i = s.find("\n", i)
            if i < 0:
                break
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "*":
            j = s.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c == '"' or c == "'":
            j = i + 1
            while j < n and s[j] != c:
                j += 2 if s[j] == "\\" else 1
            out.append((s[i:j + 1], i, j + 1))
            i = j + 1
            continue
        m = _KR_IDENT.match(s, i)
        if m:
            out.append((m.group(0), m.start(), m.end()))
            i = m.end()
            continue
        out.append((c, i, i + 1))
        i += 1
    return out


def _kr_ansi_params(param_ids: list[str], kr_decls: str) -> str:
    """Build an ANSI parameter list from a K&R identifier list plus the
    declaration block between `)` and `{`. Params with no K&R
    declaration default to `int` (the K&R rule)."""
    types: dict[str, str] = {}
    for decl in kr_decls.split(";"):
        decl = decl.strip()
        if not decl:
            continue
        toks = [t[0] for t in _kr_tokens(decl)]
        if not toks:
            continue
        # Comma-separated declarators sharing one base type. The base
        # type is whatever precedes the *name* in the first declarator
        # group (the name = the last identifier in that group, so
        # `struct foo b` / `unsigned long n` / `Point p0` all work).
        groups: list[list[str]] = []
        cur: list[str] = []
        for t in toks:
            if t == ",":
                groups.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur:
            groups.append(cur)
        if not groups:
            continue

        def _name_idx(g: list[str]) -> int:
            for idx in range(len(g) - 1, -1, -1):
                if _KR_IDENT.fullmatch(g[idx]) and g[idx] not in _KR_TYPE_SPEC:
                    return idx
            return -1

        ni = _name_idx(groups[0])
        if ni < 0:
            continue
        base_t = " ".join(x for x in groups[0][:ni] if x != "*") or "int"
        for g in groups:
            gi = _name_idx(g)
            if gi < 0:
                continue
            name = g[gi]
            stars = "".join(x for x in g if x == "*")
            arr = "[]" if "[" in g else ""
            types[name] = f"{base_t} {stars}{name}{arr}".strip()
    parts = [types.get(pid, f"int {pid}") for pid in param_ids]
    return ", ".join(parts) if parts else "void"


def _rewrite_kr_and_implicit_int(source: str) -> str:
    """Rewrite pre-ANSI function definitions into ANSI C at the source
    level. Two shapes handled, at file scope only:

      * implicit-int return:  `main() {`        -> `int main() {`
      * K&R parameter list:    `f(a, b) int a; double b; {`
                               -> `int f(int a, double b) {`

    A function whose return type is explicit but whose parameters are
    K&R-style bare identifiers gets only the parameter rewrite. Function
    prototypes/declarations with implicit int (`foo();`) get the `int`
    prefix but no parameter surgery. Conservative: anything that doesn't
    cleanly match is left untouched (max-recovery, no regressions on the
    shapes we already parse)."""
    toks = _kr_tokens(source)
    edits: list[tuple[int, str, str]] = []
    depth = 0
    i = 0
    # External-declaration boundary: start of file, or just after a
    # top-level `;` / `}` (depth back to 0).
    at_boundary = True
    while i < len(toks):
        text = toks[i][0]
        if text == "{":
            depth += 1
            at_boundary = False
            i += 1
            continue
        if text == "}":
            depth -= 1
            at_boundary = depth == 0
            i += 1
            continue
        if text == ";":
            at_boundary = depth == 0
            i += 1
            continue
        if depth != 0 or not at_boundary:
            i += 1
            continue
        # At a file-scope boundary: try to match a function head.
        # Consume a specifier prefix (storage/qual/type keywords, *,
        # and typedef/return-type identifiers) until `IDENT (`.
        j = i
        has_typespec = False
        saw_ident = False
        while j < len(toks):
            t = toks[j][0]
            nxt = toks[j + 1][0] if j + 1 < len(toks) else ""
            if _KR_IDENT.fullmatch(t):
                if t in _KR_TYPE_SPEC:
                    has_typespec = True
                    j += 1
                    continue
                if t in _KR_NONTYPE_SPEC:
                    j += 1
                    continue
                # plain identifier: function name iff next is '('
                if nxt == "(":
                    break
                # else it's a return-type typedef-name (e.g. `Foo bar(`)
                has_typespec = True
                saw_ident = True
                j += 1
                continue
            if t == "*":
                has_typespec = True  # `T *foo(` — return type present
                j += 1
                continue
            break
        # Need `NAME (` here.
        if j + 1 >= len(toks) or not _KR_IDENT.fullmatch(toks[j][0]) \
                or toks[j][0] in _KR_TYPE_SPEC or toks[j + 1][0] != "(":
            at_boundary = False
            i += 1
            continue
        name_idx = j
        # Match the parenthesised parameter list.
        pdepth = 0
        k = j + 1
        while k < len(toks):
            if toks[k][0] == "(":
                pdepth += 1
            elif toks[k][0] == ")":
                pdepth -= 1
                if pdepth == 0:
                    break
            k += 1
        if k >= len(toks) or pdepth != 0:
            at_boundary = False
            i += 1
            continue
        rparen_idx = k
        param_toks = toks[j + 2:rparen_idx]
        # Classify by the token right after `)`:
        #   `{`  -> definition, no K&R decl block
        #   `;`  -> prototype/declaration
        #   else -> K&R declaration block; the body `{` is the first
        #           brace after it (K&R param decls don't carry braces
        #           in the torture corpus).
        nxt_after = toks[rparen_idx + 1][0] if rparen_idx + 1 < len(toks) else ""
        if nxt_after == "{":
            m, term = rparen_idx + 1, "{"
        elif nxt_after == ";":
            m, term = rparen_idx + 1, ";"
        else:
            m = rparen_idx + 1
            while m < len(toks) and toks[m][0] not in ("{", "}"):
                m += 1
            term = "{" if m < len(toks) and toks[m][0] == "{" else ""
        if not term:
            at_boundary = False
            i += 1
            continue
        # Is the param list K&R (every entry a bare identifier)?
        ids: list[str] = []
        is_kr_params = len(param_toks) > 0
        grp: list[str] = []
        seq = [t[0] for t in param_toks] + [","]
        for x in seq:
            if x == ",":
                if len(grp) == 1 and _KR_IDENT.fullmatch(grp[0]) \
                        and grp[0] not in _KR_TYPE_SPEC and grp[0] != "void":
                    ids.append(grp[0])
                else:
                    is_kr_params = False
                grp = []
            else:
                grp.append(x)
        name_start = toks[name_idx][1]
        if term == ";":
            # prototype: only fix implicit int
            if not has_typespec:
                edits.append((name_start, name_start, "int "))
        else:
            # definition
            if is_kr_params:
                kr_decls = source[toks[rparen_idx][2]:toks[m][1]]
                ansi = _kr_ansi_params(ids, kr_decls)
                edits.append((toks[j + 1][1], toks[m][1],
                              f"({ansi}) "))
            if not has_typespec:
                edits.append((name_start, name_start, "int "))
        # Resume scanning after the function name; body braces handled
        # by the normal depth tracking.
        at_boundary = False
        i = name_idx + 1

    if not edits:
        return source
    edits.sort(key=lambda e: e[0])
    out: list[str] = []
    pos = 0
    for st, en, rep in edits:
        if st < pos:          # overlapping edit guard — skip
            continue
        out.append(source[pos:st])
        out.append(rep)
        pos = en
    out.append(source[pos:])
    return "".join(out)


# ---- GNU computed-goto / labels-as-values (opt-in) -------------------------
#
# GCC's "labels as values" extension (`void *p = &&lab; goto *p;`) is not
# C23 and the strict grammar rejects `&&label` / `goto *expr`. The
# gcc-c-torture corpus uses it for jump tables and threaded code, where
# label addresses are only ever combined with each other (stored in
# `void*[]`, subtracted as `&&a - &&b`, added back to a `&&base`). That
# makes a consistent integer-id encoding semantics-preserving: give each
# address-taken label (per function) a small id, lower `&&L` to
# `((void*)id)` and `goto *E` to `switch ((long)(E)) { case id: goto L; }`
# over every address-taken label in that function. Affine arithmetic
# stays correct because it is all integer math in id-space, and the
# result uses only switch/goto/cast — all of which codegen handles.
# Source-level only: no grammar change, no LR-table regen.

# Token texts after which `&&IDENT` is the logical-AND operator (its
# left operand just ended) rather than the address-of-label prefix.
_CG_OPERAND_END = frozenset({")", "]"})


def _rewrite_computed_goto(source: str) -> str:
    toks = _kr_tokens(source)
    n = len(toks)

    class _Scope:
        __slots__ = ("refs", "gotos", "names")

        def __init__(self) -> None:
            self.refs: list[tuple[int, int, str]] = []   # (start, end, name)
            self.gotos: list[tuple[int, int, str]] = []   # (start, end, exprtxt)
            self.names: set[str] = set()

    scopes: list[_Scope] = []
    fn_stack: list[_Scope] = []
    depth = 0
    k = 0
    while k < n:
        t, st, en = toks[k]
        if t == "{":
            depth += 1
            if depth == 1:
                sc = _Scope()
                scopes.append(sc)
                fn_stack.append(sc)
            k += 1
            continue
        if t == "}":
            if depth == 1 and fn_stack:
                fn_stack.pop()
            depth = max(0, depth - 1)
            k += 1
            continue
        if depth == 0 or not fn_stack:
            k += 1
            continue
        sc = fn_stack[-1]
        # `&& IDENT` as label-address (not logical-AND).
        if (
            t == "&" and k + 2 < n and toks[k + 1][0] == "&"
            and _KR_IDENT.fullmatch(toks[k + 2][0])
            and toks[k + 2][0] not in _KR_TYPE_SPEC
        ):
            prev = toks[k - 1][0] if k > 0 else ""
            prev_is_operand = (
                prev in _CG_OPERAND_END
                or _KR_IDENT.fullmatch(prev) and prev not in (
                    "return", "case", "goto"
                )
                or (prev[:1].isdigit())
                or prev[:1] in ('"', "'")
            )
            if not prev_is_operand:
                name = toks[k + 2][0]
                sc.refs.append((st, toks[k + 2][2], name))
                sc.names.add(name)
                k += 3
                continue
        # `goto * EXPR ;`
        if t == "goto" and k + 1 < n and toks[k + 1][0] == "*":
            j = k + 2
            pd = 0
            while j < n:
                jt = toks[j][0]
                if jt in "([":
                    pd += 1
                elif jt in ")]":
                    pd -= 1
                elif jt == ";" and pd == 0:
                    break
                j += 1
            if j < n:
                expr_txt = source[toks[k + 1][2]:toks[j][1]]
                # Labels referenced *only* inside the dispatch
                # expression (`goto *(&&base + n)`) are skipped by this
                # jump; harvest them so they get ids + a case arm.
                for mm in re.finditer(r"&&\s*([A-Za-z_]\w*)", expr_txt):
                    sc.names.add(mm.group(1))
                sc.gotos.append((st, toks[j][2], expr_txt))
                k = j + 1
                continue
        k += 1

    edits: list[tuple[int, str, str]] = []
    for sc in scopes:
        if not sc.names:
            continue
        ids = {nm: i for i, nm in enumerate(sorted(sc.names))}
        # `&&L` is rewritten whenever it appears — even if the matching
        # `goto *` lives in another function (the address is commonly
        # passed out / stored). The dispatch switch is only emitted
        # where a `goto *` actually occurs.
        for st, en, name in sc.refs:
            edits.append((st, en, f"((void*){ids[name]})"))
        if not sc.gotos:
            continue
        cases = "".join(
            f"case {ids[nm]}: goto {nm};" for nm in sorted(sc.names)
        )
        for st, en, expr in sc.gotos:
            # `&&L` inside the dispatch expression resolves in the same
            # id-space; substitute there too so the outer edit owns the
            # whole statement (overlapping inner ref edits are skipped).
            for nm, i in ids.items():
                expr = re.sub(rf"&&\s*{nm}\b", f"((void*){i})", expr)
            edits.append(
                (st, en,
                 f"switch((long)({expr})){{{cases}default:;}}")
            )

    if not edits:
        return source
    edits.sort(key=lambda e: e[0])
    out: list[str] = []
    pos = 0
    for st, en, rep in edits:
        if st < pos:          # inside an already-applied (goto) edit
            continue
        out.append(source[pos:st])
        out.append(rep)
        pos = en
    out.append(source[pos:])
    return "".join(out)


_TYPEOF_RE = re.compile(r"\b(?:__typeof__|__typeof|typeof|typeof_unqual)\b")


def _typeof_typename_fix(src: str, tok) -> str | None:
    """Rewrite ``typeof(type-name)`` into ``typeof(*(type-name *)0)``.

    ``tok`` is the token a parse failed at.  It must be the first token
    after ``typeof (``; any other failure returns None.  An abstract
    declarator with an array or function part gets its ``(*)`` in front
    of that part: ``typeof(int [3])`` becomes ``typeof(*(int (*)[3])0)``.
    A function type name ``typeof(int (void))`` gets the same shape:
    ``typeof(*(int (*)(void))0)``.  A parenthesised abstract declarator
    such as ``int (*)[3]`` is not handled (None).
    """
    off = getattr(tok, "offset", None)
    if off is None or not (0 <= off <= len(src)):
        return None
    toks = _kr_tokens(src)
    idx = next((i for i, t in enumerate(toks) if t[1] == off), None)
    if idx is None or idx < 2 or toks[idx - 1][0] != "(" \
            or not _TYPEOF_RE.fullmatch(toks[idx - 2][0]):
        return None
    depth = 0
    end = None
    for j in range(idx, len(toks)):
        t = toks[j][0]
        if t in "([{":
            depth += 1
        elif t in ")]}":
            if depth == 0:
                end = j
                break
            depth -= 1
    if end is None or end == idx:
        return None
    body = toks[idx:end]
    # The specifiers end at the first * [ ( outside braces.
    depth = 0
    cut = len(body)
    for k, (t, _, _) in enumerate(body):
        if t in "([{":
            if depth == 0 and t != "{":
                cut = k
                break
            depth += 1
        elif t in ")]}":
            depth -= 1
        elif t == "*" and depth == 0:
            cut = k
            break
    def text(a, b):
        return src[body[a][1]:body[b - 1][2]] if b > a else ""
    specs = text(0, cut)
    decl = body[cut:]
    if not decl:
        inner = f"*({specs} *)0"
    else:
        split = next((k for k, (t, _, _) in enumerate(decl)
                      if t in ("[", "(")), None)
        if split is None:
            inner = f"*({specs} {text(cut, len(body))} *)0"
        elif (decl[split][0] == "(" and split == 0 and len(decl) > 1
              and decl[1][0] in ("*", "(", "[", "^")):
            return None     # a parenthesised declarator: int (*)[3]
        else:
            before = src[decl[0][1]:decl[split - 1][2]] if split else ""
            after = src[decl[split][1]:decl[-1][2]]
            inner = f"*({specs} {before}(*){after})0"
    return src[:body[0][1]] + inner + src[body[-1][2]:]


# ---- block-scope typedefs ---------------------------------------------------
#
# A typedef declared in a block belongs to the block (C23 6.2.1).  The back
# ends keep one table of typedef names for a function, so ``typedef long T;``
# in an inner block, with a ``typedef int T;`` at file scope, changed the
# meaning of ``T`` for the rest of the function.  The pass below renames a
# block-scope typedef that declares a name that an enclosing scope already
# declares as a typedef, and the uses that the declaration governs, to a name
# that no other scope can use.  A block typedef of a new name keeps its name.
# A declaration ``int T;`` (after a basic type keyword) in the block hides the
# typedef for the rest of the block.  A name after ``.`` or ``->`` is a member
# and stays; in a struct or union body a name is a use only when an
# identifier or ``*`` follows.


def _typedef_declarator_offsets(toks, i):
    """(end index, [index of each declarator name]) of the typedef
    declaration that starts at ``toks[i]`` (the ``typedef`` token).  A name
    is an identifier at depth 0 before ``;`` ``,`` ``[`` or ``(`` (not
    after a tag keyword, ``.``, ``->`` or a qualifier), or the identifier
    of a ``(*name)`` group."""
    n = len(toks)
    names = []
    brace = 0
    paren = 0
    j = i + 1
    decl_kw = (K.KW_VOID, K.KW_CHAR, K.KW_SHORT, K.KW_INT, K.KW_LONG,
               K.KW_FLOAT, K.KW_DOUBLE, K.KW_SIGNED, K.KW_UNSIGNED,
               K.KW_BOOL)
    while j < n:
        k = toks[j].kind
        if k is K.LBRACE:
            brace += 1
        elif k is K.RBRACE:
            brace -= 1
        elif brace == 0:
            if k is K.LPAREN:
                paren += 1
            elif k is K.RPAREN:
                paren -= 1
            elif k is K.SEMI and paren <= 0:
                break
            elif k is K.IDENT:
                prev = toks[j - 1].kind
                nxt = toks[j + 1].kind if j + 1 < n else None
                if prev in (K.KW_STRUCT, K.KW_UNION, K.KW_ENUM, K.DOT,
                            K.ARROW):
                    pass
                elif paren == 0 and nxt in (K.SEMI, K.COMMA, K.LBRACKET):
                    names.append(j)
                elif paren == 0 and nxt is K.LPAREN and (
                        prev in decl_kw or prev is K.STAR):
                    names.append(j)
                elif paren == 1 and prev is K.STAR and nxt is K.RPAREN:
                    names.append(j)
        j += 1
    return j, names


def _rename_block_scope_typedefs(source: str, filename: str,
                                 seed_typedefs=None) -> str:
    if "typedef" not in source:
        return source
    import zlib
    salt = "%04x" % (zlib.crc32(filename.encode("utf-8", "replace")) & 0xFFFF)
    toks = list(c23_parser.scan(source, filename=filename))
    kw_tag = (K.KW_STRUCT, K.KW_UNION, K.KW_ENUM)
    decl_kw = (K.KW_VOID, K.KW_CHAR, K.KW_SHORT, K.KW_INT, K.KW_LONG,
               K.KW_FLOAT, K.KW_DOUBLE, K.KW_SIGNED, K.KW_UNSIGNED,
               K.KW_BOOL)
    file_names: set[str] = set(seed_typedefs or ())
    edits: list[tuple[int, str, str]] = []
    # stack entries: dict (a block scope: name -> new name, or the same
    # name for a hiding variable) or "tag" (the braces of a tag body)
    stack: list = []
    pending: dict[int, str] = {}      # token index -> name (a declarator)
    counter = 0
    n = len(toks)

    def scopes():
        return [d for d in stack if isinstance(d, dict)]

    def mapped(name):
        for d in reversed(scopes()):
            if name in d:
                return d[name]
        return None

    def known(name):
        # a typedef name of the file, of an enclosing block (renamed or
        # not: the key ("td", name) marks a block typedef of that name)
        return name in file_names or any(
            (name in d and d[name] != name) or ("td", name) in d
            for d in scopes())

    i = 0
    while i < n:
        t = toks[i]
        k = t.kind
        eb = stack[-1] if stack and isinstance(stack[-1], list) else None
        if eb is not None:
            if k in (K.LPAREN, K.LBRACKET):
                eb[2] += 1
            elif k in (K.RPAREN, K.RBRACKET):
                eb[2] -= 1
            elif k is K.ASSIGN and eb[2] == 0:
                eb[1] = True
            elif k is K.COMMA and eb[2] == 0:
                eb[1] = False
        if k is K.LBRACE:
            stack.append({})
        elif k is K.RBRACE:
            if stack:
                stack.pop()
        elif k in kw_tag:
            # the braces of a tag body are not a block
            j = i + 1
            if j < n and toks[j].kind is K.IDENT:
                j += 1
            if k is K.KW_ENUM and j < n and toks[j].kind is K.COLON:
                while j < n and toks[j].kind not in (K.LBRACE, K.SEMI,
                                                     K.RPAREN, K.COMMA):
                    j += 1
            if j < n and toks[j].kind is K.LBRACE:
                # an enum body: [marker, in an initializer, paren depth]
                stack.append(["enum", False, 0] if k is K.KW_ENUM
                             else "tag")
                i = j + 1
                continue
        elif k is K.KW_TYPEDEF:
            end, names = _typedef_declarator_offsets(toks, i)
            cur = scopes()
            if not cur:
                file_names.update(toks[x].text for x in names)
            else:
                for x in names:
                    if known(toks[x].text):
                        pending[x] = toks[x].text
                    else:
                        cur[-1][("td", toks[x].text)] = True
        elif k is K.IDENT:
            prev = toks[i - 1].kind if i else None
            nxt = toks[i + 1].kind if i + 1 < n else None
            top = stack[-1] if stack else None
            if prev in (K.DOT, K.ARROW) or prev in kw_tag:
                pass        # a member, or a tag: not a typedef name
            elif i in pending:
                counter += 1
                new = f"{t.text}__t{salt}_{counter}"
                scopes()[-1][t.text] = new
                edits.append((t.offset, t.text, new))
            elif top == "tag" and nxt not in (K.IDENT, K.STAR):
                pass
            elif isinstance(top, list) and not top[1]:
                pass        # an enumerator definition: not a typedef name
            else:
                cur = scopes()
                found = mapped(t.text)
                if found is not None and prev in decl_kw and cur:
                    # ``int T;``: a variable hides the typedef
                    cur[-1][t.text] = t.text
                elif found is not None and found != t.text:
                    edits.append((t.offset, t.text, found))
        i += 1
    if not edits:
        return source
    out = []
    pos = 0
    for off, tag, new in edits:
        if off < pos or source[off:off + len(tag)] != tag:
            return source
        out.append(source[pos:off])
        out.append(new)
        pos = off + len(tag)
    out.append(source[pos:])
    return "".join(out)


# ---- enumerations with a fixed underlying type ---------------------------------
#
# C23 6.7.3.3: ``enum e : unsigned char { A, B };`` and the forward
# declaration ``enum e : unsigned char;``.  The grammar has no such form, so
# this pass takes ``: type`` out of the text (the parser then sees ``enum e
# { A, B };`` and ``enum e;``) and keeps the type.  ``parse`` attaches the
# table to the unit as ``unit._enum_fixed_types``: the tag of each such
# enumeration -> the list of the type texts that its declarations gave (a
# host compares them).  An anonymous ``enum : T { ... }`` gets the tag
# ``__enum_fixed_<salt>_<n>``.  ``enum e : 3;`` in a struct body is a
# bit-field, and stays.

_ENUM_FIXED_RE = re.compile(r"\benum\s*(?:[A-Za-z_]\w*\s*)?:(?!:)")


def _strip_enum_fixed_types(source: str, filename: str):
    """(new source, {tag: [type text, ...]})."""
    if not _ENUM_FIXED_RE.search(source):
        return source, {}
    import zlib
    salt = "%04x" % (zlib.crc32(filename.encode("utf-8", "replace")) & 0xFFFF)
    toks = list(c23_parser.scan(source, filename=filename))
    n = len(toks)
    edits: list[tuple[int, int, str]] = []     # (start, end, new text)
    table: dict[str, list[str]] = {}
    stack: list[str] = []          # "agg" (struct, union body) or "blk"
    agg_wait = False
    counter = 0
    i = 0
    while i < n:
        t = toks[i]
        k = t.kind
        if k is K.LBRACE:
            stack.append("agg" if agg_wait else "blk")
            agg_wait = False
        elif k is K.RBRACE:
            if stack:
                stack.pop()
            agg_wait = False
        elif k in (K.KW_STRUCT, K.KW_UNION):
            agg_wait = True
        elif k is K.KW_ENUM:
            agg_wait = False
            j = i + 1
            tag = None
            if j < n and toks[j].kind is K.IDENT:
                tag = toks[j].text
                j += 1
            if j < n and toks[j].kind is K.COLON:
                m = j + 1
                while m < n and toks[m].kind not in (
                        K.LBRACE, K.SEMI, K.COMMA, K.RPAREN, K.ASSIGN):
                    m += 1
                words = toks[j + 1:m]
                after = toks[m].kind if m < n else None
                in_agg = bool(stack) and stack[-1] == "agg"
                first = words[0] if words else None
                is_type = bool(words) and (
                    first.kind is not K.IDENT
                    and not first.kind.name.startswith(("INT_", "CHAR_")))
                if (not is_type and words
                        and all(w.kind is K.IDENT for w in words)
                        and (after is K.LBRACE or not in_agg)):
                    is_type = True
                if is_type and (after is K.LBRACE or tag is not None):
                    text = " ".join(w.text for w in words)
                    if tag is None:
                        counter += 1
                        tag = f"__enum_fixed_{salt}_{counter}"
                        edits.append((toks[j].offset,
                                      words[-1].offset + len(words[-1].text),
                                      " " + tag))
                    else:
                        edits.append((toks[j].offset,
                                      words[-1].offset + len(words[-1].text),
                                      ""))
                    table.setdefault(tag, []).append(text)
                    i = m
                    continue
        i += 1
    if not edits:
        return source, {}
    out = []
    pos = 0
    for a, b, new in edits:
        if a < pos:
            return source, {}
        out.append(source[pos:a])
        out.append(new)
        pos = b
    out.append(source[pos:])
    return "".join(out), table


# ---- block-scope tags -------------------------------------------------------
#
# The tag of a struct, union or enum declared inside a block belongs to
# that block (C23 6.2.1, 6.7.2.3).  The back ends key their type tables by
# the bare tag, so ``struct S`` in a function and ``struct S`` at file
# scope were one type.  The pass below renames each tag that a block
# declares (with a body, or as ``struct S;``) to a name that no other scope
# can use, and renames the uses that the declaration governs.  A use with
# no declaration in an enclosing block keeps its name.
#
# The enumeration constants of a block-scope enumeration belong to the
# block too (C23 6.2.1p4), so ``enum {X = 77};`` in a block must not change
# a file-scope ``X``, not even before the block.  Each constant of such an
# enumeration gets a new name, and every identifier token of that name in
# the rest of the block follows (also an inner declaration of the same
# name, which hides the constant anyway).  A name after ``.`` or ``->`` is a
# member and stays.  In a struct or union body inside the block, only a
# name after ``[``, ``:``, ``(`` or an arithmetic operator is a use; a name
# before ``;`` ``,`` or ``[`` there is a member declarator and stays.  A
# declaration ``int X;`` in a block, after a basic type keyword, of a name
# that is an enumeration constant (at file scope or in an enclosing block)
# hides the constant: the variable gets its own name for the rest of the
# block.

_TAG_DECL_RE = re.compile(
    r"\b(?:struct|union|enum)\s+[A-Za-z_]\w*\s*[{:;]|\benum\s*\{")


def _rename_block_scope_tags(source: str, filename: str) -> str:
    if not _TAG_DECL_RE.search(source):
        return source
    import zlib
    salt = "%04x" % (zlib.crc32(filename.encode("utf-8", "replace")) & 0xFFFF)
    toks = list(c23_parser.scan(source, filename=filename))
    kw_tag = (K.KW_STRUCT, K.KW_UNION, K.KW_ENUM)
    edits: list[tuple[int, str, str]] = []
    # stack entries: dict (a block scope) or a str, "enum", "struct" or
    # "union" (the braces of a tag body)
    stack: list = []
    file_tags: set[str] = set()
    counter = 0
    n = len(toks)
    i = 0

    def block_scopes():
        return [d for d in stack if isinstance(d, dict)]

    def lookup(tag):
        for d in reversed(block_scopes()):
            if tag in d:
                return d[tag]
        return None

    file_consts: set[str] = set()
    decl_kw = (K.KW_VOID, K.KW_CHAR, K.KW_SHORT, K.KW_INT, K.KW_LONG,
               K.KW_FLOAT, K.KW_DOUBLE, K.KW_SIGNED, K.KW_UNSIGNED,
               K.KW_BOOL)
    const_ok_prev = (K.LBRACKET, K.COLON, K.LPAREN, K.PLUS, K.MINUS,
                     K.SLASH, K.LSHIFT, K.RSHIFT, K.AMP, K.PIPE, K.CARET,
                     K.TILDE, K.PERCENT)
    pend = None     # [scope, key, new name, paren depth] of an enumerator
    while i < n:
        t = toks[i]
        k = t.kind
        if pend is not None:
            if k in (K.LPAREN, K.LBRACKET):
                pend[3] += 1
            elif k in (K.RPAREN, K.RBRACKET):
                pend[3] -= 1
            elif k in (K.COMMA, K.RBRACE) and pend[3] == 0:
                pend[0][pend[1]] = pend[2]
                pend = None
        if k is K.LBRACE:
            stack.append({})
        elif k is K.RBRACE:
            if stack:
                stack.pop()
        elif k is K.IDENT:
            prev = toks[i - 1].kind if i else None
            top = stack[-1] if stack else None
            if prev in (K.DOT, K.ARROW):
                pass
            elif (top == "enum" and prev in (K.LBRACE, K.COMMA)
                  and pend is None):
                cur = block_scopes()[-1] if block_scopes() else None
                if cur is None:
                    file_consts.add(t.text)
                else:
                    counter += 1
                    new = f"{t.text}__c{salt}_{counter}"
                    # the scope of an enumerator starts after its
                    # enumerator-definition (C23 6.2.1): the name joins
                    # the scope at the comma or brace that ends it
                    pend = [cur, "#" + t.text, new, 0]
                    edits.append((t.offset, t.text, new))
            elif top in ("struct", "union") and prev not in const_ok_prev:
                pass
            else:
                scopes = block_scopes()
                found = None
                for d in reversed(scopes):
                    found = d.get("#" + t.text)
                    if found is not None:
                        break
                cur = scopes[-1] if scopes else None
                if (prev in decl_kw and cur is not None
                        and "#" + t.text not in cur
                        and (found is not None or t.text in file_consts)):
                    # ``int X;`` in a block hides a constant X: the
                    # variable gets a name of its own, like a constant.
                    counter += 1
                    found = f"{t.text}__c{salt}_{counter}"
                    cur["#" + t.text] = found
                if found is not None:
                    edits.append((t.offset, t.text, found))
        elif (k in kw_tag and i + 1 < n and toks[i + 1].kind is K.LBRACE
              and k is not K.KW_ENUM):
            stack.append("struct" if k is K.KW_STRUCT else "union")
            i += 2
            continue
        elif k is K.KW_ENUM and i + 1 < n and toks[i + 1].kind is K.LBRACE:
            # An anonymous enumeration with explicit values may need a type
            # wider than int; a back end finds that type by the tag, so the
            # enumeration gets one.
            j = i + 2
            depth = 1
            has_value = False
            while j < n and depth:
                kj = toks[j].kind
                if kj is K.LBRACE:
                    depth += 1
                elif kj is K.RBRACE:
                    depth -= 1
                elif kj is K.ASSIGN and depth == 1:
                    has_value = True
                j += 1
            if has_value:
                counter += 1
                edits.append((t.offset + len("enum"), "",
                              f" __anon_enum_{salt}_{counter}"))
            stack.append("enum")
            i += 2
            continue
        elif k in kw_tag and i + 1 < n and toks[i + 1].kind is K.IDENT:
            tag_tok = toks[i + 1]
            tag = tag_tok.text
            j = i + 2
            nxt = toks[j].kind if j < n else None
            is_enum = k is K.KW_ENUM
            if is_enum and nxt is K.COLON:
                while j < n and toks[j].kind not in (K.LBRACE, K.SEMI,
                                                     K.RPAREN, K.COMMA):
                    j += 1
                nxt = toks[j].kind if j < n else None
            is_def = nxt is K.LBRACE
            prev = toks[i - 1].kind if i else None
            is_fwd = (nxt is K.SEMI and not is_enum
                      and prev in (None, K.SEMI, K.LBRACE, K.RBRACE))
            in_block = bool(block_scopes())
            if not in_block:
                file_tags.add(tag)
            else:
                cur = block_scopes()[-1]
                known = lookup(tag)
                if is_def or is_fwd:
                    if tag in cur:
                        new = cur[tag]
                    else:
                        counter += 1
                        new = f"{tag}__b{salt}_{counter}"
                        cur[tag] = new
                else:
                    new = known
                if new is not None and new != tag:
                    edits.append((tag_tok.offset, tag, new))
            if is_def:
                # skip to the brace: it opens a tag body, not a scope
                stack.append("enum" if is_enum else
                             "struct" if k is K.KW_STRUCT else "union")
                i = j + 1
                continue
            i += 2
            continue
        i += 1
    if not edits:
        return source
    out = []
    pos = 0
    for off, tag, new in edits:
        if off < pos or source[off:off + len(tag)] != tag:
            return source          # the offsets do not match: leave as is
        out.append(source[pos:off])
        out.append(new)
        pos = off + len(tag)
    out.append(source[pos:])
    return "".join(out)


# ---- typedef tracker -------------------------------------------------------


def _make_typedef_filter(seed_typedefs: set[str] | None,
                         source: str | None = None) -> tuple[object, object]:
    """Return ``(hooks, token_filter)`` wired to a TypedefTracker.

    The tracker accumulates IDENT names declared with the ``typedef``
    storage class and rewrites subsequent IDENT lookahead to
    ``TYPEDEF_NAME``. With Phase 6, the post-reduce hook receives a
    typed :class:`ast.Declaration` (no longer a ParseNode), so the
    storage-class check walks ``decl_specs`` instead of probing for
    ``KW_TYPEDEF`` tokens in a CST.
    """
    names: set[str] = set(seed_typedefs or ())
    # Track the previous token name (keyed by source offset so the
    # parser's re-querying the same token doesn't advance the state).
    # Suppress the IDENT -> TYPEDEF_NAME rewrite when the IDENT is
    # immediately preceded by ``struct`` / ``union`` / ``enum`` (those
    # introduce a tag namespace separate from the typedef namespace;
    # a tag and a typedef can share a name, e.g.
    # ``typedef struct Regexp Regexp;``). Also suppress after ``.`` or
    # ``->`` since member references live in their own namespace too.
    # Token names after which an IDENT should NOT be rewritten to
    # TYPEDEF_NAME. Cases:
    #   * After ``struct`` / ``union`` / ``enum`` — tag namespace.
    #   * After ``.`` / ``->`` — member namespace.
    #   * After a concrete type-spec keyword. ``int byte`` is a
    #     parameter declaration where ``byte`` is the declarator name
    #     even if ``byte`` is typedef'd elsewhere; C allows re-using
    #     a typedef-name as a plain identifier in inner scopes.
    # Excluded on purpose:
    #   * Storage classes (``static``, ``extern``, ``register``,
    #     ``auto``, ``inline``, ``constexpr``, ``thread_local``) and
    #     type qualifiers (``const``, ``volatile``, ``restrict``,
    #     ``_Atomic``) — these introduce or qualify a type, and the
    #     next IDENT might legitimately be a typedef-name acting as
    #     the type-spec (``static const T x;`` etc.).
    #   * ``*`` — same reason; ``T *x;`` parses fine with T as
    #     TYPEDEF_NAME and x as IDENT.
    # Type-spec keywords whose following IDENT is a declarator name
    # (and therefore shadows the typedef for the enclosing block).
    _TYPESPEC_PREV = frozenset({
        K.KW_VOID, K.KW_CHAR, K.KW_SHORT, K.KW_INT, K.KW_LONG,
        K.KW_FLOAT, K.KW_DOUBLE, K.KW_SIGNED, K.KW_UNSIGNED,
        K.KW_BOOL, K.KW_COMPLEX, K.KW_IMAGINARY, K.KW_NULLPTR,
        K.KW_DECIMAL32, K.KW_DECIMAL64, K.KW_DECIMAL128,
        K.KW_BITINT, K.KW_INT128,
    })
    # Combined: contexts where IDENT → TYPEDEF_NAME rewrite is
    # suppressed. Includes the type-spec set plus tag/member
    # namespaces. Tag/member contexts suppress but DON'T shadow.
    _SUPPRESS_PREV = _TYPESPEC_PREV | frozenset({
        K.KW_STRUCT, K.KW_UNION, K.KW_ENUM, K.DOT, K.ARROW,
        # a label has its own namespace: ``goto s;`` with ``s`` a typedef
        K.KW_GOTO,
    })
    # Tokens that, while a type specifier is "open", continue the
    # declarator *prefix* (pointers / type-qualifiers / extra spec
    # keywords) without ending it. A typedef-named IDENT reached
    # through only these is the declarator, not a second type-spec.
    _DECL_CONTINUE = _TYPESPEC_PREV | frozenset({
        K.STAR, K.KW_CONST, K.KW_VOLATILE, K.KW_RESTRICT, K.KW_ATOMIC,
    })
    # Previous-token contexts in which an IDENT is *not* a declarator
    # even if a specifier looks open: tag- and member-namespaces.
    _NOT_DECL_PREV = frozenset({
        K.KW_STRUCT, K.KW_UNION, K.KW_ENUM, K.DOT, K.ARROW,
    })
    # state[0] = offset of the currently-being-filtered token
    # state[1] = name of the previously-distinct token (used as `prev`
    #            for the suppress check)
    # state[2] = name of the currently-being-filtered token
    # state[3] = previously-distinct token's *text* (used when we need
    #            to register a typedef-name as locally-shadowed)
    # state[4] = current paren depth (for parameter-name shadowing)
    # state[5] = current brace depth
    # state[6] = stack of (brace_depth_at_entry, set_of_shadowed_names)
    #            so when the brace depth drops back, we pop the
    #            shadow set added by the matching scope.
    # state[7] = specifier-open flag: a type specifier has been seen in
    #            the current declaration and only declarator-prefix
    #            tokens have followed (so the next typedef-named IDENT
    #            is the declarator, not the type).
    # state[8] = set of token offsets forced to stay IDENT (declarator
    #            names), so a parser re-query of the same offset is
    #            stable across backtracking.
    # The parser may re-query the same offset multiple times during
    # lookahead; only advance state when the offset changes so the
    # `prev` context stays stable for a given token.
    # state[9] = brace depth of the ``typedef`` declaration being read, or
    #            -1.  A typedef declaration that names a typedef again
    #            (``{ typedef long T; T v; }`` with a file-scope ``T``)
    #            declares a typedef, not a variable that hides it.
    state = [-1, K.UNKNOWN, K.UNKNOWN, "", 0, 0, [], False, set(), -1]
    # A label has its own namespace: ``s: stmt`` with ``s`` a typedef name
    # is a label, when it stands where a statement starts, inside a
    # function body (not in a struct body, where ``T : 3`` is a bit-field).
    # kinds = for each open brace, "agg" (the body of a struct, union or
    # enum) or "blk"; agg_wait = a struct, union or enum keyword (and
    # maybe its tag) was the last thing seen.
    kinds: list[str] = []
    agg = [0]
    _STMT_START = frozenset({K.SEMI, K.LBRACE, K.RBRACE, K.COLON, K.RPAREN,
                             K.KW_ELSE, K.KW_DO})

    def is_label(tok: Token) -> bool:
        if source is None or not kinds or kinds[-1] == "agg":
            return False
        if state[1] not in _STMT_START:
            return False
        end = tok.offset + len(tok.text)
        rest = source[end:end + 64].lstrip(" \t\r\n")
        return rest.startswith(":") and not rest.startswith("::")

    def filter_(_ctx, tok: Token) -> Token:
        offset = getattr(tok, "offset", -1)
        if offset != state[0]:
            # New token: roll previous-distinct into history slots,
            # then update depth counters.
            state[1] = state[2]
            state[3] = "" if state[1] is K.UNKNOWN else state[3]
            state[2] = tok.kind
            state[0] = offset
            # When we close a brace, drop any shadow set whose
            # entry-depth matches the brace count we just exited.
            if tok.kind is K.LBRACE:
                kinds.append("agg" if agg[0] else "blk")
                agg[0] = 0
            elif tok.kind is K.RBRACE:
                if kinds:
                    kinds.pop()
            if tok.kind in (K.KW_STRUCT, K.KW_UNION, K.KW_ENUM):
                agg[0] = 1
            elif tok.kind is not K.IDENT and tok.kind is not K.LBRACE:
                agg[0] = 0
            if tok.kind is K.RBRACE:
                state[5] -= 1
                while state[6] and state[6][-1][0] > state[5]:
                    state[6].pop()
            elif tok.kind is K.LBRACE:
                state[5] += 1
            elif tok.kind is K.LPAREN:
                state[4] += 1
            elif tok.kind is K.RPAREN:
                state[4] = max(0, state[4] - 1)
            # ---- specifier / declarator-name disambiguation ---------
            # In a declaration the *first* specifier establishes the
            # type; a later identifier reached only through pointer /
            # qualifier / extra-spec tokens is the declarator name —
            # even when its text is a known typedef-name. C keeps the
            # two uses distinct, so all of:
            #     typedef int T;  struct S { T *T; };
            #     void f(T *T) { ... }
            #     { T T = 5; }
            # are valid (the 2nd `T` is a member/parameter/local).
            # This also covers the original ``int byte`` keyword case
            # (a concrete type keyword opens the specifier too).
            prev_kind = state[1]
            base_shadowed = any(tok.text in s for _d, s in state[6])
            is_td_ident = tok.kind is K.IDENT and tok.text in names
            # Would this token, on its own, act as a type specifier
            # (i.e. be rewritten to TYPEDEF_NAME)?
            emits_typedef = (
                is_td_ident
                and prev_kind not in _SUPPRESS_PREV
                and not base_shadowed
            )
            declarator_here = (
                state[7]                       # a specifier is open
                and is_td_ident                # ... and this is typedef-named
                and (state[4] >= 1 or state[5] >= 1)
                # Inside a signature (paren_depth >= 1) or a block
                # (brace_depth >= 1); file-scope typedef declarations
                # stay unshadowed.
                and prev_kind not in _NOT_DECL_PREV
            )
            if tok.kind is K.KW_TYPEDEF:
                state[9] = state[5]
            elif tok.kind is K.SEMI and state[9] == state[5]:
                state[9] = -1
            if declarator_here and state[9] >= 0:
                # the name of a typedef declaration: stays an IDENT here,
                # and still names a type afterwards (no shadow)
                state[8].add(offset)
                state[7] = False
            elif declarator_here:
                # The declarator name shadows the typedef for the
                # enclosing block. The +1 depth makes a parameter
                # shadow survive the signature's paren pair and apply
                # to the function body.
                # A name declared in a block lives until that block
                # closes: depth = the block depth (a nested block that
                # closes must not drop it).  A parameter (a signature at
                # file scope) lives until the function body closes.
                target_depth = state[5] + (1 if state[5] == 0 else 0)
                if not state[6] or state[6][-1][0] != target_depth:
                    state[6].append((target_depth, set()))
                state[6][-1][1].add(tok.text)
                state[8].add(offset)
                state[7] = False               # declarator consumed
            elif emits_typedef or tok.kind in _TYPESPEC_PREV:
                state[7] = True                # a type specifier
            elif (tok.kind is K.IDENT and prev_kind in _NOT_DECL_PREV
                  and prev_kind in (K.KW_STRUCT, K.KW_UNION, K.KW_ENUM)):
                state[7] = True                # ``struct s``: the tag ends
                #                                the specifier, a name follows
            elif state[7] and tok.kind in _DECL_CONTINUE:
                pass                           # still in declarator prefix
            else:
                state[7] = False               # specifier context ended
        prev_kind = state[1]
        # Walk the shadow stack: if any frame contains this name, the
        # IDENT is shadowed; don't rewrite. A forced declarator offset
        # is likewise kept as IDENT.
        shadowed = any(tok.text in s for _d, s in state[6])
        if (
            tok.kind is K.IDENT
            and tok.text in names
            and prev_kind not in _SUPPRESS_PREV
            and not shadowed
            and offset not in state[8]
            and not is_label(tok)
        ):
            return Token(
                name="TYPEDEF_NAME",
                text=tok.text,
                line=tok.line,
                column=tok.column,
                offset=tok.offset,
                file_id=tok.file_id,
                kind=K.TYPEDEF_NAME,
            )
        return tok

    def record_typedef(_ctx, payload: dict) -> None:
        if payload.get("when") != "post_reduce":
            return
        node = payload.get("value")
        if not isinstance(node, ast.Declaration):
            return
        # decl_specs is a list of spec nodes. Detect typedef storage class.
        has_typedef = False
        for spec in node.decl_specs or []:
            if isinstance(spec, ast.StorageClass) and spec.kw.text == "typedef":
                has_typedef = True
                break
        if not has_typedef:
            return
        for init_decl in node.declarators or []:
            ident = _leftmost_ident(_inner_decl(init_decl))
            if ident is not None:
                names.add(ident)

    hooks = HookRegistry(ignore_missing=True)
    hooks.register("record_typedef", record_typedef)
    return hooks, filter_


def _inner_decl(init_decl):
    """Walk the InitDeclarator wrapper to the underlying declarator."""
    if isinstance(init_decl, (ast.InitDeclarator, ast.InitDeclaratorWithInit)):
        return init_decl.declarator
    return init_decl


def _leftmost_ident(node) -> str | None:
    """Walk the declarator chain to the innermost ``Declarator(name=IDENT)``."""
    while node is not None:
        if isinstance(node, ast.Declarator):
            return node.name.text
        if isinstance(node, ast.PointerDeclarator):
            node = node.inner
            continue
        if isinstance(node, ast.GroupDeclarator):
            node = node.inner
            continue
        if isinstance(node, (ast.ArrayDeclarator, ast.ArrayDeclaratorUnsized,
                             ast.ArrayDeclaratorStar, ast.ArrayDeclaratorStatic,
                             ast.ArrayDeclaratorQualStatic,
                             ast.FnDeclarator, ast.FnDeclaratorEmpty)):
            node = node.inner
            continue
        return None
    return None


# ---- public entry point -----------------------------------------------------


def parse(
    source: str,
    filename: str = "<stdin>",
    seed_typedefs: set[str] | None = None,
    kr_prepass: bool = False,
    strict_unsupported: bool = False,
    unsupported_message: str = _DEFAULT_UNSUPPORTED_MESSAGE,
    strict_allow: frozenset | set = frozenset(),
) -> ast.TranslationUnit:
    """Parse already-preprocessed C source into a typed AST.

    The caller is expected to have run :class:`uc_core.preprocessor`
    over ``source`` first.

    ``kr_prepass=True`` enables the pre-ANSI source rewrite
    (implicit-int returns + K&R parameter lists -> ANSI). Off by
    default so modern code pays zero cost; hosts compiling old code
    (e.g. the gcc-c-torture corpus) opt in explicitly.

    ``strict_unsupported=True`` turns the silent erasure of ``interrupt``
    / ``fastcall`` markers (``__interrupt``, ``interrupt``,
    ``_interrupt``, ``[[...interrupt...]]``, ``__attribute__((interrupt))``,
    bare ``__fastcall`` / ``_fastcall`` / ``fastcall``) into an
    :class:`UnsupportedConstructError` with ``file:line``.
    ``unsupported_message`` is a ``str.format`` template with ``{name}``;
    a host sets it to its own wording.  Default is off: tolerant.
    ``strict_allow`` is a set of kinds, ``"interrupt"`` and/or
    ``"fastcall"``, that strict mode lets through (they are then erased
    as in tolerant mode).  A host that supports one spelling in a later
    phase records it in its own pre-pass and passes the kind here, so
    the other kinds stay refused.  The scan covers the whole
    preprocessed text, header text included, so shipped headers must
    not use refused markers.

    ``seed_typedefs`` pre-populates the typedef tracker with names a
    host knows about from system headers it processed out-of-band
    (``FILE``, ``size_t``, ``jmp_buf``, …); the parser-time hook adds
    new typedefs as they appear in source. Without seeding, references
    to libc-typedef'd names as types parse as IDENT and fall through
    to a syntax error in declarator position.

    Returns a :class:`uc_core.ast.TranslationUnit`; ``unit.items`` is a
    list of typed declarations / function definitions, with source
    positions on every node's ``.pos`` and ``Token``-level
    ``file_id``s indexing :data:`c23_parser.FILE_TABLE`.
    """
    # AST walks recurse linearly over left-recursive lists. Bump the
    # interpreter limit so non-trivial TUs don't overflow.
    if sys.getrecursionlimit() < 50_000:
        sys.setrecursionlimit(50_000)
    # Record `__attribute__((packed))` targets BEFORE `_strip_attributes`
    # erases them. Two source-text shapes recognised: a named struct
    # (`struct foo { ... } __attribute__((packed));`) is keyed by the
    # tag name; a typedef struct is keyed by the typedef name. The
    # parser builds the AST with no `is_packed` flag on either shape,
    # so we re-attach it below.
    if strict_unsupported:
        _check_strict_unsupported(source, filename, unsupported_message,
                                  frozenset(strict_allow))
    packed_tag_names: set[str] = set()
    packed_typedef_names: set[str] = set()
    for m in _PACKED_NAMED_STRUCT_RE.finditer(source):
        packed_tag_names.add(m.group(1))
    for m in _PACKED_TYPEDEF_STRUCT_RE.finditer(source):
        packed_typedef_names.add(m.group(1))
    vector_names: set = set()
    pre = _strip_attributes(source, vector_names)
    pre = _rename_block_scope_typedefs(pre, filename, seed_typedefs)
    pre = _rename_block_scope_tags(pre, filename)
    pre, enum_fixed = _strip_enum_fixed_types(pre, filename)
    hooks, filter_ = _make_typedef_filter(seed_typedefs, pre)
    try:
        try:
            unit = c23_parser.parse(
                pre, filename=filename, hooks=hooks, token_filter=filter_
            )
        except c23_parser.ParseError as err:
            # The grammar takes ``typeof(expr)`` only.  ``typeof(type-name)``
            # (C23 6.7.3.6) fails at the first token of the type name: the
            # site is rewritten to the equivalent ``typeof(*(type-name *)0)``
            # and the parse is tried again.
            if not _TYPEOF_RE.search(pre):
                raise
            for _ in range(1000):
                fixed = _typeof_typename_fix(pre, err.token)
                if fixed is None:
                    raise
                pre = fixed
                hooks, filter_ = _make_typedef_filter(seed_typedefs, pre)
                try:
                    unit = c23_parser.parse(
                        pre, filename=filename, hooks=hooks,
                        token_filter=filter_)
                    break
                except c23_parser.ParseError as err2:
                    err = err2
            else:
                raise
    except c23_parser.ParseError:
        # Strict C23 parse failed. Only if the host opted into K&R
        # recovery do we rewrite the pre-ANSI shapes (implicit-int
        # returns / K&R parameter lists) and retry — and only for
        # *this* file, which demonstrably needs it. Modern code parses
        # on the first attempt and never reaches the scanner: zero
        # cost and zero mis-rewrite risk, even inside the torture
        # corpus where the flag is on for every file.
        if not kr_prepass:
            raise
        rewritten = _rewrite_kr_and_implicit_int(pre)
        rewritten = _rewrite_computed_goto(rewritten)
        if rewritten == pre:
            raise  # nothing pre-ANSI / GNU here — surface the real error
        # Fresh typedef-tracker state; the failed attempt left the
        # filter's scan position dirty.
        hooks, filter_ = _make_typedef_filter(seed_typedefs, rewritten)
        unit = c23_parser.parse(
            rewritten, filename=filename, hooks=hooks, token_filter=filter_
        )
    # Attach the set of names that had `__attribute__((vector_size(N)))`
    # rewritten away — downstream consumers (uc386 codegen) read this
    # to flip the GCC is_vector flag on the resolved ArrayType.
    unit._vector_typedef_names = vector_names
    unit._enum_fixed_types = enum_fixed
    if packed_tag_names or packed_typedef_names:
        _mark_packed_structs(unit, packed_tag_names, packed_typedef_names)
    return unit


def _mark_packed_structs(
    unit: ast.TranslationUnit,
    packed_tag_names: set[str],
    packed_typedef_names: set[str],
) -> None:
    """Re-attach ``is_packed=True`` to ``StructDef`` / ``StructAnon``
    AST nodes whose source-text decl carried ``__attribute__((packed))``.

    ``StructDef`` (named) is matched by ``decl.name.text``.
    ``StructAnon`` (typedef'd anonymous struct) is matched by walking
    the enclosing ``Declaration``'s ``declarators`` for a typedef name
    that's in ``packed_typedef_names``."""
    def _visit(node):
        if node is None:
            return
        if isinstance(node, c23_parser.StructDef):
            tag = getattr(node.name, "text", None)
            if tag in packed_tag_names:
                node.is_packed = True
        if isinstance(node, c23_parser.Declaration):
            # Typedef case: any typedef declarator whose name is in
            # `packed_typedef_names` marks every StructAnon in the
            # decl_specs as packed.
            decl_names = _typedef_declarator_names(node)
            if decl_names & packed_typedef_names:
                for spec in node.decl_specs or ():
                    if isinstance(spec, c23_parser.StructAnon):
                        spec.is_packed = True
                    elif isinstance(spec, c23_parser.StructDef):
                        spec.is_packed = True
        # Recurse into common container shapes.
        for attr in ("items", "members", "decl_specs", "declarators",
                     "body", "decl_specifiers", "stmt", "init"):
            child = getattr(node, attr, None)
            if isinstance(child, list):
                for c in child:
                    _visit(c)
            elif child is not None and hasattr(child, "__dataclass_fields__"):
                _visit(child)
    _visit(unit)


def _typedef_declarator_names(decl: object) -> set[str]:
    """Return the set of names declared by a ``typedef`` Declaration's
    declarators. Returns an empty set if the decl isn't a typedef."""
    decl_specs = getattr(decl, "decl_specs", None) or ()
    is_typedef = any(
        isinstance(s, c23_parser.StorageClass)
        and getattr(s.kw, "text", None) == "typedef"
        for s in decl_specs
    )
    if not is_typedef:
        return set()
    names: set[str] = set()
    for d in getattr(decl, "declarators", None) or ():
        # Walk through the declarator nest to find the inner Identifier.
        name = _innermost_declarator_name(d)
        if name is not None:
            names.add(name)
    return names


def _innermost_declarator_name(node) -> str | None:
    """Pull the innermost identifier name out of a declarator tree."""
    if node is None:
        return None
    if isinstance(node, c23_parser.Token):
        return node.text
    # Common attribute names that point at the inner declarator or
    # name token.
    for attr in ("name", "declarator", "inner", "decl"):
        child = getattr(node, attr, None)
        if isinstance(child, c23_parser.Token):
            return child.text
        if child is not None and hasattr(child, "__dataclass_fields__"):
            n = _innermost_declarator_name(child)
            if n is not None:
                return n
    return None
