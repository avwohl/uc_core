"""uplox-driven C23 front-end for uc_core.

Public entry: :func:`parse`. Takes already-preprocessed C source —
the per-compiler preprocessor (uc_core, uc80, uc386 each instantiate
their own ``Preprocessor`` with target-specific predefines) runs
upstream, mirroring the legacy ``Lexer`` + ``Parser`` contract.

Internal pipeline:

1. ``_strip_attributes`` — strip C23 ``[[...]]`` and GCC
   ``__attribute__((...))`` from the source text. uplox's c23
   grammar doesn't model attributes; the legacy parser had its own
   ``_skip_gcc_attribute`` shim that erased them after the lexer.
2. uplox c23 LR(1) parse — produces a ``ParseNode`` tree. A
   :class:`TypedefTracker` runs as a token filter / declaration hook
   so identifier references to typedef'd names lex as ``TYPEDEF_NAME``
   in cast / type-spec position.
3. ``_convert_translation_unit`` — lower the ``ParseNode`` tree into
   uc_core's ``ast`` dataclasses. All 63 AST node kinds are produced
   by the converter; the existing ``ast_optimizer`` and the
   per-backend codegens consume the same shape unchanged.

The JSON bundle at ``uc_core/data/c23.json`` is built from
``uplox/examples/c23.uplox`` via ``uplox build`` and loaded once per
process (~70 ms vs ~25 s to build the LR(1) table from grammar
source).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import cast

from uplox.hooks import TypedefTracker
from uplox.lex.scanner import Scanner, Token
from uplox.parse.runtime import HookRegistry, ParseNode, parse as _uplox_parse
from uplox.tables import balanced_from_json, dfa_from_json, table_from_json

from . import ast
from .ast import SourceLocation


_BUNDLE_PATH = Path(__file__).parent / "data" / "c23.json"


@lru_cache(maxsize=1)
def _load_c23() -> tuple[Scanner, object]:
    """Load the prebuilt c23 lexer + LR table from the JSON bundle."""
    with open(_BUNDLE_PATH) as f:
        bundle = json.load(f)
    dfa, _tokens, skip_list = dfa_from_json(bundle["lex"])
    balanced = balanced_from_json(bundle["lex"].get("balanced", {}))
    scanner = Scanner(dfa=dfa, skip_tokens=frozenset(skip_list), balanced=balanced)
    table = table_from_json(bundle["parse"])
    return scanner, table


def parse(
    source: str,
    filename: str = "<stdin>",
    seed_typedefs: set[str] | None = None,
) -> ast.TranslationUnit:
    """Parse already-preprocessed C source into an AST.

    The caller is expected to have run :class:`uc_core.preprocessor`
    (or an equivalent) over ``source`` first — the legacy ``Lexer`` /
    ``Parser`` had the same contract.

    ``seed_typedefs`` pre-populates the typedef tracker with names a
    host knows about from system headers it processed out-of-band
    (``FILE``, ``size_t``, ``jmp_buf``, …); the parser-time hook adds
    new typedefs as they appear in source. Without seeding, references
    to libc-typedef'd names as types parse as IDENT and fall through
    to a syntax error in declarator position.
    """
    _vector_names.clear()
    pre = _strip_attributes(source)
    scanner, table = _load_c23()

    tracker = TypedefTracker()
    if seed_typedefs:
        tracker.names |= seed_typedefs
    hooks = HookRegistry(ignore_missing=True)
    hooks.register("record_typedef", tracker.record_declaration)
    # The tracker rewrites IDENT-to-TYPEDEF_NAME purely on token text;
    # in C, however, the identifier following ``struct`` / ``union`` /
    # ``enum`` lives in a separate name-space (the tag namespace) and
    # must NOT be rewritten even when it shadows a typedef of the
    # same name. ``typedef struct s s; struct s { ... };`` is the
    # canonical case. Pre-scan the source for every offset where an
    # IDENT sits in tag position — i.e. ``struct``/``union``/``enum``
    # followed only by whitespace then an identifier. The set is
    # consulted by the lexer-feedback filter; doing it as a single
    # regex pass sidesteps uplox's repeated filter calls during LR
    # reduction attempts (the offset is fixed in the source even when
    # the same token is re-presented).
    # Pre-scan: walk the lexed token stream once and compute which
    # IDENT offsets must NOT be rewritten to TYPEDEF_NAME.  Three
    # categories carry their own namespaces in C:
    #
    #   * tag namespace      — IDENT after ``struct``/``union``/``enum``
    #   * label namespace    — IDENT before ``:`` or after ``goto``
    #   * declarator names   — IDENT introduced by a non-typedef
    #     declaration in some enclosing scope; subsequent uses of the
    #     same name within that scope shadow any typedef of that name.
    #
    # The walk maintains a brace-depth stack of locally-declared
    # identifier names. For each IDENT use whose name lives in any
    # active stack frame, the offset is added to ``tag_idents`` so the
    # token filter leaves it as IDENT.
    tag_idents: set[int] = _compute_keep_ident_offsets(pre, scanner)

    def _filter(ctx, tok):
        if tok.name == "IDENT" and tok.offset in tag_idents:
            return tok  # tag namespace; do not rewrite
        return tracker.filter(ctx, tok)

    tree = _uplox_parse(
        table,
        scanner.scan(pre),
        hooks=hooks,
        token_filter=_filter,
    )
    if not isinstance(tree, ParseNode):
        raise RuntimeError("c23 parse returned non-ParseNode")
    return _convert_translation_unit(tree, filename)


_BUILTIN_TYPE_KW = frozenset({
    "KW_VOID", "KW_CHAR", "KW_SHORT", "KW_INT", "KW_LONG",
    "KW_FLOAT", "KW_DOUBLE", "KW_SIGNED", "KW_UNSIGNED",
    "KW_BOOL", "KW_COMPLEX", "KW_BITINT",
    "KW_DECIMAL32", "KW_DECIMAL64", "KW_DECIMAL128",
    # Type qualifiers / storage classes that legitimately precede a
    # declarator without changing the namespace of the trailing IDENT.
    "KW_CONST", "KW_VOLATILE", "KW_RESTRICT", "KW_ATOMIC",
    "KW_STATIC", "KW_EXTERN", "KW_AUTO", "KW_REGISTER",
    "KW_INLINE", "KW_NORETURN", "KW_THREAD_LOCAL",
    "KW_CONSTEXPR", "KW_TYPEDEF", "TYPEDEF_NAME",
})

_TAG_KW = frozenset({"KW_STRUCT", "KW_UNION", "KW_ENUM"})

_DECL_TERMINATORS = frozenset({
    "SEMI", "COMMA", "ASSIGN", "LBRACKET", "LPAREN", "RPAREN",
})


def _compute_keep_ident_offsets(pre: str, scanner) -> set[int]:
    """Pre-scan ``pre`` and return the set of IDENT offsets that the
    typedef filter must leave alone — tag/label/declarator names that
    shadow some typedef of the same text.

    The implementation is a single-pass walk over the lexed token
    stream maintaining a stack of brace-scoped declared names.
    """
    keep: set[int] = set()
    tokens = list(scanner.scan(pre))
    # Stack of locally-declared name sets, one per brace scope.
    scopes: list[set[str]] = [set()]
    n = len(tokens)
    i = 0

    def name_in_scopes(name: str) -> bool:
        return any(name in s for s in scopes)

    while i < n:
        tok = tokens[i]
        name = tok.name
        if name == "LBRACE":
            scopes.append(set())
            i += 1
            continue
        if name == "RBRACE":
            if len(scopes) > 1:
                scopes.pop()
            i += 1
            continue
        # ``goto IDENT`` — label use.
        if name == "KW_GOTO" and i + 1 < n and tokens[i + 1].name == "IDENT":
            keep.add(tokens[i + 1].offset)
            i += 2
            continue
        # ``IDENT :`` at statement-prefix position is a label
        # definition. The token preceding the IDENT will be SEMI,
        # LBRACE, RBRACE, or COLON (case label). Restrict to those
        # contexts to avoid stealing the ``a ? b : c`` ternary or
        # ``case 1:``.
        if (
            name == "IDENT"
            and i + 1 < n
            and tokens[i + 1].name == "COLON"
            and (
                i == 0
                or tokens[i - 1].name in {"SEMI", "LBRACE", "RBRACE", "COLON"}
            )
        ):
            keep.add(tok.offset)
            i += 1
            continue
        # ``struct|union|enum IDENT`` — tag. Mark the tag IDENT and
        # fall through so the same ``struct …`` token can also seed
        # the declaration walk below (``struct s s;``).
        if name in _TAG_KW and i + 1 < n and tokens[i + 1].name == "IDENT":
            keep.add(tokens[i + 1].offset)
        # Declaration: a run of decl-specifier tokens followed by one
        # or more declarators. We detect this by spotting a built-in
        # type / qualifier / storage class as the first token after a
        # statement separator, then scanning forward through tokens
        # capturing every IDENT that sits at declarator position
        # (i.e. immediately followed by SEMI/COMMA/ASSIGN/LBRACKET/
        # LPAREN/RPAREN). Any such IDENT shadows a typedef of the
        # same name within the current scope.
        if name in _BUILTIN_TYPE_KW or name in _TAG_KW:
            # Walk forward until SEMI at depth 0 (top-level
            # parentheses + braces). A ``{`` here may be either a
            # function body OR a struct/union/enum body — function
            # body terminates the declaration, but a brace right
            # after a tag-keyword/IDENT belongs to the type spec and
            # must be skipped. We distinguish by tracking whether the
            # most recent non-whitespace token before the brace was a
            # tag keyword, an IDENT directly following a tag keyword,
            # or an RBRACE (closing a nested struct body).
            paren_depth = 0
            brace_depth = 0
            j = i
            top_level_idents: list = []  # only at brace_depth 0
            inner_idents: list = []      # declarators inside tag bodies
            is_typedef = False
            while j < n:
                t = tokens[j]
                tn = t.name
                if tn == "LPAREN":
                    paren_depth += 1
                elif tn == "RPAREN":
                    paren_depth -= 1
                elif tn == "LBRACE":
                    prev = tokens[j - 1].name if j > 0 else None
                    prev_prev = tokens[j - 2].name if j > 1 else None
                    is_tag_body = (
                        prev in _TAG_KW
                        or (prev == "IDENT" and prev_prev in _TAG_KW)
                        or brace_depth > 0
                    )
                    if not is_tag_body and brace_depth == 0:
                        break
                    brace_depth += 1
                elif tn == "RBRACE":
                    brace_depth -= 1
                elif tn == "KW_TYPEDEF":
                    is_typedef = True
                elif (
                    paren_depth == 0
                    and brace_depth == 0
                    and tn == "SEMI"
                ):
                    break
                elif (
                    paren_depth == 0
                    and tn == "IDENT"
                    and j + 1 < n
                    and tokens[j + 1].name in _DECL_TERMINATORS
                ):
                    if brace_depth == 0:
                        top_level_idents.append(t)
                    else:
                        # Member declarator inside a struct/union body.
                        # Members live in a separate namespace, so the
                        # IDENT must not be rewritten to TYPEDEF_NAME
                        # when its text shadows a typedef.
                        inner_idents.append(t)
                # Tag IDENTs inside the body: ``struct INNER {`` —
                # mark INNER as tag, since uplox's typedef tracker
                # would otherwise rewrite the tag name when it
                # collides with an outer typedef.
                elif (
                    tn in _TAG_KW
                    and j + 1 < n
                    and tokens[j + 1].name == "IDENT"
                    and brace_depth > 0
                ):
                    keep.add(tokens[j + 1].offset)
                j += 1
            for ident_tok in inner_idents:
                keep.add(ident_tok.offset)
            # Typedef declarations introduce *typedef* names, not
            # ordinary-namespace shadows. Leave those alone — the
            # parse-time hook records them in the tracker after the
            # reduce, and subsequent uses parse as ``TYPEDEF_NAME``.
            if not is_typedef:
                for ident_tok in top_level_idents:
                    scopes[-1].add(ident_tok.text)
                    keep.add(ident_tok.offset)
            i = j  # resume at SEMI/LBRACE; the LBRACE branch above
                   # will push a new scope on the next iteration.
            continue
        if name == "IDENT" and name_in_scopes(tok.text):
            keep.add(tok.offset)
        i += 1
    return keep


# ---------------------------------------------------------------------------
# Attribute stripping
# ---------------------------------------------------------------------------

# C23 ``[[...]]`` attribute specifier. The grammar has DOUBLE_LBRACKET /
# DOUBLE_RBRACKET tokens declared but no rules using them; the simplest
# path is to strip the whole specifier from the source pre-parse. Nested
# brackets are not allowed inside C23 attributes (only balanced parens),
# so a non-greedy match against ``]]`` is safe.
_C23_ATTR_RE = re.compile(r"\[\[.*?\]\]", re.DOTALL)

# GCC ``__attribute__ ((...))`` — paren depth counted by hand because
# attribute argument lists nest parens (``aligned(8)`` etc.).
_GCC_ATTR_RE = re.compile(r"__attribute__\s*\(\s*\(", re.IGNORECASE)

# DOS-era keyword qualifiers (memory model, calling conventions,
# function attributes) that the legacy uc_core parser silently
# absorbed. Period MS-C / Borland / Watcom headers spray these
# everywhere and they have no meaning under flat-32 / Z80, so we
# erase them at the source level — the uplox grammar then doesn't
# have to model each variant.
_DOS_QUALIFIER_RE = re.compile(
    # Underscore-prefixed forms are always strippable — never used as
    # identifiers in real code, so no ambiguity. Bare forms (``near``,
    # ``far``, ``huge``, ``cdecl``, ``pascal``, ...) clash with valid
    # variable names; gate them on a positive lookahead for another
    # type-spec/declarator token (whitespace then identifier-start
    # char or ``*``) so ``int huge = 5;`` keeps ``huge`` intact.
    r"\b(?:"
    r"(?:__|_)(?:near|far|huge)"
    r"|(?:__|_)(?:cs|ds|es|ss|seg)"
    r"|(?:__|_)(?:cdecl|pascal|stdcall|fastcall|syscall|watcall|fortran)"
    r"|(?:__|_)(?:interrupt|loadds|saveregs|export)"
    r"|__extension__"
    r")\b"
    r"|"
    r"\b(?:near|far|huge|cdecl|pascal|stdcall|fastcall|syscall|watcall"
    r"|fortran|interrupt)\b(?=\s+[A-Za-z_*])"
)
# ``__based(seg)`` / ``__Seg16(...)`` — keyword followed by a
# parenthesised argument list to discard along with the keyword.
_DOS_PAREN_QUALIFIER_RE = re.compile(
    r"\b(?:__|_)?(?:based|Seg16)\s*\("
)


def _strip_attributes(source: str) -> str:
    """Strip C23 ``[[...]]`` and GCC ``__attribute__((...))`` attribute
    specifiers, plus DOS-era qualifier keywords (near/far/huge/__cdecl/
    etc.). All of these are erased from the token stream by the legacy
    front-end; we keep that contract here so the uplox grammar doesn't
    have to model them.

    Handles ``__attribute__((vector_size(N)))`` specially: rather than
    erasing it, rewrite the surrounding declarator into an equivalent
    ``T name[N/sizeof(T)]`` array shape so codegen can lay out the
    storage. uc386 backends key on the array element type and count;
    the ``is_vector`` flag legacy carried isn't reproduced here, but
    the storage layout is identical, which is what the smoke tests
    assert."""
    source = _rewrite_vector_size(source)
    source = _C23_ATTR_RE.sub("", source)
    source = _strip_dos_paren_qualifiers(source)
    source = _DOS_QUALIFIER_RE.sub("", source)
    out = []
    i = 0
    n = len(source)
    while i < n:
        m = _GCC_ATTR_RE.match(source, i)
        if not m:
            out.append(source[i])
            i += 1
            continue
        # Found `__attribute__((`. Walk forward counting parens until
        # we land on the matching `))`.
        depth = 2
        j = m.end()
        in_str = False
        in_chr = False
        while j < n and depth > 0:
            c = source[j]
            if in_str:
                if c == "\\" and j + 1 < n:
                    j += 2
                    continue
                if c == '"':
                    in_str = False
            elif in_chr:
                if c == "\\" and j + 1 < n:
                    j += 2
                    continue
                if c == "'":
                    in_chr = False
            elif c == '"':
                in_str = True
            elif c == "'":
                in_chr = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        i = j
    return "".join(out)


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
    r"\s*__attribute__\s*\(\s*\(\s*(?:__)?vector_size(?:__)?"
    r"\s*\(\s*(\d+)\s*\)\s*\)\s*\)"
)


def _rewrite_vector_size(source: str) -> str:
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
        # Record the declarator name so the converter can flip
        # ``is_vector=True`` on the resulting ArrayType.
        _vector_names.add(name)
        if count <= 0:
            return f"{type_part} {name}"
        return f"{type_part} {name}[{count}]"

    return _VECTOR_SIZE_RE.sub(repl, source)


def _strip_dos_paren_qualifiers(source: str) -> str:
    """Erase ``__based(seg)`` / ``__Seg16(...)`` style qualifiers,
    keyword and argument list together. The argument list nests
    parens, so we count them by hand."""
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


# ---------------------------------------------------------------------------
# ParseNode → uc_core AST conversion
# ---------------------------------------------------------------------------


def _kind(n: ParseNode | Token) -> str:
    return n.kind if isinstance(n, ParseNode) else n.name


def _is_token(n: ParseNode | Token, name: str | None = None) -> bool:
    if not isinstance(n, Token):
        return False
    return name is None or n.name == name


def _tok_text(n: ParseNode | Token) -> str:
    assert isinstance(n, Token), f"expected token, got {type(n).__name__}"
    return n.text


def _loc(node: ParseNode | Token, filename: str) -> SourceLocation | None:
    """Find the leftmost token in ``node`` and build a SourceLocation."""
    t = _first_token(node)
    if t is None:
        return None
    return SourceLocation(filename=filename, line=t.line, column=t.column)


def _first_token(node: ParseNode | Token) -> Token | None:
    if isinstance(node, Token):
        return node
    for c in node.children:
        t = _first_token(c)
        if t is not None:
            return t
    return None


# --- list flattening --------------------------------------------------------


def _flatten_list(node: ParseNode, list_kind: str, item_kind: str) -> list:
    """Flatten a left-recursive ``X : X Y | Y`` chain into a list of
    ``Y`` nodes in source order."""
    out: list = []
    def visit(n):
        if not isinstance(n, ParseNode):
            return
        if n.kind == list_kind:
            for c in n.children:
                visit(c)
            return
        if n.kind == item_kind:
            out.append(n)
            return
    visit(node)
    return out


def _flatten_alt_list(node: ParseNode, list_kind: str) -> list:
    """For productions like ``X : X SEP Y | Y``, return [Y, Y, ...] in
    source order. Filters by ``list_kind`` for the wrapper."""
    out: list = []
    def visit(n):
        if not isinstance(n, ParseNode):
            return
        if n.kind == list_kind:
            cs = n.children
            if len(cs) == 1:
                visit(cs[0])
                return
            # X SEP Y or X Y — recurse left, take rightmost meaningful child
            visit(cs[0])
            out.append(cs[-1])
            return
        out.append(n)
    visit(node)
    return out


# --- translation unit ------------------------------------------------------


# Per-parse mutable state. Held in a module-level dict reset at the
# start of each ``_convert_translation_unit`` call rather than threaded
# through every helper — passing a context object would touch every
# function in the file. Single-threaded use; not safe across threads
# without external locking.
_typedefs: dict[str, ast.TypeNode] = {}
_vector_names: set[str] = set()


def _convert_translation_unit(root: ParseNode, filename: str) -> ast.TranslationUnit:
    """translation_unit : top_items"""
    _typedefs.clear()
    items_node = root.children[0]
    items = _flatten_list(items_node, "top_items", "top_item")
    decls: list[ast.Declaration] = []
    for it in items:
        for d in _convert_top_item(it, filename):
            decls.append(d)
    if _vector_names:
        _mark_vector_arraytypes(decls)
    return ast.TranslationUnit(declarations=decls, location=_loc(root, filename))


def _mark_vector_arraytypes(decls: list[ast.Declaration]) -> None:
    """Set ``is_vector=True`` on the outer ArrayType of every decl
    whose name was rewritten from ``__attribute__((vector_size(N)))``.
    The rewrite happens in source-text form (we can't tell GCC vector
    arrays from plain C arrays from grammar alone), so we re-tag
    matching nodes here."""
    for d in decls:
        name = getattr(d, "name", None)
        if name is None or name not in _vector_names:
            continue
        ty = getattr(d, "var_type", None) or getattr(d, "target_type", None)
        if isinstance(ty, ast.ArrayType):
            ty.is_vector = True


def _convert_top_item(node: ParseNode, filename: str) -> list[ast.Declaration]:
    """top_item : function_definition | declaration | static_assert_declaration | asm_declaration"""
    inner = node.children[0]
    assert isinstance(inner, ParseNode)
    if inner.kind == "function_definition":
        return [_convert_function_definition(inner, filename)]
    if inner.kind == "declaration":
        return _convert_declaration(inner, filename)
    if inner.kind == "static_assert_declaration":
        # uc_core's AST has no static_assert node — drop. Real assertion
        # would happen at semantic analysis; uc_core doesn't enforce.
        return []
    if inner.kind == "asm_declaration":
        # File-scope `asm("...")` — rare; drop for now.
        return []
    raise RuntimeError(f"unexpected top_item: {inner.kind}")


# --- function definition ---------------------------------------------------


def _convert_function_definition(
    node: ParseNode, filename: str
) -> ast.FunctionDecl:
    """function_definition : decl_specs declarator compound_stmt"""
    decl_specs = node.children[0]
    declarator = node.children[1]
    body_node = node.children[2]

    base_type, storage, is_inline = _convert_decl_specs(decl_specs, filename)
    name, full_type = _apply_declarator(declarator, base_type, filename)

    # The declarator must produce a function type.
    if not isinstance(full_type, ast.FunctionType):
        raise RuntimeError(f"function definition's declarator is not a function type")

    body = _convert_compound_stmt(body_node, filename)

    # Convert FunctionType.param_types back to ParamDecls. The declarator
    # walk records names + side effects when it builds the FunctionType;
    # we pull both back out via the legacy ParamDecl wrapper.
    params: list[ast.ParamDecl] = full_type.__dict__.pop("__uc_core_params", [])

    return ast.FunctionDecl(
        name=name,
        return_type=full_type.return_type,
        params=params,
        body=body,
        is_variadic=full_type.is_variadic,
        storage_class=storage,
        is_inline=is_inline,
        location=_loc(node, filename),
    )


# --- declaration specifiers ------------------------------------------------


def _convert_decl_specs(
    node: ParseNode, filename: str
) -> tuple[ast.TypeNode, str | None, bool]:
    """decl_specs : decl_specs decl_spec | decl_spec

    Returns ``(base_type, storage_class, is_inline)``. The base type is
    a TypeNode without pointer / array / function shape; declarator
    walking layers those on top.
    """
    specs = _flatten_list(node, "decl_specs", "decl_spec")

    storage: str | None = None
    is_inline = False
    is_const = False
    is_volatile = False
    is_signed: bool | None = None
    is_unsigned = False
    type_words: list[str] = []  # accumulated type-spec keywords
    # Special non-string type-specs (struct/union/enum/typedef-name/typeof)
    type_node: ast.TypeNode | None = None

    for sp in specs:
        assert isinstance(sp, ParseNode) and sp.kind == "decl_spec"
        inner = sp.children[0]
        if not isinstance(inner, ParseNode):
            continue
        if inner.kind == "storage_class":
            sc_kind = _kind(inner.children[0])
            if sc_kind == "KW_STATIC":
                storage = "static"
            elif sc_kind == "KW_EXTERN":
                storage = "extern"
            elif sc_kind == "KW_REGISTER":
                storage = "register"
            elif sc_kind == "KW_AUTO":
                storage = "auto"
            elif sc_kind == "KW_TYPEDEF":
                storage = "typedef"
            elif sc_kind == "KW_INLINE":
                is_inline = True
            elif sc_kind == "KW_CONSTEXPR":
                storage = "constexpr"
            elif sc_kind == "KW_THREAD_LOCAL":
                # Storage class — uc_core's VarDecl.storage_class is a
                # single string. Combine with whatever else is set, or
                # use as primary if alone.
                storage = "thread_local" if storage is None else f"{storage} thread_local"
            elif sc_kind == "KW_NORETURN":
                pass  # uc_core doesn't carry this flag separately
            elif sc_kind == "alignment_specifier":
                pass  # alignment captured separately, not stored here
            continue
        if inner.kind == "type_qualifier":
            tq_kind = _kind(inner.children[0])
            if tq_kind == "KW_CONST":
                is_const = True
            elif tq_kind == "KW_VOLATILE":
                is_volatile = True
            # KW_RESTRICT / KW_ATOMIC: accepted but not stored
            continue
        if inner.kind == "type_spec":
            ts_inner = inner.children[0]
            ts_kind = _kind(ts_inner)
            if ts_kind in (
                "KW_VOID", "KW_CHAR", "KW_SHORT", "KW_INT", "KW_LONG",
                "KW_FLOAT", "KW_DOUBLE", "KW_BOOL",
            ):
                type_words.append(_tok_text(ts_inner).lower())
                # Normalise: "_Bool" → "bool", upper variants → lower
                if type_words[-1] == "_bool":
                    type_words[-1] = "bool"
                continue
            if ts_kind == "KW_SIGNED":
                is_signed = True
                continue
            if ts_kind == "KW_UNSIGNED":
                is_unsigned = True
                continue
            if ts_kind == "KW_INT128":
                # GCC built-in 128-bit integer extension. uc386 codegen
                # keys on the canonical name ``int128`` (no underscores)
                # to pick a 16-byte slot; ``__uint128_t`` carries
                # implicit unsignedness.
                tname = _tok_text(ts_inner)
                if tname == "__uint128_t":
                    is_unsigned = True
                type_words.append("int128")
                continue
            if ts_kind == "KW_COMPLEX":
                type_words.append("_Complex")
                continue
            if ts_kind == "KW_IMAGINARY":
                type_words.append("_Imaginary")
                continue
            if ts_kind in ("KW_DECIMAL32", "KW_DECIMAL64", "KW_DECIMAL128"):
                # uc_core doesn't model decimal float types; map to
                # the closest binary float so codegen can lay them out.
                type_words.append("float" if ts_kind == "KW_DECIMAL32" else "double")
                continue
            if ts_kind == "KW_BITINT":
                # _BitInt(N) — uc_core represents as BasicType('_BitInt')
                # with N captured via... a hack on the node. Simplify
                # for now: just record the keyword.
                type_words.append("_BitInt")
                continue
            if ts_kind == "KW_TYPEOF" or ts_kind == "KW_TYPEOF_UNQUAL":
                # KW_TYPEOF LPAREN expr RPAREN
                operand = _convert_expr(inner.children[2], filename)
                type_node = ast.TypeofType(operand=operand, location=_loc(inner, filename))
                continue
            if ts_kind == "struct_or_union_specifier":
                type_node = _convert_struct_or_union(ts_inner, filename)
                continue
            if ts_kind == "enum_specifier":
                type_node = _convert_enum(ts_inner, filename)
                continue
            if ts_kind == "TYPEDEF_NAME":
                # Substitute the recorded target type so downstream code
                # sees the underlying StructType / EnumType / BasicType
                # (legacy uc_core parser inlined typedefs at parse time).
                tname = _tok_text(ts_inner)
                resolved = _typedefs.get(tname)
                if resolved is not None:
                    type_node = resolved
                else:
                    # Typedef seen by the lexer-feedback hook but not yet
                    # recorded by the converter (e.g. seeded externally
                    # via ``seed_typedefs``); fall back to a BasicType.
                    type_node = ast.BasicType(
                        name=tname, location=_loc(inner, filename)
                    )
                continue
            raise RuntimeError(f"unexpected type_spec inner: {ts_kind}")

    # Build the base type. A non-string spec (struct/union/enum/typedef)
    # wins; otherwise we combine the accumulated ``type_words`` with
    # signed-/unsigned-ness into a BasicType.
    if type_node is None:
        if "_Complex" in type_words:
            # `_Complex T` becomes ComplexType, not BasicType.
            base_words = [w for w in type_words if w != "_Complex"]
            if not base_words:
                base_words = ["double"]  # `_Complex` alone == _Complex double
            base_name = " ".join(base_words).replace(" int", "").strip() or "double"
            type_node = ast.ComplexType(
                base_type=base_name,
                is_const=is_const,
                is_volatile=is_volatile,
                location=_loc(node, filename),
            )
            return type_node, storage, is_inline
        if not type_words and (is_signed is not None or is_unsigned):
            type_words = ["int"]  # `signed` / `unsigned` alone == int
        if not type_words:
            type_words = ["int"]  # implicit int — legacy K&R
        # Canonicalise: collapse "long long" / "long int" / etc into
        # the form uc_core's BasicType expects.
        name = " ".join(type_words)
        # Normalise "long int" -> "long", "short int" -> "short", "long
        # long int" -> "long long", "long double" stays.
        name = (
            name.replace(" int", "")
            .replace("long long", "long long")
            .strip()
        )
        if name == "":
            name = "int"
        signedness: bool | None = None
        if is_unsigned:
            signedness = False
        elif is_signed:
            signedness = True
        type_node = ast.BasicType(
            name=name,
            is_signed=signedness,
            is_const=is_const,
            is_volatile=is_volatile,
            location=_loc(node, filename),
        )
    else:
        # Apply qualifiers to the typedef / struct / enum / typeof type
        # if it accepts them.
        if hasattr(type_node, "is_const"):
            type_node.is_const = type_node.is_const or is_const
        if hasattr(type_node, "is_volatile"):
            type_node.is_volatile = type_node.is_volatile or is_volatile
    return type_node, storage, is_inline


# --- struct / union --------------------------------------------------------


def _convert_struct_or_union(node: ParseNode, filename: str) -> ast.StructType:
    """struct_or_union_specifier :
    | struct_or_union IDENT LBRACE struct_decl_list RBRACE
    | struct_or_union LBRACE struct_decl_list RBRACE
    | struct_or_union IDENT
    """
    cs = node.children
    su_node = cs[0]
    is_union = _kind(su_node.children[0]) == "KW_UNION"
    name: str | None = None
    members: list[ast.StructMember] = []
    if len(cs) == 2:
        # struct_or_union IDENT — just a tag reference.
        name = _tok_text(cs[1])
    elif len(cs) == 3:
        # struct_or_union LBRACE RBRACE — anonymous, empty body (C23).
        pass
    elif len(cs) == 4:
        if _is_token(cs[2], "RBRACE"):
            # struct_or_union IDENT LBRACE RBRACE — empty body.
            name = _tok_text(cs[1])
        else:
            # struct_or_union LBRACE struct_decl_list RBRACE — anonymous
            members = _convert_struct_decl_list(cs[2], filename)
    elif len(cs) == 5:
        # struct_or_union IDENT LBRACE struct_decl_list RBRACE
        name = _tok_text(cs[1])
        members = _convert_struct_decl_list(cs[3], filename)
    return ast.StructType(
        name=name,
        is_union=is_union,
        members=members,
        location=_loc(node, filename),
    )


def _convert_struct_decl_list(
    node: ParseNode, filename: str
) -> list[ast.StructMember]:
    out: list[ast.StructMember] = []
    decls = _flatten_list(node, "struct_decl_list", "struct_decl")
    for sd in decls:
        assert isinstance(sd, ParseNode) and sd.kind == "struct_decl"
        cs = sd.children
        inner = cs[0]
        if isinstance(inner, ParseNode) and inner.kind == "static_assert_declaration":
            continue  # ignore static_assert in struct bodies
        base_type, _, _ = _convert_decl_specs(cs[0], filename)
        # Two forms: decl_specs SEMI (anonymous struct/union member) or
        # decl_specs struct_declarator_list SEMI (one or more named or
        # bit-field members).
        if len(cs) == 2 and _is_token(cs[1], "SEMI"):
            # C11 anonymous struct/union member — single member with
            # name=None whose type is the (anonymous) struct/union.
            out.append(
                ast.StructMember(
                    name=None,
                    member_type=base_type,
                    location=_loc(sd, filename),
                )
            )
            continue
        sdl = cs[1]
        for sdecl in _flatten_list(sdl, "struct_declarator_list", "struct_declarator"):
            out.append(_convert_struct_declarator(sdecl, base_type, filename))
    return out


def _convert_struct_declarator(
    node: ParseNode, base_type: ast.TypeNode, filename: str
) -> ast.StructMember:
    """struct_declarator : declarator | declarator COLON conditional_expr | COLON conditional_expr"""
    cs = node.children
    if _is_token(cs[0], "COLON"):
        # Anonymous bit-field
        bw = _convert_expr(cs[1], filename)
        return ast.StructMember(
            name=None, member_type=base_type, bit_width=bw, location=_loc(node, filename),
        )
    name, member_type = _apply_declarator(cs[0], base_type, filename)
    bit_width: ast.Expression | None = None
    if len(cs) == 3 and _is_token(cs[1], "COLON"):
        bit_width = _convert_expr(cs[2], filename)
    return ast.StructMember(
        name=name,
        member_type=member_type,
        bit_width=bit_width,
        location=_loc(node, filename),
    )


# --- enum ------------------------------------------------------------------


def _convert_enum(node: ParseNode, filename: str) -> ast.EnumType:
    """enum_specifier : KW_ENUM [IDENT] LBRACE enumerator_list [COMMA] RBRACE | KW_ENUM IDENT"""
    cs = node.children
    name: str | None = None
    values: list[ast.EnumValue] = []
    if len(cs) == 2:
        name = _tok_text(cs[1])
    else:
        idx = 1
        if _is_token(cs[1], "IDENT"):
            name = _tok_text(cs[1])
            idx = 2
        # cs[idx] is LBRACE; cs[idx+1] is enumerator_list
        en_list = cs[idx + 1]
        for en in _flatten_list(en_list, "enumerator_list", "enumerator"):
            values.append(_convert_enumerator(en, filename))
    return ast.EnumType(name=name, values=values, location=_loc(node, filename))


def _convert_enumerator(node: ParseNode, filename: str) -> ast.EnumValue:
    """enumerator : IDENT | IDENT ASSIGN conditional_expr"""
    cs = node.children
    name = _tok_text(cs[0])
    value: ast.Expression | None = None
    if len(cs) >= 3:
        value = _convert_expr(cs[2], filename)
    return ast.EnumValue(name=name, value=value, location=_loc(node, filename))


# --- declaration -----------------------------------------------------------


def _convert_declaration(
    node: ParseNode, filename: str
) -> list[ast.Declaration]:
    """declaration : decl_specs init_declarator_list_opt SEMI"""
    decl_specs = node.children[0]
    idl_opt = node.children[1]
    base_type, storage, is_inline = _convert_decl_specs(decl_specs, filename)

    # Empty init declarator: a tag-only declaration (`struct foo;`).
    assert isinstance(idl_opt, ParseNode) and idl_opt.kind == "init_declarator_list_opt"
    if not idl_opt.children:
        return _tag_declaration(base_type, filename)

    # If decl_specs introduced a *named, complete* tag (struct/union/
    # enum with body), emit a separate tag declaration alongside the
    # per-declarator nodes. Codegen needs the tag in its registry so
    # later references like ``struct s s;`` can size the type.
    out: list[ast.Declaration] = []
    if (
        isinstance(base_type, ast.StructType)
        and base_type.name is not None
        and base_type.members
    ):
        out.append(ast.StructDecl(
            name=base_type.name,
            members=base_type.members,
            is_union=base_type.is_union,
            is_definition=True,
            is_packed=base_type.is_packed,
            location=base_type.location,
        ))
    elif (
        isinstance(base_type, ast.EnumType)
        and base_type.name is not None
        and base_type.values
    ):
        out.append(ast.EnumDecl(
            name=base_type.name,
            values=base_type.values,
            is_definition=True,
            location=base_type.location,
        ))

    # Multiple declarators share the base type but get individual
    # declarator-derived shapes / names / inits.
    idl = idl_opt.children[0]
    items = _flatten_list(idl, "init_declarator_list", "init_declarator")
    for it in items:
        d = _convert_init_declarator(it, base_type, storage, is_inline, filename)
        if d is not None:
            out.append(d)
    if storage == "typedef":
        # For typedef, the converter already produced TypedefDecl nodes.
        if len(out) == 1:
            return out
        return [ast.DeclarationList(declarations=out, location=_loc(node, filename))]
    if len(out) > 1:
        return [ast.DeclarationList(declarations=out, location=_loc(node, filename))]
    return out


def _tag_declaration(
    base_type: ast.TypeNode, filename: str
) -> list[ast.Declaration]:
    """``struct foo;`` / ``enum bar;`` — emit a forward / definition node."""
    if isinstance(base_type, ast.StructType):
        return [
            ast.StructDecl(
                name=base_type.name,
                members=base_type.members,
                is_union=base_type.is_union,
                is_definition=bool(base_type.members),
                is_packed=base_type.is_packed,
                location=base_type.location,
            )
        ]
    if isinstance(base_type, ast.EnumType):
        return [
            ast.EnumDecl(
                name=base_type.name,
                values=base_type.values,
                is_definition=bool(base_type.values),
                location=base_type.location,
            )
        ]
    return []  # bare type-spec without declarator and not a tag — drop


def _convert_init_declarator(
    node: ParseNode,
    base_type: ast.TypeNode,
    storage: str | None,
    is_inline: bool,
    filename: str,
) -> ast.Declaration | None:
    """init_declarator : declarator | declarator ASSIGN initializer"""
    cs = node.children
    name, full_type = _apply_declarator(cs[0], base_type, filename)
    init: ast.Expression | None = None
    if len(cs) >= 3:
        init = _convert_initializer(cs[2], filename)

    if storage == "typedef":
        # Record so subsequent references to ``name`` substitute the
        # target type (legacy uc_core parser did this in its
        # ``self.typedefs`` table; codegen relies on it for things like
        # ``typedef struct { … } S; S v;`` where ``v.var_type`` must be
        # the StructType, not a BasicType wrapper).
        _typedefs[name] = full_type
        return ast.TypedefDecl(
            name=name, target_type=full_type, location=_loc(node, filename)
        )

    if isinstance(full_type, ast.FunctionType):
        # Function *declarations* (no body, no definition) are emitted
        # as ``VarDecl(FunctionType)`` to mirror the legacy uc_core
        # parser. uc80's codegen relies on the distinction: only real
        # ``FunctionDecl`` nodes get added to ``ctx.function_names`` in
        # its first pass, so wrapping prototypes as ``VarDecl`` keeps
        # the per-prototype ``extrn`` emission behaviour.
        full_type.__dict__.pop("__uc_core_params", None)
        return ast.VarDecl(
            name=name,
            var_type=full_type,
            init=None,
            storage_class=storage,
            location=_loc(node, filename),
        )

    return ast.VarDecl(
        name=name,
        var_type=full_type,
        init=init,
        storage_class=storage,
        location=_loc(node, filename),
    )


# --- declarator walk -------------------------------------------------------


def _apply_declarator(
    node: ParseNode | Token, base_type: ast.TypeNode, filename: str
) -> tuple[str, ast.TypeNode]:
    """Walk ``declarator`` (or ``direct_declarator``) and return the
    name + the type built up from ``base_type``.

    The grammar's left-recursive declarator structure means we walk
    inside-out: pointers are outermost in the source but innermost in
    the type tree (a pointer-to-array-of-int is built by wrapping
    ``int`` in ``ArrayType`` first then in ``PointerType``).
    """
    if not isinstance(node, ParseNode):
        raise RuntimeError(f"_apply_declarator on token: {node}")
    if node.kind == "declarator":
        cs = node.children
        if len(cs) == 2:
            # pointer direct_declarator
            inner_type = _apply_pointer(cs[0], base_type)
            return _apply_declarator(cs[1], inner_type, filename)
        # direct_declarator alone
        return _apply_declarator(cs[0], base_type, filename)
    if node.kind == "direct_declarator":
        return _apply_direct_declarator(node, base_type, filename)
    raise RuntimeError(f"unexpected declarator kind: {node.kind}")


def _apply_pointer(node: ParseNode, base_type: ast.TypeNode) -> ast.TypeNode:
    """pointer : pointer STAR type_qualifier_list_opt | STAR type_qualifier_list_opt

    Each STAR adds a pointer layer; qualifiers attach to that layer.
    """
    cs = node.children
    if _is_token(cs[0], "STAR"):
        # STAR type_qualifier_list_opt
        is_const, is_volatile = _qualifier_flags(cs[1])
        return ast.PointerType(
            base_type=base_type,
            is_const=is_const,
            is_volatile=is_volatile,
        )
    # pointer STAR type_qualifier_list_opt
    inner = _apply_pointer(cast(ParseNode, cs[0]), base_type)
    is_const, is_volatile = _qualifier_flags(cs[2])
    return ast.PointerType(
        base_type=inner, is_const=is_const, is_volatile=is_volatile
    )


def _qualifier_flags(node: ParseNode | Token) -> tuple[bool, bool]:
    is_const = False
    is_volatile = False
    if isinstance(node, ParseNode) and node.kind == "type_qualifier_list_opt":
        for c in node.children:
            for q in _flatten_list(c, "type_qualifier_list", "type_qualifier"):
                kn = _kind(q.children[0])
                if kn == "KW_CONST":
                    is_const = True
                elif kn == "KW_VOLATILE":
                    is_volatile = True
    return is_const, is_volatile


def _apply_direct_declarator(
    node: ParseNode, base_type: ast.TypeNode, filename: str
) -> tuple[str, ast.TypeNode]:
    """direct_declarator :
    | IDENT
    | LPAREN declarator RPAREN
    | direct_declarator LBRACKET conditional_expr RBRACKET
    | direct_declarator LBRACKET RBRACKET
    | direct_declarator LPAREN parameter_type_list RPAREN
    | direct_declarator LPAREN RPAREN
    """
    cs = node.children
    if len(cs) == 1 and _is_token(cs[0], "IDENT"):
        return _tok_text(cs[0]), base_type
    if _is_token(cs[0], "LPAREN") and isinstance(cs[1], ParseNode) and cs[1].kind == "declarator":
        # Parenthesised declarator — just descends.
        return _apply_declarator(cs[1], base_type, filename)
    # left-recursive forms
    if isinstance(cs[0], ParseNode) and cs[0].kind == "direct_declarator":
        if _is_token(cs[1], "LBRACKET"):
            # ``LBRACKET type_qualifier_list_opt [conditional_expr] RBRACKET``
            # — qualifiers (``int x[const 5]``) and size are both
            # optional. We carry size through; qualifiers on the array
            # bound itself are dropped (uc_core's ArrayType doesn't
            # model them — they apply to the implied pointer in
            # function parameters).
            size: ast.Expression | None = None
            for child in cs[2:-1]:
                if isinstance(child, ParseNode) and child.kind != "type_qualifier_list_opt":
                    size = _convert_expr(child, filename)
                    break
            inner_type = ast.ArrayType(base_type=base_type, size=size)
            return _apply_direct_declarator(cs[0], inner_type, filename)
        if _is_token(cs[1], "LPAREN"):
            # function
            params, is_variadic = _convert_param_type_list_opt(cs, filename)
            ftype = ast.FunctionType(
                return_type=base_type,
                param_types=[p.param_type for p in params],
                is_variadic=is_variadic,
            )
            ftype.__dict__["__uc_core_params"] = params
            return _apply_direct_declarator(cs[0], ftype, filename)
    raise RuntimeError(
        f"unexpected direct_declarator shape: {[_kind(c) for c in cs]}"
    )


def _convert_param_type_list_opt(
    cs: list, filename: str
) -> tuple[list[ast.ParamDecl], bool]:
    """Parse the parameters from a direct_declarator's
    ``LPAREN ... RPAREN`` suffix. Caller passes the full child list of
    the declarator-suffix production."""
    if len(cs) == 3:
        # LPAREN RPAREN — no parameters declared (`f()` — old K&R, treat
        # as variadic-empty per uc_core convention).
        return [], False
    # LPAREN parameter_type_list RPAREN
    ptl = cs[2]
    return _convert_parameter_type_list(ptl, filename)


def _convert_parameter_type_list(
    node: ParseNode, filename: str
) -> tuple[list[ast.ParamDecl], bool]:
    """parameter_type_list : parameter_list | parameter_list COMMA ELLIPSIS"""
    cs = node.children
    is_variadic = len(cs) >= 3 and _is_token(cs[-1], "ELLIPSIS")
    plist = cs[0]
    params: list[ast.ParamDecl] = []
    for p in _flatten_list(plist, "parameter_list", "parameter"):
        params.append(_convert_parameter(p, filename))
    # ``f(void)`` — single anonymous void parameter — is the C
    # spelling for ``no parameters``; collapse it so codegen sees
    # the same shape as ``f()``.
    if (
        len(params) == 1
        and not is_variadic
        and params[0].name is None
        and isinstance(params[0].param_type, ast.BasicType)
        and params[0].param_type.name == "void"
    ):
        params = []
    return params, is_variadic


def _convert_parameter(node: ParseNode, filename: str) -> ast.ParamDecl:
    """parameter : decl_specs declarator | decl_specs"""
    cs = node.children
    base_type, _, _ = _convert_decl_specs(cs[0], filename)
    name: str | None = None
    ptype = base_type
    if len(cs) == 2:
        # Could be a regular declarator or an abstract declarator; the
        # grammar lumps both into <declarator> here, so we walk and
        # treat a missing IDENT as anonymous.
        try:
            name, ptype = _apply_declarator(cs[1], base_type, filename)
        except RuntimeError:
            ptype = _apply_abstract_declarator(cs[1], base_type, filename)
    # C parameter type decay: T[N] → T*, T() → T(*)().
    if isinstance(ptype, ast.ArrayType):
        ptype = ast.PointerType(base_type=ptype.base_type)
    elif isinstance(ptype, ast.FunctionType):
        ptype = ast.PointerType(base_type=ptype)
    return ast.ParamDecl(
        name=name, param_type=ptype, location=_loc(node, filename)
    )


def _apply_abstract_declarator(
    node: ParseNode | Token, base_type: ast.TypeNode, filename: str
) -> ast.TypeNode:
    """Walk an abstract_declarator (or declarator with no IDENT) and
    return the wrapped type."""
    if not isinstance(node, ParseNode):
        return base_type
    if node.kind == "abstract_declarator":
        cs = node.children
        if len(cs) == 1:
            inner = cs[0]
            if isinstance(inner, ParseNode) and inner.kind == "pointer":
                return _apply_pointer(inner, base_type)
            return _apply_direct_abstract_declarator(inner, base_type, filename)
        # pointer direct_abstract_declarator
        ptr_type = _apply_pointer(cs[0], base_type)
        return _apply_direct_abstract_declarator(cs[1], ptr_type, filename)
    if node.kind == "direct_abstract_declarator":
        return _apply_direct_abstract_declarator(node, base_type, filename)
    return base_type


def _apply_direct_abstract_declarator(
    node: ParseNode | Token, base_type: ast.TypeNode, filename: str
) -> ast.TypeNode:
    if not isinstance(node, ParseNode):
        return base_type
    cs = node.children
    if _is_token(cs[0], "LPAREN") and isinstance(cs[1], ParseNode) and cs[1].kind == "abstract_declarator":
        return _apply_abstract_declarator(cs[1], base_type, filename)
    if isinstance(cs[0], ParseNode) and cs[0].kind == "direct_abstract_declarator":
        # left-recursive: apply outer-level decoration to inner
        if _is_token(cs[1], "LBRACKET"):
            size: ast.Expression | None = None
            for child in cs[2:-1]:
                if isinstance(child, ParseNode) and child.kind != "type_qualifier_list_opt":
                    size = _convert_expr(child, filename)
                    break
            inner_type = ast.ArrayType(base_type=base_type, size=size)
            return _apply_direct_abstract_declarator(cs[0], inner_type, filename)
        if _is_token(cs[1], "LPAREN"):
            params, is_variadic = _convert_param_type_list_opt(cs, filename)
            ftype = ast.FunctionType(
                return_type=base_type,
                param_types=[p.param_type for p in params],
                is_variadic=is_variadic,
            )
            return _apply_direct_abstract_declarator(cs[0], ftype, filename)
    # Leading [...] / (...) without a leading direct_abstract_declarator
    if _is_token(cs[0], "LBRACKET"):
        size: ast.Expression | None = None
        for child in cs[1:-1]:
            if isinstance(child, ParseNode) and child.kind != "type_qualifier_list_opt":
                size = _convert_expr(child, filename)
                break
        return ast.ArrayType(base_type=base_type, size=size)
    if _is_token(cs[0], "LPAREN"):
        params, is_variadic = _convert_param_type_list_opt(
            [cs[0]] + list(cs), filename  # adjust shape
        )
        return ast.FunctionType(
            return_type=base_type,
            param_types=[p.param_type for p in params],
            is_variadic=is_variadic,
        )
    return base_type


# --- type_name (for casts, sizeof, _Generic, etc) --------------------------


def _convert_type_name(node: ParseNode, filename: str) -> ast.TypeNode:
    """type_name : decl_specs | decl_specs abstract_declarator"""
    cs = node.children
    base_type, _, _ = _convert_decl_specs(cs[0], filename)
    if len(cs) == 2:
        return _apply_abstract_declarator(cs[1], base_type, filename)
    return base_type


# --- initializer -----------------------------------------------------------


def _convert_initializer(node: ParseNode, filename: str) -> ast.Expression:
    """initializer : assignment_expr | LBRACE initializer_list [COMMA] RBRACE | LBRACE RBRACE"""
    cs = node.children
    if len(cs) == 1:
        return _convert_expr(cs[0], filename)
    # Brace-enclosed initializer list (possibly empty: `LBRACE RBRACE`).
    if len(cs) == 2:
        return ast.InitializerList(values=[], location=_loc(node, filename))
    items_node = cs[1]
    values: list = []
    for di in _flatten_list(items_node, "initializer_list", "designated_initializer"):
        values.append(_convert_designated_initializer(di, filename))
    return ast.InitializerList(values=values, location=_loc(node, filename))


def _convert_designated_initializer(node: ParseNode, filename: str):
    """designated_initializer : initializer | designation initializer"""
    cs = node.children
    if len(cs) == 1:
        return _convert_initializer(cs[0], filename)
    # designation initializer
    designators = _convert_designation(cs[0], filename)
    value = _convert_initializer(cs[1], filename)
    return ast.DesignatedInit(
        designators=designators, value=value, location=_loc(node, filename)
    )


def _convert_designation(node: ParseNode, filename: str) -> list:
    """designation : designator_list ASSIGN"""
    out: list = []
    dl = node.children[0]
    for d in _flatten_list(dl, "designator_list", "designator"):
        cs = d.children
        if _is_token(cs[0], "DOT"):
            out.append(_tok_text(cs[1]))
        elif len(cs) == 5:
            # LBRACKET expr ELLIPSIS expr RBRACKET — GCC range designator.
            out.append(ast.RangeDesignator(
                start=_convert_expr(cs[1], filename),
                end=_convert_expr(cs[3], filename),
                location=_loc(d, filename),
            ))
        else:  # LBRACKET expr RBRACKET
            out.append(_convert_expr(cs[1], filename))
    return out


# --- statements ------------------------------------------------------------


def _convert_compound_stmt(node: ParseNode, filename: str) -> ast.CompoundStmt:
    """compound_stmt : LBRACE block_items RBRACE"""
    items_node = node.children[1]
    items: list = []
    for bi in _flatten_list(items_node, "block_items", "block_item"):
        items.extend(_convert_block_item(bi, filename))
    return ast.CompoundStmt(items=items, location=_loc(node, filename))


def _convert_block_item(node: ParseNode, filename: str) -> list:
    """block_item : declaration | stmt | static_assert_declaration | asm_declaration"""
    inner = node.children[0]
    assert isinstance(inner, ParseNode)
    if inner.kind == "declaration":
        return _convert_declaration(inner, filename)
    if inner.kind == "stmt":
        s = _convert_stmt(inner, filename)
        # Splice trailing declarations stashed by ``label: declaration``
        # so they land in the enclosing block's scope.
        trailing = getattr(s, "__uc_core_trailing_decls", None) or s.__dict__.pop(
            "__uc_core_trailing_decls", None
        ) if hasattr(s, "__dict__") else None
        if trailing:
            return [s] + list(trailing)
        return [s]
    return []  # static_assert / asm — drop


def _convert_stmt(node: ParseNode, filename: str) -> ast.Statement:
    """stmt : matched_stmt | unmatched_stmt"""
    return _convert_matched_or_unmatched(node.children[0], filename)


def _convert_matched_or_unmatched(
    node: ParseNode | Token, filename: str
) -> ast.Statement:
    assert isinstance(node, ParseNode)
    if node.kind == "stmt":
        return _convert_matched_or_unmatched(node.children[0], filename)
    cs = node.children
    if node.kind in ("matched_stmt", "unmatched_stmt"):
        if len(cs) == 1:
            return _convert_stmt_inner(cs[0], filename)
        # IDENT COLON stmt | KW_CASE expr COLON stmt | KW_DEFAULT COLON stmt
        if _is_token(cs[0], "IDENT"):
            label = _tok_text(cs[0])
            tail = cs[2]
            if isinstance(tail, ParseNode) and tail.kind == "declaration":
                # C23: ``label: declaration``. The declaration must
                # remain in the enclosing block's scope, not in a
                # newly introduced one. Stash the converted decls on
                # the LabelStmt so the surrounding block-items walk
                # can splice them in after the label.
                decls = _convert_declaration(tail, filename)
                stmt = ast.LabelStmt(
                    label=label,
                    stmt=ast.ExpressionStmt(expr=None, location=_loc(node, filename)),
                    location=_loc(node, filename),
                )
                stmt.__dict__["__uc_core_trailing_decls"] = list(decls)
                return stmt
            inner = _convert_matched_or_unmatched(tail, filename)
            return ast.LabelStmt(label=label, stmt=inner, location=_loc(node, filename))
        if _is_token(cs[0], "KW_CASE"):
            value = _convert_expr(cs[1], filename)
            inner = _convert_matched_or_unmatched(cs[3], filename)
            return ast.CaseStmt(value=value, stmt=inner, location=_loc(node, filename))
        if _is_token(cs[0], "KW_DEFAULT"):
            inner = _convert_matched_or_unmatched(cs[2], filename)
            return ast.CaseStmt(value=None, stmt=inner, location=_loc(node, filename))
    return _convert_stmt_inner(node, filename)


def _convert_stmt_inner(node: ParseNode, filename: str) -> ast.Statement:
    if node.kind == "matched_if":
        # KW_IF LPAREN expr RPAREN matched_stmt KW_ELSE matched_stmt
        cs = node.children
        cond = _convert_expr(cs[2], filename)
        then_b = _convert_matched_or_unmatched(cs[4], filename)
        else_b = _convert_matched_or_unmatched(cs[6], filename)
        return ast.IfStmt(condition=cond, then_branch=then_b, else_branch=else_b, location=_loc(node, filename))
    if node.kind == "unmatched_if":
        cs = node.children
        cond = _convert_expr(cs[2], filename)
        then_b = _convert_matched_or_unmatched(cs[4], filename)
        else_b: ast.Statement | None = None
        if len(cs) > 5:
            else_b = _convert_matched_or_unmatched(cs[6], filename)
        return ast.IfStmt(condition=cond, then_branch=then_b, else_branch=else_b, location=_loc(node, filename))
    if node.kind in ("matched_iteration_stmt", "unmatched_iteration_stmt"):
        cs = node.children
        if _is_token(cs[0], "KW_WHILE"):
            cond = _convert_expr(cs[2], filename)
            body = _convert_matched_or_unmatched(cs[4], filename)
            return ast.WhileStmt(condition=cond, body=body, location=_loc(node, filename))
        if _is_token(cs[0], "KW_DO"):
            body = _convert_matched_or_unmatched(cs[1], filename)
            cond = _convert_expr(cs[4], filename)
            return ast.DoWhileStmt(body=body, condition=cond, location=_loc(node, filename))
        if _is_token(cs[0], "KW_FOR"):
            # KW_FOR LPAREN for_init for_cond_opt SEMI for_step_opt RPAREN stmt
            init = _convert_for_init(cs[2], filename)
            cond = _convert_for_opt(cs[3], filename)
            step = _convert_for_opt(cs[5], filename)
            body = _convert_matched_or_unmatched(cs[7], filename)
            return ast.ForStmt(
                init=init,
                condition=cond,
                update=step,
                body=body,
                location=_loc(node, filename),
            )
    if node.kind in ("matched_switch_stmt", "unmatched_switch_stmt"):
        cs = node.children
        expr = _convert_expr(cs[2], filename)
        body = _convert_matched_or_unmatched(cs[4], filename)
        return ast.SwitchStmt(expr=expr, body=body, location=_loc(node, filename))
    if node.kind == "compound_stmt":
        return _convert_compound_stmt(node, filename)
    if node.kind == "expression_stmt":
        cs = node.children
        if len(cs) == 1:
            return ast.ExpressionStmt(expr=None, location=_loc(node, filename))
        e = _convert_expr(cs[0], filename)
        return ast.ExpressionStmt(expr=e, location=_loc(node, filename))
    if node.kind == "jump_stmt":
        cs = node.children
        head = _kind(cs[0])
        if head == "KW_GOTO":
            return ast.GotoStmt(label=_tok_text(cs[1]), location=_loc(node, filename))
        if head == "KW_BREAK":
            return ast.BreakStmt(location=_loc(node, filename))
        if head == "KW_CONTINUE":
            return ast.ContinueStmt(location=_loc(node, filename))
        if head == "KW_RETURN":
            value = None
            if len(cs) == 3:
                value = _convert_expr(cs[1], filename)
            return ast.ReturnStmt(value=value, location=_loc(node, filename))
    raise RuntimeError(f"unexpected stmt-inner kind: {node.kind}")


def _convert_for_init(node: ParseNode, filename: str):
    """for_init : declaration | expression_stmt"""
    inner = node.children[0]
    if inner.kind == "declaration":
        decls = _convert_declaration(inner, filename)
        if len(decls) == 1:
            return decls[0]
        if not decls:
            return None
        return ast.DeclarationList(declarations=decls, location=_loc(inner, filename))
    # expression_stmt
    cs = inner.children
    if len(cs) == 1:
        return None
    return _convert_expr(cs[0], filename)


def _convert_for_opt(node: ParseNode, filename: str) -> ast.Expression | None:
    """for_cond_opt / for_step_opt : expr |"""
    if not node.children:
        return None
    return _convert_expr(node.children[0], filename)


# --- expressions -----------------------------------------------------------


_BIN_OP_BY_TOK = {
    "PLUS": "+", "MINUS": "-", "STAR": "*", "SLASH": "/", "PERCENT": "%",
    "LSHIFT": "<<", "RSHIFT": ">>",
    "AMP": "&", "PIPE": "|", "CARET": "^",
    "LAND": "&&", "LOR": "||",
    "EQ": "==", "NE": "!=", "LT": "<", "GT": ">", "LE": "<=", "GE": ">=",
    "ASSIGN": "=", "PLUS_EQ": "+=", "MINUS_EQ": "-=", "STAR_EQ": "*=",
    "SLASH_EQ": "/=", "PERCENT_EQ": "%=", "AMP_EQ": "&=", "PIPE_EQ": "|=",
    "CARET_EQ": "^=", "LSHIFT_EQ": "<<=", "RSHIFT_EQ": ">>=",
    "COMMA": ",",
}


def _convert_expr(node: ParseNode | Token, filename: str) -> ast.Expression:
    if not isinstance(node, ParseNode):
        raise RuntimeError(f"_convert_expr on token: {node}")
    kind = node.kind
    cs = node.children
    if kind == "expr":
        if len(cs) == 1:
            return _convert_expr(cs[0], filename)
        # expr COMMA assignment_expr
        l = _convert_expr(cs[0], filename)
        r = _convert_expr(cs[2], filename)
        return ast.BinaryOp(op=",", left=l, right=r, location=_loc(node, filename))
    if kind == "assignment_expr":
        if len(cs) == 1:
            return _convert_expr(cs[0], filename)
        # unary_expr OP assignment_expr
        l = _convert_expr(cs[0], filename)
        op = _BIN_OP_BY_TOK[_kind(cs[1])]
        r = _convert_expr(cs[2], filename)
        return ast.BinaryOp(op=op, left=l, right=r, location=_loc(node, filename))
    if kind == "conditional_expr":
        if len(cs) == 1:
            return _convert_expr(cs[0], filename)
        # logical_or QUESTION expr COLON conditional_expr
        cond = _convert_expr(cs[0], filename)
        t = _convert_expr(cs[2], filename)
        f = _convert_expr(cs[4], filename)
        return ast.TernaryOp(condition=cond, true_expr=t, false_expr=f, location=_loc(node, filename))
    if kind in (
        "logical_or_expr", "logical_and_expr", "inclusive_or_expr",
        "exclusive_or_expr", "and_expr", "equality_expr", "relational_expr",
        "shift_expr", "additive_expr", "multiplicative_expr",
    ):
        if len(cs) == 1:
            return _convert_expr(cs[0], filename)
        l = _convert_expr(cs[0], filename)
        op = _BIN_OP_BY_TOK[_kind(cs[1])]
        r = _convert_expr(cs[2], filename)
        return ast.BinaryOp(op=op, left=l, right=r, location=_loc(node, filename))
    if kind == "cast_expr":
        if len(cs) == 1:
            return _convert_expr(cs[0], filename)
        # LPAREN type_name RPAREN cast_expr
        target_type = _convert_type_name(cs[1], filename)
        expr = _convert_expr(cs[3], filename)
        return ast.Cast(target_type=target_type, expr=expr, location=_loc(node, filename))
    if kind == "unary_expr":
        return _convert_unary_expr(node, filename)
    if kind == "postfix_expr":
        return _convert_postfix_expr(node, filename)
    if kind == "primary_expr":
        return _convert_primary_expr(node, filename)
    raise RuntimeError(f"unexpected expr kind: {kind}")


def _convert_unary_expr(node: ParseNode, filename: str) -> ast.Expression:
    cs = node.children
    if len(cs) == 1:
        return _convert_expr(cs[0], filename)
    head = _kind(cs[0])
    if head == "INC":
        return ast.UnaryOp(op="++", operand=_convert_expr(cs[1], filename), is_prefix=True, location=_loc(node, filename))
    if head == "DEC":
        return ast.UnaryOp(op="--", operand=_convert_expr(cs[1], filename), is_prefix=True, location=_loc(node, filename))
    if head in ("AMP", "STAR", "PLUS", "MINUS", "TILDE", "BANG"):
        op_map = {"AMP": "&", "STAR": "*", "PLUS": "+", "MINUS": "-", "TILDE": "~", "BANG": "!"}
        return ast.UnaryOp(
            op=op_map[head],
            operand=_convert_expr(cs[1], filename),
            is_prefix=True,
            location=_loc(node, filename),
        )
    if head == "KW_SIZEOF":
        if len(cs) == 2:
            return ast.SizeofExpr(expr=_convert_expr(cs[1], filename), location=_loc(node, filename))
        # KW_SIZEOF LPAREN type_name RPAREN
        return ast.SizeofType(target_type=_convert_type_name(cs[2], filename), location=_loc(node, filename))
    if head == "KW_ALIGNOF":
        return ast.SizeofType(
            target_type=_convert_type_name(cs[2], filename),
            is_alignof=True,
            location=_loc(node, filename),
        )
    if head in ("KW_REAL", "KW_IMAG"):
        op = "__real__" if head == "KW_REAL" else "__imag__"
        return ast.UnaryOp(
            op=op,
            operand=_convert_expr(cs[1], filename),
            is_prefix=True,
            location=_loc(node, filename),
        )
    raise RuntimeError(f"unexpected unary head: {head}")


def _convert_postfix_expr(node: ParseNode, filename: str) -> ast.Expression:
    cs = node.children
    if len(cs) == 1:
        return _convert_expr(cs[0], filename)
    # Compound literal: LPAREN type_name RPAREN LBRACE init_list [COMMA] RBRACE
    if _is_token(cs[0], "LPAREN"):
        target_type = _convert_type_name(cs[1], filename)
        # Build an InitializerList from init_list
        init_node = cs[4]
        values: list = []
        for di in _flatten_list(init_node, "initializer_list", "designated_initializer"):
            values.append(_convert_designated_initializer(di, filename))
        init = ast.InitializerList(values=values, location=_loc(init_node, filename))
        return ast.Compound(target_type=target_type, init=init, location=_loc(node, filename))
    # Left-recursive postfix forms
    base = _convert_expr(cs[0], filename)
    if _is_token(cs[1], "LBRACKET"):
        return ast.Index(array=base, index=_convert_expr(cs[2], filename), location=_loc(node, filename))
    if _is_token(cs[1], "LPAREN"):
        args: list[ast.Expression] = []
        if len(cs) == 4:
            arg_list = cs[2]
            for ae in _flatten_list(arg_list, "argument_list", "assignment_expr"):
                args.append(_convert_expr(ae, filename))
            # The grammar embeds <assignment_expr> as the leaf; the
            # flatten helper above grabs only kind=='assignment_expr'.
            # Real grammar may have raw assignment_expr children;
            # fall back if needed.
            if not args:
                # Walk arg_list collecting all top-level items
                args = _convert_arg_list(arg_list, filename)
        return ast.Call(func=base, args=args, location=_loc(node, filename))
    if _is_token(cs[1], "DOT"):
        return ast.Member(obj=base, member=_tok_text(cs[2]), is_arrow=False, location=_loc(node, filename))
    if _is_token(cs[1], "ARROW"):
        return ast.Member(obj=base, member=_tok_text(cs[2]), is_arrow=True, location=_loc(node, filename))
    if _is_token(cs[1], "INC"):
        return ast.UnaryOp(op="++", operand=base, is_prefix=False, location=_loc(node, filename))
    if _is_token(cs[1], "DEC"):
        return ast.UnaryOp(op="--", operand=base, is_prefix=False, location=_loc(node, filename))
    raise RuntimeError(f"unexpected postfix shape: {[_kind(c) for c in cs]}")


def _convert_arg_list(node: ParseNode, filename: str) -> list[ast.Expression]:
    out: list[ast.Expression] = []
    def visit(n):
        if not isinstance(n, ParseNode):
            return
        if n.kind == "argument_list":
            cs = n.children
            if len(cs) == 1:
                out.append(_convert_expr(cs[0], filename))
            else:
                visit(cs[0])
                out.append(_convert_expr(cs[2], filename))
    visit(node)
    return out


def _convert_offsetof_designator(
    node: ParseNode, filename: str
) -> ast.Expression:
    """``offsetof_designator : IDENT | designator . IDENT | designator [ expr ]``

    Returns a Member/Index chain rooted at a synthetic
    ``Identifier(name='__offsetof_root')`` so codegen can walk it to
    compute the byte offset (matching the legacy parser's encoding).
    """
    cs = node.children
    if len(cs) == 1:
        # Bare IDENT — root member of the target struct.
        return ast.Member(
            obj=ast.Identifier(name="__offsetof_root", location=_loc(node, filename)),
            member=_tok_text(cs[0]),
            is_arrow=False,
            location=_loc(node, filename),
        )
    inner = _convert_offsetof_designator(cs[0], filename)
    if _is_token(cs[1], "DOT"):
        return ast.Member(
            obj=inner, member=_tok_text(cs[2]), is_arrow=False,
            location=_loc(node, filename),
        )
    # LBRACKET expr RBRACKET
    return ast.Index(
        array=inner,
        index=_convert_expr(cs[2], filename),
        location=_loc(node, filename),
    )


def _convert_primary_expr(node: ParseNode, filename: str) -> ast.Expression:
    cs = node.children
    if len(cs) == 1:
        head = cs[0]
        head_kind = _kind(head)
        if head_kind == "IDENT":
            return ast.Identifier(name=_tok_text(head), location=_loc(node, filename))
        if head_kind == "INT_LIT":
            text = _tok_text(head)
            value, is_long, is_long_long, is_unsigned, is_hex = _parse_int_lit(text)
            return ast.IntLiteral(
                value=value,
                is_long=is_long,
                is_long_long=is_long_long,
                is_unsigned=is_unsigned,
                is_hex=is_hex,
                location=_loc(node, filename),
            )
        if head_kind == "FLOAT_LIT":
            text = _tok_text(head)
            value, is_float, is_imag = _parse_float_lit(text)
            return ast.FloatLiteral(
                value=value, is_float=is_float, is_imaginary=is_imag,
                location=_loc(node, filename),
            )
        if head_kind == "CHAR_LIT":
            return ast.CharLiteral(value=_parse_char_lit(_tok_text(head)), location=_loc(node, filename))
        if head_kind == "STRING_LIT":
            text, is_wide = _parse_string_lit(_tok_text(head))
            return ast.StringLiteral(value=text, is_wide=is_wide, location=_loc(node, filename))
        if head_kind == "string_literal":
            # One or more adjacent STRING_LITs — concatenate.
            return _convert_string_literal(head, filename)
        if head_kind == "KW_NULLPTR":
            return ast.NullptrLiteral(location=_loc(node, filename))
    if _is_token(cs[0], "LPAREN") and _is_token(cs[-1], "RPAREN"):
        inner = cs[1]
        if isinstance(inner, ParseNode) and inner.kind == "compound_stmt":
            # GCC statement expression: ``({ ... })``. The legacy parser
            # represents it as a synthetic node holding the compound
            # statement; uc_core's AST has a StatementExpr type for it.
            body = _convert_compound_stmt(inner, filename)
            return ast.StmtExpr(body=body, location=_loc(node, filename))
        return _convert_expr(inner, filename)
    if _is_token(cs[0], "KW_VA_ARG"):
        ap = _convert_expr(cs[2], filename)
        target_type = _convert_type_name(cs[4], filename)
        return ast.VaArgExpr(
            ap=ap, target_type=target_type, location=_loc(node, filename),
        )
    if _is_token(cs[0], "KW_OFFSETOF"):
        target_type = _convert_type_name(cs[2], filename)
        designator = _convert_offsetof_designator(cs[4], filename)
        return ast.OffsetofExpr(
            target_type=target_type,
            designator=designator,
            location=_loc(node, filename),
        )
    if _is_token(cs[0], "KW_GENERIC"):
        # KW_GENERIC LPAREN assignment_expr COMMA generic_assoc_list RPAREN
        controlling = _convert_expr(cs[2], filename)
        assoc_list = cs[4]
        associations: list[tuple[ast.TypeNode | None, ast.Expression]] = []
        for ga in _flatten_list(assoc_list, "generic_assoc_list", "generic_association"):
            gcs = ga.children
            if _is_token(gcs[0], "KW_DEFAULT"):
                associations.append((None, _convert_expr(gcs[2], filename)))
            else:
                t = _convert_type_name(gcs[0], filename)
                associations.append((t, _convert_expr(gcs[2], filename)))
        return ast.GenericSelection(
            controlling_expr=controlling,
            associations=associations,
            location=_loc(node, filename),
        )
    raise RuntimeError(f"unexpected primary_expr shape: {[_kind(c) for c in cs]}")


# --- literal parsing --------------------------------------------------------


def _parse_int_lit(text: str) -> tuple[int, bool, bool, bool, bool]:
    """Parse a C integer literal into (value, is_long, is_long_long, is_unsigned, is_hex).

    Strips the C23 ``'`` digit separator and the trailing
    ``[uU]?[lL]{0,2}`` / ``wb`` BitInt suffixes."""
    s = text.replace("'", "")
    is_unsigned = False
    is_long = False
    is_long_long = False
    upper = s.upper()
    while upper.endswith(("U", "L", "LL", "WB")):
        if upper.endswith("LL"):
            is_long_long = True
            is_long = True
            s = s[:-2]
            upper = upper[:-2]
        elif upper.endswith("L"):
            is_long = True
            s = s[:-1]
            upper = upper[:-1]
        elif upper.endswith("U"):
            is_unsigned = True
            s = s[:-1]
            upper = upper[:-1]
        elif upper.endswith("WB"):
            s = s[:-2]
            upper = upper[:-2]
        else:
            break
    is_hex = False
    if s.lower().startswith("0x"):
        value = int(s, 16)
        is_hex = True
    elif s.lower().startswith("0b"):
        value = int(s, 2)
    elif s.startswith("0") and len(s) > 1:
        value = int(s, 8)
        is_hex = True  # octal also gets is_hex per legacy convention
    else:
        value = int(s)
    return value, is_long, is_long_long, is_unsigned, is_hex


def _parse_float_lit(text: str) -> tuple[float, bool, bool]:
    """Parse a C floating literal. Returns (value, is_float, is_imaginary).

    Strips the suffix; ``f``/``F`` sets is_float, ``l``/``L`` is double
    width but uc_core lumps it under ``is_float=False``. A trailing
    ``i``/``I``/``j``/``J`` (GCC imaginary literal) sets is_imaginary."""
    s = text
    is_float = False
    is_imaginary = False
    if s and s[-1] in "iIjJ":
        is_imaginary = True
        s = s[:-1]
    if s and s[-1] in "fF":
        is_float = True
        s = s[:-1]
    elif s and s[-1] in "lL":
        s = s[:-1]
    elif s and len(s) >= 2 and s[-2:].lower() in ("df", "dd", "dl"):
        s = s[:-2]
    return float(s), is_float, is_imaginary


def _parse_char_lit(text: str) -> int:
    """Parse a C character literal — strip prefix + quotes, decode the
    body's escape sequences, return the integer value of the first
    decoded character (multi-character literals are implementation-
    defined; uc_core takes only the first)."""
    s = text
    # Encoding prefix
    for prefix in ("u8", "u", "U", "L"):
        if s.startswith(prefix + "'"):
            s = s[len(prefix):]
            break
    assert s.startswith("'") and s.endswith("'"), text
    body = s[1:-1]
    decoded = _decode_string(body)
    if not decoded:
        return 0
    return ord(decoded[0])


def _convert_string_literal(node: ParseNode, filename: str) -> ast.StringLiteral:
    """string_literal : STRING_LIT | string_literal STRING_LIT — flatten
    and concatenate. Wide-flag is true if any constituent had a wide
    encoding prefix."""
    parts: list[str] = []
    is_wide = False
    def visit(n):
        if isinstance(n, Token) and n.name == "STRING_LIT":
            txt, w = _parse_string_lit(n.text)
            parts.append(txt)
            nonlocal_is_wide()
            return
        if isinstance(n, ParseNode):
            for c in n.children:
                visit(c)

    def nonlocal_is_wide():
        # Closure over the outer is_wide; rebound via list to avoid
        # ``nonlocal`` inside a closure (which Python forbids when the
        # name is reassigned inside another nested function).
        pass
    flag = [False]
    def visit2(n):
        if isinstance(n, Token) and n.name == "STRING_LIT":
            txt, w = _parse_string_lit(n.text)
            parts.append(txt)
            if w:
                flag[0] = True
            return
        if isinstance(n, ParseNode):
            for c in n.children:
                visit2(c)
    visit2(node)
    return ast.StringLiteral(
        value="".join(parts),
        is_wide=flag[0],
        location=_loc(node, filename),
    )


def _parse_string_lit(text: str) -> tuple[str, bool]:
    is_wide = False
    s = text
    for prefix in ("u8", "u", "U", "L"):
        if s.startswith(prefix + '"'):
            if prefix in ("L", "U", "u"):
                is_wide = True
            s = s[len(prefix):]
            break
    assert s.startswith('"') and s.endswith('"')
    return _decode_string(s[1:-1]), is_wide


_ESCAPES = {
    "n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\",
    "'": "'", '"': '"', "a": "\a", "b": "\b", "f": "\f", "v": "\v",
    "?": "?",
}


def _decode_string(body: str) -> str:
    out: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            out.append(c)
            break
        nxt = body[i + 1]
        if nxt in _ESCAPES:
            out.append(_ESCAPES[nxt])
            i += 2
            continue
        if nxt == "x":
            j = i + 2
            while j < n and body[j] in "0123456789abcdefABCDEF":
                j += 1
            out.append(chr(int(body[i + 2:j], 16)))
            i = j
            continue
        if nxt in "01234567":
            j = i + 1
            while j < n and j - i < 4 and body[j] in "01234567":
                j += 1
            out.append(chr(int(body[i + 1:j], 8)))
            i = j
            continue
        # Unknown escape — keep the backslash + char literally.
        out.append(c)
        out.append(nxt)
        i += 2
    return "".join(out)
