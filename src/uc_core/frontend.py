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
from .c23_parser import HookRegistry, Token


# ---- C23 attributes are stripped at the source level -----------------------


_ATTRIBUTE_RE = re.compile(
    r"__attribute__\s*\(\(\s*"
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
_DOS_QUALIFIER_RE = re.compile(
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
    r"\s*__attribute__\s*\(\s*\(\s*(?:__)?vector_size(?:__)?"
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
    r"__attribute__\s*\(\s*\(\s*(?:__)?packed(?:__)?\s*\)\s*\)"
)
_PACKED_TYPEDEF_STRUCT_RE = re.compile(
    r"\btypedef\s+struct\s*(?:[A-Za-z_]\w*\s*)?\{[^{}]*\}\s*"
    r"__attribute__\s*\(\s*\(\s*(?:__)?packed(?:__)?\s*\)\s*\)\s*"
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
    source = _DOS_QUALIFIER_RE.sub("", source)
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


# ---- typedef tracker -------------------------------------------------------


def _make_typedef_filter(seed_typedefs: set[str] | None) -> tuple[object, object]:
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
        "KW_VOID", "KW_CHAR", "KW_SHORT", "KW_INT", "KW_LONG",
        "KW_FLOAT", "KW_DOUBLE", "KW_SIGNED", "KW_UNSIGNED",
        "KW_BOOL", "KW_COMPLEX", "KW_IMAGINARY", "KW_NULLPTR",
        "KW_DECIMAL32", "KW_DECIMAL64", "KW_DECIMAL128",
        "KW_BITINT", "KW_INT128",
    })
    # Combined: contexts where IDENT → TYPEDEF_NAME rewrite is
    # suppressed. Includes the type-spec set plus tag/member
    # namespaces. Tag/member contexts suppress but DON'T shadow.
    _SUPPRESS_PREV = _TYPESPEC_PREV | frozenset({
        "KW_STRUCT", "KW_UNION", "KW_ENUM", "DOT", "ARROW",
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
    # The parser may re-query the same offset multiple times during
    # lookahead; only advance state when the offset changes so the
    # `prev` context stays stable for a given token.
    state = [-1, "", "", "", 0, 0, []]

    def filter_(_ctx, tok: Token) -> Token:
        offset = getattr(tok, "offset", -1)
        if offset != state[0]:
            # New token: roll previous-distinct into history slots,
            # then update depth counters.
            state[1] = state[2]
            state[3] = "" if state[1] == "" else state[3]
            state[2] = tok.name
            state[0] = offset
            # When we close a brace, drop any shadow set whose
            # entry-depth matches the brace count we just exited.
            if tok.name == "RBRACE":
                state[5] -= 1
                while state[6] and state[6][-1][0] > state[5]:
                    state[6].pop()
            elif tok.name == "LBRACE":
                state[5] += 1
            elif tok.name == "LPAREN":
                state[4] += 1
            elif tok.name == "RPAREN":
                state[4] = max(0, state[4] - 1)
            # When an IDENT follows a concrete type-spec keyword, it's
            # a declarator name — shadow it for the rest of the
            # enclosing block. This handles ``int byte`` where byte
            # is a typedef-name being re-used as a parameter or local.
            # Tag-namespace and member-namespace contexts (KW_STRUCT,
            # DOT, ARROW, etc.) suppress the rewrite but DON'T shadow
            # the name — `struct Regexp` introduces a tag, not a
            # variable, and the typedef is still reachable.
            if (
                tok.name == "IDENT"
                and state[1] in _TYPESPEC_PREV
                and tok.text in names
                and (state[4] >= 1 or state[5] >= 1)
                # Inside a function signature (paren_depth >= 1) or
                # a block (brace_depth >= 1). File-scope typedef
                # declarations stay unshadowed.
            ):
                # Parameters declared in a signature (state[4] >= 1)
                # need to shadow the typedef in the following body, so
                # add the shadow at brace_depth + 1 — that matches the
                # body's brace depth and survives the LPAREN/RPAREN
                # pair without being popped.
                target_depth = state[5]
                if not state[6] or state[6][-1][0] != target_depth + 1:
                    state[6].append((target_depth + 1, set()))
                state[6][-1][1].add(tok.text)
        prev_name = state[1]
        # Walk the shadow stack: if any frame contains this name, the
        # IDENT is shadowed; don't rewrite.
        shadowed = any(tok.text in s for _d, s in state[6])
        if (
            tok.name == "IDENT"
            and tok.text in names
            and prev_name not in _SUPPRESS_PREV
            and not shadowed
        ):
            return Token(
                name="TYPEDEF_NAME",
                text=tok.text,
                line=tok.line,
                column=tok.column,
                offset=tok.offset,
                file_id=tok.file_id,
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
) -> ast.TranslationUnit:
    """Parse already-preprocessed C source into a typed AST.

    The caller is expected to have run :class:`uc_core.preprocessor`
    over ``source`` first.

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
    packed_tag_names: set[str] = set()
    packed_typedef_names: set[str] = set()
    for m in _PACKED_NAMED_STRUCT_RE.finditer(source):
        packed_tag_names.add(m.group(1))
    for m in _PACKED_TYPEDEF_STRUCT_RE.finditer(source):
        packed_typedef_names.add(m.group(1))
    vector_names: set = set()
    pre = _strip_attributes(source, vector_names)
    hooks, filter_ = _make_typedef_filter(seed_typedefs)
    unit = c23_parser.parse(
        pre, filename=filename, hooks=hooks, token_filter=filter_
    )
    # Attach the set of names that had `__attribute__((vector_size(N)))`
    # rewritten away — downstream consumers (uc386 codegen) read this
    # to flip the GCC is_vector flag on the resolved ArrayType.
    unit._vector_typedef_names = vector_names
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
