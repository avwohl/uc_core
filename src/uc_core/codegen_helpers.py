"""Codegen helpers shared by uc80 / uc386 (and any future target).

The uplox v3 auto-AST is declarator-shaped: a declaration carries a
decl_specs list (BasicTypeSpec / TypeQualifier / StorageClass /
StructDef / EnumDef / TypedefNameSpec / TypeofExpr / BitIntSpec /
AlignasType / AlignasValue) and a declarator chain (Declarator /
PointerDeclarator / GroupDeclarator / ArrayDeclarator{,Unsized,Star,
Static,QualStatic} / FnDeclarator{,Empty}). Per-target codegens want
to ask the resolved-type-tree question "what type is variable X" —
this module answers it by walking decl_specs + declarator together
and producing a ResolvedType.

Also exports a few small utility helpers (make_identifier,
_decode_string_literal / _string_is_wide / _decoded_str_len for the
new Token-text-bearing literal nodes, and is_function_type /
is_pointer_type / is_array_type / is_struct_type that accept both
ResolvedType and the legacy ``uc_core.ast_legacy`` shapes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import ast
from . import ast_legacy as lt


@dataclass
class ResolvedType:
    """Codegen-internal resolved-type info, derived from the auto-AST.

    kind == "basic":  name (str), is_signed (Optional[bool])
    kind == "pointer": pointee (ResolvedType)
    kind == "array":   element (ResolvedType), size_expr (auto-AST or None)
    kind == "function": return_type (ResolvedType),
                        param_types (list[ResolvedType]),
                        is_variadic (bool)
    kind == "struct":  name (Optional[str]), is_union (bool),
                       members (list[(name, ResolvedType, bit_width)])
    kind == "enum":    name (Optional[str])
    kind == "typedef": name (str)  # unresolved typedef-name reference

    ``is_vector`` is GCC's __attribute__((vector_size)) marker on array
    types — codegens use it to route through SIMD-shaped paths.
    """
    kind: str
    name: Optional[str] = None
    is_signed: Optional[bool] = None
    is_const: bool = False
    is_volatile: bool = False
    pointee: Optional["ResolvedType"] = None
    element: Optional["ResolvedType"] = None
    size_expr: Optional[object] = None
    return_type: Optional["ResolvedType"] = None
    param_types: tuple = ()
    is_variadic: bool = False
    is_union: bool = False
    members: tuple = ()
    is_vector: bool = False


_active_typedef_resolver = None


class typedef_resolver_scope:
    """Context manager installing a callable resolver(name) -> Optional[ResolvedType].

    While active, :func:`resolve_base_type` calls the resolver when it
    sees a ``TypedefNameSpec``; if the resolver returns a non-None
    :class:`ResolvedType`, that shape is substituted for the typedef-
    name. Hosts use this to expand typedef-named types into their
    underlying struct / pointer / basic shape so the downstream codegen
    never needs to track a typedef table itself.
    """
    def __init__(self, resolver):
        self.resolver = resolver
        self._prev = None

    def __enter__(self):
        global _active_typedef_resolver
        self._prev = _active_typedef_resolver
        _active_typedef_resolver = self.resolver
        return self

    def __exit__(self, *_):
        global _active_typedef_resolver
        _active_typedef_resolver = self._prev


def resolve_type_from_decl(decl_specs, declarator) -> tuple[Optional[str], ResolvedType]:
    base = resolve_base_type(decl_specs)
    return _wrap_declarator(declarator, base)


def resolve_base_type(decl_specs) -> ResolvedType:
    type_keywords: list[str] = []
    signed: Optional[bool] = None
    is_const = False
    is_volatile = False
    is_complex = False
    explicit: Optional[ResolvedType] = None
    for spec in decl_specs or []:
        if isinstance(spec, ast.BasicTypeSpec):
            kw = spec.kw.text
            if kw in ("signed", "__signed", "__signed__"):
                signed = True
            elif kw == "unsigned":
                signed = False
            elif kw in ("_Complex", "__complex", "__complex__"):
                is_complex = True
            else:
                type_keywords.append(kw)
        elif isinstance(spec, ast.TypeQualifier):
            kw = spec.kw.text
            if kw == "const" or kw.startswith("__const"):
                is_const = True
            elif kw == "volatile" or kw.startswith("__volatile"):
                is_volatile = True
        elif isinstance(spec, ast.StorageClass):
            continue
        elif isinstance(spec, ast.TypedefNameSpec):
            # If a host has installed a typedef resolver, look up the
            # underlying type so the rest of the codegen sees the
            # resolved shape (BasicType / StructType / PointerType / ...).
            # Otherwise emit the unresolved "typedef" sentinel for the
            # caller to handle.
            resolver = _active_typedef_resolver
            resolved = resolver(spec.name.text) if resolver is not None else None
            if resolved is not None:
                explicit = resolved
            else:
                explicit = ResolvedType(kind="typedef", name=spec.name.text)
        elif isinstance(spec, (ast.StructDef, ast.StructAnon, ast.StructEmpty,
                               ast.StructAnonEmpty, ast.StructRef)):
            explicit = _resolve_struct_spec(spec)
        elif isinstance(spec, (ast.EnumDef, ast.EnumAnon, ast.EnumRef)):
            explicit = _resolve_enum_spec(spec)
        elif isinstance(spec, ast.BitIntSpec):
            explicit = ResolvedType(kind="basic", name="int")
    if explicit is None:
        name = _reduce_type_keywords(type_keywords)
        explicit = ResolvedType(kind="basic", name=name, is_signed=signed)
    explicit.is_const = explicit.is_const or is_const
    explicit.is_volatile = explicit.is_volatile or is_volatile
    if is_complex and explicit.kind == "basic":
        # _Complex T is laid out as two T's; codegen treats it as a
        # struct-like with .base_type giving the element type name.
        # We store the complex spec inline so resolved_to_legacy can
        # produce a proper ComplexType. The base name lives in `name`.
        explicit.kind = "complex"
    return explicit


def _reduce_type_keywords(keywords: list[str]) -> str:
    if not keywords:
        return "int"
    long_count = keywords.count("long")
    others = [k for k in keywords if k != "long"]
    if "void" in keywords:
        return "void"
    if "double" in keywords:
        return "long double" if long_count else "double"
    if "float" in keywords:
        return "float"
    if "char" in keywords:
        return "char"
    if "short" in keywords:
        return "short"
    if "bool" in keywords or "_Bool" in keywords:
        return "bool"
    if long_count >= 2:
        return "long long"
    if long_count == 1:
        return "long"
    if others:
        kw = others[0]
        # Normalise GCC's leading-underscore aliases to the bare name
        # codegens expect (matches the legacy resolved-type tree).
        if kw == "__int128":
            return "int128"
        # C23 _Decimal32 / _Decimal64 / _Decimal128 are decimal floats;
        # codegens approximate them as the same-precision binary float.
        if kw == "_Decimal32":
            return "float"
        if kw in ("_Decimal64", "_Decimal128"):
            return "double"
        return kw
    return "int"


def _resolve_struct_spec(spec) -> ResolvedType:
    is_union = spec.kind.kw.text == "union"
    name = getattr(spec, "name", None)
    name = name.text if name is not None else None
    members: list = []
    raw_members = getattr(spec, "members", None) or []
    for m in raw_members:
        if isinstance(m, ast.StructMember):
            mbase = resolve_base_type(m.decl_specs)
            for sd in m.declarators or []:
                if isinstance(sd, ast.PlainDeclarator):
                    nm, mt = _wrap_declarator(sd.declarator, mbase)
                    members.append((nm, mt, None))
                elif isinstance(sd, ast.BitFieldDecl):
                    nm, mt = _wrap_declarator(sd.declarator, mbase)
                    members.append((nm, mt, sd.bit_width))
                elif isinstance(sd, ast.AnonBitFieldDecl):
                    members.append((None, mbase, sd.bit_width))
        elif isinstance(m, ast.StructAnonMember):
            mbase = resolve_base_type(m.decl_specs)
            members.append((None, mbase, None))
    return ResolvedType(kind="struct", name=name, is_union=is_union,
                        members=tuple(members))


def _resolve_enum_spec(spec) -> ResolvedType:
    name = getattr(spec, "name", None)
    name = name.text if name is not None else None
    return ResolvedType(kind="enum", name=name)


def _wrap_declarator(node, base: ResolvedType) -> tuple[Optional[str], ResolvedType]:
    if node is None:
        return None, base
    if isinstance(node, ast.PointerDeclarator):
        wrapped = _wrap_pointer(node.pointer, base)
        return _wrap_declarator(node.inner, wrapped)
    if isinstance(node, ast.Declarator):
        return node.name.text, base
    if isinstance(node, ast.GroupDeclarator):
        return _wrap_declarator(node.inner, base)
    if isinstance(node, ast.ArrayDeclarator):
        inner_name, inner_type = _wrap_declarator(
            node.inner, ResolvedType(kind="array", element=base, size_expr=node.size)
        )
        return inner_name, inner_type
    if isinstance(node, ast.ArrayDeclaratorUnsized):
        inner_name, inner_type = _wrap_declarator(
            node.inner, ResolvedType(kind="array", element=base, size_expr=None)
        )
        return inner_name, inner_type
    if isinstance(node, (ast.ArrayDeclaratorStar, ast.ArrayDeclaratorStatic,
                         ast.ArrayDeclaratorQualStatic)):
        size = getattr(node, "size", None)
        inner_name, inner_type = _wrap_declarator(
            node.inner, ResolvedType(kind="array", element=base, size_expr=size)
        )
        return inner_name, inner_type
    if isinstance(node, ast.FnDeclaratorEmpty):
        fn = ResolvedType(kind="function", return_type=base, param_types=(),
                          is_variadic=False)
        return _wrap_declarator(node.inner, fn)
    # Abstract declarators (used in type-names like `int (*)[3]` or
    # `int[]`). Same shape as the named variants but without an
    # innermost ``Declarator(name)``.
    if isinstance(node, ast.AbstractPointer):
        wrapped = _wrap_pointer(node.pointer, base)
        return None, wrapped
    if isinstance(node, ast.AbstractPointerInner):
        wrapped = _wrap_pointer(node.pointer, base)
        return _wrap_declarator(node.inner, wrapped)
    if isinstance(node, ast.AbstractGroup):
        return _wrap_declarator(node.inner, base)
    if isinstance(node, ast.AbstractArray):
        inner_name, inner_type = _wrap_declarator(
            node.inner,
            ResolvedType(kind="array", element=base, size_expr=node.size),
        )
        return inner_name, inner_type
    if isinstance(node, ast.AbstractArrayBare):
        return None, ResolvedType(
            kind="array", element=base, size_expr=node.size,
        )
    if isinstance(node, (ast.AbstractArrayBareUnsized, ast.AbstractArrayBareStar)):
        return None, ResolvedType(
            kind="array", element=base, size_expr=None,
        )
    if isinstance(node, (ast.AbstractArrayBareStatic, ast.AbstractArrayBareQualStatic)):
        return None, ResolvedType(
            kind="array", element=base, size_expr=node.size,
        )
    if isinstance(node, (ast.AbstractArrayUnsized, ast.AbstractArrayStar)):
        inner_name, inner_type = _wrap_declarator(
            node.inner,
            ResolvedType(kind="array", element=base, size_expr=None),
        )
        return inner_name, inner_type
    if isinstance(node, (ast.AbstractArrayStatic, ast.AbstractArrayQualStatic)):
        inner_name, inner_type = _wrap_declarator(
            node.inner,
            ResolvedType(kind="array", element=base, size_expr=node.size),
        )
        return inner_name, inner_type
    if isinstance(node, ast.AbstractFnEmpty):
        fn = ResolvedType(kind="function", return_type=base, param_types=(),
                          is_variadic=False)
        return _wrap_declarator(node.inner, fn)
    if isinstance(node, ast.AbstractFn):
        params = node.params
        is_variadic = isinstance(params, ast.VariadicParams)
        if is_variadic:
            params = params.params
        param_types: list[ResolvedType] = []
        for p in params or []:
            if isinstance(p, ast.ParamDecl):
                _, pt = resolve_type_from_decl(p.decl_specs, p.declarator)
                param_types.append(pt)
            elif isinstance(p, ast.ParamDeclAbstract):
                _, pt = resolve_type_from_decl(p.decl_specs, p.declarator)
                param_types.append(pt)
            elif isinstance(p, ast.ParamDeclTypeOnly):
                param_types.append(resolve_base_type(p.decl_specs))
        if (len(param_types) == 1 and param_types[0].kind == "basic"
                and param_types[0].name == "void"):
            param_types = []
        fn = ResolvedType(kind="function", return_type=base,
                          param_types=tuple(param_types),
                          is_variadic=is_variadic)
        return _wrap_declarator(node.inner, fn)
    if isinstance(node, ast.FnDeclarator):
        params = node.params
        is_variadic = isinstance(params, ast.VariadicParams)
        if is_variadic:
            params = params.params
        param_types: list[ResolvedType] = []
        for p in params or []:
            if isinstance(p, ast.ParamDecl):
                _, pt = resolve_type_from_decl(p.decl_specs, p.declarator)
                param_types.append(pt)
            elif isinstance(p, ast.ParamDeclAbstract):
                _, pt = resolve_type_from_decl(p.decl_specs, p.declarator)
                param_types.append(pt)
            elif isinstance(p, ast.ParamDeclTypeOnly):
                param_types.append(resolve_base_type(p.decl_specs))
        if (len(param_types) == 1 and param_types[0].kind == "basic"
                and param_types[0].name == "void"):
            param_types = []
        fn = ResolvedType(kind="function", return_type=base,
                          param_types=tuple(param_types),
                          is_variadic=is_variadic)
        return _wrap_declarator(node.inner, fn)
    return None, base


def _wrap_pointer(node, base: ResolvedType) -> ResolvedType:
    if isinstance(node, ast.PointerOne):
        is_const, is_volatile = _pointer_quals(node.quals)
        return ResolvedType(kind="pointer", pointee=base,
                            is_const=is_const, is_volatile=is_volatile)
    if isinstance(node, ast.PointerNested):
        inner = _wrap_pointer(node.outer, base)
        is_const, is_volatile = _pointer_quals(node.quals)
        return ResolvedType(kind="pointer", pointee=inner,
                            is_const=is_const, is_volatile=is_volatile)
    return base


def _pointer_quals(quals) -> tuple[bool, bool]:
    if not quals:
        return False, False
    is_const = False
    is_volatile = False
    for q in quals:
        if isinstance(q, ast.TypeQualifier):
            kw = q.kw.text
            if kw == "const" or kw.startswith("__const"):
                is_const = True
            elif kw == "volatile" or kw.startswith("__volatile"):
                is_volatile = True
    return is_const, is_volatile


def decl_storage_class(decl_specs) -> Optional[str]:
    for spec in decl_specs or []:
        if isinstance(spec, ast.StorageClass):
            kw = spec.kw.text
            if kw in ("static", "extern", "register", "auto"):
                return kw
            if kw == "typedef":
                return "typedef"
    return None


def decl_is_inline(decl_specs) -> bool:
    for spec in decl_specs or []:
        if isinstance(spec, ast.StorageClass):
            kw = spec.kw.text
            if kw == "inline" or kw.startswith("__inline"):
                return True
    return False


def declarator_ident(node) -> Optional[str]:
    while node is not None:
        if isinstance(node, ast.Declarator):
            return node.name.text
        sub = getattr(node, "inner", None)
        if sub is None:
            return None
        node = sub
    return None


def _outermost_fn_declarator(node):
    while node is not None:
        if isinstance(node, (ast.FnDeclarator, ast.FnDeclaratorEmpty)):
            return node
        sub = getattr(node, "inner", None)
        if sub is None:
            return None
        node = sub
    return None


def _make_synthetic_token(name: str, text: str):
    from .c23_parser import Token
    return Token(name=name, text=text, line=0, column=0, offset=0, file_id=0)


def make_identifier(name: str):
    return ast.Identifier(
        name=_make_synthetic_token("IDENT", name),
        pos=ast._Pos(),
    )


def is_function_type(t) -> bool:
    if isinstance(t, lt.FunctionType):
        return True
    return isinstance(t, ResolvedType) and t.kind == "function"


def is_pointer_type(t) -> bool:
    if isinstance(t, lt.PointerType):
        return True
    return isinstance(t, ResolvedType) and t.kind == "pointer"


def is_array_type(t) -> bool:
    if isinstance(t, lt.ArrayType):
        return True
    return isinstance(t, ResolvedType) and t.kind == "array"


def is_struct_type(t) -> bool:
    if isinstance(t, lt.StructType):
        return True
    return isinstance(t, ResolvedType) and t.kind == "struct"


def function_is_variadic(func) -> bool:
    fn = _outermost_fn_declarator(func.declarator) if hasattr(func, "declarator") else None
    return isinstance(fn, ast.FnDeclarator) and isinstance(fn.params, ast.VariadicParams)


def function_name(func) -> Optional[str]:
    if not hasattr(func, "declarator"):
        return None
    return declarator_ident(func.declarator)


def function_params(func) -> list:
    fn = _outermost_fn_declarator(func.declarator) if hasattr(func, "declarator") else None
    if not isinstance(fn, ast.FnDeclarator):
        return []
    params = fn.params
    if isinstance(params, ast.VariadicParams):
        params = params.params
    return list(params or [])


def function_param_names(func) -> list[str]:
    out: list[str] = []
    for p in function_params(func):
        if isinstance(p, ast.ParamDecl):
            nm = declarator_ident(p.declarator)
            if nm is not None:
                out.append(nm)
    return out


def iter_var_decls(declaration):
    """For an ``ast.Declaration``, yield (name, ResolvedType, init_or_None,
    is_func) for each init_declarator."""
    base = resolve_base_type(declaration.decl_specs)
    for init_decl in declaration.declarators or []:
        if isinstance(init_decl, ast.InitDeclarator):
            inner = init_decl.declarator
            init = None
        elif isinstance(init_decl, ast.InitDeclaratorWithInit):
            inner = init_decl.declarator
            init = init_decl.init
        else:
            continue
        name, full = _wrap_declarator(inner, base)
        yield name, full, init, full.kind == "function"


def decode_string_literal(text: str) -> str:
    """Decode a STRING_LIT source token to the bytes its content represents."""
    if text.startswith("u8"):
        text = text[2:]
    elif text.startswith(("u", "U", "L")):
        text = text[1:]
    if text.startswith('"') and text.endswith('"'):
        body = text[1:-1]
    else:
        body = text
    esc = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"',
           "0": "\0", "a": "\a", "b": "\b", "f": "\f", "v": "\v"}
    out: list[str] = []
    i = 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in esc:
                out.append(esc[nxt])
                i += 2
                continue
            out.append(nxt)
            i += 2
            continue
        out.append(body[i])
        i += 1
    return "".join(out)


def string_is_wide(text: str) -> bool:
    return text.startswith(("u", "U", "L")) and not text.startswith("u8")


def decoded_str_len(text: str) -> int:
    if text.startswith("u8"):
        text = text[2:]
    elif text.startswith(("u", "U", "L")):
        text = text[1:]
    if text.startswith('"') and text.endswith('"'):
        body = text[1:-1]
    else:
        body = text
    n = 0
    i = 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            i += 2
            n += 1
        else:
            i += 1
            n += 1
    return n


def rewrite_str_token(string_lit, rewriter) -> None:
    """Replace a StringLiteral's Token text in-place by running its
    decoded body through ``rewriter`` and re-encoding."""
    text = string_lit.value.text
    prefix = ""
    if text.startswith("u8"):
        prefix = "u8"
        text = text[2:]
    elif text.startswith(("u", "U", "L")):
        prefix = text[0]
        text = text[1:]
    assert text.startswith('"') and text.endswith('"')
    decoded = decode_string_literal(string_lit.value.text)
    new_decoded = rewriter(decoded)
    if new_decoded == decoded:
        return
    new_text = prefix + '"' + new_decoded.replace("\\", "\\\\").replace('"', '\\"') + '"'
    from .c23_parser import Token as _Tok
    string_lit.value = _Tok(
        name=string_lit.value.name,
        text=new_text,
        line=string_lit.value.line,
        column=string_lit.value.column,
        offset=string_lit.value.offset,
        file_id=string_lit.value.file_id,
    )
