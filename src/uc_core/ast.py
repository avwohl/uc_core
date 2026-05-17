"""uc_core AST — uplox v3 auto-AST re-exports.

After Phase 6, ``uc_core.ast`` is a thin re-export of the typed AST
dataclasses generated from ``examples/c23.uplox``. The previous
resolved-type-tree shapes (BasicType, PointerType, FunctionType, ...)
are gone; consumers walk the **declarator-shaped** auto-AST directly.

Key shape differences vs. the legacy AST:

* ``TranslationUnit.items`` (was ``.declarations``).
* Declarations carry ``decl_specs`` (a list of ``BasicTypeSpec`` /
  ``TypeQualifier`` / ``StorageClass`` / ``StructDef`` / ``EnumDef`` /
  ``TypedefNameSpec`` / ``TypeofExpr`` / ``BitIntSpec`` /
  ``AlignasType`` / ``AlignasValue``) and ``declarators`` (a list of
  ``InitDeclarator`` / ``InitDeclaratorWithInit``).
* Types are not resolved at parse time: a "pointer to int" is
  represented as ``PointerOne`` (or ``PointerNested``) plus the
  ``Declarator`` chain. Code that needs a single "type" should walk
  ``decl_specs`` and the declarator chain together.
* Literals carry a ``Token`` (with ``.text`` / ``.line`` / ``.file_id``)
  rather than a pre-parsed Python value. ``IntLiteral.value.text`` is
  ``"42"``; lift to ``int`` at the point of use.
* If / Return / etc. split present / absent variants into distinct
  kinds: ``IfStmt`` (no else) vs ``IfStmtElse`` (with else),
  ``ReturnStmt`` (no value) vs ``ReturnStmtValue`` (with value).
* Source position lives on ``.pos`` (a ``_Pos`` with
  ``start_line`` / ``end_line`` / ``start_column`` / ``end_column``).
"""

from __future__ import annotations

from .c23_parser import *  # noqa: F401,F403
from .c23_parser import _Pos  # noqa: F401


# Sentinel classes for the legacy resolved-type-tree shapes that no
# longer exist. The auto-AST never produces instances of these, so
# every ``isinstance(node, ast.X)`` check fires False — the legacy
# branches become unreachable. New isinstance branches against the
# declarator-shape AST (``Declaration``, ``InitDeclarator``,
# ``BasicTypeSpec``, ``PointerDeclarator``, ...) coexist alongside
# the dead branches until consumers finish migrating.
class _Removed:
    """Tombstone — instances of this class are never created."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Type has been removed in Phase 6; migrate to the auto-AST shape."
        )


VarDecl = _Removed
FunctionDecl = _Removed
DeclarationList = _Removed
TypedefDecl = _Removed
EnumDecl = _Removed
StructDecl = _Removed
ParamDecl_legacy = _Removed  # alias to disambiguate from auto-AST ParamDecl
BasicType = _Removed
PointerType = _Removed
ArrayType = _Removed
FunctionType = _Removed
StructType = _Removed
EnumType = _Removed
ComplexType = _Removed
TypeofType = _Removed
# NB: `TypesCompatibleP` is NOT tombstoned — the c23 grammar now has a
# `__builtin_types_compatible_p(type-name, type-name)` production, so
# the uplox auto-AST emits a real `TypesCompatibleP` node (t1/t2).
# `from .c23_parser import *` above brings it in; don't shadow it
# (codegen's `isinstance(expr, ast.TypesCompatibleP)` depends on it).
LabelAddr = _Removed
AsmStmt = _Removed
# Legacy alias: codegens that branch on `ast.Compound` continue to
# detect the auto-AST's `CompoundLiteral` form. The empty variant
# (`(T){}`) is also accepted.
from .c23_parser import (
    CompoundLiteral as _CompoundLiteralForAlias,
    CompoundLiteralEmpty as _CompoundLiteralEmptyForAlias,
    InitializerList as _InitializerListForAlias,
)
Compound = _CompoundLiteralForAlias


# Legacy `Compound.init` was an InitializerList wrapping the values;
# the auto-AST stores the values directly in `.items`. Expose `.init`
# as a synthetic InitializerList view so legacy `compound.init.values`
# walks find the values uniformly.
def _cl_init(self):
    return _InitializerListForAlias(values=self.items, pos=self.pos)


def _cle_init(self):
    return _InitializerListForAlias(values=[], pos=self.pos)


_CompoundLiteralForAlias.init = property(_cl_init)
_CompoundLiteralEmptyForAlias.init = property(_cle_init)
SourceLocation = _Removed
Node = _Removed
TypeNode = _Removed
Expression = _Removed
Statement = _Removed
StructMember_legacy = _Removed
EnumValue_legacy = _Removed
# C23 ``true`` / ``false`` aren't separate tokens in c23.uplox (they
# lex as IDENT and bind to host-defined macros / nullary builtins);
# legacy code that probed for ``ast.BoolLiteral`` always fires False
# under the auto-AST.
BoolLiteral = _Removed


# CallNoArgs always has zero arguments, so consumers that walk
# call.args uniformly still want an empty sequence. Attach the
# default at class level so legacy consumers reading expr.args
# work for both Call and CallNoArgs without per-site changes.
from .c23_parser import CallNoArgs as _CallNoArgs
_CallNoArgs.args = ()
# Same for FnDeclaratorEmpty — its .params field is "missing" but
# a default empty list lets callers iterate uniformly.
from .c23_parser import FnDeclaratorEmpty as _FnDeclaratorEmpty
_FnDeclaratorEmpty.params = ()


# Legacy-compat properties on FunctionDef so codegens reading
# func.name / func.return_type / func.params / func.is_variadic /
# func.storage_class / func.is_inline continue to work after the
# move from the resolved-type FunctionDecl. Each property walks the
# decl_specs + declarator chain on demand.
from .c23_parser import FunctionDef as _FunctionDef


def _fd_name(self):
    from .codegen_helpers import declarator_ident
    return declarator_ident(self.declarator)


def _fd_storage_class(self):
    from .codegen_helpers import decl_storage_class
    return decl_storage_class(self.decl_specs)


def _fd_is_inline(self):
    from .codegen_helpers import decl_is_inline
    return decl_is_inline(self.decl_specs)


def _fd_is_variadic(self):
    from .codegen_helpers import function_is_variadic
    return function_is_variadic(self)


def _fd_params(self):
    """Yield ParamDecl-shaped namespaces for each named parameter,
    matching the legacy ``FunctionDecl.params`` (list of ParamDecl
    with .name / .param_type / .size_side_effects).
    ``param_type`` is in legacy ``ast_legacy`` form so existing
    isinstance checks against BasicType / PointerType / etc. work."""
    from .codegen_helpers import (
        function_params, declarator_ident, resolve_type_from_decl,
        resolve_base_type,
    )

    class _ParamView:
        __slots__ = ("name", "param_type", "size_side_effects")

        def __init__(self, name, param_type):
            self.name = name
            self.param_type = param_type
            self.size_side_effects = None

    out = []
    from . import c23_parser as _cp
    def _decay_for_param(pt):
        # C: in a function-parameter declaration, `T name[]` and
        # `T name[N]` are both adjusted to `T *name` (the array's
        # size, if any, is ignored). Codegen that sees an ArrayType
        # for a param would otherwise treat it as a frame-local
        # array and emit lea-based loads.
        if pt is not None and pt.kind == "array":
            from .codegen_helpers import ResolvedType
            return ResolvedType(kind="pointer", pointee=pt.element)
        return pt
    for p in function_params(self):
        if isinstance(p, _cp.ParamDecl):
            _, pt = resolve_type_from_decl(p.decl_specs, p.declarator)
            out.append(_ParamView(declarator_ident(p.declarator),
                                  resolved_to_legacy(_decay_for_param(pt))))
        elif isinstance(p, _cp.ParamDeclAbstract):
            _, pt = resolve_type_from_decl(p.decl_specs, p.declarator)
            out.append(_ParamView(None,
                                  resolved_to_legacy(_decay_for_param(pt))))
        elif isinstance(p, _cp.ParamDeclTypeOnly):
            pt = resolve_base_type(p.decl_specs)
            out.append(_ParamView(None,
                                  resolved_to_legacy(_decay_for_param(pt))))
    return out


def _fd_return_type(self):
    from .codegen_helpers import resolve_type_from_decl
    _, fn_type = resolve_type_from_decl(self.decl_specs, self.declarator)
    inner = fn_type.return_type if fn_type.kind == "function" else fn_type
    return resolved_to_legacy(inner)


def resolved_to_legacy(rt):
    """Convert a ``ResolvedType`` to the equivalent ``ast_legacy`` tree
    (BasicType / PointerType / ArrayType / FunctionType / StructType /
    EnumType). Codegens that walk legacy types unchanged see them in
    the shape they expect."""
    from .codegen_helpers import ResolvedType as _RT
    from . import ast_legacy as _lt
    if rt is None:
        return None
    if not isinstance(rt, _RT):
        return rt  # already a legacy type or some other thing
    if rt.kind == "basic":
        return _lt.BasicType(
            name=rt.name or "int",
            is_signed=rt.is_signed,
            is_const=rt.is_const,
            is_volatile=rt.is_volatile,
        )
    if rt.kind == "pointer":
        return _lt.PointerType(
            base_type=resolved_to_legacy(rt.pointee),
            is_const=rt.is_const,
            is_volatile=rt.is_volatile,
        )
    if rt.kind == "array":
        out = _lt.ArrayType(
            base_type=resolved_to_legacy(rt.element),
            size=rt.size_expr,
        )
        if rt.is_vector:
            out.is_vector = True
        return out
    if rt.kind == "function":
        return _lt.FunctionType(
            return_type=resolved_to_legacy(rt.return_type),
            param_types=[resolved_to_legacy(p) for p in rt.param_types],
            is_variadic=rt.is_variadic,
        )
    if rt.kind == "struct":
        members = [
            _lt.StructMember(
                name=nm,
                member_type=resolved_to_legacy(mt),
                bit_width=bw,
            )
            for (nm, mt, bw) in rt.members
        ]
        return _lt.StructType(
            name=rt.name,
            is_union=rt.is_union,
            members=members,
            is_packed=rt.is_packed,
        )
    if rt.kind == "enum":
        return _lt.EnumType(name=rt.name)
    if rt.kind == "complex":
        return _lt.ComplexType(
            base_type=rt.name or "double",
            is_const=rt.is_const,
            is_volatile=rt.is_volatile,
        )
    if rt.kind == "typedef":
        # Unresolved typedef-name reference. uc_core has no typedef
        # table at this level; fall back to int.
        return _lt.BasicType(name=rt.name or "int")
    if rt.kind == "typeof":
        # `typeof(expr)` type specifier — hand the operand to the
        # host codegen as a legacy TypeofType; its pre-codegen pass
        # walks the operand and substitutes the concrete type.
        return _lt.TypeofType(operand=rt.typeof_operand)
    return None


_FunctionDef.name = property(_fd_name)
_FunctionDef.storage_class = property(_fd_storage_class)
_FunctionDef.is_inline = property(_fd_is_inline)
_FunctionDef.is_variadic = property(_fd_is_variadic)
_FunctionDef.params = property(_fd_params)
_FunctionDef.return_type = property(_fd_return_type)


# Legacy-compat properties on FloatLiteral. The auto-AST stores the raw
# source token in ``.value``; the legacy AST had a parsed float there
# plus is_float / is_imaginary / is_long flag bools. Re-derive them
# from the token text so codegens that read these attributes keep
# working without per-site changes.
from .c23_parser import FloatLiteral as _FloatLiteral


def _fl_is_imaginary(self):
    from ._const import float_is_imaginary
    return float_is_imaginary(self)


def _fl_is_float(self):
    from ._const import float_is_float
    return float_is_float(self)


def _fl_is_long(self):
    from ._const import float_is_long
    return float_is_long(self)


_FloatLiteral.is_imaginary = property(_fl_is_imaginary)
_FloatLiteral.is_float = property(_fl_is_float)
_FloatLiteral.is_long = property(_fl_is_long)


# Auto-AST splits unary into UnaryOp (prefix) vs PostfixOp (postfix).
# Legacy codegen branches on a unified ``is_prefix`` flag — attach the
# constant at class level so each variant answers correctly.
from .c23_parser import UnaryOp as _UnaryOp
from .c23_parser import PostfixOp as _PostfixOp
_UnaryOp.is_prefix = True
_PostfixOp.is_prefix = False


# Field aliases for legacy codegen sites:
#   GotoStmt.target  -> Token text of .label
#   StmtExpr.body    -> .stmt (the inner CompoundStmt)
#   VaArgExpr.ap     -> .va_list_expr
# The legacy AST exposed these as flat attrs; the auto-AST renamed them.
# Defining them as @property keeps the legacy callsites working.
from .c23_parser import (
    GotoStmt as _GotoStmt, StmtExpr as _StmtExpr, VaArgExpr as _VaArgExpr,
)
_GotoStmt.target = property(lambda self: self.label.text)
_StmtExpr.body = property(lambda self: self.stmt)
_VaArgExpr.ap = property(lambda self: self.va_list_expr)

# GenericSelection: legacy `.associations`, auto-AST `.assocs`.
from .c23_parser import GenericSelection as _GenericSelection
_GenericSelection.associations = property(lambda self: self.assocs)


# SequenceExpr is the auto-AST representation of `a, b` (the comma
# operator). uc386's legacy code handles the comma operator via
# BinaryOp(",", left, right). Expose a fake ``.op`` token so the
# existing `_opt(expr) == ","` dispatch sites detect SequenceExpr
# uniformly without needing to be touched.
from .c23_parser import SequenceExpr as _SequenceExpr, Token as _Token


def _seq_op(_self):
    return _Token(name="COMMA", text=",", line=0, column=0, offset=0, file_id=0)


_SequenceExpr.op = property(_seq_op)


# Auto-AST Cast / SizeofType / AlignofType / CompoundLiteral /
# CompoundLiteralEmpty carry ``target_type`` as an ``ast.TypeName`` /
# ``ast.TypeNameWithDeclarator`` wrapper. Legacy codegens read
# ``target_type`` and isinstance-check it against
# ``ast_legacy.{BasicType,PointerType,...}`` — convert at construction
# so consumers see the legacy shape directly.
def _convert_target_type(t):
    from .c23_parser import TypeName as _TN, TypeNameWithDeclarator as _TNWD
    from .codegen_helpers import (
        resolve_base_type as _rbt,
        resolve_type_from_decl as _rtfd,
    )
    if t is None:
        return None
    if isinstance(t, _TN):
        return resolved_to_legacy(_rbt(t.decl_specs))
    if isinstance(t, _TNWD):
        _, rt = _rtfd(t.decl_specs, t.declarator)
        return resolved_to_legacy(rt)
    return resolved_to_legacy(t)


def _wrap_init_convert_target(cls):
    """Wrap ``cls.__init__`` to convert ``self.target_type`` to legacy
    shape after construction. Done via __init__ wrap (not __post_init__)
    because dataclass-generated __init__ only calls __post_init__ when
    the method was present at @dataclass decoration time."""
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.target_type = _convert_target_type(self.target_type)
    cls.__init__ = __init__


from .c23_parser import (
    Cast as _Cast,
    CompoundLiteral as _CompoundLiteral,
    CompoundLiteralEmpty as _CompoundLiteralEmpty,
)
# SizeofType / AlignofType target_type is NOT converted at construction
# because the ast_optimizer's `_sizeof_type` reads it back as a
# ``ast.TypeName`` / ``ast.TypeNameWithDeclarator`` for compile-time
# folding. Codegens convert on demand via ``_to_legacy_type``.
for _c in (_Cast, _CompoundLiteral, _CompoundLiteralEmpty):
    _wrap_init_convert_target(_c)
