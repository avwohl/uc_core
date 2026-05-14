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
TypesCompatibleP = _Removed
LabelAddr = _Removed
AsmStmt = _Removed
Compound = _Removed  # legacy compound-literal (Phase 6 uses CompoundLiteral)
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
    with .name / .param_type / .size_side_effects)."""
    from .codegen_helpers import function_params, declarator_ident, resolve_type_from_decl

    class _ParamView:
        __slots__ = ("name", "param_type", "size_side_effects")

        def __init__(self, name, param_type):
            self.name = name
            self.param_type = param_type
            self.size_side_effects = None

    out = []
    for p in function_params(self):
        if isinstance(p, _CallNoArgs.__class__):  # never matches; placeholder
            continue
        from . import c23_parser as _cp
        if isinstance(p, _cp.ParamDecl):
            _, pt = resolve_type_from_decl(p.decl_specs, p.declarator)
            out.append(_ParamView(declarator_ident(p.declarator), pt))
        elif isinstance(p, _cp.ParamDeclAbstract):
            _, pt = resolve_type_from_decl(p.decl_specs, p.declarator)
            out.append(_ParamView(None, pt))
        elif isinstance(p, _cp.ParamDeclTypeOnly):
            from .codegen_helpers import resolve_base_type
            out.append(_ParamView(None, resolve_base_type(p.decl_specs)))
    return out


def _fd_return_type(self):
    from .codegen_helpers import resolve_type_from_decl
    _, fn_type = resolve_type_from_decl(self.decl_specs, self.declarator)
    return fn_type.return_type if fn_type.kind == "function" else fn_type


_FunctionDef.name = property(_fd_name)
_FunctionDef.storage_class = property(_fd_storage_class)
_FunctionDef.is_inline = property(_fd_is_inline)
_FunctionDef.is_variadic = property(_fd_is_variadic)
_FunctionDef.params = property(_fd_params)
_FunctionDef.return_type = property(_fd_return_type)
