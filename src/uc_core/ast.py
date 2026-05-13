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
