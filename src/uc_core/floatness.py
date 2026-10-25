"""Which expressions of a translation unit may have a floating type.

The AST optimizer knows no variable types, but several of its rewrites
are valid only for integers: ``x * 8`` -> ``x << 3``, ``x / 8u`` ->
``x >> 3``, ``(x * 3) * 5`` -> ``x * 15`` with a wrapped constant,
``x - x`` -> ``0``.  On a float or double ``x`` each of them gives a
wrong value or an invalid shift of a float.

``FloatInfo`` builds, in one walk over the unit, the table of names
whose declared type is (or may be) floating: objects, functions,
parameters and struct members declared with ``float``, ``double``,
``long double`` or the like, directly or through a typedef, as a
pointer or array of such, or with ``typeof``.  ``maybe_float`` then
answers for an expression.  The test is name-based and one-sided: a
name declared floating anywhere counts as floating at every use, so
shadowing can only cost an optimization, never a wrong rewrite.  An
integer-valued operator (comparison, ``%``, shift, bit operator, ``!``),
a cast to an integer or pointer type, ``sizeof`` and the arguments of a
call do not make their expression floating.
"""

from __future__ import annotations

from . import ast
from .volatility import _is_node, _text, children, declared_names

_FLOAT_KW = ("float", "double")

# Binary operators whose result is an integer in valid C.
_INT_BINOPS = frozenset(("<", ">", "<=", ">=", "==", "!=", "&&", "||",
                         "%", "<<", ">>", "&", "|", "^"))


class FloatInfo:
    """See the module docstring."""

    def __init__(self, tu=None) -> None:
        self.typedefs: set[str] = set()   # typedef names of floating types
        self.names: set[str] = set()      # objects and functions
        self.fields: set[str] = set()     # struct and union members
        if tu is not None:
            self._walk(tu)

    # ---- type tests

    def specs_float(self, root) -> bool:
        """Does a specifier list or type name mention a floating type?"""
        stack = list(root) if isinstance(root, (list, tuple)) else [root]
        while stack:
            n = stack.pop()
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            if not _is_node(n):
                continue
            cls = type(n).__name__
            if cls == "BasicTypeSpec":
                kw = _text(n.kw)
                if kw in _FLOAT_KW or kw.startswith("_Float") \
                        or kw.startswith("_Decimal") or kw == "_Complex":
                    return True
                continue
            if cls == "BasicType":       # a cast target is a resolved type
                nm = n.name or ""
                if "float" in nm or "double" in nm or nm in self.typedefs:
                    return True
                continue
            if cls in ("ComplexType", "TypeofType"):
                return True
            if cls == "TypedefName":
                if _text(n.name) in self.typedefs:
                    return True
                continue
            if cls == "TypedefNameSpec":
                if _text(n.name) in self.typedefs:
                    return True
                continue
            if cls in ("TypeofExpr", "TypeofUnqualExpr"):
                return True              # the operand's type is not known here
            stack.extend(children(n))
        return False

    # ---- queries

    def maybe_float(self, expr) -> bool:
        """May ``expr`` have a floating type?  True when unsure."""
        stack = [expr]
        while stack:
            n = stack.pop()
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            if not _is_node(n):
                continue
            cls = type(n).__name__
            if cls == "FloatLiteral":
                return True
            if cls == "Identifier":
                if _text(n.name) in self.names:
                    return True
                continue
            if cls in ("Member", "ArrowMember"):
                if _text(n.member) in self.fields:
                    return True
                continue                 # the object does not give the type
            if cls in ("SizeofExpr", "SizeofType", "AlignofType",
                       "OffsetofExpr"):
                continue
            if cls in ("Call", "CallNoArgs"):
                stack.append(n.func)     # the arguments do not give the type
                continue
            tt = getattr(n, "target_type", None)
            if tt is not None:           # Cast, compound literal, va_arg
                if self.specs_float(tt):
                    return True
                if cls == "Cast":
                    continue             # an integer or pointer cast
                stack.extend(children(n))
                continue
            if cls in ("TypeofExpr", "TypeofUnqualExpr", "StmtExpr"):
                return True
            if cls == "BinaryOp" and _text(n.op) in _INT_BINOPS:
                continue
            if cls == "UnaryOp" and _text(n.op) in ("!", "~"):
                continue
            stack.extend(children(n))
        return False

    # ---- collection

    def _walk(self, root) -> None:
        stack = [root]
        while stack:
            n = stack.pop()
            cls = type(n).__name__
            if cls == "Declaration":
                specs = n.decl_specs or []
                if self.specs_float(specs):
                    is_td = any(type(s).__name__ == "StorageClass"
                                and _text(s.kw) == "typedef" for s in specs)
                    for d in n.declarators or []:
                        (self.typedefs if is_td else self.names).update(
                            declared_names(d))
            elif cls in ("FunctionDef", "_FunctionDef"):
                if self.specs_float(n.decl_specs or []):
                    self.names.update(declared_names(n.declarator))
            elif cls in ("ParamDecl", "ParamDeclAbstract"):
                if self.specs_float(n.decl_specs or []):
                    self.names.update(declared_names(n.declarator))
            elif cls == "StructMember":
                if self.specs_float(n.decl_specs or []):
                    for d in n.declarators or []:
                        self.fields.update(declared_names(d))
            stack.extend(reversed(list(children(n, skip_params=False))))
