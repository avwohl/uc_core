"""Which objects of a translation unit are volatile.

C23 5.1.2.3 and 6.7.3: every access to a volatile object happens as
written.  No pass that rewrites expressions or statements may elide,
merge, duplicate, reorder or drop one.  Volatility is a property of the
declared type, so a pass that sees only names needs a table of the names
whose declared type mentions ``volatile``: directly, through a typedef,
as a pointer qualifier (``T *volatile p``), in a pointer's target type
(``volatile T *p``), in an array's element type, or in a struct member.

``VolatileInfo`` builds that table in one walk over the unit.  The test it
answers is name-based and one-sided.  A name declared volatile anywhere in
its function (or at file scope) counts as volatile at every use in that
function, so shadowing can only cost an optimization, never an access.  A
struct member counts as volatile in every struct, a typedef or struct tag
whose type mentions ``volatile`` makes everything declared with it
volatile, and a cast to such a type is a volatile access.  When the unit
mentions ``volatile`` nowhere, every query is a constant False and costs
nothing.
"""

from __future__ import annotations

import dataclasses

from . import ast
from . import _dcfast

_FIELD_CACHE: dict[type, tuple[str, ...]] = {}

# Classes whose ``params`` field is a parameter list: the parameters are
# not part of the declared type of the object or of the function result.
_FN_DECLARATORS = ("FnDeclarator", "FnDeclaratorEmpty", "AbstractFn",
                   "AbstractFnBare", "AbstractFnBareEmpty", "AbstractFnEmpty",
                   "_FnDeclaratorEmpty")


def _is_node(x) -> bool:
    return _dcfast.is_dataclass(x) and not isinstance(x, type)


def children(node, skip_params: bool = True):
    """The AST nodes directly below ``node`` (items of lists included)."""
    cls = type(node)
    names = _FIELD_CACHE.get(cls)
    if names is None:
        names = tuple(f.name for f in _dcfast.fields(node)
                      if f.name not in ("pos", "location"))
        _FIELD_CACHE[cls] = names
    skip = skip_params and cls.__name__ in _FN_DECLARATORS
    for n in names:
        if skip and n == "params":
            continue
        v = getattr(node, n, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            for x in v:
                if _is_node(x):
                    yield x
        elif _is_node(v):
            yield v


def _text(x) -> str:
    if x is None:
        return ""
    return x.text if hasattr(x, "text") else (x if isinstance(x, str) else "")


def declared_names(node) -> list[str]:
    """The name declared by a declarator, struct-member declarator or
    init-declarator chain: a one-element list, or empty."""
    while node is not None:
        if isinstance(node, ast.Declarator):
            return [_text(node.name)]
        nxt = None
        for attr in ("inner", "declarator"):
            nxt = getattr(node, attr, None)
            if nxt is not None:
                break
        node = nxt
    return []


def _fn_params(declarator) -> list:
    """The ParamDecl nodes of the outermost function declarator."""
    node = declarator
    while node is not None:
        if type(node).__name__ in ("FnDeclarator", "_FnDeclaratorEmpty",
                                   "FnDeclaratorEmpty"):
            params = getattr(node, "params", None)
            if params is not None and not isinstance(params, list):
                params = getattr(params, "params", None)   # VariadicParams
            return list(params or [])
        node = getattr(node, "inner", None)
    return []


class VolatileInfo:
    """See the module docstring."""

    def __init__(self, tu=None) -> None:
        self.any = False                  # `volatile` occurs in the unit
        self.typedefs: set[str] = set()   # typedef names of volatile types
        self.tags: set[str] = set()       # struct/union tags with a volatile member
        self.fields: set[str] = set()     # names of volatile struct members
        self.file_names: set[str] = set() # volatile objects of file scope
        self._locals: dict[int, "_LiveNames"] = {}
        self._all: set[str] | None = None
        if tu is not None:
            self._collect(tu)

    # ---- queries

    def names_for(self, func):
        """Names that count as volatile inside the FunctionDef ``func``."""
        return self._locals.get(id(func), self.file_names)

    def all_names(self) -> set[str]:
        """Every name declared volatile anywhere in the unit."""
        if self._all is None:
            names = set(self.file_names)
            for loc in self._locals.values():
                names |= loc._b
            self._all = names
        return self._all

    def touches(self, expr, names) -> bool:
        """May ``expr`` access a volatile object?  (Conservative: True
        when unsure.)  ``names`` is the volatile-name set in force."""
        if not self.any or expr is None:
            return False
        stack = [expr]
        while stack:
            n = stack.pop()
            if isinstance(n, (list, tuple)):        # adjacent string pieces
                stack.extend(n)
                continue
            if not _is_node(n):
                continue
            cls = type(n).__name__
            if cls == "Identifier":
                if _text(n.name) in names:
                    return True
                continue
            if cls in ("Member", "ArrowMember"):
                if _text(n.member) in self.fields:
                    return True
            tt = getattr(n, "target_type", None)
            if tt is not None and self._legacy_type_volatile(tt):
                return True
            if cls in ("TypeofExpr", "TypeofUnqualExpr", "StmtExpr"):
                return True
            stack.extend(children(n))
        return False

    # ---- type tests

    def subtree_volatile(self, root) -> bool:
        """Does a specifier list or declarator mention a volatile type?"""
        stack = list(root) if isinstance(root, (list, tuple)) else [root]
        while stack:
            n = stack.pop()
            if isinstance(n, (list, tuple)):
                stack.extend(n)
                continue
            if not _is_node(n):
                continue
            cls = type(n).__name__
            if cls == "TypeQualifier":
                if _text(n.kw) == "volatile":
                    self.any = True
                    return True
                continue
            if cls == "TypedefNameSpec":
                if _text(n.name) in self.typedefs:
                    return True
                continue
            if cls in ("StructRef", "StructDef", "StructEmpty"):
                if _text(getattr(n, "name", None)) in self.tags:
                    return True
            if cls in ("TypeofExpr", "TypeofUnqualExpr"):
                self.any = True
                return True              # the operand's type is not known here
            stack.extend(children(n))
        return False

    def _legacy_type_volatile(self, t) -> bool:
        """A cast, compound-literal or va_arg target type."""
        stack = [t]
        while stack:
            n = stack.pop()
            if not _is_node(n):
                continue
            cls = type(n).__name__
            if getattr(n, "is_volatile", False):
                return True
            nm = getattr(n, "name", None)
            if isinstance(nm, str):
                if nm in self.typedefs:
                    return True
                if cls == "StructType" and nm in self.tags:
                    return True
            if cls in ("TypeName", "TypeNameWithDeclarator"):
                if self.subtree_volatile([n]):
                    return True
                continue
            stack.extend(children(n))
        return False

    # ---- collection

    def _collect(self, tu) -> None:
        # Pass over the whole unit in source order: typedefs, tags, members
        # and file-scope objects.  Function bodies are walked with a
        # per-function sink, so a local's name stays local.
        self._walk(tu, self.file_names, None)

    def _walk(self, root, sink: set[str], func) -> None:
        stack = [(root, sink)]
        while stack:
            n, sk = stack.pop()
            cls = type(n).__name__
            if cls == "TypeQualifier":
                if _text(n.kw) == "volatile":
                    self.any = True
                continue
            if getattr(n, "is_volatile", False):
                self.any = True
            if cls == "Declaration":
                specs = n.decl_specs or []
                spec_vol = self.subtree_volatile(specs)
                is_td = any(type(s).__name__ == "StorageClass"
                            and _text(s.kw) == "typedef" for s in specs)
                for d in n.declarators or []:
                    if spec_vol or self.subtree_volatile(d):
                        (self.typedefs if is_td else sk).update(
                            declared_names(d))
            elif cls in ("StructDef", "StructAnon"):
                tag_vol = False
                for m in n.members or []:
                    if type(m).__name__ != "StructMember":
                        continue
                    mspec = self.subtree_volatile(m.decl_specs or [])
                    for d in m.declarators or []:
                        if mspec or self.subtree_volatile(d):
                            tag_vol = True
                            self.fields.update(declared_names(d))
                nm = _text(getattr(n, "name", None))
                if tag_vol and nm:
                    self.tags.add(nm)
            elif cls in ("FunctionDef", "_FunctionDef"):
                if self.subtree_volatile(n.decl_specs or []) \
                        or self.subtree_volatile(n.declarator):
                    sk.update(declared_names(n.declarator))
                loc: set[str] = set()
                for p in _fn_params(n.declarator):
                    if self.subtree_volatile(getattr(p, "decl_specs", None)
                                             or []) \
                            or self.subtree_volatile(
                                getattr(p, "declarator", None)):
                        loc.update(declared_names(
                            getattr(p, "declarator", None)))
                # Parameters of one function are walked first, then the
                # body, in source order (pop order: push reversed).
                kids = [(c, loc) for c in children(n)]
                stack.extend(reversed(kids))
                # The function's local table is final once its body has
                # been walked; record the live set (file names are added
                # when queried, below).
                self._locals[id(n)] = _LiveNames(self.file_names, loc)
                continue
            kids = [(c, sk) for c in children(n)]
            stack.extend(reversed(kids))


class _LiveNames:
    """File-scope names plus one function's names, as a set-like view
    (the file-scope set can still grow while the unit is walked)."""

    __slots__ = ("_a", "_b")

    def __init__(self, a: set[str], b: set[str]) -> None:
        self._a = a
        self._b = b

    def __contains__(self, x) -> bool:
        return x in self._b or x in self._a
