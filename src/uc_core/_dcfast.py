"""Cached ``dataclasses.is_dataclass`` and ``dataclasses.fields``.

The AST walkers of the optimizer and the code generators call these
millions of times for a program of some thousand lines; the standard
functions look the class up again on every call.  The answers depend on
the class only, so they are kept per class.  A class object (not an
instance) takes the standard path, so the results are the same as the
standard functions give.
"""
import dataclasses as _dc

_IS: dict = {}
_FIELDS: dict = {}


def is_dataclass(x) -> bool:
    t = type(x)
    r = _IS.get(t)
    if r is None:
        if issubclass(t, type):
            return _dc.is_dataclass(x)
        r = _IS[t] = hasattr(t, "__dataclass_fields__")
    return r


def fields(x):
    t = type(x)
    r = _FIELDS.get(t)
    if r is None:
        if issubclass(t, type):
            return _dc.fields(x)
        r = _dc.fields(x)           # TypeError for a non-dataclass
        _FIELDS[t] = r
    return r
