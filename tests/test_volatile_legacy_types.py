"""resolved_to_legacy keeps ``volatile`` for every kind that has the flag.

The cast ``(volatile u8 *) 0x4300`` (typedef name), ``(volatile struct d *)``
and a volatile enum type must reach the legacy tree with is_volatile set,
or a codegen drops the device access.
"""

from __future__ import annotations

import pytest

from uc_core import ast
from uc_core.codegen_helpers import ResolvedType as RT


@pytest.mark.parametrize("kind,kw", [
    ("typedef", {"name": "u8"}),
    ("struct", {"name": "dev"}),
    ("struct", {"name": "dev", "is_union": True}),
    ("enum", {"name": "e"}),
    ("basic", {"name": "char"}),
    ("complex", {"name": "double"}),
])
def test_volatile_kept(kind, kw):
    assert ast.resolved_to_legacy(RT(kind, is_volatile=True, **kw)).is_volatile
    assert not ast.resolved_to_legacy(RT(kind, **kw)).is_volatile


def test_volatile_typedef_pointee():
    p = RT("pointer", pointee=RT("typedef", name="u8", is_volatile=True))
    assert ast.resolved_to_legacy(p).base_type.is_volatile


def test_volatile_struct_pointee():
    p = RT("pointer", pointee=RT("struct", name="d", is_volatile=True))
    assert ast.resolved_to_legacy(p).base_type.is_volatile
