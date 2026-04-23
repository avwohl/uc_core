"""Backend protocol that target-specific code generators must implement.

A "backend" in uc_core is anything that consumes a post-optimized AST
(TranslationUnit) and emits target assembly as a string. The driver in a
target package (uc80, uc386, ...) owns I/O, runtime embedding, and
target-specific post-processing; uc_core only defines the seam between
the shared frontend/optimizer and whatever comes next.

This module is deliberately minimal. It's a contract, not a framework.
"""

from typing import Protocol, runtime_checkable

from . import ast


@runtime_checkable
class CodeGenerator(Protocol):
    """A target backend.

    Backends are constructed by the target's driver with whatever
    configuration they need (module name, optimization flags, printf
    feature sets, etc.) - those are not part of the shared contract
    because they vary per target.

    The one shared method is `generate`: take an optimized
    TranslationUnit, return target assembly text.
    """

    def generate(self, unit: ast.TranslationUnit) -> str:
        """Generate target assembly from an AST translation unit."""
        ...
