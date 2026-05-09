"""uc_core - shared C23 frontend and AST optimizer for uc80/uc386."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("uc_core")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
