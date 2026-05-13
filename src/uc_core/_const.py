"""Integer-literal value helpers for the declarator-shape AST.

The auto-AST's :class:`uc_core.ast.IntLiteral` carries a ``Token``
(``value`` field) whose ``text`` is the source representation
(``"42"`` / ``"0x10"`` / ``"010"`` / ``"42uLL"`` / etc.). Consumers
that need the integer value as a Python ``int`` go through
:func:`int_value`; constant-folding passes that mint new literals
go through :func:`make_int_lit` (which synthesises a fake Token).
"""

from __future__ import annotations

from . import ast


def int_value(lit) -> int:
    """Parse an ``IntLiteral``'s source text into a Python ``int``."""
    if not isinstance(lit, ast.IntLiteral):
        raise TypeError(f"int_value: expected IntLiteral, got {type(lit).__name__}")
    text = lit.value.text
    # Strip C integer-literal suffix.
    n = len(text)
    while n > 0 and text[n - 1] in "uUlLwWbB":
        n -= 1
    text = text[:n]
    text = text.replace("'", "")  # C23 digit separator
    if text.startswith(("0x", "0X")):
        return int(text, 16)
    if text.startswith(("0b", "0B")):
        return int(text, 2)
    if text.startswith("0") and len(text) > 1 and all(c in "01234567" for c in text[1:]):
        return int(text, 8)
    return int(text)


def int_flags(lit) -> tuple[bool, bool, bool, bool]:
    """Return (is_long, is_long_long, is_unsigned, is_hex) for an IntLiteral."""
    if not isinstance(lit, ast.IntLiteral):
        return False, False, False, False
    text = lit.value.text
    is_long = False
    is_long_long = False
    is_unsigned = False
    is_hex = text.startswith(("0x", "0X")) or (
        text.startswith("0") and len(text) > 1 and text[1] not in "xXbB."
    )
    n = len(text)
    while n > 0 and text[n - 1] in "uUlLwWbB":
        c = text[n - 1]
        if c in "uU":
            is_unsigned = True
        elif c in "lL":
            if is_long:
                is_long_long = True
            else:
                is_long = True
        n -= 1
    return is_long, is_long_long, is_unsigned, is_hex


def make_int_lit(value: int, *, is_hex: bool = False) -> "ast.IntLiteral":
    """Synthesise an ``IntLiteral`` for a folded constant.

    The synthetic Token has ``text = str(value)`` (or ``hex()`` when
    ``is_hex``) so future :func:`int_value` calls can decode it. The
    ``line`` / ``column`` / ``offset`` fields are zeroed.
    """
    text = hex(value) if is_hex else str(value)
    tok = _make_token("INT_LIT", text)
    pos = ast._Pos()
    return ast.IntLiteral(value=tok, pos=pos)


def _make_token(name: str, text: str):
    """Build a synthetic uplox Token (lex-side type, re-exported via c23_parser)."""
    from .c23_parser import Token
    return Token(name=name, text=text, line=0, column=0, offset=0, file_id=0)
