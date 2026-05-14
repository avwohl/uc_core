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
        # `0x` with nothing after (malformed or macro-expansion residue):
        # treat as 0 rather than crashing.
        return int(text[2:], 16) if len(text) > 2 else 0
    if text.startswith(("0b", "0B")):
        return int(text[2:], 2) if len(text) > 2 else 0
    if text.startswith("0") and len(text) > 1 and all(c in "01234567" for c in text[1:]):
        return int(text, 8)
    return int(text) if text else 0


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


def float_value(lit) -> float:
    """Parse a ``FloatLiteral``'s source text into a Python ``float``."""
    if not isinstance(lit, ast.FloatLiteral):
        raise TypeError(
            f"float_value: expected FloatLiteral, got {type(lit).__name__}"
        )
    text = lit.value.text
    # Strip suffix: f / F / l / L / i / I / j / J (and combinations like
    # `fi` for `float imaginary`). Also strip C23 decimal-float
    # suffixes: df / DF, dd / DD, dl / DL (_Decimal32 / _Decimal64 /
    # _Decimal128).
    n = len(text)
    while n > 0 and text[n - 1] in "fFlLiIjJdD":
        n -= 1
    text = text[:n]
    text = text.replace("'", "")  # C23 digit separator
    # Hex floats start with 0x / 0X and use 'p' for the exponent.
    if text.startswith(("0x", "0X")):
        return float.fromhex(text)
    return float(text)


def float_is_imaginary(lit) -> bool:
    """Return True if a FloatLiteral has an i/I/j/J imaginary suffix."""
    if not isinstance(lit, ast.FloatLiteral):
        return False
    text = lit.value.text
    return any(c in "iIjJ" for c in text)


def float_is_float(lit) -> bool:
    """Return True if a FloatLiteral has an f/F suffix (type float, vs default double)."""
    if not isinstance(lit, ast.FloatLiteral):
        return False
    text = lit.value.text
    return any(c in "fF" for c in text)


def float_is_long(lit) -> bool:
    """Return True if a FloatLiteral has an l/L suffix (type long double)."""
    if not isinstance(lit, ast.FloatLiteral):
        return False
    text = lit.value.text
    return any(c in "lL" for c in text)


def char_value(lit) -> int:
    """Decode a ``CharLiteral`` to its integer value.

    Handles the standard escape sequences (``\\n``, ``\\t``, ``\\xNN``,
    octal, ``\\u``/``\\U``) plus the encoding-prefix variants (``u'a'``,
    ``U'a'``, ``L'a'``, ``u8'a'``). Multi-character constants (``'ab'``)
    pack the bytes into a single int, MSB-first, matching gcc."""
    if not isinstance(lit, ast.CharLiteral):
        raise TypeError(
            f"char_value: expected CharLiteral, got {type(lit).__name__}"
        )
    text = lit.value.text
    # Strip encoding prefix.
    if text.startswith("u8"):
        text = text[2:]
    elif text.startswith(("u", "U", "L")):
        text = text[1:]
    # Strip the surrounding single quotes.
    if text.startswith("'") and text.endswith("'"):
        body = text[1:-1]
    else:
        body = text
    esc = {"n": 10, "t": 9, "r": 13, "\\": 92, "'": 39, '"': 34, "0": 0,
           "a": 7, "b": 8, "f": 12, "v": 11, "?": 63}
    out: list[int] = []
    i = 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in esc:
                out.append(esc[nxt])
                i += 2
                continue
            if nxt == "x":
                j = i + 2
                hex_digits = "0123456789abcdefABCDEF"
                while j < len(body) and body[j] in hex_digits:
                    j += 1
                out.append(int(body[i + 2:j], 16) & 0xFF)
                i = j
                continue
            if nxt in "01234567":
                j = i + 1
                while j < len(body) and j < i + 4 and body[j] in "01234567":
                    j += 1
                out.append(int(body[i + 1:j], 8) & 0xFF)
                i = j
                continue
            out.append(ord(nxt))
            i += 2
            continue
        out.append(ord(body[i]))
        i += 1
    if not out:
        return 0
    # Multi-character constant: pack MSB-first.
    val = 0
    for b in out:
        val = (val << 8) | (b & 0xFF)
    return val


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
