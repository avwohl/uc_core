"""Target-configurable integer type sizes.

Lets a backend driver choose int/long/pointer widths via CLI or defaults, and
exposes matching predefined macros so headers (limits.h, stdint.h, stddef.h,
inttypes.h) can adapt with `#if __SIZEOF_INT__ == 4` etc. without forking.

Defaults here are the Z80 "compact 16-bit" layout (the uc80 baseline); uc386
builds a Watcom flat-32 TypeConfig (int=4, long=4, ptr=4, long long=8).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TypeConfig:
    char_size: int = 1
    short_size: int = 2
    int_size: int = 2
    long_size: int = 4
    long_long_size: int = 8
    ptr_size: int = 2
    float_size: int = 4
    double_size: int = 4
    long_double_size: int = 4

    def __post_init__(self) -> None:
        # C requires: char <= short <= int <= long <= long long
        if not (self.char_size <= self.short_size <= self.int_size
                <= self.long_size <= self.long_long_size):
            raise ValueError(
                f"TypeConfig violates char<=short<=int<=long<=long long: "
                f"char={self.char_size} short={self.short_size} "
                f"int={self.int_size} long={self.long_size} "
                f"long long={self.long_long_size}")
        if self.char_size != 1:
            raise ValueError("char_size must be 1")

    # --- Byte sizes by type name ---------------------------------------

    def sizeof_basic(self, name: str) -> int | None:
        """Return byte size for a BasicType name, or None if not a basic type."""
        if name in ('char', 'signed char', 'unsigned char', 'bool', '_Bool'):
            return self.char_size
        if name in ('short', 'signed short', 'unsigned short',
                    'short int', 'unsigned short int'):
            return self.short_size
        if name in ('int', 'signed', 'signed int', 'unsigned', 'unsigned int'):
            return self.int_size
        if name in ('long', 'signed long', 'unsigned long',
                    'long int', 'unsigned long int'):
            return self.long_size
        if name in ('long long', 'signed long long', 'unsigned long long',
                    'long long int', 'unsigned long long int'):
            return self.long_long_size
        if name == 'float':
            return self.float_size
        if name == 'double':
            return self.double_size
        if name == 'long double':
            return self.long_double_size
        return None

    # --- Limits -------------------------------------------------------

    @staticmethod
    def _signed_max(size: int) -> int:
        return (1 << (size * 8 - 1)) - 1

    @staticmethod
    def _unsigned_max(size: int) -> int:
        return (1 << (size * 8)) - 1

    @property
    def int_max(self) -> int:  return self._signed_max(self.int_size)
    @property
    def uint_max(self) -> int: return self._unsigned_max(self.int_size)
    @property
    def long_max(self) -> int: return self._signed_max(self.long_size)
    @property
    def ulong_max(self) -> int: return self._unsigned_max(self.long_size)
    @property
    def long_long_max(self) -> int:  return self._signed_max(self.long_long_size)
    @property
    def ulong_long_max(self) -> int: return self._unsigned_max(self.long_long_size)
    @property
    def short_max(self) -> int:  return self._signed_max(self.short_size)
    @property
    def ushort_max(self) -> int: return self._unsigned_max(self.short_size)

    # --- Predefined macros --------------------------------------------

    def predefined_macros(self) -> dict[str, str]:
        """GCC-compatible __SIZEOF_* and __*_WIDTH__ macros.

        Backends should merge this into their target_predefines so period
        headers (limits.h, stdint.h) can conditionalize on width.
        """
        bits = lambda n: n * 8
        return {
            "__CHAR_BIT__":         "8",
            "__SIZEOF_CHAR__":      str(self.char_size),
            "__SIZEOF_SHORT__":     str(self.short_size),
            "__SIZEOF_INT__":       str(self.int_size),
            "__SIZEOF_LONG__":      str(self.long_size),
            "__SIZEOF_LONG_LONG__": str(self.long_long_size),
            "__SIZEOF_POINTER__":   str(self.ptr_size),
            "__SIZEOF_FLOAT__":     str(self.float_size),
            "__SIZEOF_DOUBLE__":    str(self.double_size),
            "__SIZEOF_LONG_DOUBLE__": str(self.long_double_size),
            "__SIZEOF_SIZE_T__":    str(self.ptr_size),
            "__SIZEOF_PTRDIFF_T__": str(self.ptr_size),
            "__CHAR_WIDTH__":       str(bits(self.char_size)),
            "__SHORT_WIDTH__":      str(bits(self.short_size)),
            "__INT_WIDTH__":        str(bits(self.int_size)),
            "__LONG_WIDTH__":       str(bits(self.long_size)),
            "__LONG_LONG_WIDTH__":  str(bits(self.long_long_size)),
            "__PTRDIFF_WIDTH__":    str(bits(self.ptr_size)),
            "__SIZE_WIDTH__":       str(bits(self.ptr_size)),
            "__CHAR_MAX__":         str(self._signed_max(self.char_size)),
            "__SCHAR_MAX__":        str(self._signed_max(self.char_size)),
            "__SHRT_MAX__":         str(self.short_max),
            "__INT_MAX__":          str(self.int_max),
            "__LONG_MAX__":         str(self.long_max) + "L",
            "__LONG_LONG_MAX__":    str(self.long_long_max) + "LL",
        }


# Common presets for backends to pick from --------------------------------

Z80_CPM = TypeConfig(
    char_size=1, short_size=2, int_size=2,
    long_size=4, long_long_size=8, ptr_size=2,
)

WATCOM_FLAT32 = TypeConfig(
    char_size=1, short_size=2, int_size=4,
    long_size=4, long_long_size=8, ptr_size=4,
)
