# Changelog

All notable changes to uc_core land here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the public
surface (the `uc_core` Python API, the AST node shapes re-exported from
`uc_core.ast`, and the preprocessor).

## Unreleased

### Added

- **`strict_allow` for strict mode.** `parse(..., strict_allow={"interrupt"})`
  lets one kind (`interrupt` or `fastcall`) pass, so a host can support one
  spelling and keep the others refused. A bare `interrupt` or `fastcall`
  before `*` is a marker only at the start of a declaration, so
  `interrupt * x` in an expression is a multiply. The strict scan covers
  header text, so shipped headers must not use refused markers.
- **Opt-in strict mode in `frontend.parse`.** `parse(..., strict_unsupported=True)`
  raises `UnsupportedConstructError` (`file:line: message`) for `__interrupt`,
  `_interrupt`, bare `interrupt`, `[[...interrupt...]]`,
  `__attribute__((interrupt))` and `__fastcall` / `_fastcall` / `fastcall`,
  where tolerant mode erases them silently. `unsupported_message` is a
  `str.format` template with `{name}`, so each host picks its own wording.
  Names that only contain the words, tags (`struct interrupt`) and string
  literals are not affected. The default is off; uc386 behaviour is unchanged.

### Fixed

- **Float detection is scoped to the function.** A float `x` in one
  function no longer stops `x * 8` becoming a shift on an integer `x` in
  another function (uc386 lost `shl` and `lea` that way).  A name declared
  in a function now takes its type from that function; other names take
  the file-scope declaration.
- **`(x << a) << b` and `(x >> a) >> b` are not merged when `a + b` is 16
  or more.**  The merge ignored the operand width, so
  `(y << 33) << 33` on a `long long` gave 4 and not 0.

- **The AST optimizer no longer applies integer-only rewrites to a float.**
  `x * 8` became `x << 3`, `x / 8u` became `x >> 3`, `(x * 3) * 5` folded
  to an integer constant that wraps in 16 bits, `x - x` and `x == x` folded
  to the int 0 and 1, `x * 0` and `0 / x` to the int 0, and `x + 0` to `x`.
  For a float or double `x` these gave wrong values (`float nv = 1.5f;
  nv * 4` was 1.5; uc386 converted the float to an integer, shifted and
  converted back). The optimizer knows no types, so the new
  `uc_core.floatness.FloatInfo` collects the names declared with a floating
  type (objects, functions, parameters, struct members; directly, through a
  typedef, as pointer or array, or with `typeof`). The rewrites above skip an
  operand that may be floating. A cast to a float type was never recognized
  (the cast target is a legacy `BasicType`, not `ast.BasicType`), so
  `(float)i * 8` was shifted too. Integer output of uc386 is unchanged.
- **`typeof(type-name)` parses.** The grammar takes only `typeof(expr)`, so
  `typeof(volatile unsigned char)`, `typeof(unsigned char *)` and
  `typeof(MyTypedef)` were parse errors. `frontend.parse` now rewrites the
  failing site to `typeof(*(type-name *)0)` and parses again (an array or
  function part gets its `(*)`: `typeof(int [3])` is `typeof(*(int (*)[3])0)`).
  Only a source that fails to parse and uses `typeof` pays for this.

- **The AST optimizer no longer removes, merges or repeats an access to a
  volatile object (C23 6.7.3).** It deleted adjacent stores to a volatile
  global, static, extern, local or typedef object (`g = 1; g = 2;`), a
  declaration initializer of a volatile object whose volatility came from
  a typedef or a pointer qualifier, and a store whose right side read a
  volatile object. It folded `v == v`, `v | v`, `(a | v) | v` and
  `v % 1`, turned `v * 2` into `v + v` (and, once guarded, into `v << 1`, which is
  wrong for a float; the multiply now stays), merged equal volatile reads
  through common subexpression elimination and copy propagation, and
  unrolled `for (volatile int i = 0; ...)`. The new module
  `uc_core.volatility` collects the names, typedefs, struct tags and
  struct members that are declared volatile (also through typedefs, pointer
  qualifiers and array element types); `ASTOptimizer.optimize` builds the
  table, and the optimizer keeps one test, `_touches_volatile`, for every
  rewrite that drops, repeats or merges an operand. `VolatileInfo` is
  public for other passes (uc80 uses it in its code generator). Code that
  mentions no volatile object is optimized as before. uc386 gets the fix
  with no change.

## 0.4.2

### Fixed

- **A label may precede an `asm(...)` statement.** `L: asm("nop");` was a
  parse error: the grammar (`c23.uplox` in uplox) took a label only before
  a statement or a declaration, and `asm` is an `asm_declaration` block
  item. C23 6.8.2 lets a label precede any statement. `c23.json` and
  `c23_parser.py` are regenerated from the changed grammar; the label node
  is a `LabelStmt` whose `stmt` is the `AsmDeclaration`. Found by the
  whetstone benchmark of uc80, whose timer macro is a label followed by
  an asm statement.

## 0.4.1 — 2026-08-20

Three fixes for cases where uc_core silently produced the wrong program
rather than an error: a DOS-qualifier strip that ate a struct tag, a
function-like macro invocation that was never expanded, and a lost
backslash in stringification. Plus `__has_include` / `#include_next`,
which libc++ headers gate their real includes on.

### Added

- **`__has_include` and `__has_include_next` in `#if` / `#elif`.** The
  operator was never implemented, and the condition evaluator's generic
  "unknown `IDENT(...)` becomes `0`" elision swallowed it silently — so
  `#if __has_include(<h>)` was always false. libc++'s `<ctime>` gates
  `#include <time.h>` on `__has_include(<time.h>)`, so the include was
  skipped even on hosts where the header existed and a direct
  `#include <time.h>` resolved fine; the failure surfaced later as
  missing declarations rather than as anything the preprocessor said.
  New `_expand_has_include` runs before macro expansion (the `<h>` /
  `"h"` header-name argument is not macro-expanded) and before the
  `IDENT(...)` elision, resolves the name the way `#include` would
  (angle-bracket names search the include-path list; quoted names also
  search the current file's directory) and substitutes `1` or `0`. The
  argument has to be a literal `<h>` or `"h"`: a header name that comes
  out of macro expansion, `__has_include(HEADER)`, still meets the
  `IDENT(...)` elision and reads false, and `__has_include` is not itself
  defined as a macro, so the common
  `#if defined(__has_include) && __has_include(<h>)` guard still takes
  its fallback branch.

- **`#include_next`.** The GCC/Clang extension: the header search starts
  *after* the directory holding the file that issued the directive, so a
  wrapper header can chain to the system one (libc++'s own `stddef.h`
  does `#include_next <stddef.h>`). `__has_include_next` honours the same
  skip. Implemented as `skip_current_dir=True` on `_process_include`,
  which prunes the search list up to and including the current file's
  directory.

### Changed

- **Include resolution is now shared between `#include` and
  `__has_include`.** The search loop moved out of `_process_include` into
  `Preprocessor._search_for_include` — exact match across the search
  paths first, then a case-insensitive scan so DOS-era sources that write
  `#include "global.h"` next to a real `GLOBAL.H` still resolve on
  case-sensitive hosts, with Windows-style backslashes in the name
  normalized to forward slashes. Both callers now agree on whether a
  header exists; what `#include` itself resolves to is unchanged. One
  diagnostic did move: a header that cannot be found is now named in the
  error as it was written, where 0.4.0 printed it after backslash
  normalization (`sys\stat.h`, not `sys/stat.h`).

- **The `uplox` dependency floor moves from `>=3.2.0` to `>=3.3.0`.** The
  generated parser bundle `src/uc_core/c23_parser.py` is unchanged in this
  release.

### Fixed

- **`near` / `far` / `huge` — and the other bare DOS qualifiers
  `cdecl`, `pascal`, `stdcall`, `fastcall`, `syscall`, `watcall`,
  `fortran` and `interrupt` — are no longer erased where they name a
  struct, union or enum tag.** uc_core strips the DOS-era qualifiers at
  the source level in `frontend.py` because the c23 grammar does not model
  them. The prefixed spellings (`__far`, `_huge`) are reserved, so erasing
  those is safe; the bare spellings are ordinary C identifiers — C23 has
  no `near`/`far`/`huge` — and a program is entitled to use the word. The
  lookahead already in `_DOS_QUALIFIER_RE` kept most uses safe (a name
  followed by `;`, `=` or `(` is never a qualifier, so `int huge;` and
  `huge = 1;` came through) but it cannot tell `struct huge mk` from
  `near int n`: both are the word followed by another word. Both halves
  of that were wrong. The loud one:

      struct huge { int a; };
      struct huge mk(void) { ... }

  became `struct mk(void) { ... }`, and uc80 reported
  `internal error: unexpected token 'KW_VOID'` against the line rather
  than the name — so it read as a compiler fault, not as a reserved word.
  The quiet one is worse: `struct huge v;` became `struct v;`, a
  well-formed forward declaration of a struct called `v`. Nothing was
  reported, the variable simply stopped existing, and the error landed
  later on whatever used it. A tag position is now matched first by the
  regex and put back verbatim by `_drop_dos_qualifier`. Only the word
  directly after `struct`/`union`/`enum` is treated as a tag, so
  `struct foo far *p` still loses its qualifier and `struct far far *p`
  keeps the tag and drops the qualifier — the DOS headers this pass
  exists for are unaffected.

  Still wrong: a typedef name. `typedef struct {...} huge; huge v;` has
  the same shape as `near int n`, with no keyword in front to
  disambiguate, and this pass has no typedef set to consult.

- **A function-like macro invocation whose `(` opens on the next line is
  now expanded.** [cpp.replace] puts no line constraint between a
  function-like macro's name and its `(` — only whitespace, and a newline
  is whitespace. Boost relies on it:

      BOOST_STATIC_CONSTANT
        (bool,
         value = ((sizeof(F) <= sizeof(function_buffer) && ...)));

  `_has_unclosed_macro_call` decides whether `preprocess()` glues the next
  source line on before expanding, and it was wrong in two independent
  ways, either of which alone loses the invocation. The paren scan skipped
  `' '` and `'\t'` but never `'\n'`; since the predicate is called per
  line, a macro name ending a line ran off the end and the call looked
  closed, so no continuation was ever merged. And once a line *had* been
  merged, the text reads `NAME\n   (args...`, where the name is again
  followed by `'\n'` — so the merge stopped after exactly one line and the
  invocation stayed unexpanded anyway. The skip class now includes `\n`
  and `\r`, and the predicate takes the next line, consulted only for the
  end-of-line case and only to confirm the continuation really opens with
  `(`: a bare function-like macro name at end of line is legal and is not
  an invocation, so merging unconditionally would swallow the following
  line, `#directive` and all. Both call sites are fixed — the top-level
  loop and the included-file path.

  An unexpanded invocation is not a diagnostic; it is a declaration that
  vanishes. In the ucpp386 Boost.Test build, `BOOST_STATIC_CONSTANT`
  survived unexpanded exactly once in a 4.7 MB translation unit, so the
  `value` member of
  `boost::detail::function::function_allows_small_object_optimization`
  was never declared at all. Nothing in the preprocessor's own output
  looked wrong. The Boost.Test crash first blamed on that missing member
  was afterwards traced to a separate ucpp386 fault, an unevaluable
  `alignment_of`, so what this fix removes is a silently lost
  declaration, not that crash.

- **Stringification no longer drops a backslash.**
  `_handle_stringification` passed the already-quoted argument as
  `re.sub`'s replacement string, and `re.sub` interprets backslash
  sequences there (`\1`, `\\`, …) — so the doubled backslashes the
  stringifier had just written were collapsed back to single ones.
  `EXPECT_EQ("'\"' (34, 0x22)", x)` expanded to
  `Compare("\"'\\"' (34, 0x22)\"", …)`, one `\` short before the inner
  quote, leaving an unbalanced string literal and a lexical error further
  down the pipeline. The replacement is now passed as a lambda so the text
  reaches the output verbatim.

### Documentation

- **README "Related Projects" rewritten in Simplified Technical
  English** (ASD-STE100 Issue 9): gerund heads and "via" removed,
  articles restored, semicolons replaced, and one canonical name per
  project across the sibling repos (rules 1.11 and 9.4). File names and
  test results are unchanged, and no other part of the README changed.
  The entries are not word-for-word fact-preserving: uc80 now reads as a
  C compiler rather than a C23 one, which matches its own README, and
  several entries gained a detail the old one-liners left out.

### Test surface

`tests/test_dos_tolerance.py` gains 17 cases across five functions, all
for the tag-position fix: tag declarations and references for
`struct`/`union`/`enum`, a tag through a pointer, a tag as a parameter,
`cdecl` and `interrupt` as tags, a check that the tag keeps its *name*
(so a later `struct huge` reference still resolves), a check that
`struct huge v;` still declares `v` rather than parsing as a forward
declaration, and the two mixed cases `struct foo far *p` and
`struct far far *p`.

The three preprocessor changes ship without tests in this repo; they were
found and verified downstream, in the ucpp386 Boost.Test build.

---

Earlier history predates this file — see the git log for anything before
0.4.0.
