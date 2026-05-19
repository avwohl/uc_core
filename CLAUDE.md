# uc_core — Claude operating notes

C23 compiler front-end: preprocessor, lexer/parser, AST, and a
**target-neutral** AST optimizer. The codegen backends are separate
sibling repos that consume this package: `uc386` (i386/DOS), `uc80`
(Z80/8080), `uplm80`. See `README.md` for the public overview.

## AST shape (Phase 6 / uplox v3 auto-AST)

- `uc_core.ast` is a thin re-export of the auto-generated dataclasses
  from `examples/c23.uplox` (`from .c23_parser import *`). Nodes are
  declarator-shaped; literals/operators/identifiers carry `Token`s;
  source position is `.pos` (a `_Pos`), never `.location`.
- The grammar splits "optional part present/absent" into **distinct
  node kinds**, not one node with an optional field:
  `IfStmt`/`IfStmtElse`, `ReturnStmt`/`ReturnStmtValue`,
  `Call`/`CallNoArgs`; postfix `i++` is `PostfixOp` (not `UnaryOp`).
  Match the pair; don't assume the with-field variant.
- Legacy resolved-type shapes (`BasicType`, `PointerType`,
  `Statement`, `Expression`, ...) are `_Removed` tombstones in
  `ast.py` — `isinstance` against them is always False.

## ⚠️ `src/uc_core/ast_legacy.py` is load-bearing — do NOT delete it

It looks like a Phase-6 leftover but is the **`ResolvedType` →
legacy-type-tree interchange that codegen consumes**. Active
importers: `uc_core/ast.py`, `uc_core/codegen_helpers.py`, and **both
backends** — `uc386/src/uc386/codegen.py`, `uc80/src/uc80/codegen.py`.
Removing it breaks the active modules and both backends' codegen.

The genuinely dead Phase-6 shims (`ast_optimizer_legacy.py`,
`frontend_legacy.py`) were referenced nowhere and have been deleted.
`ast_legacy.py` is intentionally retained.

## AST optimizer

`src/uc_core/ast_optimizer.py` is fully migrated to the auto-AST and
functional (it was previously a silent no-op on function bodies).
`tests/test_ast_optimizer.py` is its regression corpus: invariants
(never-raises @ -O0..-O3; no-op program unchanged; side-effecting
operands / calls never duplicated) plus one test per transform.

Copy-prop safety lives in `_types_compatible_for_copy`: propagate only
when both vars have an identical conservative type key
(`_decl_type_key`) and neither is address-taken. Conservative by
design — a false negative is a missed optimization; a false positive
is a miscompile.

## Validating an optimizer / front-end change

uc386's suite executes compiled i386 via unicorn — the strongest
semantic oracle. Run all three after any change that affects codegen:

- `cd /Users/wohl/src/uc_core && python3.14 -m pytest -q`            (≈145 passed)
- `cd /Users/wohl/src/uc386 && .venv/bin/pytest tests/ -q`           (≈1355 passed)
- `cd /Users/wohl/src/uc80  && python3.14 -m pytest tests/ -q`       (≈136 passed)

## Toolchain

- Use Homebrew `python3.14` (system `python3` is 3.9, too old —
  `dataclass(kw_only=True)` needs ≥3.10). `uc_core` is editable-
  installed, so edits here are live for the backends.
