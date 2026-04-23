# uc_core

Shared C23 frontend and AST-level optimizer for the `uc80` (Z80/CP/M) and
`uc386` (x86-32/MS-DOS) retro C compilers.

## What's in here

- **Lexer / parser** — C23 source to AST
- **Preprocessor** — C23 preprocessor with target-configurable predefined macros
- **AST** — node definitions shared across backends
- **AST optimizer** — target-independent expression simplification, constant
  folding, strength reduction, dead-code elimination at the AST level
- **Backend protocol** — the contract a target backend must implement

## Targets using uc_core

- [uc80](https://github.com/avwohl/uc80) — Z80 / CP/M
- [uc386](https://github.com/avwohl/uc386) — x86-32 / MS-DOS

## Install

```
pip install uc_core
```

Or from source:

```
pip install -e .
```

## Usage from a backend

```python
from uc_core.lexer import Lexer
from uc_core.parser import Parser
from uc_core.preprocessor import Preprocessor
from uc_core.ast_optimizer import ASTOptimizer
from uc_core.backend import CodeGenerator  # Protocol

# Target-specific predefined macros
predefines = {
    "__UC80__": "1",
    "__Z80__": "1",
    "__CPM__": "1",
}

pp = Preprocessor(include_paths=["lib/include"], target_predefines=predefines)
source = pp.preprocess(open("hello.c").read(), "hello.c")
tokens = list(Lexer(source, "hello.c").tokenize())
ast = Parser(tokens).parse()
ast = ASTOptimizer(opt_level=3).optimize(ast)

# Backend is provided by the target package (uc80, uc386, ...)
# It must implement uc_core.backend.CodeGenerator
```

## License

GPL-3.0-or-later. See `LICENSE`.
