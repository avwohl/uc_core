"""AST-level expression optimizer for uc80.

Performs bottom-up transformations on expression nodes before codegen:
- Constant folding (all integer widths: 16/32/64-bit)
- Strength reduction (multiply/divide/modulo by power-of-2 → shifts/masks)
- Algebraic simplifications (identity/zero/full-mask elements)
- Dead code elimination (constant conditions, unreachable code)
- Double negation / NOT elimination
- Comparison simplifications (x == x → 1, etc.)
- Idempotent boolean simplifications
- Nested constant folding ((x + c1) + c2 → x + (c1+c2))
- Multi-pass optimization until convergence

Level 3 (-O3) adds:
- Commutative normalization (canonical operand ordering)
- Common subexpression elimination (CSE)
- Copy propagation
- Dead store elimination
- Loop-invariant code motion
- Loop unrolling (small constant-bound loops)
"""

import copy
import dataclasses
from . import ast
from ._const import int_value, int_flags, make_int_lit
from .c23_parser import Token
from .codegen_helpers import make_identifier
from .type_config import TypeConfig, Z80_CPM


def _iv(lit):
    """Shorthand for int_value when the caller already knows lit is an IntLiteral."""
    return int_value(lit)


# --- auto-AST node helpers -------------------------------------------------
# The Phase-6 auto-AST stores BinaryOp/UnaryOp `.op` as a Token (not a
# str) and carries source position on `.pos` (no `.location`). These
# helpers let the optimizer read operators uniformly and mint
# replacement nodes in the auto-AST shape.

def _op(node) -> str:
    """Operator text of a BinaryOp/UnaryOp. Tolerates a bare str so
    optimizer-minted nodes and any legacy callers both work."""
    o = node.op
    return o.text if hasattr(o, "text") else o


def _idname(ident) -> str:
    """Name text of an Identifier. The auto-AST stores `.name` as a
    Token; the optimizer keys its dataflow tables (copy/CSE/var-type/
    address-taken) by str, so normalize here. Tolerates a bare str."""
    n = ident.name
    return n.text if hasattr(n, "text") else n


# The auto-AST splits function calls into Call (with args) and
# CallNoArgs (no args; no `.args` field) — the same present/absent
# node-split as IfStmt/IfStmtElse. A no-arg call still has side
# effects, so both must be matched everywhere a call is analysed.
_CALL = (ast.Call, ast.CallNoArgs)


def _call_args(node) -> list:
    """Argument list of a Call/CallNoArgs (CallNoArgs has none)."""
    return getattr(node, "args", None) or []


def _optok(text: str) -> Token:
    """Synthetic operator/punctuator Token for an optimizer-minted node."""
    return Token(name="OP", text=text, line=0, column=0, offset=0, file_id=0)


def _binop(op: str, left, right) -> "ast.BinaryOp":
    return ast.BinaryOp(left=left, op=_optok(op), right=right)


def _unop(op: str, operand) -> "ast.UnaryOp":
    return ast.UnaryOp(op=_optok(op), operand=operand)


def _expr_has_float(node) -> bool:
    """Lexical detection: does `node`'s subtree contain a FloatLiteral
    or a Cast to a floating type? Used by strength-reduction passes
    that mustn't fire on expressions whose runtime value is a float.

    Lexical (no type-resolution) so it's correct for the common period-
    code shapes (`-.5 * (1<<16)`, `1.0 + x`) without needing the full
    type checker. False negatives on opaque calls are fine — we'd just
    miss a strength reduction; we never WRONGLY strength-reduce."""
    if node is None:
        return False
    if isinstance(node, ast.FloatLiteral):
        return True
    if isinstance(node, ast.Cast):
        t = node.target_type
        if isinstance(t, ast.BasicType) and t.name in ("float", "double", "long double"):
            return True
    if isinstance(node, list):
        return any(_expr_has_float(c) for c in node)
    if dataclasses.is_dataclass(node):
        for f in dataclasses.fields(node):
            if _expr_has_float(getattr(node, f.name, None)):
                return True
    return False


class ASTOptimizer:
    def __init__(self, opt_level: int = 2, type_config: TypeConfig | None = None):
        self.stats: dict[str, int] = {}
        self._changed = False
        self.opt_level = opt_level
        self.type_config = type_config if type_config is not None else Z80_CPM
        # Level 3 state (reset per function):
        self._cse_cache: dict[str, ast.Expression] = {}
        self._cse_deps: dict[str, set[str]] = {}
        self._copies: dict[str, str] = {}
        self._var_types: dict[str, ast.TypeNode] = {}  # Track declared types for safe copy prop
        self._address_taken_vars: set[str] = set()

    def optimize(self, tu: ast.TranslationUnit) -> ast.TranslationUnit:
        """Optimize all expressions in the translation unit (multi-pass)."""
        max_passes = 5
        for _ in range(max_passes):
            self._changed = False
            for item in tu.items or []:
                self._optimize_decl(item)
            if not self._changed:
                break
        return tu

    # === Declaration walkers ===

    def _optimize_decl(self, decl) -> None:
        if isinstance(decl, ast.FunctionDef):
            if decl.body is not None:
                if self.opt_level >= 3:
                    self._reset_level3_state()
                    # Track parameter names from the declarator chain;
                    # types are now declarator-shaped and we no longer
                    # carry them as resolved types in _var_types.
                    for name in _function_param_names(decl.declarator):
                        self._var_types[name] = None
                    self._collect_address_taken(decl.body)
                self._optimize_stmt(decl.body)
        elif isinstance(decl, ast.Declaration):
            # A Declaration may carry zero or many init_declarators.
            # Each ``InitDeclaratorWithInit`` has an initializer expression
            # we want to fold; ``InitDeclarator`` has none.
            for init_decl in decl.declarators or []:
                if isinstance(init_decl, ast.InitDeclaratorWithInit):
                    init_decl.init = self._optimize_expr(init_decl.init)

    # === Statement walkers (with dead code elimination) ===

    def _optimize_stmt(self, stmt: ast.Statement) -> None:
        if isinstance(stmt, ast.CompoundStmt):
            new_items: list = []
            replaced = False  # a 1:1 substitution (e.g. loop unroll)
            unreachable = False
            for item in stmt.items:
                if unreachable:
                    # Keep items that contain labels/cases (reachable via goto/switch)
                    if self._contains_label(item):
                        unreachable = False
                        # ast.Statement/ast.Expression are _Removed
                        # tombstones in the auto-AST; dispatch by the
                        # one concrete category instead — a block item
                        # is a declaration or (everything else) a
                        # statement.
                        if isinstance(item, (ast.Declaration, ast.FunctionDef)):
                            self._optimize_decl(item)
                        else:
                            self._optimize_stmt(item)
                        new_items.append(item)
                    else:
                        self._stat("dead_code")
                        self._changed = True
                    continue

                # Level 3: try loop unrolling before normal optimization
                if self.opt_level >= 3 and isinstance(item, ast.ForStmt):
                    unrolled = self._try_loop_unroll(item)
                    if unrolled is not None:
                        # Unrolled into a CompoundStmt; optimize its contents
                        self._optimize_stmt(unrolled)
                        new_items.append(unrolled)
                        replaced = True  # same length, must still write back
                        continue

                if isinstance(item, (ast.Declaration, ast.FunctionDef)):
                    self._optimize_decl(item)
                else:
                    self._optimize_stmt(item)
                new_items.append(item)

                # Level 3: invalidate caches after processing statement
                if self.opt_level >= 3:
                    self._invalidate_caches_for_stmt(item)

                # Check if this statement is a terminator
                if isinstance(item, (ast.ReturnStmt, ast.ReturnStmtValue,
                                     ast.GotoStmt, ast.BreakStmt,
                                     ast.ContinueStmt)):
                    unreachable = True
            if replaced or len(new_items) != len(stmt.items):
                stmt.items = new_items

            # Level 3: dead store elimination on the final list
            if self.opt_level >= 3:
                stmt.items = self._eliminate_dead_stores(stmt.items)

        elif isinstance(stmt, ast.ExpressionStmt):
            if stmt.expr is not None:
                stmt.expr = self._optimize_expr(stmt.expr)

        elif isinstance(stmt, (ast.IfStmt, ast.IfStmtElse)):
            # The c23 grammar emits two distinct nodes: `IfStmt`
            # (no else, fields condition/then_branch) and
            # `IfStmtElse` (adds else_branch). Read the else via
            # getattr so a plain IfStmt is uniformly "no else", and
            # only write `.else_branch` back when the attribute
            # actually exists (IfStmtElse).
            has_else_attr = hasattr(stmt, "else_branch")
            stmt.condition = self._optimize_expr(stmt.condition)
            else_b = getattr(stmt, "else_branch", None)
            # Constant condition → dead code elimination
            # But only if eliminated branches don't contain labels (goto targets)
            if isinstance(stmt.condition, ast.IntLiteral):
                if _iv(stmt.condition) != 0:
                    # Always true: eliminate else if it has no labels
                    self._optimize_stmt(stmt.then_branch)
                    if else_b is not None:
                        if not self._contains_label(else_b):
                            self._stat("dead_code")
                            self._changed = True
                            if has_else_attr:
                                stmt.else_branch = None
                        else:
                            self._optimize_stmt(else_b)
                else:
                    # Always false
                    then_has_labels = self._contains_label(stmt.then_branch)
                    if not then_has_labels:
                        self._stat("dead_code")
                        self._changed = True
                        if else_b is not None:
                            self._optimize_stmt(else_b)
                            stmt.then_branch = else_b
                            if has_else_attr:
                                stmt.else_branch = None
                            stmt.condition = make_int_lit(1)
                        else:
                            stmt.then_branch = ast.CompoundStmt(items=[])
                    else:
                        # Then-branch has labels, can't eliminate
                        self._optimize_stmt(stmt.then_branch)
                        if else_b is not None:
                            self._optimize_stmt(else_b)
            else:
                self._optimize_stmt(stmt.then_branch)
                if else_b is not None:
                    self._optimize_stmt(else_b)

        elif isinstance(stmt, ast.WhileStmt):
            # The body may run multiple times and re-enter the
            # condition with mutated state. Copy/CSE caches built
            # before the loop don't apply to subsequent iterations,
            # so clear BEFORE optimizing the condition too — otherwise
            # `int *p = head; while (p) { ...; p = next; }` rewrites
            # the loop condition's `p` back to `head` (the pre-loop
            # copy) and the loop runs forever.
            if self.opt_level >= 3:
                self._clear_all_caches()
            stmt.condition = self._optimize_expr(stmt.condition)
            # while(0) → dead loop body (only if no labels inside)
            if (isinstance(stmt.condition, ast.IntLiteral) and _iv(stmt.condition) == 0
                    and not self._contains_label(stmt.body)):
                self._stat("dead_code")
                self._changed = True
                stmt.body = ast.CompoundStmt(items=[])
            else:
                if self.opt_level >= 3:
                    self._clear_all_caches()
                self._optimize_stmt(stmt.body)
                if self.opt_level >= 3:
                    self._clear_all_caches()

        elif isinstance(stmt, ast.DoWhileStmt):
            if self.opt_level >= 3:
                self._clear_all_caches()
            self._optimize_stmt(stmt.body)
            if self.opt_level >= 3:
                self._clear_all_caches()
            stmt.condition = self._optimize_expr(stmt.condition)

        elif isinstance(stmt, ast.ForStmt):
            if stmt.init is not None:
                if isinstance(stmt.init, ast.Declaration):
                    self._optimize_decl(stmt.init)
                else:
                    stmt.init = self._optimize_expr(stmt.init)
            # Clear caches BEFORE condition / update — they re-evaluate
            # on subsequent iterations after the body has potentially
            # mutated state. Same shape as the WhileStmt fix.
            if self.opt_level >= 3:
                self._clear_all_caches()
            if stmt.condition is not None:
                stmt.condition = self._optimize_expr(stmt.condition)
            if stmt.update is not None:
                stmt.update = self._optimize_expr(stmt.update)
            if self.opt_level >= 3:
                self._clear_all_caches()
            self._optimize_stmt(stmt.body)
            if self.opt_level >= 3:
                self._clear_all_caches()

        elif isinstance(stmt, ast.SwitchStmt):
            stmt.expr = self._optimize_expr(stmt.expr)
            self._optimize_stmt(stmt.body)

        elif isinstance(stmt, ast.CaseStmt):
            if stmt.value is not None:
                stmt.value = self._optimize_expr(stmt.value)
            self._optimize_stmt(stmt.stmt)

        elif isinstance(stmt, ast.LabelStmt):
            self._optimize_stmt(stmt.stmt)

        elif isinstance(stmt, ast.ReturnStmtValue):
            stmt.value = self._optimize_expr(stmt.value)

    @staticmethod
    def _contains_label(node) -> bool:
        """Check if a statement (or any nested statement) contains a label or case."""
        if isinstance(node, (ast.LabelStmt, ast.CaseStmt)):
            return True
        if isinstance(node, ast.CompoundStmt):
            return any(ASTOptimizer._contains_label(item) for item in node.items)
        if isinstance(node, (ast.IfStmt, ast.IfStmtElse)):
            if ASTOptimizer._contains_label(node.then_branch):
                return True
            else_b = getattr(node, "else_branch", None)
            if else_b and ASTOptimizer._contains_label(else_b):
                return True
        if isinstance(node, (ast.WhileStmt, ast.DoWhileStmt, ast.ForStmt)):
            return ASTOptimizer._contains_label(node.body)
        if isinstance(node, ast.SwitchStmt):
            return ASTOptimizer._contains_label(node.body)
        return False

    # === Expression optimizer (bottom-up) ===

    def _optimize_expr(self, expr: ast.Expression) -> ast.Expression:
        """Recursively optimize an expression bottom-up."""
        # Level 3: copy propagation on identifiers
        if self.opt_level >= 3 and isinstance(expr, ast.Identifier):
            nm = _idname(expr)
            if nm in self._copies:
                self._stat("copy_prop")
                self._changed = True
                return make_identifier(self._copies[nm])

        if isinstance(expr, ast.BinaryOp):
            # For assignments, protect LHS from copy propagation
            if _op(expr) in ("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
                           "^=", "<<=", ">>="):
                # Optimize sub-expressions in LHS (array indices, pointer deref)
                # but NOT a simple Identifier target (no copy prop on lvalues)
                if isinstance(expr.left, ast.Index):
                    expr.left.array = self._optimize_expr(expr.left.array)
                    expr.left.index = self._optimize_expr(expr.left.index)
                elif isinstance(expr.left, ast.Member):
                    expr.left.obj = self._optimize_expr(expr.left.obj)
                expr.right = self._optimize_expr(expr.right)
                return self._optimize_binary(expr)
            expr.left = self._optimize_expr(expr.left)
            expr.right = self._optimize_expr(expr.right)
            result = self._optimize_binary(expr)
            # Level 3: CSE after binary optimization
            if self.opt_level >= 3:
                result = self._cse_lookup(result)
            return result
        elif isinstance(expr, ast.UnaryOp):
            # Don't optimize operand of address-of or dereference or ++/--
            if _op(expr) in ("&", "*", "++", "--"):
                return expr
            expr.operand = self._optimize_expr(expr.operand)
            result = self._optimize_unary(expr)
            # Level 3: CSE after unary optimization
            if self.opt_level >= 3:
                result = self._cse_lookup(result)
            return result
        elif isinstance(expr, ast.TernaryOp):
            expr.condition = self._optimize_expr(expr.condition)
            expr.true_expr = self._optimize_expr(expr.true_expr)
            expr.false_expr = self._optimize_expr(expr.false_expr)
            # Constant condition folding
            if isinstance(expr.condition, ast.IntLiteral):
                self._stat("ternary_fold")
                self._changed = True
                return expr.true_expr if _iv(expr.condition) != 0 else expr.false_expr
            return expr
        elif isinstance(expr, _CALL):
            if isinstance(expr, ast.Call):
                expr.args = [self._optimize_expr(a) for a in expr.args]
            return expr
        elif isinstance(expr, ast.Index):
            expr.array = self._optimize_expr(expr.array)
            expr.index = self._optimize_expr(expr.index)
            return expr
        elif isinstance(expr, ast.Cast):
            expr.expr = self._optimize_expr(expr.expr)
            return expr
        elif isinstance(expr, ast.SizeofType):
            # Constant fold sizeof(type) at compile time.
            size = self._sizeof_type(expr.target_type)
            if size is not None:
                self._stat("sizeof_fold")
                self._changed = True
                return make_int_lit(size)
            return expr
        elif isinstance(expr, ast.SizeofExpr):
            # sizeof(string_literal) → array length including null terminator.
            # In the new AST a string literal in expression position is
            # either a single ``StringLiteral`` or a list of them (for
            # adjacent literal concat); compute the total decoded length.
            inner = expr.operand
            strlit_len = _string_literal_byte_length(inner)
            if strlit_len is not None:
                self._stat("sizeof_fold")
                self._changed = True
                return make_int_lit(strlit_len + 1)
            return expr
        elif isinstance(expr, ast.InitializerList):
            # _optimize_expr is total: it returns non-expression nodes
            # (nested InitializerList, designators) unchanged, so it is
            # safe to map over every value here.
            expr.values = [self._optimize_expr(v) for v in expr.values]
            return expr
        return expr

    # === Binary operation optimizer ===

    def _optimize_binary(self, expr: ast.BinaryOp) -> ast.Expression:
        op = _op(expr)
        left = expr.left
        right = expr.right

        # Skip assignment operators
        if op in ("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="):
            return expr

        # Skip logical short-circuit and comma (side effects)
        if op in ("&&", "||", ","):
            return expr

        # === Constant folding: both operands are IntLiteral ===
        if isinstance(left, ast.IntLiteral) and isinstance(right, ast.IntLiteral):
            # Use wider mask of the two operands
            mask = max(self._literal_mask(left), self._literal_mask(right))
            # C standard 6.4.4.1: hex/octal literals that exceed signed range
            # are effectively unsigned (e.g., 0xab00 is unsigned int on 16-bit)
            unsigned = (self._is_effectively_unsigned(left) or
                        self._is_effectively_unsigned(right))
            # `is_long` propagates from operand suffixes; widening from
            # `mask` is only meaningful if the mask exceeds the target
            # int width. On 32-bit-int targets, mask=0xFFFFFFFF doesn't
            # mean "long" (which is also 32-bit); only mask > uint_max
            # would force a long-or-wider promotion.
            is_long = int_flags(left)[0] or int_flags(right)[0] or mask > self.type_config.uint_max
            is_long_long = (
                int_flags(left)[1]
                or int_flags(right)[1]
                or mask > self.type_config.ulong_max
            )
            result = self._fold_constants(op, _iv(left), _iv(right), unsigned, mask)
            if result is not None:
                # Convert masked result back to signed representation so that
                # _is_long_expr doesn't incorrectly promote to 32-bit.
                # e.g., (1 - 2) & 0xFFFF = 65535 must be stored as -1.
                if not unsigned:
                    half = (mask + 1) >> 1
                    if result >= half:
                        result -= (mask + 1)
                self._stat("const_fold")
                self._changed = True
                return make_int_lit(result)

        # === Strength reduction: multiply by power-of-2 ===
        if op == "*":
            result = self._strength_reduce_mul(expr)
            if result is not None:
                return result

        # === Strength reduction: divide/modulo by power-of-2 (unsigned only) ===
        if op == "/" and isinstance(right, ast.IntLiteral):
            shift = self._log2_if_power_of_2(_iv(right))
            if shift is not None and self._is_unsigned_literal(left, right):
                self._stat("div_to_shift")
                self._changed = True
                return _binop(">>", left, make_int_lit(shift))

        if op == "%" and isinstance(right, ast.IntLiteral):
            shift = self._log2_if_power_of_2(_iv(right))
            if shift is not None and self._is_unsigned_literal(left, right):
                self._stat("mod_to_and")
                self._changed = True
                return _binop("&", left, make_int_lit(_iv(right) - 1))

        # === Algebraic identity elements ===
        r = self._simplify_identity(expr)
        if r is not None:
            return r

        # === Algebraic zero elements ===
        r = self._simplify_zero(expr)
        if r is not None:
            return r

        # === Full-mask identities ===
        r = self._simplify_full_mask(expr)
        if r is not None:
            return r

        # === Self-referential identities (only for side-effect-free operands) ===
        if self._is_same_identifier(left, right):
            if op == "&" or op == "|":
                self._stat("self_identity")
                self._changed = True
                return left
            if op == "^" or op == "-":
                self._stat("self_zero")
                self._changed = True
                return make_int_lit(0)
            # Comparison simplifications
            if op == "==":
                self._stat("self_cmp")
                self._changed = True
                return make_int_lit(1)
            if op == "!=":
                self._stat("self_cmp")
                self._changed = True
                return make_int_lit(0)
            if op == "<" or op == ">":
                self._stat("self_cmp")
                self._changed = True
                return make_int_lit(0)
            if op == "<=" or op == ">=":
                self._stat("self_cmp")
                self._changed = True
                return make_int_lit(1)

        # === Idempotent boolean simplifications ===
        r = self._simplify_idempotent(expr)
        if r is not None:
            return r

        # === Nested constant folding ===
        r = self._nested_const_fold(expr)
        if r is not None:
            return r

        # === Level 3: Commutative normalization ===
        if self.opt_level >= 3:
            r = self._normalize_commutative(expr)
            if r is not None:
                return r

        return expr

    # === Unary operation optimizer ===

    def _optimize_unary(self, expr: ast.UnaryOp) -> ast.Expression:
        op = _op(expr)
        operand = expr.operand

        # Constant folding for unary operations
        if isinstance(operand, ast.IntLiteral):
            val = _iv(operand)
            mask = self._literal_mask(operand)
            result = None
            if op == "-":
                result = (-val) & mask
            elif op == "+":
                result = val
            elif op == "~":
                result = (~val) & mask
            elif op == "!":
                result = 1 if val == 0 else 0
            # Convert masked result back to signed representation so that
            # _is_long_expr doesn't incorrectly promote to 32-bit.
            # e.g., -(1) & 0xFFFF = 65535 must be stored as -1, not 65535,
            # otherwise 65535 > 32767 triggers unwanted long promotion.
            if result is not None and not int_flags(operand)[2] and op in ("-", "~"):
                half = (mask + 1) >> 1
                if result >= half:
                    result -= (mask + 1)
            if result is not None:
                self._stat("const_fold_unary")
                self._changed = True
                # Per C99 6.5.3.3: ! operator always returns int (not unsigned, not long)
                if op == "!":
                    return make_int_lit(result)
                return make_int_lit(result)

        # Double negation: -(-x) → x, ~(~x) → x
        if isinstance(operand, ast.UnaryOp) and _op(operand) == op and op in ("-", "~"):
            self._stat("double_neg")
            self._changed = True
            return operand.operand

        # NOTE: !!x is NOT the same as x (!!5 == 1, not 5)
        # Only -(-x) and ~(~x) are safe identity eliminations

        return expr

    # === Strength reduction helpers ===

    def _strength_reduce_mul(self, expr: ast.BinaryOp) -> ast.Expression | None:
        """Reduce multiply by power-of-2 to shift or addition."""
        left = expr.left
        right = expr.right

        # x * 2^n → x << n  (also handles 2^n * x)
        const, other = None, None
        if isinstance(right, ast.IntLiteral):
            const, other = right, left
        elif isinstance(left, ast.IntLiteral):
            const, other = left, right

        if const is None:
            return None

        # Don't strength-reduce when the other operand is float-typed: shift
        # is undefined for floats and `x + x` would still produce a float.
        # The latter is fine in principle, but we'd need to know the type to
        # confirm.  Cast nodes around float values are the common case
        # ((float)N * 16) and we conservatively bail out for any cast or
        # float literal — _or_ for any expression that contains a
        # FloatLiteral subterm (e.g. `-.5 * (1<<16)` parses with the inner
        # `(1<<16)` already collapsed to `IntLiteral(65536)`, so `other`
        # there is `UnaryOp(-, FloatLiteral)` — not a Cast or FloatLiteral
        # at the top, but we still mustn't shift it).
        if isinstance(other, (ast.FloatLiteral, ast.Cast)):
            if isinstance(other, ast.Cast):
                t = other.target_type
                if isinstance(t, ast.BasicType) and t.name in ("float", "double", "long double"):
                    return None
            else:
                return None
        if _expr_has_float(other):
            return None

        val = _iv(const)

        # x * 0 → 0 (handled by zero elements)
        # x * 1 → x (handled by identity elements)
        if val <= 1:
            return None

        shift = self._log2_if_power_of_2(val)
        if shift is not None:
            self._stat("mul_to_shift")
            self._changed = True
            if shift == 1 and not self._expr_has_side_effects(other):
                # x * 2 → x + x (single ADD HL,HL).
                # Skip when `other` has side effects — duplicating a
                # `(n += 5)` or function-call operand would fire the
                # effect twice and produce the wrong value.
                return _binop("+", other, other)
            return _binop("<<", other, make_int_lit(shift))

        return None

    # === Algebraic simplification helpers ===

    def _simplify_identity(self, expr: ast.BinaryOp) -> ast.Expression | None:
        """Remove identity elements: x+0→x, x*1→x, etc.

        Skip the simplification if dropping the literal would lose a
        unsigned/long type promotion that the C standard would have
        applied. e.g. `0U ^ x` has type `unsigned int` even when `x` is
        `int` — replacing it with just `x` would silently change the
        result type back to `int`, mis-driving downstream code paths
        (sign-extension, comparison signedness, etc.).
        """
        op = _op(expr)
        left = expr.left
        right = expr.right

        # If the literal forces a wider/unsigned type via the usual
        # arithmetic conversions, eliding it would change the type of
        # the surrounding expression. Don't simplify in that case.
        # `L` alone is fine on targets where `long` and `int` share width
        # (the uc386 case); but `U` and `LL` change codegen. We're
        # intentionally conservative here — the cost is one missed
        # simplification, not a correctness bug.
        def _literal_promotes(lit: ast.Expression, other: ast.Expression) -> bool:
            if not isinstance(lit, ast.IntLiteral):
                return False
            if int_flags(lit)[2] or int_flags(lit)[1]:
                return True
            return False

        l_zero = (
            isinstance(left, ast.IntLiteral) and _iv(left) == 0
            and not _literal_promotes(left, right)
        )
        r_zero = (
            isinstance(right, ast.IntLiteral) and _iv(right) == 0
            and not _literal_promotes(right, left)
        )
        l_one = (
            isinstance(left, ast.IntLiteral) and _iv(left) == 1
            and not _literal_promotes(left, right)
        )
        r_one = (
            isinstance(right, ast.IntLiteral) and _iv(right) == 1
            and not _literal_promotes(right, left)
        )

        result = None
        if op == "+":
            if r_zero:
                result = left
            elif l_zero:
                result = right
        elif op == "-":
            if r_zero:
                result = left
        elif op == "*":
            if r_one:
                result = left
            elif l_one:
                result = right
        elif op == "/":
            if r_one:
                result = left
        elif op == "%":
            # x % 1 → 0  (only when x is side-effect-free)
            if r_one and not self._expr_has_side_effects(left):
                self._stat("identity")
                self._changed = True
                return make_int_lit(0)
        elif op in ("<<", ">>"):
            if r_zero:
                result = left
        elif op == "|":
            if r_zero:
                result = left
            elif l_zero:
                result = right
        elif op == "^":
            if r_zero:
                result = left
            elif l_zero:
                result = right
        elif op == "&":
            # x & 0 handled in _simplify_zero
            pass

        if result is not None:
            self._stat("identity")
            self._changed = True
        return result

    def _simplify_zero(self, expr: ast.BinaryOp) -> ast.Expression | None:
        """Simplify zero-producing operations: x*0→0, x&0→0, etc.

        Only fires when the discarded (non-zero) operand is side-effect-free.
        `0 % a++` must still increment a, so the simplification is unsafe
        unless we can drop the other side without losing side effects.
        """
        op = _op(expr)
        left = expr.left
        right = expr.right

        l_zero = isinstance(left, ast.IntLiteral) and _iv(left) == 0
        r_zero = isinstance(right, ast.IntLiteral) and _iv(right) == 0
        l_pure = not self._expr_has_side_effects(left)
        r_pure = not self._expr_has_side_effects(right)

        if op == "*":
            if (r_zero and l_pure) or (l_zero and r_pure):
                self._stat("zero_element")
                self._changed = True
                return make_int_lit(0)
        elif op == "&":
            if (r_zero and l_pure) or (l_zero and r_pure):
                self._stat("zero_element")
                self._changed = True
                return make_int_lit(0)
        elif op == "/" and l_zero and r_pure:
            self._stat("zero_element")
            self._changed = True
            return make_int_lit(0)
        elif op == "%" and l_zero and r_pure:
            self._stat("zero_element")
            self._changed = True
            return make_int_lit(0)
        elif op in ("<<", ">>") and l_zero and r_pure:
            self._stat("zero_element")
            self._changed = True
            return make_int_lit(0)

        return None

    def _simplify_full_mask(self, expr: ast.BinaryOp) -> ast.Expression | None:
        """Simplify full-mask identities: x & 0xFFFF → x, x | 0xFFFF → 0xFFFF, etc.

        IMPORTANT: These are only safe when the other operand is also 16-bit.
        If the other operand could be wider (32/64-bit), these are WRONG:
        - (long long)x & 0xFFFF ≠ x  (truncates upper bits)
        - (long long)x | 0xFFFF ≠ 0xFFFF  (loses upper bits)
        Only apply when the other operand is a non-long IntLiteral.
        """
        op = _op(expr)
        left = expr.left
        right = expr.right

        # Determine the effective mask value for 16-bit
        r_full = isinstance(right, ast.IntLiteral) and (_iv(right) & 0xFFFF) == 0xFFFF and not int_flags(right)[0]
        l_full = isinstance(left, ast.IntLiteral) and (_iv(left) & 0xFFFF) == 0xFFFF and not int_flags(left)[0]

        # Only safe when the OTHER operand is known to be 16-bit
        r_safe_16 = isinstance(right, ast.IntLiteral) and not int_flags(right)[0]
        l_safe_16 = isinstance(left, ast.IntLiteral) and not int_flags(left)[0]

        if op == "&":
            # x & 0xFFFF → x (only when x is 16-bit)
            if r_full and l_safe_16:
                self._stat("full_mask")
                self._changed = True
                return left
            if l_full and r_safe_16:
                self._stat("full_mask")
                self._changed = True
                return right
        elif op == "|":
            # x | 0xFFFF → 0xFFFF (only when x is 16-bit)
            if r_full and l_safe_16:
                self._stat("full_mask")
                self._changed = True
                return make_int_lit(0xFFFF)
            if l_full and r_safe_16:
                self._stat("full_mask")
                self._changed = True
                return make_int_lit(0xFFFF)
        elif op == "^":
            # x ^ 0xFFFF → ~x (only when x is 16-bit)
            if r_full and l_safe_16:
                self._stat("full_mask")
                self._changed = True
                return _unop("~", left)
            if l_full and r_safe_16:
                self._stat("full_mask")
                self._changed = True
                return _unop("~", right)

        return None

    def _simplify_idempotent(self, expr: ast.BinaryOp) -> ast.Expression | None:
        """Simplify idempotent boolean patterns: (a & b) & b → a & b, etc."""
        op = _op(expr)
        left = expr.left
        right = expr.right

        # (a OP b) OP b → a OP b   (for & and |)
        if op in ("&", "|") and isinstance(left, ast.BinaryOp) and _op(left) == op:
            if self._is_same_identifier(right, left.right):
                self._stat("idempotent")
                self._changed = True
                return left
            if self._is_same_identifier(right, left.left):
                self._stat("idempotent")
                self._changed = True
                return left

        return None

    # === Nested constant folding ===

    def _nested_const_fold(self, expr: ast.BinaryOp) -> ast.Expression | None:
        """Fold nested constants: (x + c1) + c2 → x + (c1+c2), etc."""
        op = _op(expr)
        right = expr.right
        left = expr.left

        if not isinstance(right, ast.IntLiteral):
            return None

        c2 = _iv(right)

        if isinstance(left, ast.BinaryOp) and isinstance(left.right, ast.IntLiteral):
            inner_op = _op(left)
            c1 = _iv(left.right)
            x = left.left
            # Use wider mask of the two constants
            is_long = int_flags(right)[0] or int_flags(left.right)[0]
            is_long_long = (
                int_flags(right)[1]
                or getattr(left.right, 'is_long_long', False)
            )
            is_unsigned = int_flags(right)[2] or int_flags(left.right)[2]
            mask = max(self._literal_mask(right), self._literal_mask(left.right))

            def _new(val: int) -> ast.IntLiteral:
                return make_int_lit(val)

            # (x + c1) + c2 → x + (c1 + c2)
            if op == "+" and inner_op == "+":
                combined = (c1 + c2) & mask
                # Sign-extend: if high bit set, convert to negative
                sign_bit = (mask + 1) >> 1
                if combined >= sign_bit and not (int_flags(right)[2] or int_flags(left.right)[2]):
                    combined -= (mask + 1)
                self._stat("nested_fold")
                self._changed = True
                return _binop("+", x, _new(combined))

            # (x - c1) + c2 → x + (c2 - c1)
            if op == "+" and inner_op == "-":
                combined = (c2 - c1) & mask
                sign_bit = (mask + 1) >> 1
                if combined >= sign_bit and not (int_flags(right)[2] or int_flags(left.right)[2]):
                    combined -= (mask + 1)
                self._stat("nested_fold")
                self._changed = True
                if combined == 0:
                    return x
                return _binop("+", x, _new(combined))

            # (x + c1) - c2 → x + (c1 - c2)
            if op == "-" and inner_op == "+":
                combined = (c1 - c2) & mask
                sign_bit = (mask + 1) >> 1
                if combined >= sign_bit and not (int_flags(right)[2] or int_flags(left.right)[2]):
                    combined -= (mask + 1)
                self._stat("nested_fold")
                self._changed = True
                if combined == 0:
                    return x
                return _binop("+", x, _new(combined))

            # (x - c1) - c2 → x - (c1 + c2)
            if op == "-" and inner_op == "-":
                combined = (c1 + c2) & mask
                self._stat("nested_fold")
                self._changed = True
                return _binop("-", x, _new(combined))

            # (x * c1) * c2 → x * (c1 * c2)
            if op == "*" and inner_op == "*":
                combined = (c1 * c2) & mask
                self._stat("nested_fold")
                self._changed = True
                return _binop("*", x, _new(combined))

            # (x << c1) << c2 → x << (c1 + c2)
            if op == "<<" and inner_op == "<<":
                combined = c1 + c2
                self._stat("nested_fold")
                self._changed = True
                return _binop("<<", x, make_int_lit(combined))

            # (x >> c1) >> c2 → x >> (c1 + c2)
            if op == ">>" and inner_op == ">>":
                combined = c1 + c2
                self._stat("nested_fold")
                self._changed = True
                return _binop(">>", x, make_int_lit(combined))

        return None

    # === Level 3: Helper infrastructure ===

    def _reset_level3_state(self) -> None:
        """Reset all level-3 optimization state (called at function entry)."""
        self._cse_cache.clear()
        self._cse_deps.clear()
        self._copies.clear()
        self._var_types.clear()
        self._address_taken_vars.clear()

    def _collect_address_taken(self, node) -> None:
        """Pre-pass: find all variables whose address is taken (&var)."""
        if isinstance(node, ast.UnaryOp) and _op(node) == "&":
            if isinstance(node.operand, ast.Identifier):
                self._address_taken_vars.add(_idname(node.operand))
        # Recurse into children
        if isinstance(node, ast.CompoundStmt):
            for item in node.items:
                self._collect_address_taken(item)
        elif isinstance(node, ast.ExpressionStmt):
            if node.expr is not None:
                self._collect_address_taken(node.expr)
        elif isinstance(node, (ast.IfStmt, ast.IfStmtElse)):
            self._collect_address_taken(node.condition)
            self._collect_address_taken(node.then_branch)
            else_b = getattr(node, "else_branch", None)
            if else_b is not None:
                self._collect_address_taken(else_b)
        elif isinstance(node, ast.WhileStmt):
            self._collect_address_taken(node.condition)
            self._collect_address_taken(node.body)
        elif isinstance(node, ast.DoWhileStmt):
            self._collect_address_taken(node.body)
            self._collect_address_taken(node.condition)
        elif isinstance(node, ast.ForStmt):
            if node.init is not None:
                self._collect_address_taken(node.init)
            if node.condition is not None:
                self._collect_address_taken(node.condition)
            if node.update is not None:
                self._collect_address_taken(node.update)
            self._collect_address_taken(node.body)
        elif isinstance(node, ast.SwitchStmt):
            self._collect_address_taken(node.expr)
            self._collect_address_taken(node.body)
        elif isinstance(node, ast.CaseStmt):
            self._collect_address_taken(node.stmt)
        elif isinstance(node, ast.LabelStmt):
            self._collect_address_taken(node.stmt)
        elif isinstance(node, ast.ReturnStmtValue):
            self._collect_address_taken(node.value)
        elif isinstance(node, ast.ReturnStmt):
            pass  # bare `return;` — nothing to walk
        elif isinstance(node, ast.BinaryOp):
            self._collect_address_taken(node.left)
            self._collect_address_taken(node.right)
        elif isinstance(node, (ast.UnaryOp, ast.PostfixOp)):
            self._collect_address_taken(node.operand)
        elif isinstance(node, ast.TernaryOp):
            self._collect_address_taken(node.condition)
            self._collect_address_taken(node.true_expr)
            self._collect_address_taken(node.false_expr)
        elif isinstance(node, _CALL):
            self._collect_address_taken(node.func)
            for arg in _call_args(node):
                self._collect_address_taken(arg)
        elif isinstance(node, ast.Index):
            self._collect_address_taken(node.array)
            self._collect_address_taken(node.index)
        elif isinstance(node, ast.Cast):
            self._collect_address_taken(node.expr)
        elif isinstance(node, ast.Declaration):
            for d in node.declarators or []:
                if isinstance(d, ast.InitDeclaratorWithInit):
                    self._collect_address_taken(d.init)

    @staticmethod
    def _expr_key(expr: ast.Expression) -> str | None:
        """Return a hashable key for a pure expression, or None if it has side effects."""
        if isinstance(expr, ast.IntLiteral):
            return f"INT:{_iv(expr)}:" + str(int_flags(expr))
        if isinstance(expr, ast.CharLiteral):
            from ._const import char_value
            return f"CHR:{char_value(expr)}"
        if isinstance(expr, ast.Identifier):
            return f"ID:{expr.name.text}"
        if isinstance(expr, ast.BinaryOp):
            # Skip assignment and short-circuit ops
            if _op(expr) in ("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
                           "^=", "<<=", ">>=", "&&", "||", ","):
                return None
            lk = ASTOptimizer._expr_key(expr.left)
            rk = ASTOptimizer._expr_key(expr.right)
            if lk is None or rk is None:
                return None
            return f"BIN:{_op(expr)}:{lk}:{rk}"
        if isinstance(expr, (ast.UnaryOp, ast.PostfixOp)):
            if _op(expr) in ("++", "--", "*", "&"):
                return None  # side effects or pointer deref
            ok = ASTOptimizer._expr_key(expr.operand)
            if ok is None:
                return None
            return f"UNA:{_op(expr)}:{ok}"
        if isinstance(expr, ast.Cast):
            ek = ASTOptimizer._expr_key(expr.expr)
            if ek is None:
                return None
            # Include target type to distinguish e.g. (signed char)x vs (unsigned char)x
            tt = expr.target_type
            if isinstance(tt, ast.BasicType):
                tk = f"{tt.name}:{tt.is_signed}"
            elif isinstance(tt, ast.PointerType):
                tk = "ptr"
            else:
                tk = type(tt).__name__
            return f"CAST:{tk}:{ek}"
        # Calls, Index, Member, etc. are not pure for CSE purposes
        return None

    @staticmethod
    def _get_expr_vars(expr: ast.Expression) -> set[str]:
        """Get all variable names referenced in an expression tree."""
        result: set[str] = set()
        if isinstance(expr, ast.Identifier):
            result.add(_idname(expr))
        elif isinstance(expr, ast.BinaryOp):
            result.update(ASTOptimizer._get_expr_vars(expr.left))
            result.update(ASTOptimizer._get_expr_vars(expr.right))
        elif isinstance(expr, (ast.UnaryOp, ast.PostfixOp)):
            result.update(ASTOptimizer._get_expr_vars(expr.operand))
        elif isinstance(expr, ast.TernaryOp):
            result.update(ASTOptimizer._get_expr_vars(expr.condition))
            result.update(ASTOptimizer._get_expr_vars(expr.true_expr))
            result.update(ASTOptimizer._get_expr_vars(expr.false_expr))
        elif isinstance(expr, _CALL):
            result.update(ASTOptimizer._get_expr_vars(expr.func))
            for arg in _call_args(expr):
                result.update(ASTOptimizer._get_expr_vars(arg))
        elif isinstance(expr, ast.Index):
            result.update(ASTOptimizer._get_expr_vars(expr.array))
            result.update(ASTOptimizer._get_expr_vars(expr.index))
        elif isinstance(expr, ast.Cast):
            result.update(ASTOptimizer._get_expr_vars(expr.expr))
        elif isinstance(expr, ast.Member):
            result.update(ASTOptimizer._get_expr_vars(expr.obj))
        return result

    @staticmethod
    def _expr_has_side_effects(expr: ast.Expression) -> bool:
        """Check if an expression has side effects."""
        if isinstance(expr, (ast.Call, ast.CallNoArgs, ast.StmtExpr)):
            return True
        # `va_arg(ap, T)` advances the va_list pointer — that's a
        # side effect. Without this, `va_arg(ap, int) * 2` got
        # strength-reduced to `va_arg + va_arg`, evaluating va_arg
        # twice and skipping a value.
        if isinstance(expr, ast.VaArgExpr):
            return True
        if isinstance(expr, (ast.UnaryOp, ast.PostfixOp)):
            if _op(expr) in ("++", "--"):
                return True
            return ASTOptimizer._expr_has_side_effects(expr.operand)
        if isinstance(expr, ast.BinaryOp):
            if _op(expr) in ("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
                           "^=", "<<=", ">>="):
                return True
            return (ASTOptimizer._expr_has_side_effects(expr.left) or
                    ASTOptimizer._expr_has_side_effects(expr.right))
        if isinstance(expr, ast.TernaryOp):
            return (ASTOptimizer._expr_has_side_effects(expr.condition) or
                    ASTOptimizer._expr_has_side_effects(expr.true_expr) or
                    ASTOptimizer._expr_has_side_effects(expr.false_expr))
        if isinstance(expr, ast.Index):
            return (ASTOptimizer._expr_has_side_effects(expr.array) or
                    ASTOptimizer._expr_has_side_effects(expr.index))
        if isinstance(expr, ast.Cast):
            return ASTOptimizer._expr_has_side_effects(expr.expr)
        # `s.m` and `p->m`: the access itself doesn't have a side
        # effect, but the object expression may. Without this,
        # `make().x * 2` got strength-reduced to
        # `make().x + make().x`, calling `make()` twice.
        if isinstance(expr, ast.Member):
            return ASTOptimizer._expr_has_side_effects(expr.obj)
        # Compound literals: `(T){init}` evaluates each initializer
        # when the literal is materialized. Without this,
        # `(struct S){f()}.x * 2` calls f() twice.
        if isinstance(expr, ast.Compound):
            return ASTOptimizer._expr_has_side_effects(expr.init)
        if isinstance(expr, ast.InitializerList):
            for v in expr.values:
                if isinstance(v, ast.DesignatedInit):
                    if ASTOptimizer._expr_has_side_effects(v.value):
                        return True
                elif ASTOptimizer._expr_has_side_effects(v):
                    return True
            return False
        return False

    @staticmethod
    def _expr_has_pointer_or_call(expr: ast.Expression) -> bool:
        """Does `expr` contain a pointer dereference or a function
        call? Used by the dead-store eliminator to bail when the
        second store's RHS could observe the first via aliasing.
        """
        if isinstance(expr, ast.UnaryOp) and _op(expr) == "*":
            return True
        if isinstance(expr, ast.Member) and expr.is_arrow:
            return True
        if isinstance(expr, _CALL):
            return True
        if isinstance(expr, ast.BinaryOp):
            return (
                ASTOptimizer._expr_has_pointer_or_call(expr.left)
                or ASTOptimizer._expr_has_pointer_or_call(expr.right)
            )
        if isinstance(expr, ast.UnaryOp):
            return ASTOptimizer._expr_has_pointer_or_call(expr.operand)
        if isinstance(expr, ast.TernaryOp):
            return (
                ASTOptimizer._expr_has_pointer_or_call(expr.condition)
                or ASTOptimizer._expr_has_pointer_or_call(expr.true_expr)
                or ASTOptimizer._expr_has_pointer_or_call(expr.false_expr)
            )
        if isinstance(expr, ast.Index):
            return (
                ASTOptimizer._expr_has_pointer_or_call(expr.array)
                or ASTOptimizer._expr_has_pointer_or_call(expr.index)
            )
        if isinstance(expr, ast.Cast):
            return ASTOptimizer._expr_has_pointer_or_call(expr.expr)
        if isinstance(expr, ast.Member):
            return ASTOptimizer._expr_has_pointer_or_call(expr.obj)
        return False

    @staticmethod
    def _expr_references_var(expr: ast.Expression, name: str) -> bool:
        """Check if expression tree references a variable by name."""
        if isinstance(expr, ast.Identifier):
            return _idname(expr) == name
        if isinstance(expr, ast.BinaryOp):
            return (ASTOptimizer._expr_references_var(expr.left, name) or
                    ASTOptimizer._expr_references_var(expr.right, name))
        if isinstance(expr, (ast.UnaryOp, ast.PostfixOp)):
            return ASTOptimizer._expr_references_var(expr.operand, name)
        if isinstance(expr, ast.TernaryOp):
            return (ASTOptimizer._expr_references_var(expr.condition, name) or
                    ASTOptimizer._expr_references_var(expr.true_expr, name) or
                    ASTOptimizer._expr_references_var(expr.false_expr, name))
        if isinstance(expr, _CALL):
            if ASTOptimizer._expr_references_var(expr.func, name):
                return True
            return any(ASTOptimizer._expr_references_var(a, name) for a in _call_args(expr))
        if isinstance(expr, ast.Index):
            return (ASTOptimizer._expr_references_var(expr.array, name) or
                    ASTOptimizer._expr_references_var(expr.index, name))
        if isinstance(expr, ast.Cast):
            return ASTOptimizer._expr_references_var(expr.expr, name)
        if isinstance(expr, ast.Member):
            return ASTOptimizer._expr_references_var(expr.obj, name)
        return False

    def _get_modified_vars_in_expr(self, expr: ast.Expression) -> set[str]:
        """Collect all variable names modified (assigned, incremented) in an expression."""
        result: set[str] = set()
        if isinstance(expr, ast.BinaryOp):
            if _op(expr) in ("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
                           "^=", "<<=", ">>="):
                if isinstance(expr.left, ast.Identifier):
                    result.add(_idname(expr.left))
            result.update(self._get_modified_vars_in_expr(expr.left))
            result.update(self._get_modified_vars_in_expr(expr.right))
        elif isinstance(expr, (ast.UnaryOp, ast.PostfixOp)):
            if _op(expr) in ("++", "--"):
                if isinstance(expr.operand, ast.Identifier):
                    result.add(_idname(expr.operand))
            result.update(self._get_modified_vars_in_expr(expr.operand))
        elif isinstance(expr, ast.TernaryOp):
            result.update(self._get_modified_vars_in_expr(expr.condition))
            result.update(self._get_modified_vars_in_expr(expr.true_expr))
            result.update(self._get_modified_vars_in_expr(expr.false_expr))
        elif isinstance(expr, _CALL):
            for arg in _call_args(expr):
                result.update(self._get_modified_vars_in_expr(arg))
        elif isinstance(expr, ast.Index):
            result.update(self._get_modified_vars_in_expr(expr.array))
            result.update(self._get_modified_vars_in_expr(expr.index))
        elif isinstance(expr, ast.Cast):
            result.update(self._get_modified_vars_in_expr(expr.expr))
        return result

    def _get_modified_vars_in_stmt(self, stmt: ast.Statement) -> set[str]:
        """Collect all variable names modified within a statement tree."""
        result: set[str] = set()
        if isinstance(stmt, ast.CompoundStmt):
            for item in stmt.items:
                if isinstance(item, ast.Declaration):
                    for d in item.declarators or []:
                        if isinstance(d, ast.InitDeclaratorWithInit):
                            result.update(self._get_modified_vars_in_expr(d.init))
                elif isinstance(item, ast.FunctionDef):
                    pass  # nested function: separate scope
                else:
                    result.update(self._get_modified_vars_in_stmt(item))
        elif isinstance(stmt, ast.ExpressionStmt):
            if stmt.expr is not None:
                result.update(self._get_modified_vars_in_expr(stmt.expr))
        elif isinstance(stmt, (ast.IfStmt, ast.IfStmtElse)):
            result.update(self._get_modified_vars_in_expr(stmt.condition))
            result.update(self._get_modified_vars_in_stmt(stmt.then_branch))
            else_b = getattr(stmt, "else_branch", None)
            if else_b is not None:
                result.update(self._get_modified_vars_in_stmt(else_b))
        elif isinstance(stmt, ast.WhileStmt):
            result.update(self._get_modified_vars_in_expr(stmt.condition))
            result.update(self._get_modified_vars_in_stmt(stmt.body))
        elif isinstance(stmt, ast.DoWhileStmt):
            result.update(self._get_modified_vars_in_stmt(stmt.body))
            result.update(self._get_modified_vars_in_expr(stmt.condition))
        elif isinstance(stmt, ast.ForStmt):
            if stmt.init is not None:
                if isinstance(stmt.init, ast.Declaration):
                    for d in stmt.init.declarators or []:
                        if isinstance(d, ast.InitDeclaratorWithInit):
                            result.update(self._get_modified_vars_in_expr(d.init))
                else:
                    result.update(self._get_modified_vars_in_expr(stmt.init))
            if stmt.condition is not None:
                result.update(self._get_modified_vars_in_expr(stmt.condition))
            if stmt.update is not None:
                result.update(self._get_modified_vars_in_expr(stmt.update))
            result.update(self._get_modified_vars_in_stmt(stmt.body))
        elif isinstance(stmt, ast.SwitchStmt):
            result.update(self._get_modified_vars_in_expr(stmt.expr))
            result.update(self._get_modified_vars_in_stmt(stmt.body))
        elif isinstance(stmt, ast.CaseStmt):
            result.update(self._get_modified_vars_in_stmt(stmt.stmt))
        elif isinstance(stmt, ast.LabelStmt):
            result.update(self._get_modified_vars_in_stmt(stmt.stmt))
        elif isinstance(stmt, ast.ReturnStmtValue):
            result.update(self._get_modified_vars_in_expr(stmt.value))
        return result

    @staticmethod
    def _stmt_has_calls(stmt) -> bool:
        """Check if a statement tree contains any function calls."""
        if isinstance(stmt, ast.CompoundStmt):
            return any(ASTOptimizer._stmt_has_calls(item) for item in stmt.items)
        if isinstance(stmt, ast.ExpressionStmt):
            return stmt.expr is not None and ASTOptimizer._expr_has_calls(stmt.expr)
        if isinstance(stmt, (ast.IfStmt, ast.IfStmtElse)):
            if ASTOptimizer._expr_has_calls(stmt.condition):
                return True
            if ASTOptimizer._stmt_has_calls(stmt.then_branch):
                return True
            else_b = getattr(stmt, "else_branch", None)
            if else_b and ASTOptimizer._stmt_has_calls(else_b):
                return True
            return False
        if isinstance(stmt, ast.WhileStmt):
            return ASTOptimizer._expr_has_calls(stmt.condition) or ASTOptimizer._stmt_has_calls(stmt.body)
        if isinstance(stmt, ast.DoWhileStmt):
            return ASTOptimizer._stmt_has_calls(stmt.body) or ASTOptimizer._expr_has_calls(stmt.condition)
        if isinstance(stmt, ast.ForStmt):
            if stmt.init:
                if isinstance(stmt.init, ast.Declaration):
                    for d in stmt.init.declarators or []:
                        if isinstance(d, ast.InitDeclaratorWithInit) and ASTOptimizer._expr_has_calls(d.init):
                            return True
                elif ASTOptimizer._expr_has_calls(stmt.init):
                    return True
            if stmt.condition and ASTOptimizer._expr_has_calls(stmt.condition):
                return True
            if stmt.update and ASTOptimizer._expr_has_calls(stmt.update):
                return True
            return ASTOptimizer._stmt_has_calls(stmt.body)
        if isinstance(stmt, ast.CaseStmt):
            return ASTOptimizer._stmt_has_calls(stmt.stmt)
        if isinstance(stmt, ast.LabelStmt):
            return ASTOptimizer._stmt_has_calls(stmt.stmt)
        if isinstance(stmt, ast.ReturnStmtValue):
            return ASTOptimizer._expr_has_calls(stmt.value)
        if isinstance(stmt, ast.SwitchStmt):
            return ASTOptimizer._expr_has_calls(stmt.expr) or ASTOptimizer._stmt_has_calls(stmt.body)
        return False

    @staticmethod
    def _expr_has_calls(expr: ast.Expression) -> bool:
        """Check if an expression tree contains any function calls."""
        if isinstance(expr, _CALL):
            return True
        if isinstance(expr, ast.BinaryOp):
            return ASTOptimizer._expr_has_calls(expr.left) or ASTOptimizer._expr_has_calls(expr.right)
        if isinstance(expr, (ast.UnaryOp, ast.PostfixOp)):
            return ASTOptimizer._expr_has_calls(expr.operand)
        if isinstance(expr, ast.TernaryOp):
            return (ASTOptimizer._expr_has_calls(expr.condition) or
                    ASTOptimizer._expr_has_calls(expr.true_expr) or
                    ASTOptimizer._expr_has_calls(expr.false_expr))
        if isinstance(expr, ast.Index):
            return ASTOptimizer._expr_has_calls(expr.array) or ASTOptimizer._expr_has_calls(expr.index)
        if isinstance(expr, ast.Cast):
            return ASTOptimizer._expr_has_calls(expr.expr)
        return False

    # === Level 3: Commutative normalization ===

    def _normalize_commutative(self, expr: ast.BinaryOp) -> ast.Expression | None:
        """Sort operands of commutative ops into canonical form.

        Order: Identifier < Complex < IntLiteral.
        This ensures a+5 and 5+a get the same CSE key.
        """
        if _op(expr) not in ("+", "*", "&", "|", "^", "==", "!="):
            return None

        def sort_key(e: ast.Expression) -> tuple:
            if isinstance(e, ast.Identifier):
                return (0, e.name.text)
            if isinstance(e, ast.IntLiteral):
                return (2, _iv(e))
            if isinstance(e, ast.CharLiteral):
                return (2, _iv(e))
            # Complex expression
            k = self._expr_key(e)
            return (1, k if k else "")

        lk = sort_key(expr.left)
        rk = sort_key(expr.right)
        if lk > rk:
            self._stat("commutative_norm")
            self._changed = True
            expr.left, expr.right = expr.right, expr.left
            return expr
        return None

    # === Level 3: Common Subexpression Elimination ===

    def _cse_lookup(self, expr: ast.Expression) -> ast.Expression:
        """Look up expression in CSE cache; return cached copy or cache this one."""
        key = self._expr_key(expr)
        if key is None:
            return expr  # Not a pure expression

        if key in self._cse_cache:
            self._stat("cse")
            self._changed = True
            return self._cse_cache[key]

        # Cache this expression
        self._cse_cache[key] = expr
        self._cse_deps[key] = self._get_expr_vars(expr)
        return expr

    def _invalidate_cse_for_var(self, name: str) -> None:
        """Invalidate all CSE entries that depend on a variable."""
        to_remove = [k for k, deps in self._cse_deps.items() if name in deps]
        for k in to_remove:
            del self._cse_cache[k]
            del self._cse_deps[k]

    def _clear_all_caches(self) -> None:
        """Clear all CSE and copy propagation caches (for non-linear control flow)."""
        self._cse_cache.clear()
        self._cse_deps.clear()
        self._copies.clear()

    def _invalidate_caches_for_stmt(self, item) -> None:
        """After processing a statement, invalidate caches as needed."""
        if isinstance(item, (ast.LabelStmt, ast.CaseStmt)):
            # Non-linear control flow: clear everything
            self._clear_all_caches()
            return

        if isinstance(item, ast.ExpressionStmt) and item.expr is not None:
            self._invalidate_caches_for_expr(item.expr)
        elif isinstance(item, ast.Declaration):
            # Each initialized declarator assigns its variable.
            for d in item.declarators or []:
                if not isinstance(d, ast.InitDeclaratorWithInit):
                    continue
                nm = _declarator_ident(d.declarator)
                if nm is None:
                    continue
                self._invalidate_cse_for_var(nm)
                self._invalidate_copies_for_var(nm)
                # Track copy `x = y` when the types are compatible.
                if isinstance(d.init, ast.Identifier):
                    src = _idname(d.init)
                    if self._types_compatible_for_copy(nm, src):
                        self._copies[nm] = src
        elif isinstance(item, (ast.IfStmt, ast.IfStmtElse, ast.WhileStmt,
                               ast.DoWhileStmt, ast.ForStmt, ast.SwitchStmt)):
            # Control flow structures: conservatively clear caches
            self._clear_all_caches()
        elif isinstance(item, (ast.ReturnStmt, ast.ReturnStmtValue, ast.GotoStmt)):
            self._clear_all_caches()

    def _invalidate_caches_for_expr(self, expr: ast.Expression) -> None:
        """Invalidate CSE/copy caches based on side effects in an expression."""
        if isinstance(expr, ast.BinaryOp):
            if _op(expr) in ("=", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
                           "^=", "<<=", ">>="):
                if isinstance(expr.left, ast.Identifier):
                    name = _idname(expr.left)
                    self._invalidate_cse_for_var(name)
                    self._invalidate_copies_for_var(name)
                    # Track simple copy: x = y (only if types are compatible)
                    if _op(expr) == "=" and isinstance(expr.right, ast.Identifier):
                        if self._types_compatible_for_copy(name, _idname(expr.right)):
                            self._copies[name] = _idname(expr.right)
                elif isinstance(expr.left, (ast.UnaryOp, ast.Index)):
                    # Pointer/array write: clear all caches
                    self._clear_all_caches()
                    return
            self._invalidate_caches_for_expr(expr.left)
            self._invalidate_caches_for_expr(expr.right)
        elif isinstance(expr, (ast.UnaryOp, ast.PostfixOp)):
            if _op(expr) in ("++", "--"):
                if isinstance(expr.operand, ast.Identifier):
                    self._invalidate_cse_for_var(_idname(expr.operand))
                    self._invalidate_copies_for_var(_idname(expr.operand))
                else:
                    # Pointer increment etc.
                    self._clear_all_caches()
                    return
            self._invalidate_caches_for_expr(expr.operand)
        elif isinstance(expr, _CALL):
            # Function call: clear entire cache (can modify globals, address-taken vars)
            self._clear_all_caches()
        elif isinstance(expr, ast.TernaryOp):
            self._invalidate_caches_for_expr(expr.condition)
            self._invalidate_caches_for_expr(expr.true_expr)
            self._invalidate_caches_for_expr(expr.false_expr)

    # === Level 3: Copy propagation ===

    def _invalidate_copies_for_var(self, name: str) -> None:
        """Remove copy entries involving a variable (as source or target)."""
        # Remove if name is a target
        self._copies.pop(name, None)
        # Remove entries where name is the source
        to_remove = [k for k, v in self._copies.items() if v == name]
        for k in to_remove:
            del self._copies[k]

    def _types_compatible_for_copy(self, target: str, source: str) -> bool:
        """Check if copy propagation is safe between two variables.

        Only propagate when both variables have the same basic type category
        (both int-like, both float, both pointers, etc.) to avoid miscompilation
        when assignments perform implicit type conversion (e.g., float -> int).
        """
        t1 = self._var_types.get(target)
        t2 = self._var_types.get(source)
        if t1 is None or t2 is None:
            return False  # Unknown type, don't propagate
        # Both must be BasicType with the same name, or both PointerType, etc.
        if type(t1) != type(t2):
            return False
        if isinstance(t1, ast.BasicType) and isinstance(t2, ast.BasicType):
            # Must be same type category (e.g., both "int", both "float")
            return t1.name == t2.name
        # For pointers, also require the pointee's type category to match.
        # `struct printer *pr = void_ptr;` is a legal C conversion, but if
        # the optimizer rewrites later `pr->m` references as `void_ptr->m`,
        # downstream `_type_of` loses the declared struct type. Without this
        # guard, the codegen rejects the resulting `->` lowering.
        if isinstance(t1, ast.PointerType) and isinstance(t2, ast.PointerType):
            b1, b2 = t1.base_type, t2.base_type
            # void* on either side: refuse propagation. The conversion loses
            # type information needed for member access.
            if (isinstance(b1, ast.BasicType) and b1.name == "void"):
                return False
            if (isinstance(b2, ast.BasicType) and b2.name == "void"):
                return False
            # Different kinds of pointee (struct vs basic, etc.): refuse.
            if type(b1) != type(b2):
                return False
        # For arrays, structs - same type means compatible
        return True

    # === Level 3: Dead store elimination ===

    def _eliminate_dead_stores(self, items: list) -> list:
        """Remove dead stores: consecutive assignments to same variable where first is unused."""
        if len(items) < 2:
            return items

        result = []
        i = 0
        while i < len(items):
            if i + 1 < len(items):
                # Consecutive stores to the same variable where the
                # first is overwritten before any use. `first` may be a
                # plain assignment `x = e;` or a single-init
                # declaration `T x = e;`; `second` is the overwriting
                # assignment `x = e2;`.
                first = items[i]
                second = items[i + 1]
                second_target = self._get_simple_assign_target(second)

                first_target = self._get_simple_assign_target(first)
                first_stripped = None  # set iff first is a declaration
                if first_target is not None:
                    first_rhs = self._get_assign_rhs(first)
                else:
                    ds = self._decl_init_store(first)
                    if ds is not None:
                        first_target, first_rhs, first_stripped = ds
                    else:
                        first_rhs = None

                if (first_target is not None and second_target is not None
                        and first_target == second_target):
                    second_rhs = self._get_assign_rhs(second)
                    if (first_rhs is not None
                            and not self._expr_has_side_effects(first_rhs)
                            and second_rhs is not None
                            and not self._expr_references_var(second_rhs, first_target)
                            and not self._expr_has_pointer_or_call(second_rhs)):
                        # Dead store. We're conservative when the second
                        # RHS contains a pointer dereference or function
                        # call — either could observe the first store
                        # via aliasing (e.g. `b = a; b = *p;` where p
                        # points to b).
                        self._stat("dead_store")
                        self._changed = True
                        if first_stripped is not None:
                            # Declaration: keep it (the variable must
                            # stay declared/scoped), drop only the dead
                            # initializer.
                            result.append(first_stripped)
                        i += 1  # skip the original first store
                        continue

            result.append(items[i])
            i += 1
        return result

    @staticmethod
    def _get_simple_assign_target(item) -> str | None:
        """If item is `x = expr;`, return 'x'. Otherwise None."""
        if not isinstance(item, ast.ExpressionStmt):
            return None
        expr = item.expr
        if not isinstance(expr, ast.BinaryOp) or _op(expr) != "=":
            return None
        if isinstance(expr.left, ast.Identifier):
            return _idname(expr.left)
        return None

    @staticmethod
    def _get_assign_rhs(item) -> ast.Expression | None:
        """Get the RHS of a simple assignment statement."""
        if not isinstance(item, ast.ExpressionStmt):
            return None
        expr = item.expr
        if not isinstance(expr, ast.BinaryOp) or _op(expr) != "=":
            return None
        return expr.right

    @staticmethod
    def _decl_init_store(item):
        """If `item` is `T x = expr;` — a Declaration with exactly one
        InitDeclaratorWithInit naming a plain identifier, not declared
        `volatile` — return ``(name, init_expr, decl_without_init)``.
        The returned declaration keeps the declarator so the variable
        stays declared and scoped; only the (dead) initializer is
        dropped. Otherwise None."""
        if not isinstance(item, ast.Declaration):
            return None
        decls = item.declarators or []
        if len(decls) != 1 or not isinstance(decls[0], ast.InitDeclaratorWithInit):
            return None
        for sp in item.decl_specs or []:
            if (isinstance(sp, ast.TypeQualifier)
                    and getattr(sp.kw, "text", "") == "volatile"):
                return None  # a volatile write must not be elided
        d = decls[0]
        name = _declarator_ident(d.declarator)
        if name is None:
            return None
        stripped = ast.Declaration(
            decl_specs=item.decl_specs,
            declarators=[ast.InitDeclarator(declarator=d.declarator, pos=d.pos)],
            pos=item.pos,
        )
        return name, d.init, stripped

    # === Level 3: Loop unrolling ===

    def _try_loop_unroll(self, stmt: ast.ForStmt) -> ast.CompoundStmt | None:
        """Try to unroll a small constant-bound for loop."""
        # Match: for (var = const; var < const; var++) or similar
        init = stmt.init
        cond = stmt.condition
        update = stmt.update

        # Check init: `var = const` (expression form, var declared in an
        # enclosing scope) or `T var = const;` (declaration form, var
        # scoped to the loop — must be re-declared in the unrolled block).
        init_decl = None
        if isinstance(init, ast.BinaryOp) and _op(init) == "=":
            if not isinstance(init.left, ast.Identifier):
                return None
            if not isinstance(init.right, ast.IntLiteral):
                return None
            var_name = _idname(init.left)
            start_val = _iv(init.right)
        elif isinstance(init, ast.Declaration):
            with_init = [d for d in (init.declarators or [])
                         if isinstance(d, ast.InitDeclaratorWithInit)]
            if len(with_init) != 1:
                return None  # 0 or multiple declarators: not a simple counter
            d = with_init[0]
            if not isinstance(d.init, ast.IntLiteral):
                return None
            var_name = _declarator_ident(d.declarator)
            if var_name is None:
                return None
            start_val = _iv(d.init)
            init_decl = init  # re-emit once so `var` stays declared
        else:
            return None

        # Check condition: var < const, var <= const, var != const
        if not isinstance(cond, ast.BinaryOp):
            return None
        if not isinstance(cond.left, ast.Identifier) or _idname(cond.left) != var_name:
            return None
        if not isinstance(cond.right, ast.IntLiteral):
            return None
        end_val = _iv(cond.right)
        cmp_op = _op(cond)

        # Check update: var++ / ++var (UnaryOp prefix or PostfixOp) or
        # var += 1. The auto-AST splits postfix `i++` into PostfixOp;
        # both it and prefix UnaryOp carry .op (Token) and .operand.
        step = None
        if isinstance(update, (ast.UnaryOp, ast.PostfixOp)) and _op(update) == "++" and isinstance(update.operand, ast.Identifier) and _idname(update.operand) == var_name:
            step = 1
        elif isinstance(update, ast.BinaryOp) and _op(update) == "+=" and isinstance(update.left, ast.Identifier) and _idname(update.left) == var_name and isinstance(update.right, ast.IntLiteral):
            step = _iv(update.right)
        if step is None or step <= 0:
            return None

        # Calculate iteration count
        if cmp_op == "<":
            iterations = max(0, (end_val - start_val + step - 1) // step)
        elif cmp_op == "<=":
            iterations = max(0, (end_val - start_val + step) // step)
        elif cmp_op == "!=":
            if (end_val - start_val) % step != 0:
                return None  # Would be infinite loop
            iterations = max(0, (end_val - start_val) // step)
        else:
            return None

        # Check criteria
        if iterations > 4 or iterations <= 0:
            return None

        # Check body size and no break/continue/goto/labels
        body = stmt.body
        body_stmts = self._get_flat_stmts(body)
        if body_stmts is None:
            return None
        if len(body_stmts) > 3:
            return None
        if self._contains_flow_control(body):
            return None

        # Unroll!
        self._stat("loop_unroll")
        self._changed = True

        items: list = []
        if init_decl is not None:
            # Keep the loop variable declared (and scoped to this block)
            # in the declaration-form case; the per-iteration `var = val`
            # assignments below overwrite its initializer.
            items.append(copy.deepcopy(init_decl))
        for i in range(iterations):
            val = start_val + i * step
            # var = val
            # var_name is the induction variable's name (str via _idname).
            assign = ast.ExpressionStmt(
                expr=_binop("=", make_identifier(var_name), make_int_lit(val)),
            )
            items.append(assign)
            # Deep copy of body statements
            for s in body_stmts:
                items.append(copy.deepcopy(s))

        return ast.CompoundStmt(items=items)

    @staticmethod
    def _get_flat_stmts(body) -> list | None:
        """Get flat list of statements from a loop body. Returns None if not flattenable."""
        if isinstance(body, ast.CompoundStmt):
            # All items must be statements (no declarations) for the
            # unroll to preserve scoping.
            result = []
            for item in body.items:
                if isinstance(item, (ast.Declaration, ast.FunctionDef)):
                    return None
                result.append(item)
            return result
        # A single non-compound, non-declaration statement body.
        if not isinstance(body, (ast.Declaration, ast.FunctionDef)):
            return [body]
        return None

    @staticmethod
    def _contains_flow_control(node) -> bool:
        """Check if a statement contains break, continue, goto, or labels."""
        if isinstance(node, (ast.BreakStmt, ast.ContinueStmt, ast.GotoStmt)):
            return True
        if isinstance(node, (ast.LabelStmt, ast.CaseStmt)):
            return True
        if isinstance(node, ast.CompoundStmt):
            return any(ASTOptimizer._contains_flow_control(item) for item in node.items)
        if isinstance(node, (ast.IfStmt, ast.IfStmtElse)):
            if ASTOptimizer._contains_flow_control(node.then_branch):
                return True
            else_b = getattr(node, "else_branch", None)
            if else_b and ASTOptimizer._contains_flow_control(else_b):
                return True
        if isinstance(node, (ast.WhileStmt, ast.DoWhileStmt)):
            return ASTOptimizer._contains_flow_control(node.body)
        if isinstance(node, ast.ForStmt):
            return ASTOptimizer._contains_flow_control(node.body)
        if isinstance(node, ast.ExpressionStmt):
            return False
        if isinstance(node, ast.ReturnStmt):
            return False
        return False

    # === Utility methods ===

    @staticmethod
    def _log2_if_power_of_2(n: int) -> int | None:
        """Return log2(n) if n is a power of 2, else None."""
        if n <= 0 or (n & (n - 1)) != 0:
            return None
        return n.bit_length() - 1

    @staticmethod
    def _is_same_identifier(a: ast.Expression, b: ast.Expression) -> bool:
        """Check if two expressions are the same simple identifier."""
        return (isinstance(a, ast.Identifier) and isinstance(b, ast.Identifier)
                and _idname(a) == _idname(b))

    @staticmethod
    def _is_unsigned_literal(*exprs: ast.Expression) -> bool:
        """Check if any IntLiteral is unsigned."""
        return any(isinstance(e, ast.IntLiteral) and int_flags(e)[2] for e in exprs)

    def _is_effectively_unsigned(self, lit: ast.IntLiteral) -> bool:
        """Check if a literal is effectively unsigned per C standard 6.4.4.1.

        For hex/octal constants without U suffix, the type is the first that fits:
        int → unsigned int → long int → unsigned long int → ...
        """
        if int_flags(lit)[2]:
            return True
        if not int_flags(lit)[3]:
            return False
        tc = self.type_config
        if not int_flags(lit)[0]:
            # Exceeds signed int, fits unsigned int?
            if _iv(lit) > tc.int_max and _iv(lit) <= tc.uint_max:
                return True
            if _iv(lit) > tc.long_max and _iv(lit) <= tc.ulong_max:
                return True
            return False
        # Hex with L suffix: unsigned long if > LONG_MAX but <= ULONG_MAX
        if _iv(lit) > tc.long_max and _iv(lit) <= tc.ulong_max:
            return True
        return False

    def _literal_mask(self, lit: ast.IntLiteral) -> int:
        """Get bitmask for an IntLiteral based on its effective type width.

        Per C standard 6.4.4.1, constant type is determined by value and suffix:
        - Decimal without suffix: int → long → long long
        - Hex/octal without suffix: int → unsigned int → long → unsigned long → ...
        - U suffix: unsigned int → unsigned long → unsigned long long
        - LU/UL suffix: unsigned long → unsigned long long
        - LLU/ULL suffix: unsigned long long
        If value exceeds the range of the current type, promote to next.
        """
        tc = self.type_config
        int_mask = tc.uint_max
        long_mask = tc.ulong_max
        long_long_mask = tc.ulong_long_max
        val = int_value(lit)
        is_long, is_long_long, is_unsigned, is_hex = int_flags(lit)
        if is_long_long:
            return long_long_mask
        if is_long and is_unsigned:
            # LU/UL suffix: unsigned long, then unsigned long long.
            if val > tc.ulong_max or val < 0:
                return long_long_mask
            return long_mask
        if is_long:
            # L suffix on hex/octal: long → unsigned long → long long → ulong long.
            # L suffix on decimal: long → long long.
            if is_hex:
                if val > tc.ulong_max or val < -(tc.long_max + 1):
                    return long_long_mask
                if val > tc.long_max:
                    return long_mask
                return long_mask
            if val > tc.long_max or val < -(tc.long_max + 1):
                return long_long_mask
            return long_mask
        if is_hex or is_unsigned:
            if val > tc.uint_max or val < -(tc.int_max + 1):
                if val > tc.ulong_max:
                    return long_long_mask
                return long_mask
            return int_mask
        # Decimal, no suffix
        if val > tc.int_max or val < -(tc.int_max + 1):
            if val > tc.long_max or val < -(tc.long_max + 1):
                return long_long_mask
            return long_mask
        return int_mask

    def _fold_constants(self, op: str, a: int, b: int, unsigned: bool,
                        mask: int = 0xFFFF) -> int | None:
        """Evaluate a constant binary expression. Returns None if not foldable."""
        try:
            if op == "+":
                return (a + b) & mask
            elif op == "-":
                return (a - b) & mask
            elif op == "*":
                return (a * b) & mask
            elif op == "/":
                if b == 0:
                    return None
                if unsigned:
                    return (a & mask) // (b & mask)
                else:
                    sa = self._to_signed(a, mask)
                    sb = self._to_signed(b, mask)
                    return int(sa / sb) & mask
            elif op == "%":
                if b == 0:
                    return None
                if unsigned:
                    return (a & mask) % (b & mask)
                else:
                    sa = self._to_signed(a, mask)
                    sb = self._to_signed(b, mask)
                    q = int(sa / sb)
                    return (sa - q * sb) & mask
            elif op == "&":
                return a & b
            elif op == "|":
                return a | b
            elif op == "^":
                return a ^ b
            elif op == "<<":
                return (a << b) & mask
            elif op == ">>":
                if unsigned:
                    return (a & mask) >> b
                else:
                    sa = self._to_signed(a, mask)
                    return (sa >> b) & mask
            elif op == "==":
                return 1 if (a & mask) == (b & mask) else 0
            elif op == "!=":
                return 1 if (a & mask) != (b & mask) else 0
            elif op == "<":
                if unsigned:
                    return 1 if (a & mask) < (b & mask) else 0
                else:
                    return 1 if self._to_signed(a, mask) < self._to_signed(b, mask) else 0
            elif op == ">":
                if unsigned:
                    return 1 if (a & mask) > (b & mask) else 0
                else:
                    return 1 if self._to_signed(a, mask) > self._to_signed(b, mask) else 0
            elif op == "<=":
                if unsigned:
                    return 1 if (a & mask) <= (b & mask) else 0
                else:
                    return 1 if self._to_signed(a, mask) <= self._to_signed(b, mask) else 0
            elif op == ">=":
                if unsigned:
                    return 1 if (a & mask) >= (b & mask) else 0
                else:
                    return 1 if self._to_signed(a, mask) >= self._to_signed(b, mask) else 0
        except (ZeroDivisionError, OverflowError, ValueError):
            return None
        return None

    @staticmethod
    def _to_signed(val: int, mask: int) -> int:
        """Convert unsigned int to signed using mask width."""
        val = val & mask  # Ensure value is in unsigned range first
        sign_bit = (mask >> 1) + 1
        if val & sign_bit:
            return val - (mask + 1)
        return val

    def _sizeof_type(self, t) -> int | None:
        """Compute sizeof(type) at compile time on the declarator-shape AST.

        ``t`` is a ``TypeName`` (decl_specs only) or
        ``TypeNameWithDeclarator`` (decl_specs + abstract declarator).
        Returns ``None`` for shapes the optimizer can't size statically
        (typeof, _BitInt, structs/enums by reference, arrays of unknown
        bound, etc.) so the host's backend can compute them with full
        type info at codegen time.
        """
        if isinstance(t, ast.TypeName):
            return self._sizeof_decl_specs(t.decl_specs)
        if isinstance(t, ast.TypeNameWithDeclarator):
            inner = self._sizeof_decl_specs(t.decl_specs)
            if inner is None:
                return None
            return self._wrap_abstract_declarator_size(t.declarator, inner)
        return None

    def _sizeof_decl_specs(self, decl_specs) -> int | None:
        """Walk a decl_specs list and compute size of the resulting base type."""
        type_keywords: list[str] = []
        signed_kw: bool | None = None
        for spec in decl_specs or []:
            if isinstance(spec, ast.BasicTypeSpec):
                kw = spec.kw.text
                if kw in ("signed", "__signed", "__signed__"):
                    signed_kw = True
                elif kw == "unsigned":
                    signed_kw = False
                else:
                    type_keywords.append(kw)
            elif isinstance(spec, ast.TypedefNameSpec):
                # Typedef'd name — would need a typedef table to resolve;
                # the optimizer doesn't carry one. Skip.
                return None
            elif isinstance(spec, (ast.StorageClass, ast.TypeQualifier,
                                   ast.AlignasType, ast.AlignasValue)):
                continue
            else:
                # Struct/Enum/Typeof/BitInt — punt.
                return None
        name = self._reduce_type_keywords(type_keywords)
        if name is None:
            return None
        return self.type_config.sizeof_basic(name)

    @staticmethod
    def _reduce_type_keywords(keywords: list[str]) -> str | None:
        if not keywords:
            return "int"  # bare ``signed`` / ``unsigned``
        long_count = keywords.count("long")
        keywords_no_long = [k for k in keywords if k != "long"]
        if "void" in keywords:
            return "void"
        if "double" in keywords:
            return "long double" if long_count else "double"
        if "float" in keywords:
            return "float"
        if "char" in keywords:
            return "char"
        if "short" in keywords:
            return "short"
        if "bool" in keywords or "_Bool" in keywords:
            return "bool"
        if long_count >= 2:
            return "long long"
        if long_count == 1:
            return "long"
        if keywords_no_long:
            return keywords_no_long[0]
        return "int"

    def _wrap_abstract_declarator_size(self, node, inner: int) -> int | None:
        """Compute sizeof for an abstract declarator chain wrapping a base size.

        Abstract pointers contribute ``ptr_size``; abstract arrays
        multiply by the constant size (or return None if the size is
        not a constant); abstract function declarators yield a function
        pointer size at the C decay rule.
        """
        if isinstance(node, ast.AbstractPointer):
            # outermost is a pointer chain — sizeof is one pointer per
            # level. Count the levels in the pointer list.
            return self.type_config.ptr_size
        if isinstance(node, ast.AbstractPointerInner):
            return self.type_config.ptr_size
        # Arrays / fn declarators on abstract types — punt.
        return None

    def _stat(self, name: str) -> None:
        self.stats[name] = self.stats.get(name, 0) + 1


# ---- declarator-shape AST helpers ------------------------------------------


def _function_param_names(declarator) -> list[str]:
    """From a FunctionDef.declarator chain, yield each parameter's IDENT name.

    Auto-AST shape: the outermost declarator is an ``FnDeclarator`` (or
    ``FnDeclaratorEmpty`` for ``()`` / ``(void)``); its ``params`` field
    is either a list of ``ParamDecl`` / ``ParamDeclAbstract`` /
    ``ParamDeclTypeOnly`` or a ``VariadicParams`` wrapping that list.
    """
    fn = _outermost_fn_declarator(declarator)
    if fn is None:
        return []
    if isinstance(fn, ast.FnDeclaratorEmpty):
        return []
    params = fn.params
    if isinstance(params, ast.VariadicParams):
        params = params.params
    out: list[str] = []
    for p in params or []:
        if isinstance(p, ast.ParamDecl):
            name = _declarator_ident(p.declarator)
            if name is not None:
                out.append(name)
    return out


def _outermost_fn_declarator(node):
    """Walk through PointerDeclarator wraps to find the outermost
    FnDeclarator / FnDeclaratorEmpty (or None if not a function)."""
    while node is not None:
        if isinstance(node, (ast.FnDeclarator, ast.FnDeclaratorEmpty)):
            return node
        if isinstance(node, ast.PointerDeclarator):
            node = node.inner
            continue
        if isinstance(node, ast.GroupDeclarator):
            node = node.inner
            continue
        return None
    return None


def _string_literal_byte_length(node) -> int | None:
    """Compute the decoded byte length of a string-literal AST node.

    Returns ``None`` for non-string operands or wide strings (we don't
    track wchar_t size in the TypeConfig so codegen handles those).
    """
    # The grammar's <string_literal> non-terminal is annotated as a
    # list of <string_piece> StringLiteral nodes — adjacent C string
    # literals concatenate. A single piece comes through as the bare
    # StringLiteral; multiple pieces as a list.
    if isinstance(node, ast.StringLiteral):
        return _decoded_string_len(node)
    if isinstance(node, list) and all(isinstance(p, ast.StringLiteral) for p in node):
        total = 0
        for p in node:
            n = _decoded_string_len(p)
            if n is None:
                return None
            total += n
        return total
    return None


def _decoded_string_len(lit) -> int | None:
    """Length of a StringLiteral's decoded payload (in bytes for narrow
    strings; ``None`` for wide / u8 / u / U / L variants we don't size)."""
    text = lit.value.text
    # Strip encoding prefix. u8 narrow stays bytes; u/U/L are wide.
    if text.startswith("u8"):
        text = text[2:]
    elif text.startswith(("u", "U", "L")):
        return None
    assert text.startswith('"') and text.endswith('"'), f"bad string lit {lit.value.text!r}"
    body = text[1:-1]
    n = 0
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            # One escape consumes (up to) two source chars and contributes
            # one byte. \xNN / \NNN escapes consume more but we don't
            # parse those fully here — approximate by 2 chars / 1 byte.
            i += 2
            n += 1
            continue
        i += 1
        n += 1
    return n


def _declarator_ident(node) -> str | None:
    """The innermost ``Declarator(name=IDENT)`` of a declarator chain."""
    while node is not None:
        if isinstance(node, ast.Declarator):
            return node.name.text
        for attr in ("inner",):
            sub = getattr(node, attr, None)
            if sub is not None:
                node = sub
                break
        else:
            return None
    return None

