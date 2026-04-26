"""AST node definitions for C24 parser."""

from dataclasses import dataclass, field
from typing import Optional, Union
from .tokens import SourceLocation


@dataclass(kw_only=True)
class Node:
    """Base class for all AST nodes."""
    location: Optional[SourceLocation] = field(default=None, repr=False)


# === Type Nodes ===

@dataclass(kw_only=True)
class TypeNode(Node):
    """Base class for type nodes."""
    pass


@dataclass(kw_only=True)
class BasicType(TypeNode):
    """Basic type (int, char, void, etc.)."""
    name: str  # "int", "char", "void", "short", "long", "float", "double"
    is_signed: Optional[bool] = None  # None = default, True = signed, False = unsigned
    is_const: bool = False
    is_volatile: bool = False


@dataclass(kw_only=True)
class PointerType(TypeNode):
    """Pointer to another type."""
    base_type: TypeNode
    is_const: bool = False
    is_volatile: bool = False


@dataclass(kw_only=True)
class ArrayType(TypeNode):
    """Array of another type."""
    base_type: TypeNode
    size: Optional['Expression'] = None  # None for unsized arrays


@dataclass(kw_only=True)
class FunctionType(TypeNode):
    """Function type."""
    return_type: TypeNode
    param_types: list[TypeNode] = field(default_factory=list)
    is_variadic: bool = False


@dataclass(kw_only=True)
class StructType(TypeNode):
    """Struct type reference."""
    name: Optional[str] = None  # None for anonymous structs
    is_union: bool = False
    members: list['StructMember'] = field(default_factory=list)  # For inline definitions
    is_const: bool = False
    is_volatile: bool = False


@dataclass(kw_only=True)
class EnumType(TypeNode):
    """Enum type reference."""
    name: Optional[str] = None
    values: list['EnumValue'] = field(default_factory=list)  # For inline definitions
    is_const: bool = False
    is_volatile: bool = False


@dataclass(kw_only=True)
class ComplexType(TypeNode):
    """Complex number type (_Complex)."""
    base_type: str  # "float", "double", or "long double"
    is_const: bool = False
    is_volatile: bool = False


# === Expression Nodes ===

@dataclass(kw_only=True)
class Expression(Node):
    """Base class for expression nodes."""
    pass


@dataclass(kw_only=True)
class IntLiteral(Expression):
    """Integer literal."""
    value: int
    is_long: bool = False     # True if has L or LL suffix
    is_long_long: bool = False  # True if has LL suffix (forces long-long width)
    is_unsigned: bool = False  # True if has U suffix
    is_hex: bool = False      # True if hex/octal literal (affects type promotion)


@dataclass(kw_only=True)
class FloatLiteral(Expression):
    """Floating-point literal."""
    value: float
    is_float: bool = False  # True if 'f'/'F' suffixed (type float), False = type double
    is_imaginary: bool = False  # True if 'i'/'I' suffixed (gcc extension)


@dataclass(kw_only=True)
class CharLiteral(Expression):
    """Character literal."""
    value: int  # ASCII/char code


@dataclass(kw_only=True)
class StringLiteral(Expression):
    """String literal."""
    value: str
    is_wide: bool = False


@dataclass(kw_only=True)
class BoolLiteral(Expression):
    """Boolean literal (true/false)."""
    value: bool


@dataclass(kw_only=True)
class NullptrLiteral(Expression):
    """nullptr literal."""
    pass


@dataclass(kw_only=True)
class Identifier(Expression):
    """Identifier reference."""
    name: str


@dataclass(kw_only=True)
class BinaryOp(Expression):
    """Binary operation."""
    op: str  # "+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^",
             # "&&", "||", "==", "!=", "<", ">", "<=", ">=", ",", "="
    left: Expression
    right: Expression


@dataclass(kw_only=True)
class UnaryOp(Expression):
    """Unary operation."""
    op: str  # "-", "+", "!", "~", "*", "&", "++", "--" (prefix)
    operand: Expression
    is_prefix: bool = True  # False for postfix ++ and --


@dataclass(kw_only=True)
class TernaryOp(Expression):
    """Ternary conditional expression."""
    condition: Expression
    true_expr: Expression
    false_expr: Expression


@dataclass(kw_only=True)
class Call(Expression):
    """Function call."""
    func: Expression
    args: list[Expression] = field(default_factory=list)


@dataclass(kw_only=True)
class Index(Expression):
    """Array indexing."""
    array: Expression
    index: Expression


@dataclass(kw_only=True)
class Member(Expression):
    """Struct/union member access."""
    obj: Expression
    member: str
    is_arrow: bool = False  # True for ->, False for .


@dataclass(kw_only=True)
class Cast(Expression):
    """Type cast."""
    target_type: TypeNode
    expr: Expression


@dataclass(kw_only=True)
class SizeofExpr(Expression):
    """sizeof with expression."""
    expr: Expression


@dataclass(kw_only=True)
class SizeofType(Expression):
    """sizeof with type."""
    target_type: TypeNode


@dataclass(kw_only=True)
class VaArgExpr(Expression):
    """`va_arg(ap, type-name)` — read the next variadic arg of the
    given type from `ap` and advance `ap` past it.

    `ap` is an arbitrary expression naming a `va_list` (typically an
    Identifier). `target_type` is a type-name parsed via the same path
    as `sizeof(type)` and casts. Treated as a builtin form by the
    parser because the second operand is a type, not an expression."""
    ap: 'Expression'
    target_type: TypeNode


@dataclass(kw_only=True)
class Compound(Expression):
    """Compound literal (C99): (type){initializer}."""
    target_type: TypeNode
    init: 'InitializerList'


# === Statement Nodes ===

@dataclass(kw_only=True)
class Statement(Node):
    """Base class for statement nodes."""
    pass


@dataclass(kw_only=True)
class ExpressionStmt(Statement):
    """Expression statement."""
    expr: Optional[Expression] = None  # None for empty statement


@dataclass(kw_only=True)
class CompoundStmt(Statement):
    """Compound statement (block)."""
    items: list[Union[Statement, 'Declaration']] = field(default_factory=list)


@dataclass(kw_only=True)
class IfStmt(Statement):
    """If statement."""
    condition: Expression
    then_branch: Statement
    else_branch: Optional[Statement] = None


@dataclass(kw_only=True)
class WhileStmt(Statement):
    """While loop."""
    condition: Expression
    body: Statement


@dataclass(kw_only=True)
class DoWhileStmt(Statement):
    """Do-while loop."""
    body: Statement
    condition: Expression


@dataclass(kw_only=True)
class ForStmt(Statement):
    """For loop."""
    body: Statement
    init: Optional[Union[Expression, 'Declaration']] = None
    condition: Optional[Expression] = None
    update: Optional[Expression] = None


@dataclass(kw_only=True)
class SwitchStmt(Statement):
    """Switch statement."""
    expr: Expression
    body: Statement  # Usually a CompoundStmt with case labels


@dataclass(kw_only=True)
class CaseStmt(Statement):
    """Case label in switch."""
    value: Optional[Expression]  # None for default
    stmt: Statement


@dataclass(kw_only=True)
class LabelStmt(Statement):
    """Labeled statement."""
    label: str
    stmt: Statement


@dataclass(kw_only=True)
class GotoStmt(Statement):
    """Goto statement.

    For named goto, `label` is the destination and `target` is None.
    For GCC computed goto (`goto *expr;`), `target` is the address
    expression and `label` is "".
    """
    label: str = ""
    target: Optional[Expression] = None


@dataclass(kw_only=True)
class LabelAddr(Expression):
    """GCC `&&label` — address of a function-local label, usable as
    `void *` and as the operand of computed goto."""
    label: str


@dataclass(kw_only=True)
class TypeofType(TypeNode):
    """`typeof(expr)` — the type of `expr`. Resolved at codegen time
    by walking the operand and computing its type. Used in places
    where a type-name is expected (declarations, casts, sizeof,
    va_arg).
    """
    operand: Expression


@dataclass(kw_only=True)
class TypesCompatibleP(Expression):
    """`__builtin_types_compatible_p(T1, T2)` — compile-time int 0/1
    indicating whether two types are C-compatible. Backends fold to a
    constant."""
    t1: TypeNode
    t2: TypeNode


@dataclass(kw_only=True)
class OffsetofExpr(Expression):
    """`__builtin_offsetof(T, designator)` — compile-time byte offset
    of a member within a struct/union type.

    `designator` is encoded as a Member/Index chain rooted at a
    synthetic `Identifier(name='__offsetof_root')` of `target_type`.
    The codegen walks the chain to compute the integer offset and
    emits it as a constant.
    """
    target_type: TypeNode
    designator: Expression


@dataclass(kw_only=True)
class BreakStmt(Statement):
    """Break statement."""
    pass


@dataclass(kw_only=True)
class ContinueStmt(Statement):
    """Continue statement."""
    pass


@dataclass(kw_only=True)
class ReturnStmt(Statement):
    """Return statement."""
    value: Optional[Expression] = None


@dataclass(kw_only=True)
class AsmStmt(Statement):
    """Inline asm statement (`asm("..." : ... : ...)`).

    Backends typically treat this as a no-op since uc_core doesn't
    interpret the asm template. The `operands` list captures the
    expressions in the output/input operand groups so a backend can
    at least evaluate them for side effects (which gcc semantics
    require — the constraints define register/memory binding, but
    the expression itself still executes per call site).

    `outputs` and `inputs` carry `(constraint, expr)` pairs so a
    backend that wants to honor simple asm idioms (`asm("" : "=r"(o)
    : "0"(i))` ≡ `o = i`) has the constraint metadata available.
    """
    template: str = ""
    is_volatile: bool = False
    operands: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    inputs: list = field(default_factory=list)


# === Declaration Nodes ===

@dataclass(kw_only=True)
class Declaration(Node):
    """Base class for declarations."""
    pass


@dataclass(kw_only=True)
class VarDecl(Declaration):
    """Variable declaration."""
    name: str
    var_type: TypeNode
    init: Optional[Expression] = None
    storage_class: Optional[str] = None  # "static", "extern", "register", "auto"


@dataclass(kw_only=True)
class ParamDecl(Declaration):
    """Function parameter declaration."""
    name: Optional[str]  # Can be None in prototypes
    param_type: TypeNode


@dataclass(kw_only=True)
class FunctionDecl(Declaration):
    """Function declaration or definition."""
    name: str
    return_type: TypeNode
    params: list['ParamDecl'] = field(default_factory=list)
    body: Optional[CompoundStmt] = None  # None for declarations
    is_variadic: bool = False
    storage_class: Optional[str] = None
    is_inline: bool = False


@dataclass(kw_only=True)
class StructDecl(Declaration):
    """Struct/union declaration."""
    name: Optional[str] = None
    members: list['StructMember'] = field(default_factory=list)
    is_union: bool = False
    is_definition: bool = False  # True if members are defined


@dataclass(kw_only=True)
class StructMember(Node):
    """Struct/union member."""
    name: Optional[str]  # None for anonymous
    member_type: TypeNode
    bit_width: Optional[Expression] = None


@dataclass(kw_only=True)
class EnumDecl(Declaration):
    """Enum declaration."""
    name: Optional[str] = None
    values: list['EnumValue'] = field(default_factory=list)
    is_definition: bool = False


@dataclass(kw_only=True)
class EnumValue(Node):
    """Enum value."""
    name: str
    value: Optional[Expression] = None


@dataclass(kw_only=True)
class TypedefDecl(Declaration):
    """Typedef declaration."""
    name: str
    target_type: TypeNode


@dataclass(kw_only=True)
class DeclarationList(Declaration):
    """Multiple declarations from a single declaration statement (e.g., 'int a, b;')."""
    declarations: list[Declaration] = field(default_factory=list)


# === Initializers ===

@dataclass(kw_only=True)
class InitializerList(Expression):
    """Initializer list { ... }."""
    values: list[Union[Expression, 'DesignatedInit']] = field(default_factory=list)


@dataclass(kw_only=True)
class DesignatedInit(Node):
    """Designated initializer."""
    designators: list[Union[str, Expression]] = field(default_factory=list)  # .member or [index]
    value: Expression


@dataclass(kw_only=True)
class RangeDesignator(Node):
    """Range designator for array initialization [start ... end]."""
    start: Expression
    end: Expression


@dataclass(kw_only=True)
class StmtExpr(Expression):
    """Statement expression (GCC extension): ({ ... })."""
    body: 'CompoundStmt'


@dataclass(kw_only=True)
class GenericSelection(Expression):
    """C11 _Generic selection expression."""
    controlling_expr: Expression
    associations: list[tuple[Optional[TypeNode], Expression]]  # (type, expr) pairs, None type = default


# === Top Level ===

@dataclass(kw_only=True)
class TranslationUnit(Node):
    """Top-level compilation unit."""
    declarations: list[Declaration] = field(default_factory=list)


# Type aliases for convenience
Decl = Declaration
Stmt = Statement
Expr = Expression
