"""The candidate-expression grammar: parser and numpy evaluator.

This module is the Python twin of the OpenFOAM-side parser in
``src/kOmegaSSTBasis/basisExpr``. The two implementations MUST agree to 1e-12
on every state (enforced by tests/test_expr_parity.py against golden files from
``src/testBasisExpr``). Any change here is a grammar change and is forbidden
after the Phase-1 freeze.

Grammar v1 (pre-declared widening over the plan: sqrt, abs):

    expr    := term (('+' | '-') term)*
    term    := unary (('*' | '/') unary)*
    unary   := '-' unary | primary
    primary := NUMBER | VAR | CONST | FUNC1 '(' expr ')'
             | FUNC2 '(' expr ',' expr ')' | '(' expr ')'
    VAR     := I1 I2 I3 I4 I5 Ret F1 PoE
    CONST   := c0 .. c7          (free constants, tuned by CMA-ES)
    FUNC1   := tanh exp sqrt abs
    FUNC2   := min max

Guarded semantics (identical on both sides):
    a / b     ->  (a*b) / (b*b + GUARD_EPS)      sign-preserving, smooth
    exp(x)    ->  exp(min(x, EXP_CLAMP))
    sqrt(x)   ->  sqrt(max(x, 0))
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

GUARD_EPS = 1.0e-12
EXP_CLAMP = 50.0

VARIABLES = ("I1", "I2", "I3", "I4", "I5", "Ret", "F1", "PoE",
             # comparator-only: the RITA classifier and its k-budget ratios
             # (Buchanan et al., arXiv:2504.06758). No discovered model uses
             # these; they exist so that model can be expressed and compared
             # in the same framework.
             "sigmaSL", "phiDkPk", "phiDkCk", "phik", "ReOmega",
             # FORGE V3 physical-state extension (tedp.features): slots 13..19
             # in the C++ variable table. Definitions live in one place,
             # src/tedp/features.py, mirrored by kOmegaSSTBasis.C.
             "Gp", "Gk", "Apk", "Psn", "Ksn", "Rf", "Rw")
RITA_VARIABLES = ("sigmaSL", "phiDkPk", "phiDkCk", "phik", "ReOmega")
V3_VARIABLES = ("Gp", "Gk", "Apk", "Psn", "Ksn", "Rf", "Rw")
LEGACY_VARIABLES = ("I1", "I2", "I3", "I4", "I5", "Ret", "F1", "PoE")
#: One-point variables a candidate may name (comparator-only RITA variables excluded).
CANDIDATE_VARIABLES = LEGACY_VARIABLES + V3_VARIABLES
# c0..c7 are the candidate constants (the C++ parser accepts exactly these);
# c8..c31 exist ONLY so library comparator structures can carry their larger
# published coefficient sets through the same evaluator — candidates are
# capped at MAX_FREE_CONSTANTS by spec.validate()/tier0.
CONSTANTS = tuple(f"c{i}" for i in range(32))
#: Slot layout shared with exprNode.H: state variables first, then c0..c7.
N_STATE_VARS = len(VARIABLES)
FIRST_RITA_VAR = VARIABLES.index("sigmaSL")
FIRST_V3_VAR = VARIABLES.index("Gp")
FUNCTIONS_1 = ("tanh", "exp", "sqrt", "abs")
FUNCTIONS_2 = ("min", "max")
MAX_FREE_CONSTANTS = 8


class GrammarError(ValueError):
    """Raised on any token or construct outside the frozen grammar."""


@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Var:
    name: str  # element of VARIABLES


@dataclass(frozen=True)
class Const:
    index: int  # 0..7, the free constant c<index>


@dataclass(frozen=True)
class Unary:
    op: str  # "-"
    arg: "Node"


@dataclass(frozen=True)
class Binary:
    op: str  # + - * /
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class Call:
    func: str
    args: tuple["Node", ...]


Node = Num | Var | Const | Unary | Binary | Call

_TOKEN = re.compile(
    r"\s*(?:(?P<num>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<op>[-+*/(),]))"
)


def _tokenize(expr: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if m is None:
            rest = expr[pos:].lstrip()
            if not rest:
                break
            raise GrammarError(f"illegal character {rest[0]!r} at column {pos} in {expr!r}")
        pos = m.end()
        if m.group("num") is not None:
            tokens.append(("num", m.group("num")))
        elif m.group("name") is not None:
            tokens.append(("name", m.group("name")))
        else:
            tokens.append(("op", m.group("op")))
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], source: str):
        self.tokens = tokens
        self.pos = 0
        self.source = source

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise GrammarError(f"unexpected end of expression in {self.source!r}")
        self.pos += 1
        return tok

    def expect(self, op: str) -> None:
        tok = self.next()
        if tok != ("op", op):
            raise GrammarError(f"expected {op!r}, got {tok[1]!r} in {self.source!r}")

    def parse(self) -> Node:
        node = self.expr()
        if self.peek() is not None:
            raise GrammarError(f"trailing tokens after expression in {self.source!r}")
        return node

    def expr(self) -> Node:
        node = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.next()[1]
            node = Binary(op, node, self.term())
        return node

    def term(self) -> Node:
        node = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.next()[1]
            node = Binary(op, node, self.unary())
        return node

    def unary(self) -> Node:
        if self.peek() == ("op", "-"):
            self.next()
            return Unary("-", self.unary())
        return self.primary()

    def primary(self) -> Node:
        kind, text = self.next()
        if kind == "num":
            return Num(float(text))
        if kind == "op" and text == "(":
            node = self.expr()
            self.expect(")")
            return node
        if kind == "name":
            if text in VARIABLES:
                return Var(text)
            if text in CONSTANTS:
                return Const(int(text[1:]))
            if text in FUNCTIONS_1:
                self.expect("(")
                arg = self.expr()
                self.expect(")")
                return Call(text, (arg,))
            if text in FUNCTIONS_2:
                self.expect("(")
                a = self.expr()
                self.expect(",")
                b = self.expr()
                self.expect(")")
                return Call(text, (a, b))
            raise GrammarError(f"unknown identifier {text!r} in {self.source!r}")
        raise GrammarError(f"unexpected token {text!r} in {self.source!r}")


def parse(expr: str) -> Node:
    """Parse ``expr`` or raise GrammarError. The only entry point."""
    if not expr or not expr.strip():
        raise GrammarError("empty expression")
    return _Parser(_tokenize(expr), expr).parse()


def evaluate(
    node: Node,
    variables: dict[str, np.ndarray],
    constants: tuple[float, ...] | list[float] | None = None,
) -> np.ndarray:
    """Evaluate with guarded semantics. `variables` maps VARIABLES to arrays
    (broadcastable); `constants` supplies c0..c7 where the node uses them."""
    match node:
        case Num(value=v):
            return np.asarray(v, dtype=np.float64)
        case Var(name=name):
            return np.asarray(variables[name], dtype=np.float64)
        case Const(index=i):
            if constants is None or i >= len(constants):
                raise GrammarError(f"constant c{i} used but not supplied")
            return np.asarray(constants[i], dtype=np.float64)
        case Unary(op="-", arg=arg):
            return -evaluate(arg, variables, constants)
        case Binary(op=op, left=left, right=right):
            a = evaluate(left, variables, constants)
            b = evaluate(right, variables, constants)
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            # guarded division, identical formula to the C++ side
            return (a * b) / (b * b + GUARD_EPS)
        case Call(func=func, args=args):
            a = evaluate(args[0], variables, constants)
            if func == "tanh":
                return np.tanh(a)
            if func == "exp":
                return np.exp(np.minimum(a, EXP_CLAMP))
            if func == "sqrt":
                return np.sqrt(np.maximum(a, 0.0))
            if func == "abs":
                return np.abs(a)
            b = evaluate(args[1], variables, constants)
            if func == "min":
                return np.minimum(a, b)
            return np.maximum(a, b)
    raise GrammarError(f"unhandled node {node!r}")


def used_constants(node: Node) -> set[int]:
    match node:
        case Const(index=i):
            return {i}
        case Unary(arg=arg):
            return used_constants(arg)
        case Binary(left=left, right=right):
            return used_constants(left) | used_constants(right)
        case Call(args=args):
            out: set[int] = set()
            for a in args:
                out |= used_constants(a)
            return out
    return set()


def substitute_constants(node: Node, values: tuple[float, ...] | list[float]) -> Node:
    """Replace c<i> with numeric literals — what the OpenFOAM dict receives."""
    match node:
        case Const(index=i):
            if i >= len(values):
                raise GrammarError(f"constant c{i} has no value")
            return Num(float(values[i]))
        case Unary(op=op, arg=arg):
            return Unary(op, substitute_constants(arg, values))
        case Binary(op=op, left=left, right=right):
            return Binary(op, substitute_constants(left, values), substitute_constants(right, values))
        case Call(func=func, args=args):
            return Call(func, tuple(substitute_constants(a, values) for a in args))
    return node


def to_string(node: Node) -> str:
    """Deterministic serialization (fully parenthesized) — the canonical form
    used for spec hashing and for rendering the OpenFOAM dictionary."""
    match node:
        case Num(value=v):
            return repr(v)
        case Var(name=name):
            return name
        case Const(index=i):
            return f"c{i}"
        case Unary(op=op, arg=arg):
            return f"(-{to_string(arg)})"
        case Binary(op=op, left=left, right=right):
            return f"({to_string(left)}{op}{to_string(right)})"
        case Call(func=func, args=args):
            return f"{func}({','.join(to_string(a) for a in args)})"
    raise GrammarError(f"unhandled node {node!r}")
