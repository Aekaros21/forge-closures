"""Candidate specification: the unit of discovery.

A candidate is two channels of (tensor, expression) terms plus tuned constants:

    b^Delta = sum_n  g_n(I1..I5, Ret, F1, PoE) * T^(n)(S_hat, W_hat)
    R       = 2 k (sum_m h_m(...) * T^(m)) : grad(U)

Expressions live in the frozen grammar of tedp.expr and reference free
constants c0..c7; ``constants`` supplies their values. The OpenFOAM dictionary
receives numeric literals only (substitute_constants).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace

from . import expr

TENSORS = tuple(f"T{i}" for i in range(1, 11))
# Only additive/multiplicative identities are free.  Even familiar epsilon
# literals count: exempting a palette of "structural" numbers lets a proposer
# synthesize arbitrary uncharged amplitudes from their ratios.
STRUCTURAL_LITERALS = frozenset({0.0, 1.0})
SEARCH_GMAX = 10.0
SEARCH_R_MAX_FACTOR = 5.0


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Term:
    tensor: str  # "T1".."T10"
    expression: str  # grammar string, may use c0..c7

    def __post_init__(self) -> None:
        if self.tensor not in TENSORS:
            raise SpecError(f"unknown tensor {self.tensor!r}")
        expr.parse(self.expression)  # raises GrammarError on violation


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    bdelta: tuple[Term, ...] = ()
    rsource: tuple[Term, ...] = ()
    constants: tuple[float, ...] = ()
    gmax: float = 10.0
    r_max_factor: float = 5.0

    def all_terms(self) -> tuple[Term, ...]:
        return self.bdelta + self.rsource

    def used_constants(self) -> set[int]:
        used: set[int] = set()
        for term in self.all_terms():
            used |= expr.used_constants(expr.parse(term.expression))
        return used

    def used_variables(self) -> set[str]:
        """Every flow variable named anywhere in this spec.

        Walked here rather than in the frozen expression module for the same
        reason as ``_contains_fitted_or_flow_symbol`` below: which variables a
        candidate may name is a search-admission rule, not grammar semantics.
        """
        used: set[str] = set()

        def visit(node: expr.Node) -> None:
            match node:
                case expr.Var(name=n):
                    used.add(n)
                case expr.Unary(arg=arg):
                    visit(arg)
                case expr.Binary(left=left, right=right):
                    visit(left)
                    visit(right)
                case expr.Call(args=args):
                    for a in args:
                        visit(a)

        for term in self.all_terms():
            visit(expr.parse(term.expression))
        return used

    def n_constants(self) -> int:
        return len(self.used_constants())

    def numeric_parameters(self) -> set[float]:
        values: set[float] = set()

        def visit(node: expr.Node) -> None:
            match node:
                case expr.Num(value=value):
                    value = abs(float(value))
                    if value not in STRUCTURAL_LITERALS:
                        values.add(value)
                case expr.Unary(arg=arg):
                    visit(arg)
                case expr.Binary(left=left, right=right):
                    visit(left)
                    visit(right)
                case expr.Call(args=args):
                    for arg in args:
                        visit(arg)

        for term in self.all_terms():
            visit(expr.parse(term.expression))
        return values

    def n_parameters(self) -> int:
        """Effective fitted complexity, including non-structural literals."""
        return self.n_constants() + len(self.numeric_parameters())

    def ast_nodes(self) -> int:
        def count(node: expr.Node) -> int:
            match node:
                case expr.Unary(arg=arg):
                    return 1 + count(arg)
                case expr.Binary(left=left, right=right):
                    return 1 + count(left) + count(right)
                case expr.Call(args=args):
                    return 1 + sum(count(arg) for arg in args)
            return 1

        return sum(count(expr.parse(t.expression)) for t in self.all_terms())

    def validate(self, search_policy: bool = False, allowed_variables=None) -> None:
        """``allowed_variables`` (search policy only) restricts the one-point inputs a
        candidate may name, e.g. to the legacy eight before the V3 solver state is
        deployed. It defaults to every candidate variable of the grammar."""
        used = self.used_constants()
        if len(used) > expr.MAX_FREE_CONSTANTS:
            raise SpecError(f"{len(used)} free constants (max {expr.MAX_FREE_CONSTANTS})")
        if used and max(used) >= expr.MAX_FREE_CONSTANTS:
            raise SpecError(
                f"candidates may only use c0..c{expr.MAX_FREE_CONSTANTS - 1} "
                f"(found c{max(used)})"
            )
        if used and max(used) >= len(self.constants):
            raise SpecError(
                f"constants c{max(used)} used but only {len(self.constants)} values given"
            )
        expected = set(range(len(self.constants)))
        if used != expected:
            raise SpecError(
                "constants must be contiguous, used exactly once as a parameter set "
                f"(used={sorted(used)}, supplied={len(self.constants)})"
            )
        if not all(math.isfinite(v) for v in self.constants):
            raise SpecError("constants must all be finite")
        if search_policy:
            # The RITA comparator variables were added to the grammar after both
            # search campaigns closed, so that one published model could be
            # expressed and run through the same solver. They are not functions
            # of a one-point state (they need wall distance, k, omega and
            # grad(k)), so a candidate naming one could be neither screened
            # a-priori nor judged novel against the library. No discovered model
            # uses them; this makes that a rule instead of a fact.
            comparator_only = self.used_variables() & set(expr.RITA_VARIABLES)
            if comparator_only:
                raise SpecError(
                    "comparator-only variables are not admissible in a "
                    f"candidate: {', '.join(sorted(comparator_only))}"
                )
            permitted = set(expr.CANDIDATE_VARIABLES if allowed_variables is None else allowed_variables)
            unavailable = self.used_variables() - permitted
            if unavailable:
                raise SpecError(
                    "variables not available in the declared interface: "
                    f"{', '.join(sorted(unavailable))}"
                )
            if self.gmax != SEARCH_GMAX:
                raise SpecError(
                    f"gmax is protocol-fixed at {SEARCH_GMAX:g}, got {self.gmax:g}"
                )
            if self.r_max_factor != SEARCH_R_MAX_FACTOR:
                raise SpecError(
                    "rMaxFactor is protocol-fixed at "
                    f"{SEARCH_R_MAX_FACTOR:g}, got {self.r_max_factor:g}"
                )
            arithmetic_reasons = search_policy_reasons(self)
            if arithmetic_reasons:
                raise SpecError("; ".join(arithmetic_reasons))


def _contains_fitted_or_flow_symbol(node: expr.Node) -> bool:
    """Return whether *node* depends on a flow variable or ``cN``.

    Kept here, rather than in the frozen expression module, because this is a
    search-admission rule and not part of grammar/evaluator semantics.
    """

    match node:
        case expr.Var() | expr.Const():
            return True
        case expr.Unary(arg=arg):
            return _contains_fitted_or_flow_symbol(arg)
        case expr.Binary(left=left, right=right):
            return (
                _contains_fitted_or_flow_symbol(left)
                or _contains_fitted_or_flow_symbol(right)
            )
        case expr.Call(args=args):
            return any(_contains_fitted_or_flow_symbol(arg) for arg in args)
    return False


def _is_direct_numeric_literal(node: expr.Node) -> bool:
    """A numeric leaf, optionally with its grammar-level unary minus."""

    return isinstance(node, expr.Num) or (
        isinstance(node, expr.Unary)
        and node.op == "-"
        and isinstance(node.arg, expr.Num)
    )


def _negated_signature(signature: tuple) -> tuple:
    if signature[0] == "num":
        value = -signature[1]
        return ("num", 0.0 if value == 0.0 else value)
    if signature[0] == "neg":
        return signature[1]
    return ("neg", signature)


def _signature_sort_key(signature: tuple) -> str:
    # repr(tuple) is deterministic for the strings and finite floats emitted
    # by the parser and avoids requiring unlike tuple payloads to compare.
    return repr(signature)


def _normalized_signature(node: expr.Node) -> tuple:
    """Small structural normal form used only to expose coefficient gaming.

    It removes the exact/structural identities 0 and 1, flattens associative
    addition and multiplication, orders commutative operands, hoists unary
    signs, and reduces idempotent min/max calls.  It intentionally does not
    perform general symbolic algebra or use flow-domain facts.
    """

    match node:
        case expr.Num(value=value):
            value = float(value)
            return ("num", 0.0 if value == 0.0 else value)
        case expr.Var(name=name):
            return ("var", name)
        case expr.Const(index=index):
            return ("const", index)
        case expr.Unary(op="-", arg=arg):
            return _negated_signature(_normalized_signature(arg))
        case expr.Binary(op=op, left=left, right=right):
            lhs = _normalized_signature(left)
            rhs = _normalized_signature(right)
            if op in {"+", "-"}:
                if op == "-":
                    rhs = _negated_signature(rhs)
                operands: list[tuple] = []
                for signature in (lhs, rhs):
                    if signature[0] == "add":
                        operands.extend(signature[1])
                    elif signature != ("num", 0.0):
                        operands.append(signature)
                if not operands:
                    return ("num", 0.0)
                operands.sort(key=_signature_sort_key)
                if len(operands) == 1:
                    return operands[0]
                return ("add", tuple(operands))
            if op == "*":
                if lhs == ("num", 0.0) or rhs == ("num", 0.0):
                    return ("num", 0.0)
                negative = False
                factors: list[tuple] = []
                for signature in (lhs, rhs):
                    if signature[0] == "neg":
                        negative = not negative
                        signature = signature[1]
                    if signature[0] == "num" and signature[1] < 0.0:
                        negative = not negative
                        signature = ("num", -signature[1])
                    if signature == ("num", 1.0):
                        continue
                    if signature[0] == "mul":
                        factors.extend(signature[1])
                    else:
                        factors.append(signature)
                factors.sort(key=_signature_sort_key)
                if not factors:
                    product = ("num", 1.0)
                elif len(factors) == 1:
                    product = factors[0]
                else:
                    product = ("mul", tuple(factors))
                return _negated_signature(product) if negative else product
            # The guarded evaluator makes x/1 differ from x by ~1e-12, but 1
            # is nevertheless the declared structural division identity for
            # search-policy normalization.  No other cancellation is done.
            negative = False
            if lhs[0] == "neg":
                negative = not negative
                lhs = lhs[1]
            if lhs[0] == "num" and lhs[1] < 0.0:
                negative = not negative
                lhs = ("num", -lhs[1])
            if rhs[0] == "neg":
                negative = not negative
                rhs = rhs[1]
            if rhs[0] == "num" and rhs[1] < 0.0:
                negative = not negative
                rhs = ("num", -rhs[1])
            quotient = lhs if rhs == ("num", 1.0) else ("div", lhs, rhs)
            return _negated_signature(quotient) if negative else quotient
        case expr.Call(func=func, args=args):
            normalized = tuple(_normalized_signature(arg) for arg in args)
            if func == "abs":
                argument = normalized[0]
                if argument[0] == "neg":
                    argument = argument[1]
                if argument[0] == "call" and argument[1] == "abs":
                    return argument
                return ("call", func, (argument,))
            if func in {"min", "max"}:
                ordered = tuple(sorted(normalized, key=_signature_sort_key))
                if ordered[0] == ordered[1]:
                    return ordered[0]
                return ("call", func, ordered)
            return ("call", func, normalized)
    raise TypeError(f"unhandled expression node {node!r}")


def _additive_operands(node: expr.Node) -> list[expr.Node]:
    """Flatten a +/- tree; signs cannot make a repeated atom independent."""

    if isinstance(node, expr.Binary) and node.op in {"+", "-"}:
        return _additive_operands(node.left) + _additive_operands(node.right)
    if isinstance(node, expr.Unary) and node.op == "-":
        return _additive_operands(node.arg)
    return [node]


def _unsigned_signature(node: expr.Node) -> tuple:
    signature = _normalized_signature(node)
    if signature[0] == "neg":
        return signature[1]
    if signature[0] == "num" and signature[1] < 0.0:
        return ("num", -signature[1])
    return signature


def _repeated_additive_operand(nodes: list[expr.Node]) -> str | None:
    seen: set[tuple] = set()
    for node in nodes:
        for operand in _additive_operands(node):
            signature = _unsigned_signature(operand)
            if signature == ("num", 0.0):
                continue
            if signature in seen:
                return expr.to_string(operand)
            seen.add(signature)
    return None


def search_policy_reasons(spec: CandidateSpec) -> list[str]:
    """Return AST-level arithmetic/parsimony policy violations.

    Numeric leaves remain legal (and non-0/1 values count through
    :meth:`CandidateSpec.n_parameters`).  What is forbidden is computing a
    number from numeric-only arithmetic, or spelling an integer coefficient
    by repeating the same normalized additive atom.  The latter is checked
    both inside one expression and across terms that share a channel/tensor.
    """

    reasons: list[str] = []

    def visit(node: expr.Node, location: str) -> None:
        if (
            not _contains_fitted_or_flow_symbol(node)
            and not _is_direct_numeric_literal(node)
        ):
            reasons.append(
                f"computed numeric subexpression in {location} is forbidden; "
                "use one literal or cN"
            )
            return
        if isinstance(node, expr.Binary):
            if node.op in {"+", "-"}:
                repeated = _repeated_additive_operand([node])
                if repeated is not None:
                    reasons.append(
                        f"repeated additive operand {repeated!r} in {location}; "
                        "use one literal or cN coefficient"
                    )
            visit(node.left, location)
            visit(node.right, location)
        elif isinstance(node, expr.Unary):
            visit(node.arg, location)
        elif isinstance(node, expr.Call):
            for argument in node.args:
                visit(argument, location)

    channels = (("bdelta", spec.bdelta), ("rsource", spec.rsource))
    for channel, terms in channels:
        grouped: dict[str, list[expr.Node]] = {}
        for term in terms:
            node = expr.parse(term.expression)
            location = f"{channel} {term.tensor} expression {term.expression!r}"
            visit(node, location)
            grouped.setdefault(term.tensor, []).append(node)
        for tensor, nodes in grouped.items():
            if len(nodes) < 2:
                continue
            repeated = _repeated_additive_operand(nodes)
            if repeated is not None:
                reasons.append(
                    f"repeated additive operand {repeated!r} across "
                    f"{channel} {tensor} terms; merge the terms and use one "
                    "literal or cN coefficient"
                )
    return list(dict.fromkeys(reasons))


def canonical(spec: CandidateSpec) -> CandidateSpec:
    """Deterministic form: expressions serialized from their ASTs, terms
    sorted by (tensor index, expression). Constants are NOT renumbered."""

    def canon_terms(terms: tuple[Term, ...]) -> tuple[Term, ...]:
        canon = [
            Term(t.tensor, expr.to_string(expr.parse(t.expression))) for t in terms
        ]
        return tuple(sorted(canon, key=lambda t: (int(t.tensor[1:]), t.expression)))

    return replace(spec, bdelta=canon_terms(spec.bdelta), rsource=canon_terms(spec.rsource))


def _hash_payload(spec: CandidateSpec, with_constants: bool) -> str:
    c = canonical(spec)
    payload = {
        "bdelta": [[t.tensor, t.expression] for t in c.bdelta],
        "rsource": [[t.tensor, t.expression] for t in c.rsource],
        "gmax": c.gmax,
        "r_max_factor": c.r_max_factor,
    }
    if with_constants:
        payload["constants"] = list(c.constants)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def spec_hash(spec: CandidateSpec) -> str:
    """Identity of the evaluated model (structure + constants)."""
    return _hash_payload(spec, with_constants=True)


def struct_hash(spec: CandidateSpec) -> str:
    """Structure-only identity — used for island diversity/niching."""
    return _hash_payload(spec, with_constants=False)


def to_json(spec: CandidateSpec) -> str:
    return json.dumps(
        {
            "name": spec.name,
            "bdelta": [[t.tensor, t.expression] for t in spec.bdelta],
            "rsource": [[t.tensor, t.expression] for t in spec.rsource],
            "constants": list(spec.constants),
            "gmax": spec.gmax,
            "r_max_factor": spec.r_max_factor,
        },
        indent=2,
        sort_keys=True,
    )


def from_json(text: str) -> CandidateSpec:
    d = json.loads(text)
    spec = CandidateSpec(
        name=d["name"],
        bdelta=tuple(Term(t, e) for t, e in d.get("bdelta", [])),
        rsource=tuple(Term(t, e) for t, e in d.get("rsource", [])),
        constants=tuple(float(v) for v in d.get("constants", [])),
        gmax=float(d.get("gmax", 10.0)),
        r_max_factor=float(d.get("r_max_factor", 5.0)),
    )
    spec.validate()
    return spec


def to_foam_coeffs(spec: CandidateSpec, mode: str = "expressions") -> str:
    """Render the kOmegaSSTBasisCoeffs sub-dictionary. Free constants are
    substituted with numeric literals — the C++ side never sees c0..c7."""
    spec.validate()

    def render(terms: tuple[Term, ...]) -> str:
        rendered = []
        for t in terms:
            node = expr.substitute_constants(expr.parse(t.expression), spec.constants)
            rendered.append(f'            ({t.tensor} "{expr.to_string(node)}")')
        return "\n".join(rendered)

    return (
        "    kOmegaSSTBasisCoeffs\n"
        "    {\n"
        f"        mode            {mode};\n"
        "        bDelta\n"
        "        {\n"
        "            terms\n"
        "            (\n"
        f"{render(spec.bdelta)}\n"
        "            );\n"
        "        }\n"
        "        rSource\n"
        "        {\n"
        "            terms\n"
        "            (\n"
        f"{render(spec.rsource)}\n"
        "            );\n"
        "        }\n"
        f"        gMax            {spec.gmax};\n"
        f"        rMaxFactor      {spec.r_max_factor};\n"
        "        bDeltaRelax     0.5;\n"
        "        rampIterations  200;\n"
        "        nonlinearProduction on;\n"
        "        realizabilityClip   on;\n"
        '        a1Expr          "";\n'
        "    }\n"
    )
