"""Constant tuning on CFD: CMA-ES over a candidate's free constants.

Every gate-legal structure gets the same treatment before it is compared
with anything else: its constants are tuned by CMA-ES against the real
solver on a small calibration subset, with a budget proportional to the
number of constants.  Points that leave the physics gates are charged a
penalty instead of CFD, so the optimiser stays inside the feasible set and
the search never spends solver time on an inadmissible model.

Constants are optimised as multiplicative factors around the proposer's
values (``c = c0 * exp(z)``; additive ``0.1 * z`` for a zero start), which
keeps every constant's sign and scale, and the CMA state is pickled into the
checkpoint so a tuner resumes exactly where it stopped.
"""

from __future__ import annotations

import base64
import json
import math
import pickle
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import cma
import numpy as np

from .. import scoring
from ..spec import CandidateSpec

TUNING_CASES: tuple[str, ...] = ("case_1p0", "case_1p5")
EVALS_PER_CONSTANT = 6
MAX_EVALS = 48
Z_BOUND = 3.0                # |z| <= 3 allows a factor of e^3 ~ 20 either way
SIGMA0 = 0.3
INFEASIBLE_PENALTY = 1.1     # worse than any diverged evaluation (10 x 0.1)
IMPROVEMENT_MARGIN = 1.0e-4  # tuned must beat the original by this on the subset

GateFn = Callable[[CandidateSpec], tuple[bool, list[str]]]


def tuning_budget(n_constants: int) -> int:
    return int(min(MAX_EVALS, EVALS_PER_CONSTANT * max(1, n_constants)))


def encode_constants(c0: np.ndarray, constants: np.ndarray) -> np.ndarray:
    z = np.zeros_like(c0, dtype=float)
    for i, base in enumerate(c0):
        if base != 0.0:
            ratio = constants[i] / base
            z[i] = math.log(ratio) if ratio > 0 else Z_BOUND
        else:
            z[i] = constants[i] / 0.1
    return np.clip(z, -Z_BOUND, Z_BOUND)


def decode_constants(c0: np.ndarray, z: np.ndarray) -> list[float]:
    out = []
    for base, value in zip(c0, z):
        if base != 0.0:
            out.append(float(base * math.exp(float(value))))
        else:
            out.append(float(0.1 * float(value)))
    return out


def subset_objective(per_case: dict[str, float | None]) -> float:
    """Mean composite over the tuning cases with the frozen divergence rule:
    a failed case costs ten times the worst completed composite (floor 1)."""
    completed = [float(v) for v in per_case.values() if v is not None]
    failed = sum(1 for v in per_case.values() if v is None)
    worst = max(completed) if completed else scoring.DIVERGED_FLOOR
    total = completed + [
        max(scoring.DIVERGENCE_MULTIPLIER * worst, scoring.DIVERGED_FLOOR)
    ] * failed
    if not total:
        raise ValueError("subset objective needs at least one case")
    return float(np.mean(total))


@dataclass
class TunerPoint:
    index: int
    z: list[float]
    constants: list[float]
    feasible: bool
    gate_reasons: list[str] = field(default_factory=list)


class CfdTuner:
    """Ask/tell CMA-ES over one candidate's constants with durable state."""

    def __init__(
        self, spec: CandidateSpec, seed: int, budget: int | None = None,
        es: Any | None = None, state: dict[str, Any] | None = None,
    ):
        if not spec.constants:
            raise ValueError("cannot tune a candidate without free constants")
        self.spec = spec
        self.c0 = np.asarray(spec.constants, dtype=float)
        self.seed = int(seed)
        self.budget = int(budget if budget is not None else tuning_budget(len(spec.constants)))
        n = len(self.c0)
        if es is None:
            es = cma.CMAEvolutionStrategy(
                np.zeros(n), SIGMA0,
                {"seed": self.seed, "verbose": -9, "bounds": [-Z_BOUND, Z_BOUND],
                 "maxfevals": 10 * self.budget + 100},
            )
            # The proposer's own constants are always evaluated on the subset.
            es.inject([np.zeros(n)])
        self.es = es
        self.max_rounds = int(math.ceil(self.budget / self.es.popsize)) + 2
        self.state: dict[str, Any] = state or {
            "rounds_done": 0, "cfd_evals": 0, "history": [],
            "original_f": None, "best": None, "status": "active",
            "pending": None,
        }

    # ---- persistence -------------------------------------------------------
    def to_state(self) -> dict[str, Any]:
        payload = dict(self.state)
        payload.update({
            "seed": self.seed, "budget": self.budget,
            "es_pickle_b64": base64.b64encode(pickle.dumps(self.es)).decode(),
            "popsize": int(self.es.popsize), "max_rounds": self.max_rounds,
        })
        return payload

    @classmethod
    def from_state(cls, spec: CandidateSpec, payload: dict[str, Any]) -> "CfdTuner":
        es = pickle.loads(base64.b64decode(payload["es_pickle_b64"]))
        state = {
            key: payload[key] for key in (
                "rounds_done", "cfd_evals", "history", "original_f", "best",
                "status", "pending",
            )
        }
        return cls(spec, int(payload["seed"]), int(payload["budget"]), es=es, state=state)

    # ---- ask / tell ---------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.state["status"] == "active"

    def spec_for(self, constants: list[float]) -> CandidateSpec:
        return replace(self.spec, constants=tuple(float(c) for c in constants))

    def ask(self, gate: GateFn) -> list[TunerPoint]:
        """Propose one CMA population; gate each point before any CFD."""
        if not self.active:
            return []
        if self.state.get("pending"):
            # A round was asked but never told (interrupted): replay it.
            return [TunerPoint(**point) for point in self.state["pending"]]
        solutions = self.es.ask()
        points: list[TunerPoint] = []
        for index, z in enumerate(solutions):
            constants = decode_constants(self.c0, np.asarray(z, dtype=float))
            candidate = self.spec_for(constants)
            try:
                feasible, reasons = gate(candidate)
            except Exception as exc:  # a gate crash is a rejection, not a loss
                feasible, reasons = False, [f"gate error: {type(exc).__name__}: {exc}"[:200]]
            points.append(TunerPoint(
                index=index, z=[float(v) for v in z], constants=constants,
                feasible=bool(feasible), gate_reasons=list(reasons),
            ))
        self.state["pending"] = [vars(point) for point in points]
        return points

    def pending_points(self) -> list[TunerPoint]:
        return [TunerPoint(**point) for point in self.state.get("pending") or []]

    def tell(self, per_case_by_index: dict[int, dict[str, float | None]]) -> None:
        """Report subset scores for the feasible points of the pending round."""
        points = self.pending_points()
        if not points:
            raise RuntimeError("tell() without a pending ask()")
        solutions: list[np.ndarray] = []
        values: list[float] = []
        round_index = int(self.state["rounds_done"])
        for point in points:
            if point.feasible:
                per_case = per_case_by_index.get(point.index)
                if per_case is None:
                    raise RuntimeError(f"missing subset result for point {point.index}")
                f = subset_objective(per_case)
                self.state["cfd_evals"] += 1
            else:
                per_case, f = None, INFEASIBLE_PENALTY
            solutions.append(np.asarray(point.z, dtype=float))
            values.append(f)
            record = {
                "round": round_index, "index": point.index,
                "constants": point.constants, "feasible": point.feasible,
                "gate_reasons": point.gate_reasons, "f": f, "per_case": per_case,
            }
            self.state["history"].append(record)
            if round_index == 0 and all(abs(v) < 1e-12 for v in point.z):
                self.state["original_f"] = f
            best = self.state["best"]
            if point.feasible and (best is None or f < best["f"]):
                self.state["best"] = {
                    "constants": point.constants, "f": f, "per_case": per_case,
                    "round": round_index, "index": point.index,
                }
        self.es.tell(solutions, values)
        self.state["rounds_done"] = round_index + 1
        self.state["pending"] = None
        if (
            self.state["cfd_evals"] >= self.budget
            or self.state["rounds_done"] >= self.max_rounds
            or self.es.stop()
        ):
            self.state["status"] = "done"

    # ---- outcome -------------------------------------------------------------
    def tuned_spec(self) -> CandidateSpec | None:
        """Best feasible constants if they beat the original on the subset."""
        best = self.state.get("best")
        original = self.state.get("original_f")
        if best is None:
            return None
        if original is not None and best["f"] > original - IMPROVEMENT_MARGIN:
            return None
        constants = [float(c) for c in best["constants"]]
        if np.allclose(constants, self.c0, rtol=1e-9, atol=1e-12):
            return None
        return self.spec_for(constants)

    def summary(self) -> dict[str, Any]:
        return {
            "budget": self.budget, "cfd_evals": self.state["cfd_evals"],
            "rounds": self.state["rounds_done"], "status": self.state["status"],
            "original_f": self.state.get("original_f"),
            "best_f": None if self.state.get("best") is None else self.state["best"]["f"],
            "best_constants": None if self.state.get("best") is None
            else self.state["best"]["constants"],
            "infeasible_points": sum(
                not record["feasible"] for record in self.state["history"]
            ),
            "popsize": int(self.es.popsize),
        }
