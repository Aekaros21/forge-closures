"""Constant tuning: CMA-ES (pycma) on a fast a-priori objective.

The objective fits the Phase-2 frozen discrepancy fields. Full Tier-0/Tier-1
feasibility is checked by the search loop before and after tuning; if tuning
leaves the feasible set, the original legal proposal is retained. HOM23 fit
is a reported diagnostic, not a hidden optimizer penalty. The tuned child then
earns exactly one Tier-2 case (hills case_1p0) as a sanity gate before the full
suite.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from .. import candidate as cand
from .. import tier1
from ..spec import CandidateSpec


def deterministic_seed(
    base_seed: int, generation: int, island: str, parent_hash: str
) -> int:
    """Stable pycma seed for one parent, independent of evaluation order."""
    payload = f"{base_seed}:{generation}:{island}:{parent_hash}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**31 - 2) + 1


def apriori_objective(
    spec: CandidateSpec,
    states: cand.States,
    b_target: np.ndarray,
    r_target: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Case-balanced misfit to deployable frozen-RANS targets."""
    b = cand.eval_bdelta(spec, states)
    rhat = cand.eval_rhat(spec, states)
    # Fit the target actually representable under this candidate's rMax.
    r_bound = spec.r_max_factor * cand.BETA_STAR
    r_deployable = np.clip(r_target, -r_bound, r_bound)
    w = (
        np.full(len(states), 1.0 / len(states))
        if weights is None else np.asarray(weights, dtype=np.float64)
    )
    w = w / max(float(np.sum(w)), 1.0e-300)
    b_rms = max(float(np.sqrt(np.sum(w[:, None, None] * b_target**2))), 1e-12)
    r_rms = max(float(np.sqrt(np.sum(w * r_deployable**2))), 1e-12)
    e_b = float(np.sqrt(np.sum(w[:, None, None] * (b - b_target) ** 2))) / b_rms
    e_r = float(np.sqrt(np.sum(w * (rhat - r_deployable) ** 2))) / r_rms
    return 0.7 * e_b + 0.3 * e_r


def tune(
    spec: CandidateSpec,
    states: cand.States,
    b_target: np.ndarray,
    r_target: np.ndarray,
    baseline_channel: tier1.Channel1DResult,
    budget: int = 40,
    sigma0: float = 0.3,
    seed: int = 0,
    weights: np.ndarray | None = None,
) -> tuple[CandidateSpec, float]:
    """Return (tuned spec, objective). Free constants only; a spec without
    constants is returned unchanged."""
    # Kept in the API because the search owns the common Tier-1 baseline.  The
    # full channel solve is deliberately not repeated for every CMA sample;
    # exact channel feasibility is enforced around this optional tuning step.
    _ = baseline_channel
    n = len(spec.constants)
    if n == 0:
        return spec, apriori_objective(spec, states, b_target, r_target, weights)

    import cma

    def objective(x: np.ndarray) -> float:
        s = replace(spec, constants=tuple(float(v) for v in x))
        return apriori_objective(s, states, b_target, r_target, weights)

    # pycma raises if x0 sits outside the bounds; proposals may use large
    # saturation knees (e.g. Ret scales) — clip into range
    x0 = np.clip(np.asarray(spec.constants, dtype=float), -49.9, 49.9)
    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "maxfevals": budget,
            "bounds": [-50.0, 50.0],
            "verbose": -9,
            # pycma treats zero as "choose a time-dependent seed".  Normalize
            # every caller value into its deterministic positive range.
            "seed": max(1, int(seed) % (2**31 - 1)),
        },
    )
    es.optimize(objective)
    best_x = tuple(float(v) for v in es.result.xbest)
    best = replace(spec, constants=best_x)
    return best, float(es.result.fbest)
