"""The objective. FROZEN BY SHA-256 AT THE END OF PHASE 1.

After the freeze (recorded in results/FREEZE_scoring.json and checked by
evaldb at import), this file must never change. Everything here is pure
numpy on sampled arrays — no OpenFOAM, no I/O.

Per case:
    E_U  = sqrt(mean |U - U_ref|^2) / U_bulk          over DNS points
    E_Cf = sqrt(mean (Cf - Cf_ref)^2)                 over the wall profile
    E_uv = sqrt(mean (uv - uv_ref)^2) / U_bulk^2      over DNS points
    composite = 0.5 E_U + 0.3 E_Cf + 0.2 E_uv

Objective over a case set:
    J = mean(composite) + 0.005 * n_free_constants
    a diverged/unconverged case contributes 10 x the worst converged
    composite in the same evaluation batch (and at least DIVERGED_FLOOR).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

W_U = 0.5
W_CF = 0.3
W_UV = 0.2
PARSIMONY = 0.005
DIVERGENCE_MULTIPLIER = 10.0
DIVERGED_FLOOR = 1.0


@dataclass(frozen=True)
class CaseRef:
    """Reference (DNS/LES/experiment) data for one case, non-dimensional."""

    name: str
    u_bulk: float
    u_ref: np.ndarray       # (n_pts, 3) or (n_pts, 2)
    cf_ref: np.ndarray      # (n_wall,)
    uv_ref: np.ndarray      # (n_pts,)


@dataclass(frozen=True)
class CaseSample:
    """Model fields sampled at the reference points."""

    name: str
    u: np.ndarray
    cf: np.ndarray
    uv: np.ndarray


@dataclass(frozen=True)
class CaseScore:
    name: str
    e_u: float
    e_cf: float
    e_uv: float

    @property
    def composite(self) -> float:
        return W_U * self.e_u + W_CF * self.e_cf + W_UV * self.e_uv


def case_score(sample: CaseSample, ref: CaseRef) -> CaseScore:
    if sample.u.shape != ref.u_ref.shape:
        raise ValueError(
            f"{ref.name}: sample U shape {sample.u.shape} != ref {ref.u_ref.shape}"
        )
    du = np.asarray(sample.u - ref.u_ref, dtype=np.float64)
    du = du.reshape(du.shape[0], -1)  # (n_pts,) -> (n_pts, 1); vectors kept
    e_u = float(np.sqrt(np.mean(np.sum(du**2, axis=1)))) / ref.u_bulk
    e_cf = float(np.sqrt(np.mean((sample.cf - ref.cf_ref) ** 2)))
    e_uv = float(np.sqrt(np.mean((sample.uv - ref.uv_ref) ** 2))) / ref.u_bulk**2
    return CaseScore(ref.name, e_u, e_cf, e_uv)


def objective(
    scores: dict[str, CaseScore],
    n_constants: int,
    diverged: set[str] = frozenset(),
) -> float:
    """J over one case set. `scores` holds converged cases only; `diverged`
    names the cases that failed (each contributes 10 x the worst converged
    composite, floored at DIVERGED_FLOOR when nothing converged)."""
    composites = [s.composite for s in scores.values()]
    worst = max(composites) if composites else DIVERGED_FLOOR
    total = composites + [
        max(DIVERGENCE_MULTIPLIER * worst, DIVERGED_FLOOR) for _ in diverged
    ]
    if not total:
        raise ValueError("objective needs at least one case")
    return float(np.mean(total)) + PARSIMONY * n_constants
