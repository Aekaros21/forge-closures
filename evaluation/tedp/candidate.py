"""Numpy-side evaluation of candidate specs on sampled or DNS states.

This is the Python twin of what kOmegaSSTBasis computes in the solver:
identical grammar (tedp.expr), identical basis (tedp.tensors.integrity_basis),
identical gMax clamping. Used by Tier 0, Tier 1 a-priori, the sparse
regression, the novelty filter, and CMA-ES.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import expr, features, tensors
from .spec import CandidateSpec

PoE_CLAMP = 10.0  # mirrors the model's clamp on P/eps
BETA_STAR = 0.09


@dataclass(frozen=True)
class States:
    """A batch of one-point states for the grammar + basis.

    ``extra`` carries the FORGE V3 physical-state inputs (``a``, ``g``, ``ohat``,
    ``zhat``; see tedp.features) and the seven derived features. A state batch
    built without them describes a flow with no resolved pressure/TKE gradient
    and no system rotation: every V3 feature is then exactly zero, so legacy
    expressions evaluate exactly as before.
    """

    shat: np.ndarray  # (N,3,3) traceless symmetric
    what: np.ndarray  # (N,3,3) antisymmetric
    T: np.ndarray     # (N,10,3,3)
    inv: np.ndarray   # (N,5)
    ret: np.ndarray   # (N,)
    f1: np.ndarray    # (N,)
    poe: np.ndarray   # (N,)
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return self.shat.shape[0]

    def variables(self) -> dict[str, np.ndarray]:
        out = {
            "I1": self.inv[:, 0],
            "I2": self.inv[:, 1],
            "I3": self.inv[:, 2],
            "I4": self.inv[:, 3],
            "I5": self.inv[:, 4],
            "Ret": self.ret,
            "F1": self.f1,
            # mirror the solver exactly: its PoE = min(G/(beta* k w), 10)
            # with G = nut*2|S|^2 >= 0 — never negative (DNS-table PoE uses
            # the signed true production and reaches -8.6; candidates never
            # see that in-solver, so the evaluator must not either)
            "PoE": np.clip(self.poe, 0.0, PoE_CLAMP),
        }
        zeros = None
        for name in features.NAMES:
            value = self.extra.get(name)
            if value is None:
                if zeros is None:
                    zeros = np.zeros(len(self))
                value = zeros
            out[name] = value
        return out

    def inputs(self) -> dict[str, np.ndarray]:
        """The V3 feature inputs (zero vectors when the batch carries none)."""
        if all(name in self.extra for name in features.INPUT_NAMES):
            return {name: self.extra[name] for name in features.INPUT_NAMES}
        return features.zero_inputs(len(self))

    def take(self, index) -> "States":
        """Row subset (boolean mask or index array) of every array, including extras."""
        return States(self.shat[index], self.what[index], self.T[index], self.inv[index],
                      self.ret[index], self.f1[index], self.poe[index],
                      {key: np.asarray(value)[index] for key, value in self.extra.items()})


def make_states(
    shat: np.ndarray,
    what: np.ndarray,
    ret: np.ndarray,
    f1: np.ndarray,
    poe: np.ndarray,
    inputs: dict | None = None,
) -> States:
    """Build a batch; ``inputs`` (tedp.features.INPUT_NAMES) adds the V3 state."""
    T, inv = tensors.integrity_basis(shat, what)
    extra = {}
    if inputs is not None:
        extra = {name: np.asarray(inputs[name], dtype=np.float64) for name in features.INPUT_NAMES}
        extra.update(features.dimensionless_features(extra["a"], extra["g"], np.asarray(shat, dtype=np.float64),
                                                     extra["ohat"], extra["zhat"]))
    return States(np.asarray(shat), np.asarray(what), T, inv, np.asarray(ret), np.asarray(f1),
                  np.asarray(poe), extra)


def sample_states(n: int = 100_000, seed: int = 0) -> States:
    """Random one-point states: log-uniform strain/rotation magnitudes in
    [1e-3, 30] (measured as sqrt(tr(S^2)), sqrt(-tr(W^2))), Haar-rotated
    directions, Ret log-uniform [1e-5, 1e5], and F1 uniform.  PoE is sampled
    on the SST solver manifold rather than independently: an F2 draw in
    [0,1] determines nut*omega/k and hence PoE=2*(nut*omega/k)*I1/betaStar.
    Endpoints F2=0 and F2=1 are included explicitly."""
    rng = np.random.default_rng(seed)

    a = rng.normal(size=(n, 3, 3))
    s = tensors.dev(a)
    w = 0.5 * (a - np.swapaxes(a, -1, -2))

    s_norm = np.sqrt(np.einsum("nij,nij->n", s, s))
    w_norm = np.sqrt(np.einsum("nij,nij->n", w, w))
    s_mag = 10.0 ** rng.uniform(-5.0, np.log10(30.0), size=n)
    w_mag = 10.0 ** rng.uniform(-5.0, np.log10(30.0), size=n)
    s *= (s_mag / np.maximum(s_norm, 1e-300))[:, None, None]
    w *= (w_mag / np.maximum(w_norm, 1e-300))[:, None, None]

    ret = 10.0 ** rng.uniform(-5.0, 5.0, size=n)  # down to the viscous sublayer
    f1 = rng.uniform(0.0, 1.0, size=n)
    # SST: nut*omega/k = a1/max(a1, F2*sqrt(2 I1)), with F2 in [0,1].
    # Independent PoE draws create impossible combinations (PoE=O(1) at
    # I1~0) and falsely reject ratios that are regular on the solver manifold.
    i1 = np.einsum("nij,nij->n", s, s)
    f2 = rng.uniform(0.0, 1.0, size=n)
    if n:
        f2[0::3] = 0.0
        f2[1::3] = 1.0
    nut_hat = A1 / np.maximum(A1, f2 * np.sqrt(2.0 * i1))
    poe = np.clip(2.0 * nut_hat * i1 / BETA_STAR, 0.0, PoE_CLAMP)
    # V3 state inputs are drawn AFTER every legacy draw, so the legacy arrays of a
    # given (n, seed) are byte-identical to the pre-V3 sampler: gate verdicts of
    # expressions that do not use the new features are unchanged.
    inputs = features.sample_inputs(rng, w) if n else features.zero_inputs(0)
    return make_states(s, w, ret, f1, poe, inputs)


def _eval_channel(
    terms,
    constants: tuple[float, ...],
    states: States,
    gmax: float,
) -> np.ndarray:
    """sum_n clamp(g_n) * T^(n)  -> (N,3,3)"""
    out = np.zeros((len(states), 3, 3))
    variables = states.variables()
    for term in terms:
        node = expr.parse(term.expression)
        g = expr.evaluate(node, variables, constants)
        g = np.clip(np.broadcast_to(g, (len(states),)), -gmax, gmax)
        idx = int(term.tensor[1:]) - 1
        out += g[:, None, None] * states.T[:, idx]
    return out


def eval_bdelta(spec: CandidateSpec, states: States) -> np.ndarray:
    """b^Delta on each state, (N,3,3)."""
    return _eval_channel(spec.bdelta, spec.constants, states, spec.gmax)


def eval_rsource_bR(spec: CandidateSpec, states: States) -> np.ndarray:
    """The R-channel tensor b^R on each state, (N,3,3)."""
    return _eval_channel(spec.rsource, spec.constants, states, spec.gmax)


def eval_rhat_raw(spec: CandidateSpec, states: States) -> np.ndarray:
    """Unclipped normalized source R/(k omega) = 2 b^R:Shat, (N,)."""
    bR = eval_rsource_bR(spec, states)
    return 2.0 * np.einsum("nij,nij->n", bR, states.shat)


def eval_rhat(spec: CandidateSpec, states: States) -> np.ndarray:
    """Deployed normalized production source ``R/(k omega)``, (N,).

    The dimensional R in the solver is 2 k b^R : grad(U); contracting with
    Shat (both sides traceless-symmetric) gives the k- and omega-free
    observable used for comparisons.  The final clip mirrors
    ``rMaxFactor*betaStar*k*omega`` in kOmegaSSTBasis.
    """
    bound = spec.r_max_factor * BETA_STAR
    return np.clip(eval_rhat_raw(spec, states), -bound, bound)


# SST constants for the Boussinesq part of realizability checks
A1 = 0.31
B1_F23 = 1.0  # the F2/F23 factor spans [0,1]; both extremes are tested


def boussinesq_b(states: States, f2: float) -> np.ndarray:
    """b_B = -(nut*omega/k) Shat with the SST limiter,
    nut*omega/k = a1 / max(a1, f2 * sqrt(2 I1))."""
    i1 = np.maximum(states.inv[:, 0], 0.0)
    factor = A1 / np.maximum(A1, f2 * np.sqrt(2.0 * i1))
    return -factor[:, None, None] * states.shat


def realizability_violation(b: np.ndarray) -> np.ndarray:
    """Depth of Lumley-triangle violation per state (0 = realizable)."""
    ev = np.linalg.eigvalsh(b)
    low = np.maximum(-(1.0 / 3.0) - ev[..., 0], 0.0)
    high = np.maximum(ev[..., -1] - 2.0 / 3.0, 0.0)
    return np.maximum(low, high)
