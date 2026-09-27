"""Tier 0: millisecond-scale hard gates.

1. grammar legality + <=8 free constants   (spec construction/validation)
2. boundedness: aggregate bDelta and the contracted R observable finite and
   bounded on solver-consistent states; material reliance on either solver
   clamp is rejected
3. realizability: b_B + b^Delta must not violate the Lumley triangle any
   deeper than b_B alone, at both F2 extremes, on sampled AND DNS states
4. stock-SST recovery: the correction vanishes as I1, I2 -> 0
5. novelty: no library model can reproduce the candidate to <3% RMS on
   both channels over both state sets (see tedp.library)

Any failure is a hard reject with a reason string.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import candidate as cand
from . import expr, features
from .spec import CandidateSpec, SpecError

GMAX_CAP = 50.0        # a candidate may not declare its way out of boundedness
RMAX_FACTOR_CAP = 20.0
# Realizability thresholds are calibrated against the five converged frozen-
# RANS truth cases.  cbfs13700 is deliberately excluded from truth gates: its
# frozen omega residual remains O(1) at the prescribed endpoint, so it is only
# a Tier-2 stability case.  Across the five valid targets the worst violation
# fraction is 6.78e-5, gross depth 0.00605, and total bDelta norm 0.8181.
# These margins are pinned by tests/test_gate_calibration.py.
REALIZABILITY_ENVELOPE = 3.0  # |Shat|,|What| range for the Lumley-triangle gate
REALIZABILITY_FRAC = 1.0e-3   # per-case cap (valid-truth max: 6.78e-5)
REALIZABILITY_GROSS = 0.02    # max violation depth (valid-truth max: 0.00605)
CHANNEL_NORM_CAP = 1.65       # >=2x valid-truth Frobenius max 0.8181
# Backward-compatible name for external reports; the gate is aggregate now.
TERM_NORM_CAP = CHANNEL_NORM_CAP
CLAMP_EFFECT_TOL = 0.01
R_CLAMP_FRAC_TOL = 5.0e-3
R_CLAMP_EFFECT_TOL = 0.01
RMS_FLOOR_BOUND = 1.0e-12
REALIZABILITY_ABS_TOL = 1.0e-6
RECOVERY_SCALE = 1.0e-4
RECOVERY_TOL = 1.0e-3
NOVELTY_RMS_REJECT = 0.03


@dataclass
class Tier0Result:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    max_g: float = 0.0
    new_violation: float = 0.0
    recovery_residual: float = 0.0
    clamp_effect: float = 0.0
    r_clamp_fraction: float = 0.0
    r_clamp_effect: float = 0.0
    equivalent_to: str = ""
    novelty_best_rms: float = float("inf")

    def fail(self, reason: str) -> None:
        self.passed = False
        self.reasons.append(reason)


def check_boundedness(
    spec: CandidateSpec, states: cand.States
) -> tuple[bool, float, str]:
    """Two conditions.

    1. every g_n finite on every state;
    2. aggregate bDelta satisfies the calibrated tensor cap; R is checked in
       its deployed observable, rhat=2*bR:Shat.  A latent bR norm is neither
       calibrated nor representation-invariant because only its contraction
       enters the equations.
    """
    variables = states.variables()
    worst = 0.0
    for label, terms in (("bDelta", spec.bdelta), ("R", spec.rsource)):
        total = np.zeros((len(states), 3, 3))
        for term in terms:
            g = expr.evaluate(expr.parse(term.expression), variables, spec.constants)
            g = np.broadcast_to(np.asarray(g, dtype=np.float64), (len(states),))
            if not np.all(np.isfinite(g)):
                return False, float("inf"), f"{term.tensor}: non-finite g"
            idx = int(term.tensor[1:]) - 1
            gc = np.clip(g, -spec.gmax, spec.gmax)
            total += gc[:, None, None] * states.T[:, idx]
        if label == "bDelta":
            observable = np.sqrt(np.einsum("nij,nij->n", total, total))
        else:
            observable = np.abs(
                2.0 * np.einsum("nij,nij->n", total, states.shat)
            )
        if not np.all(np.isfinite(observable)):
            return False, float("inf"), f"{label}: non-finite observable"
        channel_worst = float(np.max(observable))
        worst = max(worst, channel_worst)
        if label == "bDelta" and channel_worst > CHANNEL_NORM_CAP:
            return False, worst, f"{label}: |sum(g T)| = {channel_worst:.3g}"

    return True, worst, ""


def clamp_effect(spec: CandidateSpec, states: cand.States) -> float:
    """Relative RMS influence of gMax on deployed channel observables."""
    worst = 0.0
    for evaluator in (cand.eval_bdelta, cand.eval_rhat_raw):
        clamped = evaluator(spec, states)
        unclamped = evaluator(replace(spec, gmax=1.0e30), states)
        rms = float(np.sqrt(np.mean(clamped**2)))
        diff = float(np.sqrt(np.mean((clamped - unclamped) ** 2)))
        if rms > RMS_FLOOR_BOUND:
            worst = max(worst, diff / rms)
    return worst


def r_clamp_stats(spec: CandidateSpec, states: cand.States) -> tuple[float, float]:
    """Return (clipped fraction, relative RMS effect) of rMaxFactor."""
    raw = cand.eval_rhat_raw(spec, states)
    deployed = cand.eval_rhat(spec, states)
    changed = np.abs(raw - deployed) > 1.0e-12
    rms = float(np.sqrt(np.mean(deployed**2)))
    effect = float(np.sqrt(np.mean((raw - deployed) ** 2))) / max(
        rms, RMS_FLOOR_BOUND
    )
    return float(np.mean(changed)), effect


def check_realizability(
    spec: CandidateSpec, states: cand.States
) -> tuple[bool, float]:
    """Two-part gate.

    F2=1 (the SST limiter active, its own realizability guarantee): the
    total b must lie inside the Lumley triangle, absolutely.

    F2=0 (limiter off; the linear part alone already violates at extreme
    sampled states): the candidate may not amplify the violation —
    violation(total) <= 1.02 * violation(base) + 0.02.

    Both parts are POPULATION statements, not worst-single-cell ones: the
    true DNS correction reaches |b^Delta| ~ 0.7 at separation and rides the
    triangle edge, so any useful candidate violates *somewhere* among 3.6e5
    states. Gate: violating fraction <= REALIZABILITY_FRAC (mirroring the
    finalists' clipFraction ~ 0 requirement) and no violation deeper than
    REALIZABILITY_GROSS anywhere; the in-solver clip backstops the rest."""
    bdelta = cand.eval_bdelta(spec, states)

    b1 = cand.boussinesq_b(states, f2=1.0)
    abs_v = cand.realizability_violation(b1 + bdelta)

    b0 = cand.boussinesq_b(states, f2=0.0)
    base = cand.realizability_violation(b0)
    total = cand.realizability_violation(b0 + bdelta)
    rel_v = total - (1.02 * base + 0.02)

    # Population failure is the union.  Taking max of the two marginal
    # fractions can hide disjoint violation populations.
    frac = float(np.mean((abs_v > REALIZABILITY_ABS_TOL) | (rel_v > 0.0)))
    gross = max(float(np.max(abs_v)), float(np.max(rel_v)))
    ok = frac <= REALIZABILITY_FRAC and gross <= REALIZABILITY_GROSS
    return ok, (gross if gross > REALIZABILITY_GROSS else frac)


def check_recovery(spec: CandidateSpec, states: cand.States) -> tuple[bool, float]:
    """Scale Shat, What by RECOVERY_SCALE: the correction must vanish."""
    # At this scale the SST viscosity limiter is inactive for every F2, so
    # PoE tends to 2*I1/betaStar.  Keeping the original PoE while shrinking
    # strain would test an impossible solver state.
    poe_small = np.clip(
        2.0 * states.inv[:, 0] * RECOVERY_SCALE**2 / cand.BETA_STAR,
        0.0,
        cand.PoE_CLAMP,
    )
    # The V3 state follows the same limit: relative vorticity and frame rotation
    # vanish with the velocity gradient; pressure/TKE gradients may persist.
    small = cand.make_states(
        states.shat * RECOVERY_SCALE,
        states.what * RECOVERY_SCALE,
        states.ret,
        states.f1,
        poe_small,
        features.scaled_inputs(states.inputs(), RECOVERY_SCALE),
    )
    worst = float(np.max(np.abs(cand.eval_bdelta(spec, small))))
    worst = max(worst, float(np.max(np.abs(cand.eval_rhat(spec, small)))))
    return worst <= RECOVERY_TOL, worst


def run_tier0(
    spec: CandidateSpec,
    states: cand.States,
    dns_states: cand.States | None = None,
    library=None,
) -> Tier0Result:
    result = Tier0Result(passed=True)

    # 1. legality (constructing the spec already parsed every expression)
    try:
        spec.validate(search_policy=True)
    except (SpecError, expr.GrammarError) as e:
        result.fail(f"legality: {e}")
        return result
    if not np.isfinite(spec.gmax) or spec.gmax <= 0.0:
        result.fail(f"legality: gmax must be finite and positive (got {spec.gmax})")
        return result
    if not np.isfinite(spec.r_max_factor) or spec.r_max_factor <= 0.0:
        result.fail(
            "legality: rMaxFactor must be finite and positive "
            f"(got {spec.r_max_factor})"
        )
        return result
    if spec.gmax > GMAX_CAP:
        result.fail(f"legality: gmax {spec.gmax} > cap {GMAX_CAP}")
        return result
    if spec.r_max_factor > RMAX_FACTOR_CAP:
        result.fail(
            f"legality: rMaxFactor {spec.r_max_factor} > cap {RMAX_FACTOR_CAP}"
        )
        return result

    if dns_states is None:
        dns_sets: list[cand.States] = []
    elif isinstance(dns_states, cand.States):
        dns_sets = [dns_states]
    else:
        dns_sets = list(dns_states)
    state_sets = [states, *dns_sets]

    # 2. boundedness
    for ss in state_sets:
        ok, worst, why = check_boundedness(spec, ss)
        result.max_g = max(result.max_g, worst)
        if not ok:
            result.fail(f"boundedness: {why}")
            return result
    for ss in state_sets:
        result.clamp_effect = max(result.clamp_effect, clamp_effect(spec, ss))
        rfrac, reffect = r_clamp_stats(spec, ss)
        result.r_clamp_fraction = max(result.r_clamp_fraction, rfrac)
        result.r_clamp_effect = max(result.r_clamp_effect, reffect)
    if result.clamp_effect > CLAMP_EFFECT_TOL:
        result.fail(
            f"boundedness: gMax changes correction RMS by "
            f"{result.clamp_effect:.2%}"
        )
        return result
    if (
        result.r_clamp_fraction > R_CLAMP_FRAC_TOL
        or result.r_clamp_effect > R_CLAMP_EFFECT_TOL
    ):
        result.fail(
            f"boundedness: rMax clips {result.r_clamp_fraction:.2%} of states "
            f"(RMS effect {result.r_clamp_effect:.2%})"
        )
        return result

    # 3. realizability — on the DNS states and a physically plausible
    # envelope (|Shat|, |What| <= REALIZABILITY_ENVELOPE, an order of
    # magnitude beyond anything observed in the calibration flows). The
    # full extreme battery serves boundedness; testing the Lumley triangle
    # at |Shat| = 30 rejects any useful amplitude at states that cannot
    # occur, and the in-solver clip (clipFraction ~ 0 required of
    # finalists) backstops the remainder.
    real_sets = []
    mask = (
        np.sqrt(np.maximum(states.inv[:, 0], 0.0)) <= REALIZABILITY_ENVELOPE
    ) & (np.sqrt(np.abs(states.inv[:, 1])) <= REALIZABILITY_ENVELOPE)
    if np.any(mask):
        real_sets.append(states.take(mask))
    real_sets.extend(dns_sets)
    for ss in real_sets:
        ok, worst = check_realizability(spec, ss)
        result.new_violation = max(result.new_violation, worst)
        if not ok:
            result.fail(f"realizability: new Lumley violation {worst:.3g}")
            return result

    # 4. stock recovery
    ok, worst = check_recovery(spec, states)
    result.recovery_residual = worst
    if not ok:
        result.fail(f"recovery: correction does not vanish (|b|={worst:.3g})")
        return result

    # 5. novelty
    if library is not None:
        verdict = library.novelty(spec, states, dns_sets)
        result.novelty_best_rms = verdict.best_rms
        if verdict.equivalent_to:
            result.equivalent_to = verdict.equivalent_to
            result.fail(
                f"novelty: equivalent to {verdict.equivalent_to} "
                f"(rms {verdict.best_rms:.4f})"
            )
            return result

    return result
