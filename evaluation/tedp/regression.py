"""Phase-2 sparse regression: inferred fields -> compact symbolic forms.

Feature library: monomials of (I1, I2) to degree 3 plus saturating features
(tanh(I1), tanh(|I2|), F1, PoE), per basis tensor T1..T4 for b^Delta and
T1..T3 for R (the 2-D independent set; the calibration flows are planar).
Elastic-net path (L1-heavy) prunes terms; each support is refit by plain
least squares (debiasing); the Pareto front is complexity vs a-priori RMS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import candidate as cand
from .spec import CandidateSpec, Term

MONOMIALS: tuple[tuple[str, ...], ...] = (
    ("1",), ("I1",), ("I2",), ("I1", "I1"), ("I1", "I2"), ("I2", "I2"),
    ("I1", "I1", "I1"), ("I1", "I1", "I2"), ("I1", "I2", "I2"),
    ("I2", "I2", "I2"),
)
SATURATORS = ("tanh(I1)", "tanh(abs(I2))", "F1", "PoE")
BDELTA_TENSORS = ("T1", "T2", "T3", "T4")
R_TENSORS = ("T1", "T2", "T3")
SYMM6_FROBENIUS_WEIGHTS = np.array([1.0, 2.0, 2.0, 1.0, 2.0, 1.0])
R_BOUND = 5.0 * cand.BETA_STAR


def _feature_matrix(fields: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    i1, i2 = fields["I1"], fields["I2"]
    cols: list[np.ndarray] = []
    names: list[str] = []
    for mono in MONOMIALS:
        val = np.ones_like(i1)
        for v in mono:
            val = val * (i1 if v == "I1" else i2 if v == "I2" else 1.0)
        cols.append(val)
        names.append("*".join(mono))
    cols.append(np.tanh(i1)); names.append("tanh(I1)")
    cols.append(np.tanh(np.abs(i2))); names.append("tanh(abs(I2))")
    cols.append(fields["F1out"]); names.append("F1")
    cols.append(np.minimum(fields["PoE"], 10.0)); names.append("PoE")
    return np.stack(cols, axis=1), names


@dataclass
class FitResult:
    channel: str                      # "bdelta" | "rsource"
    terms: list[tuple[str, str]]      # (tensor, expression with literals)
    n_terms: int
    apriori_rms: float                # relative to target rms
    coefficients: dict[tuple[str, str], float]


def _design_bdelta(fields) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    feats, names = _feature_matrix(fields)
    n = len(feats)
    cols, labels = [], []
    for tname in BDELTA_TENSORS:
        Tn = fields[tname]  # (n, 6) symm components from fluidfoam
        for fi, fname in enumerate(names):
            cols.append((feats[:, fi][:, None] * Tn).ravel())
            labels.append((tname, fname))
    A = np.stack(cols, axis=1)
    y = fields["bDelta"].ravel()
    return A, y, labels


def _design_r(fields) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    feats, names = _feature_matrix(fields)
    # R = 2 k omega (sum h_m T_m : Shat)  =>  dimensionless target
    # y = R/(k omega) = 2 sum h_m (T_m : Shat).  Fit the target exposed by
    # the fixed deployment safeguard, not the unrepresentable raw tail.
    # The contraction over the 6 symm components doubles off-diagonals.
    w6 = SYMM6_FROBENIUS_WEIGHTS
    shat = fields["T1"]  # T1 IS Shat
    y = np.clip(
        fields["RField"]
        / np.maximum(fields["k"] * fields["omega"], 1e-30),
        -R_BOUND,
        R_BOUND,
    )
    cols, labels = [], []
    for tname in R_TENSORS:
        contr = np.einsum("nc,nc,c->n", fields[tname], shat, w6)
        for fi, fname in enumerate(names):
            cols.append(2.0 * feats[:, fi] * contr)
            labels.append((tname, fname))
    A = np.stack(cols, axis=1)
    return A, y, labels


PATH_SUBSAMPLE = 80_000   # rows used for the support search; refits use all


def _path_fit(A: np.ndarray, y: np.ndarray, labels, channel: str,
              max_terms: int = 12, seed: int = 0,
              sample_weight: np.ndarray | None = None,
              prediction_clip: float | None = None) -> list[FitResult]:
    """Elastic-net path on a row subsample -> unique supports -> OLS refit
    of every support on the FULL data (the a-priori RMS reported is the
    full-data one)."""
    from sklearn.linear_model import ElasticNet

    weights = (
        np.ones(len(y), dtype=np.float64)
        if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    weights = weights / max(float(np.mean(weights)), 1.0e-300)
    # Scale features under the same measure as the regression objective.
    # Unweighted scaling would let raw case cardinality influence support
    # selection even after the loss itself is case-balanced.
    scale = np.sqrt(
        np.sum(weights[:, None] * A**2, axis=0) / float(np.sum(weights))
    )
    scale[scale == 0] = 1.0
    An = A / scale
    y_rms = float(np.sqrt(np.average(y**2, weights=weights))) or 1.0

    rng = np.random.default_rng(seed)
    if len(y) > PATH_SUBSAMPLE:
        # Uniform subsampling plus sample_weight applies case balancing once.
        # Sampling with p proportional to the same weights would square the
        # correction and overemphasize the small periodic-hills cases.
        idx = rng.choice(len(y), size=PATH_SUBSAMPLE, replace=False)
        As, ys, ws = An[idx], y[idx], weights[idx]
    else:
        As, ys, ws = An, y, weights

    supports: list[tuple[int, ...]] = []
    for alpha in np.logspace(-5, 0, 16):
        en = ElasticNet(alpha=alpha, l1_ratio=0.9, fit_intercept=False,
                        max_iter=1500, tol=1e-5, warm_start=False)
        en.fit(As, ys, sample_weight=ws)
        sup = tuple(np.flatnonzero(np.abs(en.coef_) > 1e-10))
        if 0 < len(sup) <= max_terms and sup not in supports:
            supports.append(sup)

    fits: list[FitResult] = []
    for sup in supports:
        root_w = np.sqrt(weights)
        coef, *_ = np.linalg.lstsq(
            A[:, sup] * root_w[:, None], y * root_w, rcond=None
        )
        prediction = A[:, sup] @ coef
        if prediction_clip is not None:
            prediction = np.clip(prediction, -prediction_clip, prediction_clip)
        resid = float(
            np.sqrt(np.average((prediction - y) ** 2, weights=weights))
        ) / y_rms
        terms: dict[str, list[str]] = {}
        cmap: dict[tuple[str, str], float] = {}
        for ci, c in zip(sup, coef):
            tname, fname = labels[ci]
            piece = f"{c:.6g}" if fname == "1" else f"{c:.6g}*{fname}"
            terms.setdefault(tname, []).append(piece)
            cmap[(tname, fname)] = float(c)
        term_list = [
            (tname, "+".join(pieces).replace("+-", "-"))
            for tname, pieces in sorted(terms.items())
        ]
        fits.append(FitResult(channel, term_list, len(sup), resid, cmap))
    return fits


def fit_bdelta(fields, max_terms: int = 12) -> list[FitResult]:
    weights = _case_balanced_weights(fields)
    fits = _path_fit(
        *_design_bdelta(fields), channel="bdelta", max_terms=max_terms,
        sample_weight=(
            np.repeat(weights, 6)
            * np.tile(SYMM6_FROBENIUS_WEIGHTS, len(weights))
        ),
    )
    _rescore_deployed(fields, fits)
    return fits


def fit_r(fields, max_terms: int = 8) -> list[FitResult]:
    fits = _path_fit(
        *_design_r(fields), channel="rsource", max_terms=max_terms,
        sample_weight=_case_balanced_weights(fields),
        prediction_clip=R_BOUND,
    )
    _rescore_deployed(fields, fits)
    return fits


def _sym6_to_33(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.array(
        [[values[:, 0], values[:, 1], values[:, 2]],
         [values[:, 1], values[:, 3], values[:, 4]],
         [values[:, 2], values[:, 4], values[:, 5]]]
    ).transpose(2, 0, 1)


def _states_from_fields(fields) -> cand.States:
    tensors = np.stack(
        [_sym6_to_33(fields[f"T{i}"]) for i in range(1, 11)], axis=1
    )
    invariants = np.stack(
        [fields[f"I{i}"] for i in range(1, 6)], axis=1
    )
    return cand.States(
        shat=tensors[:, 0],
        what=np.zeros_like(tensors[:, 0]),
        T=tensors,
        inv=invariants,
        ret=np.asarray(fields["Ret"]),
        f1=np.asarray(fields["F1out"]),
        poe=np.asarray(fields["PoE"]),
    )


def _rescore_deployed(fields, fits: list[FitResult]) -> None:
    """Replace linear-design RMS with exact fixed-clamp evaluator RMS."""
    if not fits:
        return
    states = _states_from_fields(fields)
    weights = _case_balanced_weights(fields)
    weights = weights / max(float(np.sum(weights)), 1.0e-300)
    b_target = _sym6_to_33(fields["bDelta"])
    r_target = np.clip(
        fields["RField"]
        / np.maximum(fields["k"] * fields["omega"], 1.0e-30),
        -R_BOUND,
        R_BOUND,
    )
    b_scale = max(
        float(np.sqrt(np.sum(weights[:, None, None] * b_target**2))),
        1.0e-12,
    )
    r_scale = max(float(np.sqrt(np.sum(weights * r_target**2))), 1.0e-12)
    for fit in fits:
        spec = CandidateSpec(
            name="regression_rescore",
            bdelta=(
                tuple(Term(t, e) for t, e in fit.terms)
                if fit.channel == "bdelta" else ()
            ),
            rsource=(
                tuple(Term(t, e) for t, e in fit.terms)
                if fit.channel == "rsource" else ()
            ),
            gmax=10.0,
            r_max_factor=5.0,
        )
        if fit.channel == "bdelta":
            prediction = cand.eval_bdelta(spec, states)
            fit.apriori_rms = float(np.sqrt(np.sum(
                weights[:, None, None] * (prediction - b_target) ** 2
            ))) / b_scale
        else:
            prediction = cand.eval_rhat(spec, states)
            fit.apriori_rms = float(
                np.sqrt(np.sum(weights * (prediction - r_target) ** 2))
            ) / r_scale


def _case_balanced_weights(fields) -> np.ndarray:
    """Equal total regression weight per flow case, if labels are present."""
    if "case_id" not in fields:
        return np.ones(len(fields["I1"]), dtype=np.float64)
    ids = np.asarray(fields["case_id"])
    unique = np.unique(ids)
    weights = np.zeros(len(ids), dtype=np.float64)
    for case in unique:
        mask = ids == case
        weights[mask] = 1.0 / (len(unique) * int(mask.sum()))
    return weights


def pareto_front(fits: list[FitResult]) -> list[FitResult]:
    """Non-dominated in (n_terms, apriori_rms), sorted by complexity."""
    front: list[FitResult] = []
    for f in sorted(fits, key=lambda f: (f.n_terms, f.apriori_rms)):
        if all(f.apriori_rms < g.apriori_rms for g in front):
            front.append(f)
    return front
