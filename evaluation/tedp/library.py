"""Known-model library: comparators and the Tier-0 novelty filter.

Two kinds of entries:

  * PUBLISHED models — exact constitutive relations from the literature,
    expressed in our normalization (Shat = S/omega, What = W/omega,
    half-convention, invariants I1 = tr(S^2), I2 = tr(W^2) <= 0). Every
    coefficient below was verified against a fetched source (see
    results/library_sources.md). Since these run inside the SST transport
    harness, their b^Delta is (model's total b) minus (SST's Boussinesq b),
    with nut*omega/k = a1/max(a1, sqrt(2 I1)) — the F2=1 form of the SST
    limiter (F2 is not a state variable; exact in boundary layers).
    Conversion from k-epsilon-family normalizations: their tau = k/eps =
    1/(beta* omega), so their s = Shat/beta*, invariants scale by 1/beta*^2.

  * STRUCTURAL nets — free-coefficient families fitted to a candidate by
    (linear or nonlinear) least squares. The poly nets are LINEAR in their
    coefficients and fitted exactly by lstsq; they catch any rediscovery in
    the polynomial x T1..T4 family (SpaRTA/GEP territory) regardless of the
    particular constants.

A candidate is "equivalent to X" (hard Tier-0 reject) if X reproduces it to
< NOVELTY_RMS_REJECT relative RMS on both channels over every state set.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from . import candidate as cand
from . import expr as expr_mod
from .spec import CandidateSpec, Term

NOVELTY_RMS_REJECT = 0.03
RMS_FLOOR = 1.0e-8
NEGLIGIBLE_RMS = 1.0e-4
MULTISTARTS = 5

BSTAR = 0.09
B2 = BSTAR * BSTAR  # 0.0081

# SST limiter factor nut*omega/k at F2=1, as a grammar snippet
NUT_HAT = "(0.31/max(0.31,sqrt(2.0*I1)))"


# --------------------------------------------------------------------------
# entry definition


@dataclass(frozen=True)
class LibraryModel:
    name: str
    kind: str                       # "published" | "structural"
    structure: CandidateSpec        # exprs may use c0..c31 when fitting
    n_free: int
    init: tuple[float, ...]
    bounds: tuple[float, float] = (-400.0, 400.0)
    note: str = ""
    # This describes the stored expression, not a claim of full paper fidelity.
    # CandidateSpec stays unchanged so historical model hashes remain stable.
    tensor_convention: Literal["paper", "solver"] = "paper"


def _spec(name, bdelta, rsource, constants=()) -> CandidateSpec:
    return CandidateSpec(
        name=f"lib_{name}",
        bdelta=tuple(Term(t, e) for t, e in bdelta),
        rsource=tuple(Term(t, e) for t, e in rsource),
        constants=tuple(constants),
        gmax=1.0e9,          # comparators are not candidates: clamp OFF
        r_max_factor=20.0,
    )


# --------------------------------------------------------------------------
# published models (coefficients verified — see results/library_sources.md)

def _published() -> list[LibraryModel]:
    out: list[LibraryModel] = []

    # ---- stock SST: the empty candidate
    out.append(LibraryModel("sst", "published", _spec("sst", (), ()), 0, ()))

    # ---- QCR2000 (Spalart 2000; TMR). tau_QCR = tau - Ccr1[O tau + (O tau)^T],
    # O = 2W/|gradU|, Ccr1 = 0.3. With tau's Boussinesq part only (the 2/3 k I
    # part cancels), in our variables:
    #   b^Delta = -2*Ccr1 * (nut*omega/k) * T2 / sqrt(I1 + |I2|)
    out.append(LibraryModel(
        "qcr2000", "published",
        _spec("qcr2000",
              (("T2", f"(0.0-c0)*{NUT_HAT}/sqrt(max(I1+abs(I2),1e-12))"),),
              (), constants=(0.6,)),
        1, (0.6,),          # c0 = 2*Ccr1 = 0.6
        note="fitted amplitude; published c0=0.6",
    ))

    # ---- RITA: zonally-augmented k-omega SST (Buchanan, Lacatus, West &
    # Dwight 2025, arXiv:2504.06758). A binary shear-layer classifier
    #   sigma_SL = 1 iff phi_Dk/Pk < 0.55 and phi_k >= 0.12 and Re_Omega >= 0.02
    # switches on SpaRTA-derived corrections (their Eq. 23-24):
    #   b^Delta = 3.69 phi_Dk/Pk T2 - 5.092 [x/(1+x^2)] T2,  x = I2/0.01247
    #   R       = 0.426 [y/(1+y^2)] eps,                     y = phi_Dk/Ck/0.1248
    # Their tensors use Pope's convention with tau = 1/omega, matching our
    # normalisation, so I1 and I2 transfer unchanged; T2 is odd in the rotation
    # tensor, so its coefficient is negated here (see to_solver_convention).
    # R is carried on T1: with b_R = g*Shat, R = 2k g I1 omega, so g =
    # betaStar*C/(2 I1) = 0.045 C / I1 reproduces R = C*eps exactly.
    out.append(LibraryModel(
        "rita2025", "published",
        _spec("rita2025",
              (("T2", "0.0-(sigmaSL*(3.69*phiDkPk-5.092*(I2/0.01247)"
                      "/(1.0+(I2/0.01247)*(I2/0.01247))))"),),
              (("T1", "sigmaSL*0.426*(phiDkCk/0.1248)"
                      "/(1.0+(phiDkCk/0.1248)*(phiDkCk/0.1248))*0.045"
                      "/max(I1,0.000001)"),)),
        0, (),
        note="zonal SpaRTA corrections gated by the RITA shear-layer classifier",
        tensor_convention="solver",  # legacy entry already negates its T2 term
    ))

    # ---- Wallin & Johansson 2000 EARSM, 2-D closed form (JFM 403; verified
    # via Baungaard/Wallin WES preprint + Hellsten thesis). Their tau = 1/(b*w):
    # s = Shat/b*, II_S = I1/b*^2, II_Omega = I2/b*^2 (<= 0).
    # b_total = (beta1/2) s + (beta4/2)(s w - w s)  [b = a/2]
    #         = (beta1/(2 b*)) T1 + (beta4/(2 b*^2)) T2
    # beta1 = -(6/5) N / (N^2 - 2 II_Omega), beta4 = -(6/5)/(N^2 - 2 II_Omega)
    # N via the self-consistency relation N = c1' + (9/4) P/eps (c1' = 1.8),
    # P/eps == our PoE state variable (exact at convergence).
    wj_n = "(c0+2.25*PoE)"
    wj_den = f"({wj_n}*{wj_n}-2.0*I2/{B2})"
    out.append(LibraryModel(
        "wj2000", "published",
        _spec("wj2000",
              (
                  ("T1", f"(0.0-1.2)*{wj_n}/{wj_den}/(2.0*{BSTAR})+{NUT_HAT}"),
                  ("T2", f"(0.0-1.2)/{wj_den}/(2.0*{B2})"),
              ),
              (), constants=(1.8,)),
        1, (1.8,),          # c0 = c1' = 1.8
        note="2-D form; N from PoE self-consistency; Hellsten 2005 differs in "
             "c1'(state) + omega-equation constants (transport-side, not "
             "constitutive)",
    ))

    # ---- Shih, Zhu & Lumley 1995 quadratic (NASA TM-106644, verified):
    # b = -(Cmu/b*) T1 - (C2/b*^2) T2
    # Cmu = 1/(6.5 + As* U~), U~ = sqrt(I1+|I2|)/b*
    # As* = sqrt(6) cos(phi), phi = acos(sqrt6 W*)/3, W* = I3/I1^1.5
    # C2 = sqrt(1 - 9 Cmu^2 (sqrt(I1)/b*)^2) / (1 + 6 sqrt(I1)sqrt(|I2|)/b*^2)
    # acos is outside the grammar: As*(W*) enters the runnable spec through a
    # cubic minimax fit (max err < 1% over the attainable W* range); the
    # native evaluator (novelty path) uses the exact formula.
    # fit: As* ~ 1.8341 + 0.9153 x - 0.0147 x^2 - 0.2846 x^3, x = sqrt(6) W*
    ustar = f"(sqrt(max(I1+abs(I2),1e-12))/{BSTAR})"
    wstar6 = "(2.449489743*I3/max(sqrt(I1)*I1,1e-12))"
    astar = f"(1.8341+0.9153*{wstar6}-0.0147*{wstar6}*{wstar6}-0.2846*{wstar6}*{wstar6}*{wstar6})"
    cmu_szl = f"(1.0/(6.5+{astar}*{ustar}))"
    c2_szl = (
        f"(sqrt(max(1.0-9.0*{cmu_szl}*{cmu_szl}*I1/{B2},0.0))"
        f"/(1.0+6.0*sqrt(max(I1,0.0))*sqrt(abs(I2))/{B2}))"
    )
    out.append(LibraryModel(
        "szl1995", "published",
        _spec("szl1995",
              (
                  ("T1", f"(0.0-{cmu_szl}/{BSTAR})+{NUT_HAT}"),
                  ("T2", f"(0.0-{c2_szl}/{B2})"),
              ),
              ()),
        0, (),
        note="As* via cubic fit of sqrt6*cos(acos(sqrt6 W*)/3); A0=6.5, C0=1",
    ))

    # ---- Craft, Launder & Suga 1996 cubic (via Craft's notes + Apsley,
    # half-convention coefficients): a = -2 Cmu s + Cmu[-0.4 dev(s^2)
    # + 0.4(ws-sw) - 1.04 dev(w^2)] - Cmu^3[40({s^2}) + 40({w^2})]s
    # - Cmu^3 (-80)(w s^2 - s^2 w);  b = a/2, s = Shat/b*.
    # (ws - sw) = -T2/b*^2; (w s^2 - s^2 w) = T5/b*^3; {s^2} = I1/b*^2.
    # Cmu = 0.3(1 - exp(-0.36 exp(0.75 eta)))/(1 + 0.35 eta^1.5),
    # eta = max(sqrt(2 I1), sqrt(2 |I2|))/b*.  High-Re (f_mu = 1).
    eta = f"(max(sqrt(2.0*I1),sqrt(2.0*abs(I2)))/{BSTAR})"
    cmu_cls = (
        f"(0.3*(1.0-exp(0.0-0.36*exp(0.75*{eta})))"
        f"/(1.0+0.35*{eta}*sqrt({eta})))"
    )
    cls3 = f"({cmu_cls}*{cmu_cls}*{cmu_cls})"
    out.append(LibraryModel(
        "cls1996", "published",
        _spec("cls1996",
              (
                  ("T1", f"(0.0-{cmu_cls}/{BSTAR})+{NUT_HAT}"
                         f"-{cls3}*20.0*(I1+I2)/({B2}*{B2}*{BSTAR})"),
                  ("T2", f"(0.0-0.2)*{cmu_cls}/{B2}"),
                  ("T3", f"(0.0-0.2)*{cmu_cls}/{B2}"),
                  ("T4", f"(0.0-0.52)*{cmu_cls}/{B2}"),
                  ("T5", f"40.0*{cls3}/({B2}*{BSTAR}*{B2})"),
              ),
              ()),
        0, (),
        note="high-Re form (f_mu=1, no Yap); Apsley half-convention: "
             "(b1,b2,b3)=Cmu(-0.4,0.4,-1.04), (g1,g2,g4)=Cmu^3(40,40,-80)",
    ))

    # ---- Gatski & Speziale 1993 regularized EARSM (ICASE 92-58, verified):
    # b = alpha1 * -[3(1+eta^2)/D] [S* + (S*W*-W*S*) - 2 dev(S*^2)]
    # S* = 0.5 g (2-C3) s = 0.9709 Shat, W* = 0.5 g (2-C4) w = 2.0711 What
    # (g=0.233, C1=6.8, C2=0.36, C3=1.25, C4=0.40, alpha1=1.2978)
    # eta^2 = 0.9426 I1, zeta^2 = 4.289 |I2|, D = 3 + eta^2 + 6 zeta^2 eta^2
    # + 6 zeta^2
    gs_e2 = "(0.9426*I1)"
    gs_z2 = "(4.289*abs(I2))"
    gs_pre = (
        f"(0.0-1.2978*3.0*(1.0+{gs_e2})"
        f"/(3.0+{gs_e2}+6.0*{gs_z2}*{gs_e2}+6.0*{gs_z2}))"
    )
    out.append(LibraryModel(
        "gs1993", "published",
        _spec("gs1993",
              (
                  ("T1", f"{gs_pre}*0.9709+{NUT_HAT}"),
                  ("T2", f"{gs_pre}*2.0108"),
                  ("T3", f"{gs_pre}*(0.0-1.8852)"),
              ),
              ()),
        0, (),
        note="SSG-based constants; ICASE 92-58 Eq. 60/70",
    ))

    # ---- Lien, Chen & Leschziner 1996 (via Apsley + OpenFOAM LienCubicKE):
    # Cmu = (2/3)/(1.25 + sb + 0.9 wb), sb = sqrt(2 I1)/b*, wb = sqrt(2|I2|)/b*
    # quadratic: (3, 15, -19)/(1000 + sb^3) on (dev s^2, ws-sw, dev w^2)
    # cubic: Cmu^3 (16{s^2}+16{w^2}) s, -80 -> (w s^2 - s^2 w). b = a/2.
    sb = f"(sqrt(2.0*I1)/{BSTAR})"
    wb = f"(sqrt(2.0*abs(I2))/{BSTAR})"
    cmu_lcl = f"(0.6666667/(1.25+{sb}+0.9*{wb}))"
    lcl_q = f"(1.0/(1000.0+{sb}*{sb}*{sb}))"
    lcl3 = f"({cmu_lcl}*{cmu_lcl}*{cmu_lcl})"
    out.append(LibraryModel(
        "lcl1996", "published",
        _spec("lcl1996",
              (
                  ("T1", f"(0.0-{cmu_lcl}/{BSTAR})+{NUT_HAT}"
                         f"-{lcl3}*8.0*(I1+I2)/({B2}*{B2}*{BSTAR})"),
                  ("T2", f"(0.0-7.5)*{lcl_q}/{B2}"),
                  ("T3", f"1.5*{lcl_q}/{B2}"),
                  ("T4", f"(0.0-9.5)*{lcl_q}/{B2}"),
                  ("T5", f"40.0*{lcl3}/({B2}*{BSTAR}*{B2})"),
              ),
              ()),
        0, (),
        note="high-Re form (f_mu=1); Apsley: (b1,b2,b3)=(3,15,-19)/(1000+sb^3),"
             " (g1,g2,g4)=Cmu^3(16,16,-80)",
    ))

    # ---- SpaRTA M(1)..M(3) (Schmelzer et al. 2020, arXiv:1905.07510v2,
    # verified digit-for-digit). Same normalization as ours (tau = 1/omega).
    out.append(LibraryModel(
        "sparta_m1", "published",
        _spec("sparta_m1",
              (
                  ("T1", "24.94*I1*I1+2.65*I2"),
                  ("T2", "2.96"),
                  ("T3", "2.49*I2+20.05"),
                  ("T4", "2.49*I1+14.93"),
              ),
              (("T1", "0.4"),)),
        0, (),
    ))
    out.append(LibraryModel(
        "sparta_m2", "published",
        _spec("sparta_m2",
              (
                  ("T1", "0.46*I1*I1+11.68*I2-0.30*I2*I2+0.37"),
                  ("T2", "0.0-12.25*I1-0.63*I2*I2+8.23"),
                  ("T3", "0.0-1.36*I2-2.44"),
                  ("T4", "0.0-1.36*I1+0.41*I1*I1*I1-6.52"),
              ),
              (("T1", "1.4"),)),
        0, (),
    ))
    out.append(LibraryModel(
        "sparta_m3", "published",
        _spec("sparta_m3",
              (
                  ("T1", "0.11*I1*I2+0.27*I1*I2*I2-0.13*I1*I2*I2*I2"
                         "+0.07*I1*I2*I2*I2*I2+17.48*I1+0.01*I1*I1*I2"
                         "+1.251*I1*I1*I1+3.67*I2+7.52*I2*I2-0.3"),
                  ("T2", "0.17*I1*I2*I2-0.16*I1*I2*I2*I2-36.25*I1"
                         "-2.39*I1*I1+19.22*I2+7.04"),
                  ("T3", "0.0-0.22*I1*I1+1.8*I2+0.07*I2*I2+2.65"),
                  ("T4", "0.2*I1*I1-5.23*I2-2.93"),
              ),
              (("T1", "0.93"),)),
        0, (),
    ))

    # ---- GEP (Zhao et al. 2020, arXiv:1902.09075 — verified; the exact
    # Weatheritt & Sandberg 2016 JCP string is paywalled, these are the same
    # method/basis from the same group). Same tau = 1/omega normalization.
    out.append(LibraryModel(
        "gep_zhao_cfd", "published",
        _spec("gep_zhao_cfd",
              (
                  ("T1", "0.0-2.57+I1"),
                  ("T2", "4.0"),
                  ("T3", "0.0-0.11+0.09*I1*I2+I1*I2*I2"),
              ),
              ()),
        0, (),
    ))
    out.append(LibraryModel(
        "gep_zhao_frozen", "published",
        _spec("gep_zhao_frozen",
              (
                  ("T1", "0.0-1.334+0.438*I1+2.653*I2+0.0102*I1*I1"
                         "-1.021*I2*I2+12.280*I1*I2"),
                  ("T2", "0.573-1.096*I1+8.985*I2-0.1102*I1*I1"
                         "+2.876*I2*I2+90.633*I1*I2"),
                  ("T3", "12.861-25.094*I1+6.449*I2+1.020*I1*I1"
                         "-304.979*I1*I2-184.519*I2*I2"),
              ),
              ()),
        0, (),
    ))

    return out


# --------------------------------------------------------------------------
# structural nets

POLY_FEATURES = ("1", "I1", "I2", "I1*I1", "I1*I2", "I2*I2")
POLY_TENSORS = ("T1", "T2", "T3", "T4")


def _structural() -> list[LibraryModel]:
    out = []
    out.append(LibraryModel(
        "linear_retune", "structural",
        _spec("linear_retune", (("T1", "c0"),), (), constants=(0.0,)), 1, (0.0,),
        tensor_convention="solver",
    ))
    # poly4 net: linear in its coefficients -> exact lstsq fit (see novelty);
    # catches the whole SpaRTA/GEP polynomial family with any constants
    bterms = []
    ci = 0
    for tn in POLY_TENSORS:
        for f in POLY_FEATURES:
            bterms.append((tn, f"c{ci}*{f}" if f != "1" else f"c{ci}"))
            ci += 1
    rterms = [("T1", f"c{ci}"), ("T2", f"c{ci+1}"), ("T3", f"c{ci+2}")]
    out.append(LibraryModel(
        "poly4_net", "structural",
        _spec("poly4_net", tuple(bterms), tuple(rterms),
              constants=tuple([0.0] * (ci + 3))),
        ci + 3, tuple([0.0] * (ci + 3)),
        note="fitted by exact linear lstsq",
        tensor_convention="solver",
    ))
    return out


LIBRARY: tuple[LibraryModel, ...] = tuple(_published() + _structural())


def published_models() -> list[LibraryModel]:
    return [m for m in LIBRARY if m.kind == "published"]


# --------------------------------------------------------------------------
# novelty machinery


@dataclass
class NoveltyVerdict:
    equivalent_to: str = ""
    best_rms: float = float("inf")
    per_model: dict[str, float] = field(default_factory=dict)
    #: library models that could not be compared here, and why. A comparator
    #: expressed in solver-only variables cannot be evaluated on one-point
    #: states, so it is skipped -- recorded rather than silently dropped, so a
    #: novelty verdict always says what it was NOT checked against.
    skipped: dict[str, str] = field(default_factory=dict)


def solver_only_variables(spec: CandidateSpec) -> set[str]:
    """Variables in *spec* that exist only inside the solver.

    The RITA classifier and its k-budget ratios are built from wall distance,
    k, omega, U and grad(k). None of those is a function of the one-point state
    the a-priori path evaluates on, so a spec naming one can be run in the
    solver but not screened or compared here.
    """
    return spec.used_variables() & set(expr_mod.RITA_VARIABLES)


def _channel_targets(spec: CandidateSpec, states: cand.States):
    # Novelty is about the algebraic model, not whether two different raw
    # forms happen to collapse to the same rMax safety plateau.
    return cand.eval_bdelta(spec, states), cand.eval_rhat_raw(spec, states)


def _rel_rms(diff: np.ndarray, target: np.ndarray) -> float:
    rms_d = float(np.sqrt(np.mean(diff**2)))
    rms_t = float(np.sqrt(np.mean(target**2)))
    # A numerically irrelevant channel must not let an otherwise exact clone
    # evade the novelty gate.  Conversely, a comparator that introduces a
    # material channel where the candidate has none still receives a large
    # error through the NEGLIGIBLE_RMS denominator.
    if rms_d < NEGLIGIBLE_RMS and rms_t < NEGLIGIBLE_RMS:
        return 0.0
    return rms_d / max(rms_t, NEGLIGIBLE_RMS, RMS_FLOOR)


def _worst_rel_rms(model, constants, spec, state_sets, targets) -> float:
    fitted = CandidateSpec(
        name="fit", bdelta=model.structure.bdelta,
        rsource=model.structure.rsource, constants=tuple(constants),
        gmax=model.structure.gmax, r_max_factor=spec.r_max_factor,
    )
    worst = 0.0
    for ss, (b_t, r_t) in zip(state_sets, targets):
        b_f = cand.eval_bdelta(fitted, ss)
        r_f = cand.eval_rhat_raw(fitted, ss)
        worst = max(worst, _rel_rms(b_f - b_t, b_t))
        worst = max(worst, _rel_rms(r_f - r_t, r_t))
    return worst


def _fit_poly_net(spec, state_sets, targets) -> float:
    """One exact linear least-squares fit shared by every state set."""
    b_systems = []
    r_systems = []
    for ss, (b_t, r_t) in zip(state_sets, targets):
        variables = ss.variables()
        i1, i2 = variables["I1"], variables["I2"]
        feats = np.stack(
            [np.ones_like(i1), i1, i2, i1 * i1, i1 * i2, i2 * i2], axis=1
        )
        cols = []
        for ti, _tn in enumerate(POLY_TENSORS):
            Tn = ss.T[:, ti]  # T1..T4 are indices 0..3
            for fi in range(feats.shape[1]):
                cols.append((feats[:, fi][:, None, None] * Tn).reshape(len(ss), -1))
        A = np.stack([c.ravel() for c in cols], axis=1)
        y = b_t.reshape(len(ss), -1).ravel()
        # R channel: rhat = 2 * sum c_n T_n : Shat
        rcols = []
        for ti in range(3):
            contraction = 2.0 * np.einsum(
                "nij,nij->n", ss.T[:, ti], ss.shat
            )
            for fi in range(feats.shape[1]):
                rcols.append(contraction * feats[:, fi])
        Ar = np.stack(rcols, axis=1)
        b_systems.append((A, y))
        r_systems.append((Ar, r_t))

    coef, *_ = np.linalg.lstsq(
        np.concatenate([a for a, _ in b_systems], axis=0),
        np.concatenate([y for _, y in b_systems], axis=0),
        rcond=None,
    )
    coef_r, *_ = np.linalg.lstsq(
        np.concatenate([a for a, _ in r_systems], axis=0),
        np.concatenate([y for _, y in r_systems], axis=0),
        rcond=None,
    )
    worst = 0.0
    for (A, y), (Ar, r_t) in zip(b_systems, r_systems):
        worst = max(worst, _rel_rms(A @ coef - y, y))
        worst = max(worst, _rel_rms(Ar @ coef_r - r_t, r_t))
    return worst


def fit_library_model(model, spec, state_sets, seed=0) -> float:
    targets = [_channel_targets(spec, ss) for ss in state_sets]

    if model.name == "poly4_net":
        return _fit_poly_net(spec, state_sets, targets)

    if model.n_free == 0:
        return _worst_rel_rms(model, model.init, spec, state_sets, targets)

    def residuals(x):
        fitted = CandidateSpec(
            name="fit", bdelta=model.structure.bdelta,
            rsource=model.structure.rsource,
            constants=tuple(float(v) for v in x),
            gmax=model.structure.gmax,
            r_max_factor=spec.r_max_factor,
        )
        parts = []
        for ss, (b_t, r_t) in zip(state_sets, targets):
            b_f = cand.eval_bdelta(fitted, ss)
            r_f = cand.eval_rhat_raw(fitted, ss)
            parts.append(
                (b_f - b_t).ravel() / max(np.sqrt(np.mean(b_t**2)), RMS_FLOOR)
            )
            parts.append(
                (r_f - r_t).ravel() / max(np.sqrt(np.mean(r_t**2)), RMS_FLOOR)
            )
        return np.concatenate(parts)

    rng = np.random.default_rng(seed)
    best = float("inf")
    starts = [np.asarray(model.init, dtype=float)]
    for _ in range(MULTISTARTS - 1):
        starts.append(rng.uniform(-2.0, 2.0, size=model.n_free))
    for x0 in starts:
        try:
            sol = least_squares(
                residuals, x0, bounds=model.bounds, method="trf", max_nfev=200
            )
            best = min(
                best,
                _worst_rel_rms(model, tuple(map(float, sol.x)), spec,
                               state_sets, targets),
            )
        except Exception:
            continue
        if best < NOVELTY_RMS_REJECT / 3.0:
            break
    return best


def novelty(
    spec: CandidateSpec,
    states: cand.States,
    dns_states: cand.States | list[cand.States] | tuple[cand.States, ...] | None = None,
    subsample: int = 4096,
    seed: int = 0,
) -> NoveltyVerdict:
    unevaluable = solver_only_variables(spec)
    if unevaluable:
        raise ValueError(
            "novelty() cannot judge a spec written in solver-only variables "
            f"({', '.join(sorted(unevaluable))}): they are not functions of a "
            "one-point state, so there is nothing to evaluate here. Such a model "
            "is comparable only through the solver."
        )
    rng = np.random.default_rng(seed)

    def sub(ss: cand.States) -> cand.States:
        if len(ss) <= subsample:
            return ss
        idx = rng.choice(len(ss), size=subsample, replace=False)
        return ss.take(idx)

    if dns_states is None:
        dns_sets = []
    elif isinstance(dns_states, cand.States):
        dns_sets = [dns_states]
    else:
        dns_sets = list(dns_states)
    state_sets = [sub(states), *(sub(ss) for ss in dns_sets)]

    verdict = NoveltyVerdict()

    # a correction of negligible magnitude IS stock SST, whatever its shape
    magnitudes = []
    for ss in state_sets:
        b0, r0 = _channel_targets(spec, ss)
        magnitudes.extend(
            [float(np.sqrt(np.mean(b0**2))), float(np.sqrt(np.mean(r0**2)))]
        )
    if max(magnitudes, default=0.0) < NEGLIGIBLE_RMS:
        verdict.equivalent_to = "sst"
        verdict.best_rms = 0.0
        verdict.per_model["sst"] = 0.0
        return verdict
    for model in LIBRARY:
        unevaluable = solver_only_variables(model.structure)
        if unevaluable:
            verdict.skipped[model.name] = (
                "solver-only variables: " + ", ".join(sorted(unevaluable))
            )
            continue
        rms = fit_library_model(model, spec, state_sets, seed=seed)
        verdict.per_model[model.name] = rms
        if rms < verdict.best_rms:
            verdict.best_rms = rms
        if rms < NOVELTY_RMS_REJECT:
            verdict.equivalent_to = model.name
            break
    return verdict


# ---------------------------------------------------------------------------
# Rotation-tensor convention
# ---------------------------------------------------------------------------
#: Basis tensors odd in the rotation tensor: T2 = SW - WS, T5, T7, T8, T10.
#: They change sign when W changes sign.
ODD_IN_W = ("T2", "T5", "T7", "T8", "T10")


def to_solver_convention(spec: CandidateSpec) -> CandidateSpec:
    """Re-express a published model for the solver's rotation-tensor sign.

    This is an explicit sign-change operation, not an idempotent admission
    function: CandidateSpec has no convention metadata. Use
    published_models_solver_convention() for the registry, whose RITA entry
    was already stored in solver convention before this adapter was added.

    The paper-convention models above were transcribed using Pope's
    convention W_ij = 1/2 (dU_i/dx_j - dU_j/dx_i). The solver builds its
    rotation tensor from ``skew(fvc::grad(U))`` and OpenFOAM's
    grad(U)_ij = dU_j/dx_i, so the solver's W is minus Pope's. Every basis
    tensor odd in W therefore enters with the opposite sign, and a published
    coefficient has to be negated to reproduce the published model.

    This was found after the holdout was opened, from the square duct: QCR
    2000 produced secondary flow anti-correlated with the DNS (-0.90), which
    is the reverse of the corner-directed pattern it is known for. It affects
    only the transcribed comparators. The discovered models were fitted and
    are evaluated through the same solver, so their constants are correct as
    they stand; expressing them in Pope's convention would negate their own
    odd-in-W coefficients.
    """
    def flip(terms):
        return tuple(
            Term(t.tensor, f"0.0-({t.expression})") if t.tensor in ODD_IN_W else t
            for t in terms
        )

    return replace(spec, bdelta=flip(spec.bdelta), rsource=flip(spec.rsource))


def published_models_solver_convention() -> list[LibraryModel]:
    """Return solver-convention entries without converting one twice.

    The legacy registry and its specs are preserved. In particular, RITA's
    stored T2 coefficient is already negated; flipping every entry would
    silently reverse its stress correction.
    """
    converted = []
    for model in published_models():
        if model.tensor_convention == "solver":
            converted.append(model)
        elif model.tensor_convention == "paper":
            converted.append(replace(
                model,
                structure=to_solver_convention(model.structure),
                tensor_convention="solver",
            ))
        else:
            raise ValueError(
                f"unknown tensor convention {model.tensor_convention!r} "
                f"for {model.name}"
            )
    return converted
