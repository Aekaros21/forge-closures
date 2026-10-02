"""Tests of the independent BSL-EARSM reference (reference.py).

Run (from the repository root):  python3 -m pytest -q closures/comparators/earsm_reference

Page and equation numbers refer to Menter, Garbaruk and Egorov (2012) [M12] and (2009) [M09]; see the
module docstring of reference.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference as ref  # noqa: E402

RNG = np.random.default_rng(20261001)
FULL_WJ = ref.Coefficients(include_t9=True)           # A1 = 1.245, beta_9 = 1/Q1
C1P = ref.BSL_EARSM.c1p


def random_s_w(n, scale=1.0, two_d=False, rng=RNG):
    """Random traceless S and antisymmetric W (nondimensional), magnitudes spread over decades."""
    g = rng.normal(size=(n, 3, 3))
    if two_d:
        g[:, 2, :] = 0.0
        g[:, :, 2] = 0.0
    mag = scale * 10.0 ** rng.uniform(-1.5, 1.0, size=(n, 1, 1))
    g = g * mag
    S = 0.5 * (g + np.swapaxes(g, 1, 2))
    if two_d:      # in-plane, divergence-free: S11 + S22 = 0, S33 = 0
        S -= np.trace(S, axis1=1, axis2=2)[:, None, None] * np.diag([1.0, 1.0, 0.0]) / 2.0
    else:
        S -= np.trace(S, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3.0
    W = 0.5 * (g - np.swapaxes(g, 1, 2))
    # random relative weight of strain and rotation
    w = 10.0 ** rng.uniform(-1.0, 1.0, size=(n, 1, 1))
    return S, w * W


def rel_err(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


# ---------------------------------------------------------------------------------------------
# Constants and algebra
# ---------------------------------------------------------------------------------------------

def test_lrr_with_c2_5_9_gives_the_printed_constants():
    """M12 p.91-92: WJ is LRR-based with A1 = 1.2, C1' = (9/4)(C1 - 1), 9/4 on P/eps (eq.5), no A2 term."""
    c = ref.lrr_algebraic_coefficients(c1=1.8, c2=5.0 / 9.0)
    assert c["a1"] == pytest.approx(1.2, abs=1e-14)
    assert c["c1_prime"] == pytest.approx(1.8, abs=1e-14)
    assert c["pk_factor"] == pytest.approx(2.25, abs=1e-14)
    assert c["a2"] == pytest.approx(0.0, abs=1e-14)
    assert ref.BSL_EARSM.c1p == pytest.approx(1.8)
    # the printed numbers of the N equation are (9/4) A1, (3/8) A1, (3/4) A1 with A1 = 1.2
    assert 2.25 * 1.2 == pytest.approx(2.7) and 2.25 * 1.2 / 6 == pytest.approx(9 / 20)
    assert 2.25 * 1.2 / 3 == pytest.approx(9 / 10)


def test_basis_is_symmetric_and_traceless():
    S, W = random_s_w(200)
    for i, T in ref.basis(S, W).items():
        assert np.abs(T - np.swapaxes(T, 1, 2)).max() <= 1e-12 * max(1, np.abs(T).max()), i
        assert np.abs(np.trace(T, axis1=1, axis2=2)).max() <= 1e-11 * max(1, np.abs(T).max()), i


def test_regrouped_t6_t9_vanish_and_beta3_zero_in_2d():
    """M12 p.92 below eq.(3): the regrouping separates 2D and 3D parts; IV = 0 in 2D so beta_3 = 0."""
    S, W = random_s_w(200, two_d=True)
    T = ref.basis(S, W)
    scale = np.abs(S).max(axis=(1, 2)) ** 3 + np.abs(W).max(axis=(1, 2)) ** 3
    assert (np.abs(T[6]).max(axis=(1, 2)) <= 1e-12 * scale).all()
    assert (np.abs(T[9]).max(axis=(1, 2)) <= 1e-12 * (scale ** (4 / 3))).all()
    st = ref.anisotropy(S, W)
    assert np.abs(st["betas"][3]).max() <= 1e-12


@pytest.mark.parametrize("n_source", ["cubic", "arbitrary"])
def test_explicit_betas_solve_the_implicit_relation_in_3d(n_source):
    """beta_1..beta_9 (M09 eq.7) with T (M12 eq.3) solve N a = -A1 S + (aW - Wa) exactly, any N."""
    S, W = random_s_w(300)
    if n_source == "cubic":
        n = None
    else:
        n = 10.0 ** RNG.uniform(-0.5, 1.5, size=300)
    st = ref.anisotropy(S, W, FULL_WJ, n=n)
    a_imp = ref.implicit_anisotropy(S, W, st["N"], FULL_WJ.a1)
    for i in range(300):
        assert rel_err(st["a"][i], a_imp[i]) < 1e-11, i


def test_m12_printed_beta4_is_a_misprint():
    """M12 p.92 prints beta_4 = -N/Q; it does not solve the implicit relation, M09's -1/Q does."""
    S, W = ref.simple_shear(3.0)
    st = ref.anisotropy(S, W, FULL_WJ)
    a_imp = ref.implicit_anisotropy(S, W, st["N"], FULL_WJ.a1)
    assert rel_err(st["a"], a_imp) < 1e-13
    b = st["betas"]
    a_misprint = st["a"] + (-st["N"] / b["Q"] - b[4]) * st["T"][4]
    assert rel_err(a_misprint, a_imp) > 0.5


def test_bsl_earsm_differs_from_full_wj_only_by_t9():
    """M12 p.93 / M09 p.5: beta_9 = 0 in the BSL-EARSM."""
    S, W = random_s_w(200)
    full = ref.anisotropy(S, W, FULL_WJ)
    bsl = ref.anisotropy(S, W, ref.BSL_EARSM)
    np.testing.assert_allclose(full["N"], bsl["N"], rtol=0, atol=0)
    diff = full["a"] - bsl["a"]
    expect = (1.0 / full["betas"]["Q1"])[:, None, None] * full["T"][9]
    assert np.abs(diff - expect).max() <= 1e-13 * max(1.0, np.abs(full["a"]).max())


# ---------------------------------------------------------------------------------------------
# N: root selection and continuity
# ---------------------------------------------------------------------------------------------

def invariant_grid():
    vals = np.concatenate([[0.0], np.logspace(-8, 4, 97)])
    iis, mw = np.meshgrid(vals, vals, indexing="ij")
    return iis.ravel(), -mw.ravel()


@pytest.mark.parametrize("cubic_a1", [1.2, 1.245])
def test_n_is_the_largest_real_root_over_a_sweep(cubic_a1):
    """Eq.(7) of M12 equals the largest real root of eq.(6), for P2 >= 0 and P2 < 0."""
    iis, iiw = invariant_grid()
    n, p1, p2 = ref.solve_n(iis, iiw, C1P, cubic_a1)
    assert np.isfinite(n).all()
    assert (p2 < 0).sum() > 100 and (p2 >= 0).sum() > 100     # both branches exercised
    assert (p1 > 0).all()                                      # P1 >= C1'^3/27 > 0
    roots = ref.largest_real_root(iis, iiw, C1P, cubic_a1)
    assert np.max(np.abs(n - roots) / np.maximum(1.0, roots)) < 1e-9
    # unique root with non-negative production: N >= C1'
    assert (n >= C1P * (1 - 1e-13)).all()
    c = ref.cubic_coefficients(iis, iiw, C1P, cubic_a1)
    f = n**3 + c[1] * n**2 + c[2] * n + c[3]
    fprime = 3 * n**2 + 2 * c[1] * n + c[2]
    assert np.max(np.abs(f / fprime) / n) < 1e-13           # Newton step relative to N
    # the printed form (cbrt(P1 - sqrt P2)) loses ~1e-11 where P2 ~ P1^2; bound it
    n_printed = ref.solve_n(iis, iiw, C1P, cubic_a1, stable=False)[0]
    assert np.max(np.abs(n_printed - n) / n) < 1e-10
    # Q and Q1 positive (no singularity)
    b = ref.betas(n, iis, iiw, np.zeros_like(n))
    assert (b["Q"] > 0).all() and (b["Q1"] > 0).all()


def p2_zero_crossings():
    """Points (II_S, II_W) on P2 = 0, found by bisection in II_W at fixed II_S."""
    pts = []
    for iis in np.logspace(-3, 3, 13):
        lo, hi = -10.0 * iis - 10.0, 0.0       # P2 > 0 at strong rotation, P2 < 0 at II_W = 0
        p = lambda w: ref.solve_n(iis, w)[2]    # noqa: E731
        if not (p(lo) > 0 > p(hi)):
            continue
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if p(mid) > 0:
                lo = mid
            else:
                hi = mid
        pts.append((iis, lo, hi))
    return pts


def test_n_is_continuous_across_p2_zero():
    pts = p2_zero_crossings()
    assert len(pts) >= 10
    for iis, lo, hi in pts:
        n_lo = ref.solve_n(iis, lo)[0]       # Cardano branch
        n_hi = ref.solve_n(iis, hi)[0]       # trigonometric branch
        assert abs(n_lo - n_hi) < 1e-7 * n_lo, (iis, n_lo, n_hi)


def test_n_is_continuous_along_paths_through_both_branches():
    """Along fine paths the step in N is bounded by |dN/dx| dx (implicit differentiation of eq.6)."""
    t = np.linspace(0.0, 1.0, 20001)
    paths = []
    for iis in (1e-2, 1.0, 5.0, 50.0, 500.0):
        paths.append((np.full_like(t, iis), -3.0 * iis * t))           # II_W from 0 to -3 II_S
    for lam in (0.0, 0.3, 1.0, 1.7, 3.0):
        x = 1e3 * t**2
        paths.append((x, -lam * x))                                    # fixed flow type, growing rate
    for iis, iiw in paths:
        n, _, p2 = ref.solve_n(iis, iiw)
        dn = np.abs(np.diff(n))
        # dN = -(f_IIS dIIS + f_IIW dIIW)/f_N with f of eq.(6)
        nm = 0.5 * (n[1:] + n[:-1])
        f_n = 3 * nm**2 - 2 * C1P * nm - (2.7 * 0.5 * (iis[1:] + iis[:-1]) + 2 * 0.5 * (iiw[1:] + iiw[:-1]))
        f_s = -2.7 * nm
        f_w = -2 * nm + 2 * C1P
        bound = (np.abs(f_s * np.diff(iis)) + np.abs(f_w * np.diff(iiw))) / np.abs(f_n)
        assert (dn <= 2.0 * bound + 1e-12).all()
        assert np.isfinite(n).all()


def test_n_closed_forms_at_the_edges():
    # pure strain (II_W = 0): N (N^2 - C1' N - 2.7 II_S) = 0, largest root
    iis = np.logspace(-6, 4, 50)
    n = ref.solve_n(iis, 0.0)[0]
    np.testing.assert_allclose(n, 0.5 * (C1P + np.sqrt(C1P**2 + 4 * 2.7 * iis)), rtol=1e-12)
    # pure rotation (II_S = 0): (N - C1')(N^2 - 2 II_W) = 0, only real root C1'
    n = ref.solve_n(0.0, -np.logspace(-6, 4, 50))[0]
    np.testing.assert_allclose(n, C1P, rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# Weak-strain limit: a linear eddy-viscosity model
# ---------------------------------------------------------------------------------------------

def test_weak_strain_limit_is_a_linear_eddy_viscosity_model():
    """As S, W -> 0: N -> C1', a -> -(A1/C1') S, nonlinear part O(eps) relative to the linear part."""
    S0, W0 = random_s_w(50)
    S0 /= np.linalg.norm(S0, axis=(1, 2), keepdims=True)              # |S| = 1
    W0 *= 3.0 / np.linalg.norm(W0, axis=(1, 2), keepdims=True)        # |W| = 3
    cmu_lim = ref.BSL_EARSM.a1 / (2 * C1P)          # 0.34583 (in units of k tau)
    prev_ratio = None
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        st = ref.anisotropy(eps * S0, eps * W0)
        assert np.abs(st["N"] - C1P).max() < 10 * eps**2
        assert np.abs(st["cmu_eff"] - cmu_lim).max() < 10 * eps**2
        lin = np.linalg.norm(st["a_linear"], axis=(1, 2))
        ex = np.linalg.norm(st["a_extra"], axis=(1, 2))
        ratio = (ex / lin).max()
        assert ratio < 5 * eps
        if prev_ratio is not None:
            assert ratio < 0.2 * prev_ratio            # first order in eps
        prev_ratio = ratio
        a_levm = -(ref.BSL_EARSM.a1 / C1P) * eps * S0
        assert rel_err(st["a"], a_levm) < 5 * eps


# ---------------------------------------------------------------------------------------------
# Homogeneous shear and log-layer equilibrium
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("coeffs", [ref.BSL_EARSM, ref.Coefficients(cubic_a1=1.245),
                                    ref.Coefficients(a1=1.2), FULL_WJ])
@pytest.mark.parametrize("R", [0.5, 1.0, ref.homogeneous_shear_pk_over_eps(1.0),
                               ref.homogeneous_shear_pk_over_eps(0.0), 3.0])
def test_shear_equilibrium_matches_the_closed_form(coeffs, R):
    cf = ref.shear_equilibrium_closed_form(R, coeffs)
    S, W = ref.simple_shear(cf["s_tau"])
    st = ref.anisotropy(S, W, coeffs)
    assert st["N"] == pytest.approx(cf["N"], rel=1e-12)
    assert st["pk_over_eps"] == pytest.approx(R, rel=1e-12)
    a = st["a"]
    assert a[0, 0] == pytest.approx(cf["a11"], rel=1e-12)
    assert a[1, 1] == pytest.approx(-cf["a11"], rel=1e-12)
    assert a[0, 1] == pytest.approx(cf["a12"], rel=1e-12)
    assert abs(a[2, 2]) < 1e-14 and abs(a[0, 2]) < 1e-14 and abs(a[1, 2]) < 1e-14
    assert st["cmu_eff"] == pytest.approx(cf["cmu_eff"], rel=1e-12)
    # streamwise normal stress largest, shear stress negative for dU/dy > 0 (sign of beta_4 and W)
    assert a[0, 0] > 0 > a[1, 1] and a[0, 1] < 0
    # a_11 = R/N in simple shear
    assert a[0, 0] == pytest.approx(R / st["N"], rel=1e-12)


def test_log_layer_equilibrium_reproduces_the_published_calibration():
    """P/eps = 1. M12 p.94-95: in equilibrium layers -uv = sqrt(Cmu) k = 0.3 k (Cmu = 0.09), and
    p.93: A1 = 1.245 matches the log layer without changing the BSL constants.

    Numbers (closed form, printed N equation): N = 3.968675, -a12 = 0.305553, Cmu_eff = 0.093363,
    a11 = -a22 = 0.251973, a33 = 0, kappa_eff = 0.4214. With WJ's A1 = 1.2: -a12 = 0.295266,
    Cmu_eff = 0.087182, kappa_eff = 0.4003. The recalibration raises -uv/k across 0.3 as stated.
    """
    bsl = ref.shear_equilibrium_closed_form(1.0, ref.BSL_EARSM)
    wj = ref.shear_equilibrium_closed_form(1.0, ref.Coefficients(a1=1.2))
    assert bsl["minus_uv_over_k"] == pytest.approx(0.3, rel=0.02)
    assert bsl["cmu_eff"] == pytest.approx(0.09, rel=0.04)
    assert wj["minus_uv_over_k"] < 0.3 < bsl["minus_uv_over_k"]
    assert ref.log_layer_kappa(bsl["minus_uv_over_k"]) == pytest.approx(0.41, rel=0.03)
    assert ref.log_layer_kappa(0.3) == pytest.approx(0.41, rel=1e-14)
    np.testing.assert_allclose(
        [bsl["N"], bsl["minus_uv_over_k"], bsl["cmu_eff"], bsl["a11"]],
        [3.968675, 0.305553, 0.093363, 0.251973], atol=2e-6)


def test_homogeneous_shear_equilibrium_of_the_bsl_omega_equation():
    """Structural equilibrium P/eps = beta/(gamma beta*) (constant S tau, M12 p.93 omega equation)."""
    assert ref.gamma(1.0) == pytest.approx(0.553167, abs=1e-6)
    assert ref.gamma(0.0) == pytest.approx(0.440355, abs=1e-6)
    R = ref.homogeneous_shear_pk_over_eps(0.0)           # free shear, F1 = 0
    assert R == pytest.approx(2.08923, abs=1e-5)
    cf = ref.shear_equilibrium_closed_form(R)
    # recorded state (anisotropy and effective Cmu); compare published_targets.json
    np.testing.assert_allclose(
        [cf["N"], cf["s_tau"], cf["a11"], cf["a12"], cf["cmu_eff"]],
        [6.330850, 6.724592, 0.330007, -0.310684, 0.046201], atol=2e-6)


def test_production_consistency_in_2d():
    """In 2D, -a_ij S_ij = (A1/cubic_a1)(4/9)(N - C1') exactly (M12 eqs.5-6)."""
    S, W = random_s_w(300, two_d=True)
    for coeffs in (ref.BSL_EARSM, ref.Coefficients(cubic_a1=1.245)):
        st = ref.anisotropy(S, W, coeffs)
        expect = coeffs.a1 / coeffs.cubic_a1 * st["pk_over_eps_N"]
        np.testing.assert_allclose(st["pk_over_eps"], expect, rtol=1e-11, atol=1e-14)


# ---------------------------------------------------------------------------------------------
# Kinematics, OpenFOAM split, frame rotation, transport constants
# ---------------------------------------------------------------------------------------------

def test_openfoam_gradient_convention_and_sign_of_w():
    """U = (G y, 0, 0): OpenFOAM g_ij = dU_j/dx_i has g_yx = G; W_xy = +tau G/2 (M12 p.92)."""
    G, tau = 2.0, 0.7
    g = np.zeros((3, 3))
    g[1, 0] = G
    S, W = ref.strain_rotation(g, tau)
    assert S[0, 1] == pytest.approx(0.5 * tau * G) and W[0, 1] == pytest.approx(0.5 * tau * G)
    assert W[1, 0] == pytest.approx(-0.5 * tau * G)


def test_time_scale_and_limiter():
    k, om, nu = 1.0, 10.0, 1e-5
    assert ref.time_scale(k, om, nu) == pytest.approx(1 / 0.9)
    k, om, nu = 1e-6, 1e4, 1e-5          # near-wall: Kolmogorov limiter active
    t = ref.time_scale(k, om, nu)
    assert t == pytest.approx(6 * np.sqrt(nu / (0.09 * k * om)))
    assert t > 1 / (0.09 * om)


def test_earsm_state_openfoam_split():
    """R = (2/3) k I - nut twoSymm(gradU) + nonlinearStress equals k (a + 2/3 I)."""
    g = RNG.normal(size=(100, 3, 3)) * 10.0 ** RNG.uniform(-2, 2, size=(100, 1, 1))
    g -= np.trace(g, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3.0
    k = 10.0 ** RNG.uniform(-4, 0, size=100)
    om = 10.0 ** RNG.uniform(0, 4, size=100)
    st = ref.earsm_state(g, k, om, 1e-5)
    two_symm = g + np.swapaxes(g, 1, 2)
    R = 2.0 / 3.0 * k[:, None, None] * np.eye(3) - st["nut"][:, None, None] * two_symm \
        + st["nonlinear_stress"]
    np.testing.assert_allclose(R, st["reynolds_stress"], rtol=1e-11,
                               atol=1e-12 * np.abs(st["reynolds_stress"]).max())
    assert np.abs(np.trace(st["nonlinear_stress"], axis1=1, axis2=2)).max() < 1e-12 * k.max()
    assert (st["nut"] > 0).all()


def test_lrr_frame_rotation_derivation():
    """Explicit a with W* = W - (13/4) F solves the LRR(C2 = 5/9) relation with Coriolis terms."""
    assert ref.lrr_frame_factor() == pytest.approx(13.0 / 4.0)
    S, W = random_s_w(50)
    F = 0.5 * ref.frame_tensor(RNG.normal(size=(50, 3)))
    st = ref.anisotropy(S, W - 3.25 * F, FULL_WJ)
    eye3 = np.eye(3)
    cr = 5.0 / 9.0
    for i in range(50):
        w, f, wabs = W[i], F[i], W[i] - F[i]
        # (9/4)[(aW - Wa) - 2(aF - Fa) - cr(aWabs - Wabs a)]
        def comm(m):
            return np.kron(eye3, m.T) - np.kron(m, eye3)    # vec(a m - m a)
        A = st["N"][i] * np.eye(9) - 2.25 * (comm(w) - 2.0 * comm(f) - cr * comm(wabs))
        a = np.linalg.solve(A, -FULL_WJ.a1 * S[i].reshape(9)).reshape(3, 3)
        assert rel_err(st["a"][i], a) < 1e-11


def test_frame_modes_in_strain_rotation():
    g = RNG.normal(size=(3, 3))
    Om = np.array([0.0, 0.0, 0.5])
    _, Wr = ref.strain_rotation(g, 2.0, Om, "relative")
    _, Wa = ref.strain_rotation(g, 2.0, Om, "absolute")
    _, Wl = ref.strain_rotation(g, 2.0, Om, "lrr")
    F = 2.0 * ref.frame_tensor(Om)
    np.testing.assert_allclose(Wa, Wr - F, atol=1e-15)
    np.testing.assert_allclose(Wl, Wr - 3.25 * F, atol=1e-15)
    # absolute vorticity: rotating frame Omega_z with fluid at rest in the inertial frame
    # (relative velocity -Omega x x) has W_abs = 0
    gs = np.zeros((3, 3))
    gs[0, 1], gs[1, 0] = -0.5, 0.5          # u = (0.5 y, -0.5 x, 0): g_xy = dUy/dx = -0.5
    _, Wa = ref.strain_rotation(gs, 1.0, Om, "absolute")
    np.testing.assert_allclose(Wa, 0.0, atol=1e-15)


def test_f1_and_sources():
    assert ref.f1_blending(1e-4, 1e6, 1e-5, 1e-5, 1.0) == pytest.approx(1.0)    # near wall
    assert ref.f1_blending(1e-2, 1.0, 10.0, 1e-5, 1.0) < 1e-6                  # far field
    sk, sw = ref.transport_sources(1.0, 10.0, 0.9, 1.0, 0.0)
    assert sk == pytest.approx(0.9 - 0.9)
    assert sw == pytest.approx(ref.gamma(1.0) * 10.0 * 0.9 - 0.075 * 100.0)


# ---------------------------------------------------------------------------------------------
# Mapping to the repository basis (closures/kOmegaSSTBasis/basisTensors, Python twin tedp.tensors)
# ---------------------------------------------------------------------------------------------

def test_mapping_to_the_kOmegaSSTBasis_integrity_basis():
    """basis_mapping.md: with r = tau omega, S = r Shat, W_abs = -r What, and
    T1_M = r T1, T2_M = r^2 T3, T3_M = r^2 T4, T4_M = -r^2 T2, T6_M = r^3 (T6 - I2 T1),
    T9_M = -r^4 (T7 + (I2/2) T2); IIS = r^2 I1, IIW = r^2 I2, IV = r^3 I4."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'evaluation'))   # tedp
    from tedp import tensors

    n = 200
    g = RNG.normal(size=(n, 3, 3)) * 10.0 ** RNG.uniform(-1, 2, size=(n, 1, 1))
    k = 10.0 ** RNG.uniform(-6, 0, size=n)
    om = 10.0 ** RNG.uniform(0, 5, size=n)
    nu = 1e-5
    frame = RNG.normal(size=(n, 3)) * (RNG.uniform(size=(n, 1)) < 0.5)
    # the basis library (kOmegaSSTBasis.C): Shat = dev(symm(g))/omega,
    # What = (skew(g) + eps_ijk Omega_k)/omega
    sym = 0.5 * (g + np.swapaxes(g, 1, 2))
    shat = (sym - np.trace(sym, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3.0) / om[:, None, None]
    what = (0.5 * (g - np.swapaxes(g, 1, 2)) + ref.frame_tensor(frame)) / om[:, None, None]
    T, inv = tensors.integrity_basis(shat, what)
    tau = ref.time_scale(k, om, nu)
    r = (tau * om)[:, None, None]
    S, W = ref.strain_rotation(g, tau, frame, "absolute")
    np.testing.assert_allclose(S, r * shat, rtol=1e-12, atol=1e-14 * np.abs(S).max())
    np.testing.assert_allclose(W, -r * what, rtol=1e-12, atol=1e-14 * np.abs(W).max())
    TM = ref.basis(S, W)
    i1, i2, i4 = inv[:, 0][:, None, None], inv[:, 1][:, None, None], inv[:, 3][:, None, None]
    expect = {
        1: r * T[:, 0],
        2: r**2 * T[:, 2],
        3: r**2 * T[:, 3],
        4: -r**2 * T[:, 1],
        6: r**3 * (T[:, 5] - i2 * T[:, 0]),
        9: -r**4 * (T[:, 6] + 0.5 * i2 * T[:, 1]),
    }
    for i in expect:
        for j in range(n):
            assert rel_err(TM[i][j], expect[i][j]) < 1e-11, (i, j)
    iis, iiw, iv = ref.invariants(S, W)
    np.testing.assert_allclose(iis, r[:, 0, 0]**2 * i1[:, 0, 0], rtol=1e-12)
    np.testing.assert_allclose(iiw, r[:, 0, 0]**2 * i2[:, 0, 0], rtol=1e-12)
    np.testing.assert_allclose(iv, r[:, 0, 0]**3 * i4[:, 0, 0], rtol=1e-10,
                               atol=1e-12 * np.abs(iv).max())


# ---------------------------------------------------------------------------------------------
# C++ unit-test output (written by tests/comparators/earsm/testEARSM -states ... -out ...)
# ---------------------------------------------------------------------------------------------

# written by tests/comparators/earsm/check_earsm_unit.py (default output folder build/qualification/earsm)
CPP_CSV = Path(__file__).resolve().parents[3] / "build/qualification/earsm/cpp_states.csv"


@pytest.mark.skipif(not CPP_CSV.exists(), reason="C++ unit-test output not present")
def test_cpp_states_agree_with_the_reference():
    import compare_cpp as cc

    data = cc.read_csv(CPP_CSV)
    shared = cc.read_csv(Path(__file__).resolve().parent / "shared_states.csv")
    for col in cc.IN_COLS:
        if col in shared and col != "id":
            assert np.array_equal(data[col], shared[col]), col
    rep = cc.compare(data, cc.reference_for(data, ref.BSL_EARSM, "relative"))
    assert len(rep) >= 13
    failed = {k: v for k, v in rep.items() if not v["passed"]}
    assert not failed, failed
