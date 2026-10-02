"""Independent NumPy reference for the BSL-EARSM comparator.

The model is the Wallin-Johansson explicit algebraic Reynolds-stress model (EARSM) in the k-omega
BSL framework, as published by

    [M12] F. R. Menter, A. V. Garbaruk and Y. Egorov, "Explicit algebraic Reynolds stress models for
          anisotropic wall-bounded flows", Progress in Flight Physics 3 (2012) 89-104,
          doi:10.1051/eucass/201203089 (archive/paper_pof/literature/pdfs/menter2012earsm.pdf; page
          numbers are the journal's printed page numbers 89-104).
    [M09] F. R. Menter, A. V. Garbaruk and Y. Egorov, same title, 3rd EUCASS, Versailles, 6-9 July 2009
          (archive/paper_pof/literature/pdfs/menter2009earsm.pdf; the paper has no printed page
          numbers, so "p." is the PDF page).
    [B94] F. R. Menter, AIAA J. 32(8) 1598-1605 (1994): the BSL constants that M12 quotes.

This file is an independent implementation (1-2 Oct 2026). It was written from
the two PDFs without reading the C++ comparator in closures/comparators/earsm/. Every definition carries its
page and equation. Where the two printings differ, or where the printed text is ambiguous, the choice
and the reason are stated in the docstring of the function concerned:

    * beta_4. M12 p.92 prints beta_4 = -N/Q; M09 p.3 eq.(7) prints beta_4 = -1/Q. -1/Q is correct:
      it is the only value for which the explicit anisotropy solves the implicit Wallin-Johansson
      relation N a = -A1 S + (a W - W a) (see ``implicit_anisotropy`` and the tests; in simple shear
      the implicit relation gives a_11 = 2 s^2/Q exactly). M12's -N/Q is a typesetting error.
    * beta_9. M12 p.93 and M09 p.5 state that the BSL-EARSM omits T9 ("beta_0 = 0" in M12 is a typo
      for beta_9). ``Coefficients.include_t9`` is False by default; True gives the full WJ solution.
    * A1. M12 p.93: the BSL-EARSM uses A1 = 1.245 instead of Wallin-Johansson's 1.2, in eq.(4) (Q).
      The cubic for N, eq.(6), and its solution, eqs.(7)-(8), are printed with the numbers 2.7, 9/20,
      9/10, which are (9/4) A1, (3/8) A1 and (3/4) A1 for A1 = 1.2. They are kept as printed
      (``Coefficients.cubic_a1 = 1.2``); ``cubic_a1 = A1`` gives the self-consistent variant.
    * Branch labels. M12 p.94 calls subscript 1 the "k-epsilon" branch, but the numbers it lists
      (sigma_k1 = 0.5, sigma_w1 = 0.5, beta_1 = 0.075) are the inner, k-omega branch of B94, active
      where F1 = 1. The numbers are used as printed, with F1 = 1 selecting set 1.
    * System rotation. Neither M12 nor M09 includes a frame-rotation term (M12 p.90: EARSMs "do not
      naturally account for swirling and rotating flows"). ``strain_rotation`` offers the relative,
      the absolute and the LRR-consistent rotation tensor; the last is derived in ``lrr_frame_factor``.

Conventions. Arrays carry tensors in their last two axes, (..., 3, 3), in index order T[i, j] with
i the row. The Wallin-Johansson/Menter rotation tensor is W_ij = (tau/2)(dU_i/dx_j - dU_j/dx_i)
(M12 p.92). OpenFOAM's fvc::grad(U) has components g_ij = dU_j/dx_i, the transpose of the velocity
gradient L_ij = dU_i/dx_j; ``strain_rotation`` takes the OpenFOAM form and transposes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------

#: M12 p.93 (and M09 p.4): BSL-EARSM recalibration of A1 ("a 4 percent increase").
A1_BSL_EARSM = 1.245
#: M12 p.92 below eq.(5), M09 p.3 below eq.(9): Wallin-Johansson's A1, which also fixes the printed
#: numbers 2.7, 9/20 and 9/10 of the N equation (M12 eq.(6)-(8)).
A1_WALLIN_JOHANSSON = 1.2
#: M12 p.92: C1 = 1.8 and C1' = (9/4)(C1 - 1).
C1 = 1.8
#: M12 p.92 / M09 p.3: Cmu in the time scale tau; equal to beta* (M12 p.94: beta* = Cmu = 0.09).
CMU = 0.09
#: M12 p.94 / M09 p.5: beta* = Cmu = 0.09.
BETA_STAR = 0.09
#: von Karman constant of the BSL gamma (M12 p.94 uses kappa without a value; B94 gives 0.41).
KAPPA = 0.41
#: M12 p.94 / M09 p.5: BSL constants, set 1 (F1 = 1, inner k-omega branch) and set 2 (outer).
SIGMA_K1, SIGMA_K2 = 0.5, 1.0
SIGMA_W1, SIGMA_W2 = 0.5, 0.856
BETA_W1, BETA_W2 = 0.075, 0.0828
#: M12 p.93 eq. for P~k / M09 p.4 eq.(14): production limiter factor (10 beta* k omega).
PK_LIMIT_FACTOR = 10.0
#: M12 p.92 / M09 eq.(5): factor of the Kolmogorov time-scale limiter.
C_TAU = 6.0


@dataclass(frozen=True)
class Coefficients:
    """Coefficients of the stress-strain relation.

    a1        : A1 in Q = (N^2 - 2 II_W)/A1, M12 p.92 eq.(4); 1.245 for the BSL-EARSM (M12 p.93).
    c1        : C1, M12 p.92; C1' = (9/4)(C1 - 1) = 1.8.
    cubic_a1  : A1 implied by the printed N equation (2.7 = (9/4)*1.2 in M12 eq.(6), 9/20 and 9/10 in
                eq.(8)). 1.2 reproduces the printed equation; set it equal to ``a1`` for the variant
                whose N equation is the exact 2D consistency condition with the recalibrated A1.
    include_t9: M12 p.93 / M09 p.5: the BSL-EARSM sets beta_9 = 0. True restores beta_9 = 1/Q1.
    cmu       : Cmu of the time scale (M12 p.92).
    c_tau     : the factor 6 of the Kolmogorov limiter (M12 p.92).
    """

    a1: float = A1_BSL_EARSM
    c1: float = C1
    cubic_a1: float = A1_WALLIN_JOHANSSON
    include_t9: bool = False
    cmu: float = CMU
    c_tau: float = C_TAU

    @property
    def c1p(self) -> float:
        """C1' = (9/4)(C1 - 1) (M12 p.92 below eq.(5); M09 p.3 below eq.(9))."""
        return 2.25 * (self.c1 - 1.0)


BSL_EARSM = Coefficients()
WALLIN_JOHANSSON = Coefficients(a1=A1_WALLIN_JOHANSSON, include_t9=True)


# --------------------------------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------------------------------

def time_scale(k, omega, nu, cmu: float = CMU, c_tau: float = C_TAU):
    """Turbulence time scale with the Kolmogorov limiter.

    tau = max( 1/(Cmu omega), 6 sqrt( nu/(Cmu k omega) ) )      M12 p.92; M09 p.3 eq.(5).

    With eps = Cmu k omega the second argument is 6 (nu/eps)^(1/2), Durbin's limiter (M12 ref. [9]).
    """
    k = np.asarray(k, dtype=float)
    omega = np.asarray(omega, dtype=float)
    return np.maximum(1.0 / (cmu * omega), c_tau * np.sqrt(nu / (cmu * k * omega)))


def frame_tensor(frame_omega) -> np.ndarray:
    """F_ij = eps_ijk Omega_k for a frame angular velocity Omega (rad/s), shape (..., 3, 3).

    This is the matrix the basis library builds as ``frameTensor`` in
    closures/kOmegaSSTBasis/kOmegaSSTBasis.C (rows (0, Oz, -Oy), (-Oz, 0, Ox), (Oy, -Ox, 0)).
    """
    o = np.asarray(frame_omega, dtype=float)
    f = np.zeros(o.shape[:-1] + (3, 3))
    f[..., 0, 1], f[..., 0, 2] = o[..., 2], -o[..., 1]
    f[..., 1, 0], f[..., 1, 2] = -o[..., 2], o[..., 0]
    f[..., 2, 0], f[..., 2, 1] = o[..., 1], -o[..., 0]
    return f


def lrr_frame_factor(c2: float = 5.0 / 9.0) -> float:
    """Coefficient of the frame tensor in the effective rotation of an LRR-based ARSM.

    Not in M12 or M09 (both are silent on system rotation). Derived here so that a frame term, if the
    comparator uses one, can be verified. In a frame rotating with Omega, the Reynolds-stress equation
    gains the Coriolis term C = -2 k (a F - F a) (F_ij = tau eps_ijk Omega_k, nondimensional), and the
    rapid pressure-strain of Launder-Reece-Rodi acts on the absolute rotation W_abs = W - F with the
    coefficient -c (a W_abs - W_abs a), c = (10 - 7 C2)/11. Production contributes +(a W - W a). The
    rotational part of the algebraic relation is therefore

        (1 - c)(a W - W a) - (2 - c)(a F - F a) = (1 - c)(a W* - W* a),
        W* = W - (2 - c)/(1 - c) F = W_abs - F/(1 - c).

    With Wallin and Johansson's C2 = 5/9 (the value that removes the a S + S a term and gives
    A1 = 6/5, see ``lrr_algebraic_coefficients``), c = 5/9 and W* = W - (13/4) F = W_abs - (9/4) F.
    Returns (2 - c)/(1 - c), the factor on F relative to the RELATIVE rotation tensor.
    """
    c = (10.0 - 7.0 * c2) / 11.0
    return (2.0 - c) / (1.0 - c)


def lrr_algebraic_coefficients(c1: float = C1, c2: float = 5.0 / 9.0) -> dict:
    """Coefficients of the algebraic relation that the LRR model gives (cross-check of M12 eq.(4)-(5)).

    Weak-equilibrium LRR algebraic stress model, nondimensional with tau = k/eps:
        (C1 - 1 + P/eps) a = c_S S + c_R (a W - W a) + c_A (a S + S a - (2/3) tr(aS) I)
    with c_S = -4/3 + (4/3)(9 C2 + 6)/11 - 2 (30 C2 - 2)/55, c_R = 1 - (10 - 7 C2)/11,
    c_A = -1 + (9 C2 + 6)/11. Dividing by c_R gives N a = -A1 S + (aW - Wa) - A2 (...), with
    N = (C1 - 1)/c_R + P/eps / c_R. For C2 = 5/9: c_R = 4/9, so C1' = (9/4)(C1 - 1) and the factor 9/4
    on P/eps (M12 eq.(5)), A1 = -c_S/c_R = 6/5 = 1.2 and A2 = 0 (M12 p.91: "slightly simplified LRR").
    """
    c_s = -4.0 / 3.0 + (4.0 / 3.0) * (9.0 * c2 + 6.0) / 11.0 - 2.0 * (30.0 * c2 - 2.0) / 55.0
    c_r = 1.0 - (10.0 - 7.0 * c2) / 11.0
    c_a = -1.0 + (9.0 * c2 + 6.0) / 11.0
    return {
        "c1_prime": (c1 - 1.0) / c_r,
        "pk_factor": 1.0 / c_r,
        "a1": -c_s / c_r,
        "a2": -c_a / c_r,
    }


def strain_rotation(grad_u_of, tau, frame_omega=None, frame_mode: str = "relative"):
    """Nondimensional strain and rotation tensors S, W of M12 p.92 from OpenFOAM's grad(U).

    S_ij = (tau/2)(dU_i/dx_j + dU_j/dx_i),  W_ij = (tau/2)(dU_i/dx_j - dU_j/dx_i)   M12 p.92; M09 eq.(4).

    grad_u_of: (..., 3, 3) with g_ij = dU_j/dx_i (OpenFOAM fvc::grad). L = g^T. S is made traceless
    (dev), which is exact for incompressible flow and matches the basis library's dev(symm(gradU)).
    In OpenFOAM terms: S = tau dev(symm(g)) and W = -tau skew(g) (note the sign).

    frame_mode (not in M12/M09, see module docstring):
        "relative": W as printed (frame rotation ignored),
        "absolute": W_abs = W - F,  F_ij = tau eps_ijk Omega_k,
        "lrr"     : W* = W - lrr_frame_factor() F = W - (13/4) F.
    """
    g = np.asarray(grad_u_of, dtype=float)
    L = np.swapaxes(g, -1, -2)
    tau = np.asarray(tau, dtype=float)[..., None, None]
    sym = 0.5 * (L + np.swapaxes(L, -1, -2))
    tr = np.trace(sym, axis1=-2, axis2=-1)[..., None, None]
    S = tau * (sym - tr * np.eye(3) / 3.0)
    W = tau * 0.5 * (L - np.swapaxes(L, -1, -2))
    if frame_omega is not None and frame_mode != "relative":
        F = tau * frame_tensor(np.broadcast_to(np.asarray(frame_omega, float), g.shape[:-1]))
        if frame_mode == "absolute":
            W = W - F
        elif frame_mode == "lrr":
            W = W - lrr_frame_factor() * F
        else:
            raise ValueError(f"unknown frame_mode {frame_mode!r}")
    return S, W


def invariants(S, W):
    """II_S = S_ij S_ji, II_W = W_ij W_ji, IV = S_ik W_kj W_ji.   M12 p.92; M09 p.3 eq.(6).

    II_W <= 0 for any antisymmetric W. IV = tr(S W W).
    """
    iis = np.einsum("...ij,...ji->...", S, S)
    iiw = np.einsum("...ij,...ji->...", W, W)
    iv = np.einsum("...ik,...kj,...ji->...", S, W, W)
    return iis, iiw, iv


def basis(S, W):
    """Menter's regrouped tensor basis T1, T2, T3, T4, T6, T9.   M12 p.92 eq.(3); M09 p.3 eq.(3).

    T1 = S
    T2 = S S - (1/3) II_S I
    T3 = W W - (1/3) II_W I
    T4 = S W - W S
    T6 = S W W + W W S - (2/3) IV I - II_W S
    T9 = W S W W - W W S W + (1/2) II_W (S W - W S)

    M12 p.92 below eq.(3): the regrouping (the "- II_W S" in T6 and the "+ II_W T4/2" in T9) makes T6
    and T9 vanish in two-dimensional mean flow; it changes beta_1 and beta_4 relative to Wallin and
    Johansson (2000) but not the model. Returns a dict {1: T1, 2: T2, 3: T3, 4: T4, 6: T6, 9: T9}.
    """
    iis, iiw, iv = invariants(S, W)
    eye = np.eye(3)
    SS = S @ S
    WW = W @ W
    SW = S @ W
    WS = W @ S
    t4 = SW - WS
    return {
        1: S,
        2: SS - iis[..., None, None] * eye / 3.0,
        3: WW - iiw[..., None, None] * eye / 3.0,
        4: t4,
        6: S @ WW + WW @ S - 2.0 * iv[..., None, None] * eye / 3.0 - iiw[..., None, None] * S,
        9: W @ S @ WW - WW @ S @ W + 0.5 * iiw[..., None, None] * t4,
    }


# --------------------------------------------------------------------------------------------------
# N: the cubic and its root
# --------------------------------------------------------------------------------------------------

def cubic_coefficients(iis, iiw, c1p: float = 1.8, cubic_a1: float = A1_WALLIN_JOHANSSON):
    """Coefficients (1, b, c, d) of N^3 + b N^2 + c N + d = 0.   M12 p.92 eq.(6); M09 p.4 eq.(10).

        N^3 - C1' N^2 - (2.7 II_S + 2 II_W) N + 2 C1' II_W = 0,   2.7 = (9/4) cubic_a1.

    Origin: in two-dimensional mean flow (IV = 0) the explicit solution gives
    P/eps = -a_ij S_ij = -beta_1 II_S = A1 N II_S/(N^2 - 2 II_W); inserting this in
    N = C1' + (9/4) P/eps (M12 eq.(5)) gives the cubic with 2.7 = (9/4) A1.
    """
    iis = np.asarray(iis, dtype=float)
    iiw = np.asarray(iiw, dtype=float)
    one = np.ones_like(iis + iiw)
    return (one, -c1p * one, -(2.25 * cubic_a1 * iis + 2.0 * iiw), 2.0 * c1p * iiw)


def solve_n(iis, iiw, c1p: float = 1.8, cubic_a1: float = A1_WALLIN_JOHANSSON, stable: bool = True):
    """N from the closed-form root.   M12 p.93 eqs.(7)-(8); M09 p.4 eqs.(11)-(12).

        P1 = C1' ( C1'^2/27 + (9/20) II_S - (2/3) II_W )
        P2 = P1^2 - ( C1'^2/9 + (9/10) II_S + (2/3) II_W )^3
        P2 >= 0:  N = C1'/3 + (P1 + sqrt(P2))^(1/3) + sign(P1 - sqrt(P2)) |P1 - sqrt(P2)|^(1/3)
        P2 <  0:  N = C1'/3 + 2 (P1^2 - P2)^(1/6) cos( (1/3) arccos( P1/sqrt(P1^2 - P2) ) )

    (9/20 = (3/8) cubic_a1 and 9/10 = (3/4) cubic_a1; with cubic_a1 = 1.2 these are the printed numbers.)

    Root selection. Eq.(7) is Cardano's formula for the cubic of eq.(6) after the shift N = t + C1'/3:
    P2 is the discriminant q^2/4 + p^3/27 and P1 = -q/2. For P2 >= 0 the cubic has one real root, which
    the first line gives. For P2 < 0 it has three real roots t_m = 2 sqrt(-p/3) cos(theta/3 - 2 pi m/3),
    and the second line is m = 0, the LARGEST root. So eq.(7) always selects the largest real root.
    It is the physical one: with f(N) the cubic, f(C1') = -2.7 C1' II_S <= 0 and f -> +inf, so a root
    lies at or above C1'; the three roots sum to C1' > 0, so no more than one root can exceed C1'
    (strictly, for II_S > 0). The largest root is therefore the unique root with
    P/eps = (4/9)(N - C1') >= 0 (eq.(5)): non-negative production. As II_W <= 0, Q = (N^2 - 2 II_W)/A1
    > 0 and Q1 > 0 for this root, so beta_1..beta_9 are never singular. Since II_S >= 0 and II_W <= 0,
    P1 >= C1'^3/27 > 0, so P1 + sqrt(P2) > 0 and the printed real 1/3 power is well defined; the
    second cube root uses sign(x)|x|^(1/3) = numpy.cbrt. The branches meet at P2 = 0, where both give
    C1'/3 + 2 P1^(1/3), so N is continuous. In the weak-strain limit P1 -> C1'^3/27, P2 -> 0 and
    N -> C1'. The arccos argument is clipped to [-1, 1] against round-off.

    Round-off (not in the papers): where B = C1'^2/9 + (9/10) II_S + (2/3) II_W is near 0, P2 ~ P1^2
    and P1 - sqrt(P2) cancels; the printed form then loses about 1e-11 relative accuracy in N. With
    stable=True (default) the second cube root is evaluated as B/cbrt(P1 + sqrt(P2)), which is the
    same number exactly, because (P1 - sqrt P2)(P1 + sqrt P2) = P1^2 - P2 = B^3. stable=False is the
    printed form; the two agree to 1e-10 (test_earsm_reference.py).
    """
    iis = np.asarray(iis, dtype=float)
    iiw = np.asarray(iiw, dtype=float)
    k1 = 2.25 * cubic_a1 / 6.0        # 9/20 for cubic_a1 = 1.2
    k2 = 2.25 * cubic_a1 / 3.0        # 9/10 for cubic_a1 = 1.2
    p1 = c1p * (c1p**2 / 27.0 + k1 * iis - (2.0 / 3.0) * iiw)
    base = c1p**2 / 9.0 + k2 * iis + (2.0 / 3.0) * iiw
    p2 = p1**2 - base**3
    with np.errstate(invalid="ignore", divide="ignore"):
        sq = np.sqrt(np.where(p2 >= 0.0, p2, 0.0))
        c_plus = np.cbrt(p1 + sq)
        c_minus = base / c_plus if stable else np.cbrt(p1 - sq)
        n_pos = c1p / 3.0 + c_plus + c_minus
        r = np.where(p2 < 0.0, p1**2 - p2, 1.0)
        arg = np.clip(p1 / np.sqrt(r), -1.0, 1.0)
        n_neg = c1p / 3.0 + 2.0 * r ** (1.0 / 6.0) * np.cos(np.arccos(arg) / 3.0)
    n = np.where(p2 >= 0.0, n_pos, n_neg)
    return n, p1, p2


def largest_real_root(iis, iiw, c1p: float = 1.8, cubic_a1: float = A1_WALLIN_JOHANSSON):
    """Largest real root of the cubic of M12 eq.(6) by numpy.roots (companion matrix), for tests."""
    iis = np.atleast_1d(np.asarray(iis, dtype=float))
    iiw = np.atleast_1d(np.asarray(iiw, dtype=float))
    iis, iiw = np.broadcast_arrays(iis, iiw)
    out = np.empty(iis.shape)
    for idx in np.ndindex(iis.shape):
        c = [float(x) for x in cubic_coefficients(iis[idx], iiw[idx], c1p, cubic_a1)]
        roots = np.roots(c)
        real = roots[np.abs(roots.imag) <= 1e-9 * max(1.0, np.abs(roots).max())].real
        out[idx] = real.max()
    return out


# --------------------------------------------------------------------------------------------------
# beta coefficients and anisotropy
# --------------------------------------------------------------------------------------------------

def betas(n, iis, iiw, iv, coeffs: Coefficients = BSL_EARSM) -> dict:
    """beta_1 ... beta_9 of the regrouped basis.   M09 p.3 eqs.(7)-(8); M12 p.92 eq.(4).

        Q  = (N^2 - 2 II_W)/A1,      Q1 = (Q/6)(2 N^2 - II_W)
        beta_1 = -N/Q,  beta_2 = 0,  beta_3 = -2 IV/(N Q1),  beta_4 = -1/Q,
        beta_6 = -N/Q1, beta_9 = 1/Q1 (0 in the BSL-EARSM, M12 p.93, M09 p.5),
        beta_5 = beta_7 = beta_8 = beta_10 = 0 (absent from eq.(2)).

    beta_4: M12 p.92 prints -N/Q; that is a misprint (see the module docstring). M09's -1/Q is the
    value that solves the implicit relation (``implicit_anisotropy``).
    """
    n = np.asarray(n, dtype=float)
    q = (n**2 - 2.0 * iiw) / coeffs.a1
    q1 = q * (2.0 * n**2 - iiw) / 6.0
    zero = np.zeros_like(n)
    return {
        1: -n / q,
        2: zero,
        3: -2.0 * iv / (n * q1),
        4: -1.0 / q,
        6: -n / q1,
        9: (1.0 / q1) if coeffs.include_t9 else zero,
        "Q": q,
        "Q1": q1,
    }


def anisotropy(S, W, coeffs: Coefficients = BSL_EARSM, n=None) -> dict:
    """Anisotropy a = sum beta_i T_i.   M12 p.91 eqs.(1)-(2), with tau_ij = k (a_ij + (2/3) delta_ij).

    a is twice Lumley's b. If n is None, N is the root of M12 eq.(7). Returns a dict with keys
    a, a_linear (beta_1 T1), a_extra (the rest), N, P1, P2, betas, T, IIS, IIW, IV, cmu_eff
    (= -beta_1/2: nu_t = cmu_eff k tau), pk_over_eps (= -a_ij S_ij, the production the stress gives)
    and pk_over_eps_N (= (4/9)(N - C1'), what eq.(5) implies).
    """
    iis, iiw, iv = invariants(S, W)
    if n is None:
        n, p1, p2 = solve_n(iis, iiw, coeffs.c1p, coeffs.cubic_a1)
    else:
        n = np.asarray(n, dtype=float) * np.ones_like(iis)
        p1 = p2 = np.full_like(iis, np.nan)
    b = betas(n, iis, iiw, iv, coeffs)
    T = basis(S, W)
    a_lin = b[1][..., None, None] * T[1]
    a_ex = sum(b[i][..., None, None] * T[i] for i in (2, 3, 4, 6, 9))
    a = a_lin + a_ex
    return {
        "a": a,
        "a_linear": a_lin,
        "a_extra": a_ex,
        "N": n,
        "P1": p1,
        "P2": p2,
        "betas": b,
        "T": T,
        "IIS": iis,
        "IIW": iiw,
        "IV": iv,
        "cmu_eff": -0.5 * b[1],
        "pk_over_eps": -np.einsum("...ij,...ij->...", a, S),
        "pk_over_eps_N": (4.0 / 9.0) * (n - coeffs.c1p),
    }


def implicit_anisotropy(S, W, n, a1: float):
    """Solve the implicit Wallin-Johansson relation for a, at given N, by linear algebra.

        N a = -A1 S + (a W - W a)

    This is the algebraic relation behind M12 eqs.(2)-(5) (M12 p.91: WJ's EARSM is the explicit
    solution of an algebraic model from a "slightly simplified LRR" model; ``lrr_algebraic_coefficients``
    derives N, C1' and A1 from LRR with C2 = 5/9). It is solved here as a 9x9 linear system, with no
    tensor basis, so that it checks beta_1..beta_9 and T1..T9 independently. For a given N the explicit
    solution with beta_9 = 1/Q1 is exact in 3D; the BSL-EARSM's beta_9 = 0 differs by beta_9 T9.
    Row-major vec: vec(a W) = (I kron W^T) vec(a), vec(W a) = (W kron I) vec(a).
    """
    S = np.asarray(S, dtype=float)
    W = np.asarray(W, dtype=float)
    n = np.asarray(n, dtype=float) * np.ones(S.shape[:-2])
    out = np.empty_like(S)
    eye3 = np.eye(3)
    for idx in np.ndindex(S.shape[:-2]):
        w = W[idx]
        A = n[idx] * np.eye(9) - np.kron(eye3, w.T) + np.kron(w, eye3)
        out[idx] = np.linalg.solve(A, -a1 * S[idx].reshape(9)).reshape(3, 3)
    return out


# --------------------------------------------------------------------------------------------------
# Pointwise model state, as the C++ comparator computes it per cell
# --------------------------------------------------------------------------------------------------

def earsm_state(grad_u_of, k, omega, nu, frame_omega=None, coeffs: Coefficients = BSL_EARSM,
                frame_mode: str = "relative") -> dict:
    """Everything the stress-strain relation produces for a cell.

    tau from ``time_scale`` (M12 p.92), S and W from ``strain_rotation`` (M12 p.92), N from eq.(7),
    a from eq.(2). Effective eddy viscosity and the OpenFOAM split:

        tau_ij = k (a_ij + (2/3) delta_ij)                                    M12 eq.(1)
        nu_t,eff = -(1/2) beta_1 k tau                                        (beta_1 T1 = -2 nu_t S_dim/k)
        R = (2/3) k I - nu_t,eff twoSymm(grad U) + nonlinearStress,
        nonlinearStress = k (beta_3 T3 + beta_4 T4 + beta_6 T6 + beta_9 T9)   [m2/s2]

    M12 pp.100-102 calls beta_1 T1 alone "the isotropic contribution" (the "beta_1-limited variant"); its
    eddy viscosity is nu_t,eff. In 3D, T6 contains -II_W S, a part proportional to S: it stays in the
    nonlinear stress (it carries beta_6, not beta_1). The diffusion of k and omega does NOT use
    nu_t,eff but nu_t = k/omega (M12 p.93), returned as nut_diffusion.
    """
    g = np.asarray(grad_u_of, dtype=float)
    k = np.asarray(k, dtype=float)
    omega = np.asarray(omega, dtype=float)
    tau = time_scale(k, omega, nu, coeffs.cmu, coeffs.c_tau)
    S, W = strain_rotation(g, tau, frame_omega, frame_mode)
    st = anisotropy(S, W, coeffs)
    st["tau"] = tau
    st["S"] = S
    st["W"] = W
    st["nut"] = -0.5 * st["betas"][1] * k * tau
    st["nonlinear_stress"] = k[..., None, None] * st["a_extra"]
    st["reynolds_stress"] = k[..., None, None] * (st["a"] + 2.0 * np.eye(3) / 3.0)
    st["nut_diffusion"] = k / omega
    # production -tau_ij dU_i/dx_j (M12 p.93) with L_ij = dU_i/dx_j = g_ji
    L = np.swapaxes(g, -1, -2)
    st["Pk"] = -np.einsum("...ij,...ij->...", st["reynolds_stress"], L)
    st["Pk_limited"] = np.minimum(st["Pk"], PK_LIMIT_FACTOR * BETA_STAR * k * omega)
    return st


# --------------------------------------------------------------------------------------------------
# Transport equations (M12 p.93-94, M09 p.4-5 eqs.(13)-(15))
# --------------------------------------------------------------------------------------------------

def blend(f1, phi1, phi2):
    """phi = F1 phi1 + (1 - F1) phi2.   M12 p.94; M09 p.5."""
    return f1 * phi1 + (1.0 - f1) * phi2


def gamma(f1, kappa: float = KAPPA):
    """gamma = beta/beta* - sigma_w kappa^2/sqrt(beta*) with blended beta, sigma_w.   M12 p.94.

    gamma(1) = 0.075/0.09 - 0.5*0.41^2/0.3 = 0.553167; gamma(0) = 0.0828/0.09 - 0.856*0.41^2/0.3 = 0.440355.
    """
    beta = blend(f1, BETA_W1, BETA_W2)
    sigma_w = blend(f1, SIGMA_W1, SIGMA_W2)
    return beta / BETA_STAR - sigma_w * kappa**2 / np.sqrt(BETA_STAR)


def f1_blending(k, omega, d, nu, gradk_dot_gradw, cmu: float = CMU):
    """F1 = tanh(arg1^4),  arg1 = min( max( sqrt(k)/(Cmu omega d), 500 nu/(omega d^2) ),
                                         2 k omega/(d^2 grad k . grad omega) )        M12 p.94; M09 p.5 eq.(15).

    The last argument equals B94's 4 sigma_w2 k/(CD_kw d^2) with CD_kw = 2 sigma_w2 grad k.grad omega/omega.
    B94 bounds CD_kw below (1e-20; OpenFOAM: 1e-10); M12 prints no bound. Here a non-positive
    grad k . grad omega makes the last argument +inf (no effect), the limit of the bounded form.
    """
    k = np.asarray(k, float)
    omega = np.asarray(omega, float)
    d = np.asarray(d, float)
    cross = np.asarray(gradk_dot_gradw, float)
    a = np.maximum(np.sqrt(k) / (cmu * omega * d), 500.0 * nu / (omega * d**2))
    with np.errstate(divide="ignore"):
        b = np.where(cross > 0.0, 2.0 * k * omega / (d**2 * np.where(cross > 0.0, cross, 1.0)), np.inf)
    return np.tanh(np.minimum(a, b) ** 4)


def transport_sources(k, omega, pk_tilde, f1, gradk_dot_gradw):
    """Local source terms of the BSL-EARSM transport equations (diffusion excluded).   M12 p.93-94.

        k:     P~k - beta* k omega
        omega: (gamma omega/k) P~k - beta omega^2 + (sigma_d/omega) grad k . grad omega,
               sigma_d = 2 (1 - F1) sigma_w2
    Diffusion: div((nu + sigma_k nu_t) grad k), div((nu + sigma_w nu_t) grad omega) with nu_t = k/omega
    (M12 p.93) and sigma_k, sigma_w from ``blend``.
    """
    beta = blend(f1, BETA_W1, BETA_W2)
    sigma_d = 2.0 * (1.0 - f1) * SIGMA_W2
    sk = pk_tilde - BETA_STAR * k * omega
    sw = gamma(f1) * omega / k * pk_tilde - beta * omega**2 + sigma_d / omega * gradk_dot_gradw
    return sk, sw


# --------------------------------------------------------------------------------------------------
# Equilibria with closed forms (targets for the tests and for published_targets.json)
# --------------------------------------------------------------------------------------------------

def simple_shear(s_tau: float):
    """Nondimensional S, W of the simple shear U = (G y, 0, 0) with tau G = s_tau.

    S = (s_tau/2)(e1 e2 + e2 e1), W = (s_tau/2)(e1 e2 - e2 e1); II_S = -II_W = s_tau^2/2, IV = 0.
    """
    S = np.zeros((3, 3))
    W = np.zeros((3, 3))
    S[0, 1] = S[1, 0] = 0.5 * s_tau
    W[0, 1], W[1, 0] = 0.5 * s_tau, -0.5 * s_tau
    return S, W


def shear_equilibrium_closed_form(pk_over_eps: float, coeffs: Coefficients = BSL_EARSM) -> dict:
    """Closed-form EARSM state of a simple shear in which the stress gives P/eps = R.

    Derivation (from M12 eqs.(2)-(6), independent of ``solve_n``): in simple shear II_W = -II_S, IV = 0,
    T6 = T9 = 0 and tr(T4 S) = 0, so P/eps = -a_ij S_ij = -beta_1 II_S = A1 N II_S/(N^2 + 2 II_S) = R,
    i.e. II_S = R N^2/(A1 N - 2 R). The cubic with II_W = -II_S reads (N - C1')(N^2 + 2 II_S)
    = (9/4) cubic_a1 II_S N. Eliminating II_S: N = C1' + (9/4) cubic_a1 R/A1. Then
        cmu_eff = -beta_1/2 = R/(2 II_S),   s = sqrt(II_S/2) (= tau dU/dy / 2),
        a_12 = beta_1 s = -2 cmu_eff s,   a_11 = -a_22 = -2 beta_4 s^2 = 2 s^2/Q,   a_33 = 0,
        -a_12 = sqrt(cmu_eff R)  (the structure parameter -uv/k).
    With cubic_a1 = A1 this is the self-consistent N = C1' + (9/4) R of M12 eq.(5).
    """
    R = float(pk_over_eps)
    n = coeffs.c1p + 2.25 * coeffs.cubic_a1 * R / coeffs.a1
    iis = R * n**2 / (coeffs.a1 * n - 2.0 * R)
    cmu_eff = R / (2.0 * iis)
    s = np.sqrt(iis / 2.0)
    q = (n**2 + 2.0 * iis) / coeffs.a1
    a12 = -2.0 * cmu_eff * s
    a11 = 2.0 * s**2 / q
    return {
        "N": n,
        "IIS": iis,
        "s_tau": 2.0 * s,
        "cmu_eff": cmu_eff,
        "a11": a11,
        "a22": -a11,
        "a33": 0.0,
        "a12": a12,
        "minus_uv_over_k": -a12,
    }


def homogeneous_shear_pk_over_eps(f1: float = 0.0, kappa: float = KAPPA) -> float:
    """P/eps of the structural equilibrium of homogeneous shear for the BSL omega equation.

    Constant S tau needs constant omega (tau = 1/(beta* omega) away from the limiter), so the omega
    equation of M12 p.93 without diffusion gives gamma omega/k P - beta omega^2 = 0, i.e.
    P/eps = beta/(gamma beta*) with eps = beta* k omega. Far from walls F1 -> 0 (outer constants):
    0.0828/(0.440355*0.09) = 2.0892; with the inner constants (F1 = 1): 1.5065.
    """
    beta = blend(f1, BETA_W1, BETA_W2)
    return float(beta / (gamma(f1, kappa) * BETA_STAR))


def log_layer_kappa(minus_uv_over_k: float, kappa0: float = KAPPA) -> float:
    """Effective von Karman constant of the k-omega log layer for a given structure parameter c = -uv/k.

    Log layer: k constant, P = eps, U' = u_tau/(kappa y), -uv = u_tau^2 = c k, omega = A/y with diffusion
    nu_t = k/omega (M12 p.93). P = eps gives A = u_tau c/(beta* kappa); the omega equation
    gamma beta* omega^2 - beta omega^2 + sigma_w k/y^2 = 0 then gives kappa^2 = c^3 (beta - gamma beta*)
    /(beta*^2 sigma_w). With gamma of M12 p.94, beta - gamma beta* = sigma_w kappa0^2 sqrt(beta*), so
    kappa = kappa0 (c/sqrt(beta*))^(3/2), independent of F1. c = sqrt(beta*) = 0.3 recovers kappa0.
    """
    return float(kappa0 * (minus_uv_over_k / np.sqrt(BETA_STAR)) ** 1.5)


__all__ = [name for name in dir() if not name.startswith("_")]
