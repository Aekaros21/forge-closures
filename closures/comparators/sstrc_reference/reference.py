"""Independent NumPy reference of the Spalart-Shur rotation-curvature correction of SST.

This module is the verifier's reference for the comparator ``kOmegaSSTRC`` (SST-RC, called
SST-CC by its authors). It is written from the publications alone and shares no code with the
C++ model in ``closures/comparators/sstrc``. Every quantity is evaluated pointwise from

* ``A``      the velocity gradient, ``A[..., i, j] = du_i/dx_j`` (paper convention, row i,
             column j; NOT OpenFOAM's ``fvc::grad(U)``, which is the transpose), of the
             velocity relative to the frame of the calculation;
* ``DSDt``   the Lagrangian derivative of the strain-rate tensor in that frame, WITHOUT the
             system-rotation terms (they are added here, as Eq. (6) of [SM09] writes them);
* ``Om``     the angular velocity of the frame of the calculation, ``Omega^rot`` [rad/s];
* ``omega``  the SST specific dissipation rate [1/s] (enters only ``D``).

Sources (PDFs in ``archive/paper_pof/literature/pdfs``; page numbers are the journals'):

[SM09] P. E. Smirnov and F. R. Menter, "Sensitization of the SST turbulence model to rotation
       and curvature by applying the Spalart-Shur correction term", J. Turbomach. 131(4),
       041010 (2009), doi:10.1115/1.3070573 (smirnov2009.pdf). The model implemented here.
[SS97] P. R. Spalart and M. Shur, "On the sensitization of turbulence models to rotation and
       curvature", Aerosp. Sci. Technol. 1(5), 297-302 (1997) (spalart1997.pdf). Origin of
       the measure; used here only for independent cross-checks (Eqs. (2) and (4)).
[S00]  M. L. Shur, M. K. Strelets, A. K. Travin and P. R. Spalart, "Turbulence modeling in
       rotating and curved channels: assessing the Spalart-Shur correction", AIAA J. 38(5),
       784-792 (2000) (shur2000.pdf). The SA form (SARC); its normalisation differs from
       [SM09] and is provided only as ``variant="shur2000"`` for comparison.

Definitions implemented ([SM09] unless stated):

* f_rotation = (1 + c_r1) 2 r*/(1 + r*) [1 - c_r3 tan^-1(c_r2 r~)] - c_r1
                                   Eq. (1), p. 041010-1 (= [SS97] Eq. (5), p. 301; [S00] p. 785)
* f_r1 = max{min(f_rotation, 1.25), 0.0}                      Eq. (4), p. 041010-2
  (multiplies P_k in the k equation, Eq. (2), p. 041010-1, and rho P_k/mu_t in the omega
  equation, Eq. (3), p. 041010-2)
* r* = S/Omega                                                Eq. (5), p. 041010-2
* r~ = 2 Omega_ik S_jk [DS_ij/Dt + (eps_imn S_jn + eps_jmn S_in) Omega^rot_m] / (Omega D^3)
                                                              Eq. (6), p. 041010-2
* S_ij = (du_i/dx_j + du_j/dx_i)/2                            Eq. (7), p. 041010-2
* Omega_ij = ((du_i/dx_j - du_j/dx_i) + 2 eps_mji Omega^rot_m)/2   Eq. (8), p. 041010-2
* S^2 = 2 S_ij S_ij                                           Eq. (9), p. 041010-2
* Omega^2 = 2 Omega_ij Omega_ij                               Eq. (10), p. 041010-2
* D^2 = max(S^2, 0.09 omega^2)                                Eq. (11), p. 041010-2
* "all the variables and their derivatives are defined with respect to the reference frame of
  the calculation, which is rotating with a rate Omega^rot"   p. 041010-2, above Eq. (5)
* c_r1 = 1.0, c_r2 = 2.0, c_r3 = 1.0                          p. 041010-2, end of Sec. 2
  ([SS97] p. 301 and [S00] p. 785 use c_r2 = 12 for the SA model)
* DS_ij/Dt for steady flow: the convective derivative only, D/Dt int S_ij dtau =
  int S_ij V_n dsigma (Eqs. (12)-(13), p. 041010-2), i.e. u_k dS_ij/dx_k for a divergence-free
  velocity; the local time derivative is omitted "since it is zero in converged solution".

Conventions fixed here and checked by the tests: ``eps`` is the Levi-Civita symbol with
eps_123 = +1; Omega_ij is the ABSOLUTE rotation-rate tensor (relative vorticity plus frame);
the frame term of Eq. (6) is (Om_F S - S Om_F)_ij with Om_F,ij = eps_mji Omega^rot_m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# [SM09] p. 041010-2, last paragraph of Sec. 2
CR1 = 1.0
CR2 = 2.0
CR3 = 1.0
# [SM09] Eq. (4), p. 041010-2
FR1_MAX = 1.25
FR1_MIN = 0.0
# [SM09] Eq. (11), p. 041010-2 (0.09 = beta* of SST)
D_OMEGA_COEFF = 0.09

#: Levi-Civita symbol, EPS[i, j, k] = eps_ijk with eps_123 = +1 (indices 0..2 here).
EPS = np.zeros((3, 3, 3))
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    EPS[_i, _j, _k] = 1.0
    EPS[_i, _k, _j] = -1.0


def _t(a):
    return np.swapaxes(a, -1, -2)


def strain_rate(A):
    """S_ij = (du_i/dx_j + du_j/dx_i)/2, [SM09] Eq. (7), p. 041010-2."""
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + _t(A))


def frame_rotation_tensor(Om):
    """Om_F,ij = eps_mji Omega^rot_m, the frame part of [SM09] Eq. (8), p. 041010-2.

    Equal to -eps_ijm Omega_m; for Omega = (0, 0, Oz): Om_F,12 = -Oz, Om_F,21 = +Oz."""
    Om = np.asarray(Om, dtype=float)
    return np.einsum('mji,...m->...ij', EPS, Om)


def rotation_rate_absolute(A, Om):
    """Omega_ij = ((du_i/dx_j - du_j/dx_i) + 2 eps_mji Omega^rot_m)/2, [SM09] Eq. (8), p. 041010-2.

    Identical to omega_ij of [S00] p. 785 and to the bracket of [SS97] Eq. (4), p. 300
    (there written dU_i/dx_k - dU_k/dx_i + 2 eps_lki Omega_l = 2 Omega_ik)."""
    A = np.asarray(A, dtype=float)
    return 0.5 * (A - _t(A)) + frame_rotation_tensor(np.broadcast_to(Om, A.shape[:-1]))


def frame_term(S, Om):
    """T_ij = (eps_imn S_jn + eps_jmn S_in) Omega^rot_m, [SM09] Eq. (6), p. 041010-2.

    The same term is Omega_o [eps_iop S_jp + eps_jop S_ip] in [SS97] Eq. (4), p. 300 and
    (eps_imn S_jn + eps_jmn S_in) Omega_m in [S00] p. 785."""
    S = np.asarray(S, dtype=float)
    Om = np.broadcast_to(np.asarray(Om, dtype=float), S.shape[:-1])
    return (np.einsum('imn,...jn,...m->...ij', EPS, S, Om)
            + np.einsum('jmn,...in,...m->...ij', EPS, S, Om))


def magnitude(T):
    """sqrt(2 T_ij T_ij): S for the strain ([SM09] Eq. (9)), Omega for the rotation (Eq. (10))."""
    T = np.asarray(T, dtype=float)
    return np.sqrt(2.0 * np.einsum('...ij,...ij->...', T, T))


def d_scale(S_mag, omega):
    """D = sqrt(max(S^2, 0.09 omega^2)), [SM09] Eq. (11), p. 041010-2 (omega: SST variable)."""
    S_mag = np.asarray(S_mag, dtype=float)
    omega = np.asarray(omega, dtype=float)
    return np.sqrt(np.maximum(S_mag ** 2, D_OMEGA_COEFF * omega ** 2))


def strain_gradient(H):
    """dS_ij/dx_k from H[..., i, j, k] = d^2 u_i/(dx_j dx_k): (H_ijk + H_jik)/2."""
    H = np.asarray(H, dtype=float)
    return 0.5 * (H + np.swapaxes(H, -2, -3))


def lagrangian_strain_derivative(u, H, dSdt=None):
    """DS_ij/Dt = dS_ij/dt + u_k dS_ij/dx_k in the frame of the calculation.

    Steady form ([SM09] Eqs. (12)-(13), p. 041010-2): ``dSdt`` omitted (zero at convergence);
    the flux form int S_ij V_n dsigma equals u_k dS_ij/dx_k for a divergence-free u. ``u`` is
    the velocity relative to the frame; the frame terms are NOT included (see frame_term)."""
    u = np.asarray(u, dtype=float)
    out = np.einsum('...k,...ijk->...ij', u, strain_gradient(H))
    if dSdt is not None:
        out = out + np.asarray(dSdt, dtype=float)
    return out


def r_tilde_numerator(A, DSDt, Om):
    """2 Omega_ik S_jk [DS_ij/Dt + (eps_imn S_jn + eps_jmn S_in) Omega^rot_m], numerator of
    [SM09] Eq. (6), p. 041010-2 (and of [S00] p. 785)."""
    S = strain_rate(A)
    W = rotation_rate_absolute(A, Om)
    M = np.asarray(DSDt, dtype=float) + frame_term(S, Om)
    return 2.0 * np.einsum('...ik,...jk,...ij->...', W, S, M)


def f_rotation(r_star, r_tilde, cr1=CR1, cr2=CR2, cr3=CR3):
    """(1 + c_r1) 2 r*/(1 + r*) [1 - c_r3 tan^-1(c_r2 r~)] - c_r1, [SM09] Eq. (1), p. 041010-1.

    ``r_star = inf`` (Omega = 0) is taken in the limit 2 r*/(1 + r*) -> 2."""
    r_star = np.asarray(r_star, dtype=float)
    with np.errstate(invalid='ignore'):
        ratio = np.where(np.isinf(r_star), 2.0, 2.0 * r_star / (1.0 + r_star))
    return (1.0 + cr1) * ratio * (1.0 - cr3 * np.arctan(cr2 * np.asarray(r_tilde, dtype=float))) - cr1


def f_r1(f_rot, upper=FR1_MAX, lower=FR1_MIN):
    """max{min(f_rotation, 1.25), 0.0}, [SM09] Eq. (4), p. 041010-2."""
    return np.maximum(np.minimum(np.asarray(f_rot, dtype=float), upper), lower)


@dataclass(frozen=True)
class Result:
    S: np.ndarray          # sqrt(2 S_ij S_ij)
    W: np.ndarray          # sqrt(2 Omega_ij Omega_ij), absolute rotation
    D: np.ndarray          # sqrt(max(S^2, 0.09 omega^2))
    numerator: np.ndarray  # numerator of Eq. (6)
    rStar: np.ndarray
    rTilde: np.ndarray
    fRotation: np.ndarray
    fr1: np.ndarray

    def as_dict(self):
        return {k: getattr(self, k) for k in ('S', 'W', 'D', 'numerator', 'rStar', 'rTilde', 'fRotation', 'fr1')}


def evaluate(A, DSDt, Om, omega, *, cr1=CR1, cr2=CR2, cr3=CR3, upper=FR1_MAX, lower=FR1_MIN,
             variant='smirnov2009'):
    """All quantities of the correction at the given states (broadcast over leading axes).

    ``variant="smirnov2009"`` is the model ([SM09] Eqs. (1), (4)-(11)). ``variant="shur2000"``
    is the SARC normalisation of [S00] p. 785 (D^2 = (S^2 + Omega^2)/2, denominator D^4, no
    limiter; pass cr2=12 for the published SARC constant) and is provided for comparison only.
    No floors are applied: Omega = 0 gives r* = inf and r~ = nan (undefined), D = 0 likewise.
    """
    A = np.asarray(A, dtype=float)
    S_t = strain_rate(A)
    S = magnitude(S_t)
    W = magnitude(rotation_rate_absolute(A, Om))
    N = r_tilde_numerator(A, DSDt, Om)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_star = np.where(W > 0, S / np.where(W > 0, W, 1.0), np.inf)
        if variant == 'smirnov2009':
            D = d_scale(S, omega)
            r_t = N / (W * D ** 3)
        elif variant == 'shur2000':
            D = np.sqrt(0.5 * (S ** 2 + W ** 2))
            r_t = N / D ** 4
        else:
            raise ValueError(f'unknown variant {variant!r}')
    f_rot = f_rotation(r_star, r_t, cr1, cr2, cr3)
    fr = f_r1(f_rot, upper, lower) if variant == 'smirnov2009' else f_rot
    return Result(S=S, W=W, D=D, numerator=N, rStar=r_star, rTilde=r_t, fRotation=f_rot, fr1=fr)


# ---------------------------------------------------------------------------------------------
# Independent cross-check formulas from [SS97]; used by the tests, not by evaluate().
# ---------------------------------------------------------------------------------------------
def spalart_shur_e3d(A, DSDt, Om):
    """[SS97] Eq. (4), p. 300, evaluated with explicit index loops as printed:

    e = 1/(2 S_mn S_mn) [dU_i/dx_k - dU_k/dx_i + 2 eps_lki Omega_l] S_jk
        (DS_ij/Dt + Omega_o [eps_iop S_jp + eps_jop S_ip]).

    Pointwise (single state, A 3x3). [SM09]'s numerator equals 2 S_mn S_mn e = S^2 e."""
    A = np.asarray(A, dtype=float)
    D_ = np.asarray(DSDt, dtype=float)
    Om = np.asarray(Om, dtype=float)
    S = 0.5 * (A + A.T)
    total = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bracket = A[i, k] - A[k, i] + 2.0 * sum(EPS[l, k, i] * Om[l] for l in range(3))
                paren = D_[i, j] + sum(Om[o] * (EPS[i, o, p] * S[j, p] + EPS[j, o, p] * S[i, p])
                                       for o in range(3) for p in range(3))
                total += bracket * S[j, k] * paren
    smn = sum(S[m, n] * S[m, n] for m in range(3) for n in range(3))
    return total / (2.0 * smn)


def spalart_shur_dalpha_dt_2d(S11, S12, DS11, DS12, Omega):
    """[SS97] Eq. (2), p. 299: Dalpha/Dt = Omega + [S11 DS12/Dt - S12 DS11/Dt]/(2(S11^2 + S12^2)),
    the rate of turn of the strain principal axes in an inertial frame, for 2D incompressible
    flow computed in a frame rotating at Omega about z."""
    return Omega + (S11 * DS12 - S12 * DS11) / (2.0 * (S11 ** 2 + S12 ** 2))
