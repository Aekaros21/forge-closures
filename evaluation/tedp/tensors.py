"""Small, explicit tensor utilities and conventions used by every experiment."""

from __future__ import annotations

import numpy as np

I3 = np.eye(3)


def sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + np.swapaxes(a, -1, -2))


def dev(a: np.ndarray) -> np.ndarray:
    """Symmetric deviatoric part of a tensor or a stack of tensors."""
    s = sym(a)
    tr = np.trace(s, axis1=-2, axis2=-1)[..., None, None]
    return s - tr * I3 / 3.0


def split_gradient(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean strain S and mean rotation W for A_ij=dU_i/dx_j."""
    return sym(a), 0.5 * (a - np.swapaxes(a, -1, -2))


def turbulent_kinetic_energy(r: np.ndarray) -> np.ndarray:
    return 0.5 * np.trace(r, axis1=-2, axis2=-1)


def normalized_stress(r: np.ndarray) -> np.ndarray:
    """Return R/tr(R), using the isotropic representative at R=0.

    The anisotropy of an exactly quiescent state is undefined, but closures in
    this project have a unique zero-output limit when both R and epsilon vanish.
    Assigning I/3 here makes that removable numerical limit explicit.  A
    nonzero tensor with zero trace remains invalid.
    """
    r = np.asarray(r, dtype=float)
    k = turbulent_kinetic_energy(r)
    if np.any(k < 0.0):
        raise ValueError("Reynolds stress cannot have negative trace")
    zero = k == 0.0
    if np.any(zero & (np.linalg.norm(r, axis=(-2, -1)) != 0.0)):
        raise ValueError("a nonzero Reynolds stress cannot have zero trace")
    denominator = np.where(zero, 1.0, 2.0 * k)
    out = r / denominator[..., None, None]
    return np.where(zero[..., None, None], I3 / 3.0, out)


def anisotropy(r: np.ndarray) -> np.ndarray:
    return normalized_stress(r) - I3 / 3.0


def anisotropy_invariants(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = anisotropy(r)
    ii = np.einsum("...ij,...ji->...", b, b)
    iii = np.einsum("...ij,...jk,...ki->...", b, b, b)
    return ii, iii


def barycentric_coordinates(r: np.ndarray) -> np.ndarray:
    """Weights of one-, two-, and three-component limiting states.

    Input must be positive semidefinite.  Returned ordering is (C1,C2,C3).
    """
    lam = np.linalg.eigvalsh(anisotropy(r))[..., ::-1]
    return np.stack((lam[..., 0] - lam[..., 1],
                     2.0 * (lam[..., 1] - lam[..., 2]),
                     3.0 * lam[..., 2] + 1.0), axis=-1)


def psd_margin(r: np.ndarray) -> np.ndarray:
    """Smallest eigenvalue of R/(2k); zero is the realizability boundary."""
    return np.linalg.eigvalsh(normalized_stress(r))[..., 0]


def symmetric_components(a: np.ndarray) -> np.ndarray:
    """Six-component isometric vectorization of a symmetric tensor."""
    return np.stack((a[..., 0, 0], a[..., 1, 1], a[..., 2, 2],
                     np.sqrt(2.0) * a[..., 0, 1],
                     np.sqrt(2.0) * a[..., 0, 2],
                     np.sqrt(2.0) * a[..., 1, 2]), axis=-1)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    # NumPy's QR sign convention is deterministic (for example q[0, 0] is
    # non-positive), so Q is not Haar distributed unless the arbitrary column
    # signs are restored from R.  This matters both for benchmark orientations
    # and for adversarial covariance sampling.
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    q *= signs[None, :]
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def rotate_tensor(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    return q @ a @ q.T


# --------------------------------------------------------------------------
# Integrity basis (added for the two-equation discovery project; everything
# above is verbatim from the archived pressure-strain repo).


def integrity_basis(s_hat: np.ndarray, w_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pope's ten basis tensors and five invariants of (S_hat, W_hat).

    s_hat, w_hat: (..., 3, 3) normalized strain (symmetric, traceless) and
    rotation (antisymmetric) tensors. Returns (T, inv) with T of shape
    (..., 10, 3, 3) and inv of shape (..., 5) ordered
    I1=tr(S^2), I2=tr(W^2), I3=tr(S^3), I4=tr(W^2 S), I5=tr(W^2 S^2).

    The C++ twin is src/kOmegaSSTBasis/basisTensors — formulas must match.
    """
    s = np.asarray(s_hat, dtype=np.float64)
    w = np.asarray(w_hat, dtype=np.float64)
    s2 = s @ s
    w2 = w @ w
    sw = s @ w
    ws = w @ s

    def tr(a: np.ndarray) -> np.ndarray:
        return np.trace(a, axis1=-2, axis2=-1)[..., None, None]

    t1 = s
    t2 = sw - ws
    t3 = s2 - tr(s2) * I3 / 3.0
    t4 = w2 - tr(w2) * I3 / 3.0
    t5 = w @ s2 - s2 @ w
    t6 = w2 @ s + s @ w2 - 2.0 * tr(s @ w2) * I3 / 3.0
    t7 = w @ s @ w2 - w2 @ s @ w
    t8 = s @ w @ s2 - s2 @ w @ s
    t9 = w2 @ s2 + s2 @ w2 - 2.0 * tr(s2 @ w2) * I3 / 3.0
    t10 = w @ s2 @ w2 - w2 @ s2 @ w

    T = np.stack([t1, t2, t3, t4, t5, t6, t7, t8, t9, t10], axis=-3)
    inv = np.stack(
        [
            tr(s2)[..., 0, 0],
            tr(w2)[..., 0, 0],
            tr(s2 @ s)[..., 0, 0],
            tr(w2 @ s)[..., 0, 0],
            tr(w2 @ s2)[..., 0, 0],
        ],
        axis=-1,
    )
    return T, inv
