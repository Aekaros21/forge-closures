"""FORGE V3 physical-state features: pressure-gradient, TKE-gradient and system-rotation scalars.

Python twin of the V3 feature block in ``src/kOmegaSSTBasis/kOmegaSSTBasis.C``
(``computeV3Features``). The two implementations must agree to 1e-12 on the same
inputs; ``tests/test_features.py`` pins the definitions and ``tests/foam`` compares
them on solved OpenFOAM fields.

Conventions (all quantities are evaluated from the CURRENT iterate of the coupled
candidate solution, with the same lag as I1..I5/PoE: one SIMPLE iteration):

* ``p`` is the solver's kinematic pressure (p/rho, m^2/s^2) as transported by the
  incompressible solver. ``grad(p)`` is therefore the physical acceleration
  ``(grad p_phys)/rho``. The uniform driving body force of streamwise-periodic cases
  (``meanVelocityForce``) is NOT part of the solved pressure and is not added; in a
  rotating frame ``p`` is the modified pressure that absorbs the centrifugal
  potential. Pressure is only used through its gradient, so any constant offset is
  irrelevant. A dimensional (non-kinematic) pressure field is refused by the solver.
* ``k`` is the transported turbulent kinetic energy (m^2/s^2); ``grad(k)`` has the
  same dimension as an acceleration.
* The scale ``A = omega_s * sqrt(k_s)`` (m/s^2) uses the solver's own floors
  ``omega_s = max(omega, omegaMin)`` and ``k_s = max(k, kMin)``.
* ``Shat = dev(symm(grad U))/omega_s`` is the existing dimensionless strain tensor.
* ``Omega`` is the system (frame) angular velocity taken from the solver
  configuration (``frameOmega`` in kOmegaSSTBasisCoeffs, cross-checked against the
  rotating-frame source in fvOptions); it is exactly zero for inertial cases.
* ``zeta = curl(U)`` is the RELATIVE vorticity of the solved velocity field. The
  existing rotation tensor W (and I2, T2, T4, ...) already contains the frame
  rotation through ``W_abs``; the features below add the explicit, signed
  frame/relative relationship without redefining W or I2.

Dimensionless inputs (index notation, all rotation invariant):

    a   = grad(p) / A            g = grad(k) / A
    ohat = Omega / omega_s       zhat = curl(U) / omega_s

Features (``EPS = 1e-3`` is a fixed, declared regularisation; every zero-gradient
limit is finite and equal to zero):

    Gp  = |a| / (1 + |a|)                       pressure-gradient strength, [0, 1)
    Gk  = |g| / (1 + |g|)                       TKE-gradient strength, [0, 1)
    Apk = (a . g) / (|a| |g| + EPS)             alignment of grad p with grad k, [-1, 1]
    Psn = (a . Shat . a) / (|a|^2 + EPS)        normal strain rate along grad p
    Ksn = (g . Shat . g) / (|g|^2 + EPS)        normal strain rate along grad k
    Rf  = |ohat|                                system-rotation strength, >= 0
    Rw  = (ohat . zhat) / (|ohat| |zhat| + EPS) signed frame/relative-vorticity alignment, [-1, 1]

Rw is positive where the relative vorticity is parallel to the frame rotation
(stabilised, suction side of a spanwise-rotating channel) and negative where it is
anti-parallel (destabilised, pressure side); it is identically zero in a
non-rotating configuration and tends to zero where the relative vorticity vanishes.
"""
from __future__ import annotations

import numpy as np

EPS = 1.0e-3
NAMES = ("Gp", "Gk", "Apk", "Psn", "Ksn", "Rf", "Rw")
INPUT_NAMES = ("a", "g", "ohat", "zhat")
#: Bumped whenever any definition above changes; enters the evaluator/cache identity.
FEATURE_VERSION = "forge-v3-state-features-1"

_LEVI_CIVITA = np.zeros((3, 3, 3))
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _LEVI_CIVITA[_i, _j, _k] = 1.0
    _LEVI_CIVITA[_i, _k, _j] = -1.0


def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("...i,...i->...", v, v))


def dimensionless_features(a, g, shat, ohat, zhat) -> dict[str, np.ndarray]:
    """The seven features from dimensionless inputs (shapes (N,3), (N,3,3), (N,3), (N,3))."""
    a = np.asarray(a, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    shat = np.asarray(shat, dtype=np.float64)
    ohat = np.asarray(ohat, dtype=np.float64)
    zhat = np.asarray(zhat, dtype=np.float64)
    ma, mg = _norm(a), _norm(g)
    mo, mz = _norm(ohat), _norm(zhat)
    return {
        "Gp": ma / (1.0 + ma),
        "Gk": mg / (1.0 + mg),
        "Apk": np.einsum("...i,...i->...", a, g) / (ma * mg + EPS),
        "Psn": np.einsum("...i,...ij,...j->...", a, shat, a) / (ma * ma + EPS),
        "Ksn": np.einsum("...i,...ij,...j->...", g, shat, g) / (mg * mg + EPS),
        "Rf": mo,
        "Rw": np.einsum("...i,...i->...", ohat, zhat) / (mo * mz + EPS),
    }


def dimensionless_inputs(grad_p, grad_k, k, omega, omega_frame, curl_u, *, k_min, omega_min):
    """Dimensionless input vectors from dimensional solver fields (the solver's exact recipe).

    grad_p, grad_k, curl_u: (N,3); k, omega: (N,); omega_frame: (3,) or (N,3).
    """
    k_s = np.maximum(np.asarray(k, dtype=np.float64), float(k_min))
    omega_s = np.maximum(np.asarray(omega, dtype=np.float64), float(omega_min))
    scale = (omega_s * np.sqrt(k_s))[..., None]
    omega_frame = np.broadcast_to(np.asarray(omega_frame, dtype=np.float64), np.shape(grad_p))
    return {
        "a": np.asarray(grad_p, dtype=np.float64) / scale,
        "g": np.asarray(grad_k, dtype=np.float64) / scale,
        "ohat": omega_frame / omega_s[..., None],
        "zhat": np.asarray(curl_u, dtype=np.float64) / omega_s[..., None],
    }


def from_fields(grad_p, grad_k, k, omega, shat, omega_frame, curl_u, *, k_min, omega_min):
    """Features from dimensional fields; ``shat`` is the solver's dev(symm(grad U))/omega_s."""
    inputs = dimensionless_inputs(grad_p, grad_k, k, omega, omega_frame, curl_u,
                                  k_min=k_min, omega_min=omega_min)
    return dimensionless_features(inputs["a"], inputs["g"], shat, inputs["ohat"], inputs["zhat"])


def axial_vector(what: np.ndarray) -> np.ndarray:
    """Dimensionless ABSOLUTE vorticity zhat_abs from the solver's rotation tensor.

    The solver builds ``What_ij = (skew(gradU)_ij + eps_ijk Omega_k)/omega_s`` with
    OpenFOAM's ``grad(U)_ij = dU_j/dx_i``, i.e. ``What_ij = 1/2 eps_ijk (zeta + 2 Omega)_k/omega_s``.
    Hence ``zhat_abs,k = sum_ij eps_ijk What_ij``.
    """
    return np.einsum("ijk,...ij->...k", _LEVI_CIVITA, np.asarray(what, dtype=np.float64))


def relative_vorticity(what: np.ndarray, ohat: np.ndarray) -> np.ndarray:
    """zhat_rel = zhat_abs - 2 ohat, consistent with a What that already contains the frame."""
    return axial_vector(what) - 2.0 * np.asarray(ohat, dtype=np.float64)


def sample_inputs(rng: np.random.Generator, what: np.ndarray, *, rotating_fraction: float = 0.5,
                  magnitude_range=(1.0e-3, 30.0), frame_range=(1.0e-3, 3.0)) -> dict[str, np.ndarray]:
    """Admissible random feature inputs consistent with a batch of rotation tensors.

    Pressure/TKE gradient directions are isotropic with log-uniform dimensionless
    magnitudes; a fraction of states carries a frame rotation (log-uniform strength,
    isotropic axis) and the relative vorticity is derived from the SAME ``what`` the
    tensors use, so no impossible (what, ohat, zhat) combination is generated. A
    trailing sample set: legacy draws from the caller's stream must precede this call.
    """
    n = int(np.shape(what)[0])

    def directions(count):
        v = rng.normal(size=(count, 3))
        return v / np.maximum(_norm(v), 1e-300)[:, None]

    def magnitudes(count, low, high):
        return 10.0 ** rng.uniform(np.log10(low), np.log10(high), size=count)

    a = directions(n) * magnitudes(n, *magnitude_range)[:, None]
    g = directions(n) * magnitudes(n, *magnitude_range)[:, None]
    ohat = directions(n) * magnitudes(n, *frame_range)[:, None]
    ohat[rng.uniform(size=n) >= rotating_fraction] = 0.0
    zhat = relative_vorticity(what, ohat)
    return {"a": a, "g": g, "ohat": ohat, "zhat": zhat}


def zero_inputs(n: int) -> dict[str, np.ndarray]:
    """Inputs of a state with no pressure/TKE gradient and no system rotation."""
    zeros = np.zeros((int(n), 3))
    return {name: zeros.copy() for name in INPUT_NAMES}


def scaled_inputs(inputs: dict[str, np.ndarray], gradient_scale: float) -> dict[str, np.ndarray]:
    """The inputs of the SST-recovery limit: the mean velocity gradient (relative vorticity)
    and the frame rotation vanish together with Shat/What; pressure and TKE gradients may
    persist (a fluid at rest can still carry them)."""
    return {"a": np.asarray(inputs["a"], dtype=np.float64),
            "g": np.asarray(inputs["g"], dtype=np.float64),
            "ohat": np.asarray(inputs["ohat"], dtype=np.float64) * gradient_scale,
            "zhat": np.asarray(inputs["zhat"], dtype=np.float64) * gradient_scale}


def rotate_inputs(inputs: dict[str, np.ndarray], q: np.ndarray) -> dict[str, np.ndarray]:
    """Apply a proper rotation q (3x3) to every input vector; features must be unchanged."""
    return {name: np.einsum("ij,...j->...i", q, np.asarray(value, dtype=np.float64))
            for name, value in inputs.items()}
