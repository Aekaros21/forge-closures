"""Tier 1: second-scale physics gates.

channel1d: wall-resolved 1-D k-omega SST channel at Re_tau (default 5200),
with the candidate's b^Delta and R channels wired in exactly as the 3-D
solver wires them (same numpy evaluator, same clamps). Hard gates compare
against the SST-1D baseline: C_f (via bulk velocity) within 2%, log-law
slope sane, solver converged.

Everything is in wall units: half-width delta = 1, u_tau = 1, nu = 1/Re_tau.
The wall shear is fixed by the imposed pressure gradient, so C_f differences
appear through the bulk velocity: Cf = 2 (u_tau/U_b)^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import candidate as cand
from . import features
from .spec import CandidateSpec

# SST constants (Menter 2003)
BETA_STAR = 0.09
A1 = 0.31
SIGMA_K1, SIGMA_K2 = 0.85, 1.0
SIGMA_W1, SIGMA_W2 = 0.5, 0.856
BETA_1, BETA_2 = 0.075, 0.0828
GAMMA_1 = BETA_1 / BETA_STAR - SIGMA_W1 * 0.41**2 / np.sqrt(BETA_STAR)
GAMMA_2 = BETA_2 / BETA_STAR - SIGMA_W2 * 0.41**2 / np.sqrt(BETA_STAR)


def _blend(f1: np.ndarray, v1: float, v2: float) -> np.ndarray:
    return f1 * v1 + (1.0 - f1) * v2


def _stretched_grid(n: int, retau: float) -> np.ndarray:
    """Node coordinates in [0, 1], first spacing ~0.3 wall units."""
    target_dy1 = 0.3 / retau
    # geometric ratio r solving (r-1)/(r^n - 1) = target_dy1 via bisection
    lo, hi = 1.0001, 1.2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        dy1 = (mid - 1.0) / (mid ** (n - 1) - 1.0)
        if dy1 > target_dy1:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    y = np.concatenate([[0.0], np.cumsum(r ** np.arange(n - 1))])
    return y / y[-1]


def _tridiag(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm; a is sub-, b main, c super-diagonal."""
    n = len(b)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


@dataclass
class Channel1DResult:
    converged: bool
    y: np.ndarray          # wall-normal coordinate, [0, 1]
    yplus: np.ndarray
    uplus: np.ndarray
    k: np.ndarray
    omega: np.ndarray
    nut: np.ndarray
    u_bulk: float
    cf: float              # 2 (u_tau / U_b)^2
    loglaw_slope: float    # dU+/d ln y+ averaged over the log region
    iterations: int

    @property
    def kappa_eff(self) -> float:
        return 1.0 / self.loglaw_slope if self.loglaw_slope > 0 else float("inf")


def _diffusion_matrix(y: np.ndarray, gamma_face: np.ndarray):
    """Second-order conservative diffusion d/dy(gamma d./dy) on a nonuniform
    grid; returns (sub, main, super) for interior nodes 1..n-2."""
    dy = np.diff(y)
    sub = gamma_face[:-1] / (dy[:-1] * 0.5 * (dy[:-1] + dy[1:]))
    sup = gamma_face[1:] / (dy[1:] * 0.5 * (dy[:-1] + dy[1:]))
    return sub, -(sub + sup), sup


def solve_channel1d(
    spec: CandidateSpec | None = None,
    retau: float = 5200.0,
    n: int = 192,
    max_iters: int = 40000,
    tol: float = 1.0e-9,
) -> Channel1DResult:
    nu = 1.0 / retau
    y = _stretched_grid(n, retau)
    dy = np.diff(y)
    yc = y.copy()
    yc[0] = y[1] * 1.0e-3  # avoid /0 in wall-distance formulas at the wall node

    # initial guesses: log-law-ish U, k from a crude profile, omega blend
    yplus = y * retau
    u = np.where(yplus < 11.0, yplus, (1.0 / 0.41) * np.log(np.maximum(yplus, 1.0)) + 5.2)
    k = np.maximum(0.01, 1.0 / np.sqrt(BETA_STAR) * np.minimum(1.0, yplus / 30.0) ** 2)
    k[0] = 0.0
    omega_wall = 10.0 * 6.0 * nu / (BETA_1 * dy[0] ** 2)
    omega = np.maximum(1.0 / (0.41 * np.maximum(yc, 1e-8)) / np.sqrt(BETA_STAR), 1.0)
    omega[0] = omega_wall

    dt = 0.5
    iterations = 0
    # deferred-correction relaxation of the candidate channels (PoE-dependent
    # forms like WJ oscillate under an unrelaxed lagged feedback), mirroring
    # the 3-D model's bDeltaRelax
    corr_relax = 0.5  # generated OpenFOAM dictionaries use bDeltaRelax 0.5
    bdelta_xy_prev = np.zeros(n)
    r_src_prev = np.zeros(n)
    old_numpy_errors = np.seterr(all="ignore")  # restored before returning
    for it in range(max_iters):
        iterations = it + 1
        dudy = np.gradient(u, y)
        smag = np.abs(dudy)

        # blending functions
        with np.errstate(divide="ignore", over="ignore"):
            dkdy = np.gradient(k, y)
            dwdy = np.gradient(omega, y)
            cd_kw = np.maximum(2.0 * SIGMA_W2 / np.maximum(omega, 1e-30) * dkdy * dwdy, 1.0e-10)
            arg1 = np.minimum(
                np.maximum(
                    np.sqrt(np.maximum(k, 0.0)) / (BETA_STAR * omega * yc),
                    500.0 * nu / (yc**2 * omega),
                ),
                4.0 * SIGMA_W2 * k / (cd_kw * yc**2),
            )
            f1 = np.tanh(np.minimum(arg1, 20.0) ** 4)
            arg2 = np.maximum(
                2.0 * np.sqrt(np.maximum(k, 0.0)) / (BETA_STAR * omega * yc),
                500.0 * nu / (yc**2 * omega),
            )
            f2 = np.tanh(np.minimum(arg2, 20.0) ** 2)

        nut = A1 * k / np.maximum(A1 * omega, f2 * smag)
        nut = np.clip(nut, 0.0, 1.0)  # generous cap in wall units

        # candidate corrections on the 1-D shear state
        bdelta_xy = np.zeros(n)
        r_src = np.zeros(n)
        if spec is not None and (spec.bdelta or spec.rsource):
            om_safe = np.maximum(omega, 1e-12)
            grad = np.zeros((n, 3, 3))
            grad[:, 0, 1] = dudy
            shat = (0.5 * (grad + np.swapaxes(grad, 1, 2))) / om_safe[:, None, None]
            what = (0.5 * (grad - np.swapaxes(grad, 1, 2))) / om_safe[:, None, None]
            prod = nut * smag**2
            # V3 state of the 1-D channel: the only resolved gradient is dk/dy
            # (the driving pressure gradient is an imposed body force, not part
            # of the solved pressure, and there is no frame rotation).
            grad_k = np.zeros((n, 3))
            grad_k[:, 1] = np.gradient(k, y)
            v3_inputs = features.dimensionless_inputs(
                np.zeros((n, 3)), grad_k, k, omega, np.zeros(3), np.zeros((n, 3)),
                k_min=1e-14, omega_min=1e-12)
            states = cand.make_states(
                shat,
                what,
                ret=k / np.maximum(nu * om_safe, 1e-30),
                f1=f1,
                poe=np.minimum(prod / np.maximum(BETA_STAR * k * om_safe, 1e-30), cand.PoE_CLAMP),
                inputs=v3_inputs,
            )
            ramp = min(1.0, (it + 1) / 200.0)
            if spec.bdelta:
                bdelta_xy = ramp * cand.eval_bdelta(spec, states)[:, 0, 1]
            if spec.rsource:
                bR = cand.eval_rsource_bR(spec, states)
                r_src = 2.0 * k * np.einsum("nij,nij->n", bR, grad)
                bound = spec.r_max_factor * BETA_STAR * k * om_safe
                r_src = ramp * np.clip(r_src, -bound, bound)
            bdelta_xy = (1 - corr_relax) * bdelta_xy_prev + corr_relax * bdelta_xy
            r_src = (1 - corr_relax) * r_src_prev + corr_relax * r_src
            bdelta_xy_prev, r_src_prev = bdelta_xy, r_src

        # ---- momentum: (nu+nut) dU/dy = (1 - y) + 2 k bDelta_xy
        dudy_new = ((1.0 - y) + 2.0 * k * bdelta_xy) / (nu + nut)
        u_new = np.concatenate([[0.0], np.cumsum(0.5 * (dudy_new[1:] + dudy_new[:-1]) * dy)])

        # ---- k equation (implicit diffusion + destruction)
        prod_k = np.minimum(nut * smag**2 - 2.0 * k * bdelta_xy * dudy, 10.0 * BETA_STAR * k * omega)
        prod_k = prod_k + r_src
        sigk = _blend(f1, SIGMA_K1, SIGMA_K2)
        gam_k = nu + sigk * nut
        gkf = 0.5 * (gam_k[:-1] + gam_k[1:])
        sub, main, sup = _diffusion_matrix(y, gkf)
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)
        d = np.zeros(n)
        a[1:-1] = -dt * sub
        b[1:-1] = 1.0 + dt * (-main + BETA_STAR * omega[1:-1])
        c[1:-1] = -dt * sup
        d[1:-1] = k[1:-1] + dt * prod_k[1:-1]
        b[0], d[0] = 1.0, 0.0                     # k_wall = 0
        a[-1], b[-1], d[-1] = -1.0, 1.0, 0.0      # symmetry: dk/dy = 0
        k_new = np.maximum(_tridiag(a, b, c, d), 1.0e-14)

        # ---- omega equation
        gamma = _blend(f1, GAMMA_1, GAMMA_2)
        sigw = _blend(f1, SIGMA_W1, SIGMA_W2)
        beta = _blend(f1, BETA_1, BETA_2)
        nut_safe = np.maximum(nut, 1.0e-3 * nu)
        base_w = (
            np.minimum(nut * smag**2, 10.0 * BETA_STAR * k * omega)
            / np.maximum(nut, 1.0e-30)
        )
        extra_w = (-2.0 * k * bdelta_xy * dudy + r_src) / nut_safe
        prod_w = gamma * (base_w + extra_w)
        cross = 2.0 * (1.0 - f1) * SIGMA_W2 / np.maximum(omega, 1e-30) * dkdy * dwdy
        gam_w = nu + sigw * nut
        gwf = 0.5 * (gam_w[:-1] + gam_w[1:])
        sub, main, sup = _diffusion_matrix(y, gwf)
        a = np.zeros(n)
        b = np.ones(n)
        c = np.zeros(n)
        d = np.zeros(n)
        a[1:-1] = -dt * sub
        b[1:-1] = 1.0 + dt * (-main + beta[1:-1] * omega[1:-1])
        c[1:-1] = -dt * sup
        d[1:-1] = omega[1:-1] + dt * (prod_w[1:-1] + np.maximum(cross[1:-1], 0.0)
                                      + np.minimum(cross[1:-1], 0.0))
        b[0], d[0] = 1.0, omega_wall
        a[-1], b[-1], d[-1] = -1.0, 1.0, 0.0
        omega_new = np.maximum(_tridiag(a, b, c, d), 1.0e-8)

        if not (np.all(np.isfinite(u_new)) and np.all(np.isfinite(k_new))
                and np.all(np.isfinite(omega_new))):
            # blown up (unbounded candidate): stop, report non-convergence
            du = dk = dw = float("inf")
            u = np.full_like(u, np.nan)
            break
        du = np.max(np.abs(u_new - u)) / max(np.max(np.abs(u)), 1.0)
        dk = np.max(np.abs(k_new - k)) / max(np.max(k), 1e-10)
        dw = np.max(np.abs(omega_new - omega) / np.maximum(omega, 1e-8))
        relax = 0.7
        u = u_new
        k = (1 - relax) * k + relax * k_new
        omega = (1 - relax) * omega + relax * omega_new

        if max(du, dk, dw) < tol and it > 100:
            break

    converged = max(du, dk, dw) < 10 * tol and bool(np.all(np.isfinite(u)))
    u_bulk = float(np.trapezoid(u, y))
    cf = 2.0 / u_bulk**2

    # log-law slope over 100 < y+ < 0.1 Re_tau
    yp = y * retau
    mask = (yp > 100.0) & (yp < 0.1 * retau)
    slope = float(
        np.polyfit(np.log(yp[mask]), u[mask], 1)[0]
    ) if np.count_nonzero(mask) > 5 else float("nan")

    np.seterr(**old_numpy_errors)
    return Channel1DResult(
        converged=converged,
        y=y,
        yplus=yp,
        uplus=u,
        k=k,
        omega=omega,
        nut=nut,
        u_bulk=u_bulk,
        cf=cf,
        loglaw_slope=slope,
        iterations=iterations,
    )


@dataclass
class Tier1Gates:
    passed: bool
    reasons: list[str]
    cf_ratio: float
    kappa_eff: float
    homogeneous_error: float = float("nan")
    homogeneous_stock_error: float = float("nan")
    homogeneous_max_realizability: float = float("nan")
    homogeneous_components: dict[str, dict[str, float]] = field(default_factory=dict)


CF_TOL = 0.02


def channel_gates(
    spec: CandidateSpec, baseline: Channel1DResult, retau: float = 5200.0
) -> Tier1Gates:
    """Hard Tier-1 gate: the candidate's 1-D channel must converge and keep
    C_f within CF_TOL of the SST-1D baseline."""
    reasons: list[str] = []
    result = solve_channel1d(spec, retau=retau)
    if not result.converged:
        reasons.append("channel1d: not converged")
        return Tier1Gates(False, reasons, float("inf"), float("nan"))
    ratio = result.cf / baseline.cf
    if abs(ratio - 1.0) > CF_TOL:
        reasons.append(f"channel1d: Cf ratio {ratio:.4f} outside ±{CF_TOL:.0%}")
    if not (0.3 < result.kappa_eff < 0.6):
        reasons.append(f"channel1d: effective kappa {result.kappa_eff:.3f} unphysical")
    return Tier1Gates(not reasons, reasons, ratio, result.kappa_eff)


# ---------------------------------------------------------------------------
# Homogeneous-flow ODE gates (decay, shear) — far-from-wall limit: F1 = 0,
# F2 = 0, nut = k/omega. Mean gradient A is constant; everything is algebraic
# in the one-point state, so the candidate channels enter exactly as in the
# 3-D solver.

from scipy.integrate import solve_ivp  # noqa: E402


def _homogeneous_observables(k, omega, A, spec, nu):
    """One homogeneous state with the same algebraic limits as deployment."""
    k = max(float(k), 1e-12)
    omega = max(float(omega), 1e-8)
    nut = k / omega

    S = 0.5 * (A + A.T)
    p_base = 2.0 * nut * float(np.sum(S * S))
    p_stress = p_base
    r_src = 0.0
    bdelta = np.zeros((3, 3))

    if spec is not None and (spec.bdelta or spec.rsource):
        shat = S[None, :, :] / omega
        what = (0.5 * (A - A.T))[None, :, :] / omega
        states = cand.make_states(
            shat,
            what,
            ret=np.array([k / (nu * omega)]),
            f1=np.array([0.0]),
            poe=np.array([
                min(p_base / (BETA_STAR * k * omega), cand.PoE_CLAMP)
            ]),
        )
        if spec.bdelta:
            bdelta = cand.eval_bdelta(spec, states)[0]
            p_stress += -2.0 * k * float(np.sum(bdelta * A))
        if spec.rsource:
            bR = cand.eval_rsource_bR(spec, states)[0]
            r_src = 2.0 * k * float(np.sum(bR * A))
            bound = spec.r_max_factor * BETA_STAR * k * omega
            r_src = float(np.clip(r_src, -bound, bound))

    # kOmegaSSTBasis::Pk limits the constitutive production before the
    # separately bounded R source is added.
    p_limited = min(p_stress, 10.0 * BETA_STAR * k * omega)
    b_total = -(nut / k) * S + bdelta
    realization = float(cand.realizability_violation(b_total[None])[0])
    return p_stress, p_limited, r_src, b_total, realization


def _homogeneous_rhs(t, state, A, spec, nu=1.0e-4):
    k, omega = state
    if not np.isfinite(k) or not np.isfinite(omega) or k <= 0.0 or omega <= 0.0:
        return [np.nan, np.nan]
    nut = k / omega
    p_stress, p_limited, r_src, _, _ = _homogeneous_observables(
        k, omega, A, spec, nu
    )

    dk = p_limited + r_src - BETA_STAR * k * omega
    # Deployment uses two different limiter paths.  Pk() limits the combined
    # constitutive production in the k equation, whereas BaseType's omega
    # production limits only the Boussinesq G and omegaSource() then adds
    # (Gnl + R)/nutSafe without applying Pk's limiter a second time.
    S = 0.5 * (A + A.T)
    p_base = 2.0 * nut * float(np.sum(S * S))
    g_nl = p_stress - p_base
    p_base_limited = min(p_base, 10.0 * BETA_STAR * k * omega)
    nut_safe = max(nut, 1.0e-3 * nu)
    # Homogeneous states are fully turbulent (F1=0 -> gamma2, beta2).
    dw = GAMMA_2 * (
        p_base_limited / max(nut, 1.0e-300)
        + (g_nl + r_src) / nut_safe
    ) - BETA_2 * omega**2
    return [dk, dw]


NU_HOM = 1.0e-4  # sets Ret in homogeneous integrations; results are inviscid-scaled


@dataclass
class HomogeneousResult:
    ok: bool
    stress_peq: float   # physical stress production / epsilon
    total_peq: float    # (limited P + clipped R) / epsilon (diagnostic only)
    r_over_eps: float
    growth_rate: float  # d ln(k) / d(S t)
    decay_n: float      # decay exponent k ~ t^-n (decay) or nan
    b12: float
    realizability: float

    @property
    def peq(self) -> float:
        """Compatibility alias: the non-tautological stress production."""
        return self.stress_peq


def solve_homogeneous_shear(
    spec: CandidateSpec | None, s_norm: float = 1.0, st_end: float = 20.0
) -> HomogeneousResult:
    A = np.zeros((3, 3))
    A[0, 1] = s_norm
    k0, w0 = 1.0, s_norm / 3.0   # moderate initial S k / eps
    sol = solve_ivp(
        _homogeneous_rhs, (0.0, st_end / s_norm), [k0, w0],
        args=(A, spec), method="RK45", rtol=1e-8, atol=1e-12, dense_output=False,
    )
    if not sol.success or not np.all(np.isfinite(sol.y)):
        return HomogeneousResult(
            False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        )
    k, w = sol.y[0, -1], sol.y[1, -1]
    p_stress, p_limited, r_src, b_total, realization = (
        _homogeneous_observables(k, w, A, spec, NU_HOM)
    )
    eps = BETA_STAR * k * w
    growth = (p_limited + r_src - eps) / (k * s_norm)
    return HomogeneousResult(
        True,
        p_stress / eps,
        (p_limited + r_src) / eps,
        r_src / eps,
        growth,
        np.nan,
        float(b_total[0, 1]),
        realization,
    )


def solve_homogeneous_decay(spec: CandidateSpec | None) -> HomogeneousResult:
    """k ~ t^-n asymptotically; the fit needs beta2*omega0*t >> 1, so
    integrate to t = 1e4 and fit the last decade in log t."""
    A = np.zeros((3, 3))
    sol = solve_ivp(
        _homogeneous_rhs, (0.0, 1.0e4), [1.0, 1.0],
        args=(A, spec), method="LSODA", rtol=1e-10, atol=1e-14,
    )
    if not sol.success:
        return HomogeneousResult(
            False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        )
    mask = sol.t > 1.0e3
    t = sol.t[mask]
    k = sol.y[0, mask]
    n = -float(np.polyfit(np.log(t), np.log(np.maximum(k, 1e-300)), 1)[0])
    return HomogeneousResult(
        True, np.nan, np.nan, np.nan, np.nan, n, np.nan, np.nan
    )


HOM23_DIR = Path(__file__).resolve().parents[2] / "data" / "external" / "agard_HOM23"
HOM23_SERIES = (
    ("hom23xU.dat", 14.142135, 0.005),
    ("hom23uU.dat", 28.284270, 0.010),
    ("hom23wU.dat", 56.568540, 0.020),
)


@dataclass
class HomogeneousDNSScore:
    ok: bool
    error: float
    max_realizability: float
    reasons: list[str]
    components: dict[str, dict[str, float]] = field(default_factory=dict)


def _load_hom23(path: Path) -> np.ndarray:
    # Five text header rows, then: St, uu, vv, ww, uv, epsilon.
    return np.loadtxt(path, skiprows=5)


def homogeneous_dns_score(spec: CandidateSpec | None) -> HomogeneousDNSScore:
    """Error against the bundled HOM23 homogeneous-shear DNS histories.

    The score combines log histories of k and epsilon with all Reynolds-stress
    anisotropy components present in HOM23 (b11, b22, b33, and b12).  Comparing
    a candidate to stock under the exact same metric avoids inventing another
    endpoint threshold; positivity and full anisotropy realizability remain
    absolute requirements.
    """
    errors: list[float] = []
    reasons: list[str] = []
    components: dict[str, dict[str, float]] = {}
    max_realization = 0.0
    for filename, shear, nu in HOM23_SERIES:
        data = _load_hom23(HOM23_DIR / filename)
        st = data[:, 0]
        k_dns = 0.5 * np.sum(data[:, 1:4], axis=1)
        eps_dns = data[:, 5]
        bdiag_dns = data[:, 1:4] / (2.0 * k_dns[:, None]) - 1.0 / 3.0
        b12_dns = data[:, 4] / (2.0 * k_dns)
        k0 = float(k_dns[0])
        omega0 = float(eps_dns[0] / (BETA_STAR * k0))
        A = np.zeros((3, 3))
        A[0, 1] = shear
        sol = solve_ivp(
            _homogeneous_rhs,
            (0.0, float(st[-1] / shear)),
            [k0, omega0],
            t_eval=st / shear,
            args=(A, spec, nu),
            method="RK45",
            rtol=1.0e-8,
            atol=1.0e-12,
            dense_output=True,
            max_step=0.25 / shear,
        )
        if (
            not sol.success
            or sol.y.shape[1] != len(st)
            or not np.all(np.isfinite(sol.y))
            or np.any(sol.y <= 0.0)
        ):
            reasons.append(f"{filename}: nonpositive/nonfinite trajectory")
            continue
        k_model, omega_model = sol.y
        # DNS records are spaced by Delta(St)=2.  Gate safety on a much finer
        # trajectory grid so a sharp invariant switch cannot hide a transient
        # positivity or realizability failure between observations.
        safety_st = np.arange(st[0], st[-1] + 0.125, 0.25)
        safety_y = sol.sol(safety_st / shear)
        if not np.all(np.isfinite(safety_y)) or np.any(safety_y <= 0.0):
            reasons.append(f"{filename}: unsafe between-sample trajectory")
            continue
        bdiag_model = np.empty((len(st), 3))
        b12_model = np.empty(len(st))
        for i, (k, omega) in enumerate(zip(k_model, omega_model)):
            _, _, _, b_total, realization = _homogeneous_observables(
                k, omega, A, spec, nu
            )
            bdiag_model[i] = np.diag(b_total)
            b12_model[i] = b_total[0, 1]
            max_realization = max(max_realization, realization)
        for k, omega in zip(*safety_y):
            _, _, _, _, realization = _homogeneous_observables(
                k, omega, A, spec, nu
            )
            max_realization = max(max_realization, realization)
        # Skip St<4, where an algebraic-stress model cannot reproduce the
        # DNS's initially isotropic Reynolds-stress transient.
        mask = st >= 4.0
        log_k = np.sqrt(np.mean((
            np.log(k_model[mask] / k0) - np.log(k_dns[mask] / k0)
        ) ** 2))
        eps_model = BETA_STAR * k_model * omega_model
        log_eps = np.sqrt(np.mean((
            np.log(eps_model[mask] / eps_dns[0])
            - np.log(eps_dns[mask] / eps_dns[0])
        ) ** 2))
        # Frobenius RMS over the available symmetric tensor components.  The
        # off-diagonal b12 appears twice in ||b||_F and therefore has weight 2.
        diag_sq = np.sum((bdiag_model[mask] - bdiag_dns[mask]) ** 2, axis=1)
        shear_sq = (b12_model[mask] - b12_dns[mask]) ** 2
        b_err = np.sqrt(np.mean((diag_sq + 2.0 * shear_sq) / 5.0))
        series_error = float((log_k + log_eps + b_err / 0.1) / 3.0)
        errors.append(series_error)
        components[filename] = {
            "log_k": float(log_k),
            "log_epsilon": float(log_eps),
            "anisotropy_frobenius": float(b_err),
            "composite": series_error,
        }

    if max_realization > 1.0e-6:
        reasons.append(
            f"homogeneous anisotropy violation depth {max_realization:.3g}"
        )
    if len(errors) != len(HOM23_SERIES):
        reasons.append("incomplete HOM23 trajectory set")
    return HomogeneousDNSScore(
        not reasons,
        float(np.mean(errors)) if errors else float("inf"),
        max_realization,
        reasons,
        components,
    )


def homogeneous_gates(spec: CandidateSpec) -> Tier1Gates:
    """Absolute homogeneous-trajectory safety plus a reported HOM23 metric.

    Homogeneous decay is retained as a stock-recovery unit diagnostic, not a
    selection gate: with A=0 every integrity-basis correction vanishes and all
    legal candidates necessarily produce the same exponent.  HOM23 fit is not
    a hard non-inferiority gate: the component scale, averaging, transient
    cutoff, and experimental/DNS uncertainty have not been calibrated into a
    defensible acceptance delta.  It is recorded for Pareto analysis.
    """
    reasons: list[str] = []
    base = homogeneous_dns_score(None)
    score = homogeneous_dns_score(spec)
    reasons.extend(score.reasons)
    return Tier1Gates(
        not reasons,
        reasons,
        float("nan"),
        float("nan"),
        score.error,
        base.error,
        score.max_realizability,
        score.components,
    )
