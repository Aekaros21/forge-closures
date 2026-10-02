"""QCR2000 on the SST Reynolds stress, expressed as a basis-grammar spec, and its verification.

Source of the closure (Spalart 2000, IJHFF 21, 252-263, as reproduced by Prudenzano, Gand and Deck
2025, AIAA J. 63(5), p. 1821, Eqs. (5)-(6)):

    tau_QCR_ij = tau_ij - C_cr1 (O_ik tau_jk + O_jk tau_ik),
    O_ik = 2 W_ik / sqrt(dU_m/dx_n dU_m/dx_n),   W_ik = (dU_i/dx_k - dU_k/dx_i)/2,   C_cr1 = 0.3,

with tau the linear (Boussinesq) stress of the host model. The added term is traceless for any
symmetric tau (O_ik tau_ik = 0), and the isotropic part of tau drops out of it (O_ik delta_jk +
O_jk delta_ik = O_ij + O_ji = 0), so it is the same whether tau carries -(2/3) k delta or not.

Applied to the SST stress R = (2/3) k I - nu_t dev(2 S) of OpenFOAM v2312 (R = <u'u'> = -tau), the
correction to R is

    Delta R = C_cr1 (O_ik tau_jk + O_jk tau_ik) = -4 C_cr1 nu_t (S W - W S)/|grad U|,

which is Pope's T2 structure. In the solver's variables (S_hat = dev(S)/omega_s, W_hat_solver =
-W/omega_s, T2 = S_hat W_hat - W_hat S_hat, I1 = tr S_hat^2, I2 = tr W_hat^2 <= 0) and the coupling
tau^Delta = 2 k b^Delta of libkOmegaSSTBasis,

    b^Delta = g2 T2,   g2 = 2 C_cr1 (nu_t omega_s / k) / sqrt(I1 - I2).

nu_t omega_s/k is not a grammar variable, but the production-to-dissipation ratio is:
Pi = 2 nu_t |symm grad U|^2 / (beta* k omega_s) = 2 (nu_t omega_s/k) I1 / beta* for a solenoidal
gradient, so g2 = C_cr1 beta* Pi / (I1 sqrt(I1 - I2)) with beta* = 0.09. PoE is Pi clipped at 10;
above the clip Pi is recovered from the unclipped comparator variable phiDkPk = 1/(1 + Pi):
Pi = PoE + max(0, (1 - phiDkPk)/phiDkPk - 10). This is exact for every value of the SST limiter
(F2 never enters) and of the omega floor (omega_s cancels). The literal 1e12 factors scale
numerator and denominator of the grammar's guarded divisions a/b -> ab/(b^2 + 1e-12); they leave
the algebra unchanged and reduce the guard's relative effect on each division from 1e-12/b^2 to
1e-36/b^2.

Usage (no OpenFOAM, no CFD):
    PYTHONPATH=evaluation:closures/comparators/qcr2000 python closures/comparators/qcr2000/qcr2000_comparator.py --out <dir>
writes <dir>/qcr2000_spec.json and <dir>/qcr2000_verification.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tedp import expr, tensors
from tedp.spec import CandidateSpec, Term, from_json, to_foam_coeffs, to_json

REPO = Path(__file__).resolve().parents[3]   # repository root

C_CR1 = 0.3                 # Spalart (2000); Prudenzano et al. (2025), p. 1821
BETA_STAR = 0.09            # SST beta*, the constant inside PoE and phiDkPk
A1 = 0.31                   # SST a1 (used only to generate realistic nu_t)
SMALL = 1.0e-15             # OpenFOAM double SMALL, the floor in PoE and phiDkPk
EFFECTIVELY_UNBOUNDED = 1.0e30   # as tedp.published_closures: gMax clamp inactive

#: Pi = 2 nu_t |symm grad U|^2/(beta* k omega_s), exact on both sides of the PoE clip.
PI_EXPR = "(PoE+max(0,(1000000000000*(1-phiDkPk))/(1000000000000*phiDkPk)-10))"
#: g2 on the solver's T2; c0 = C_cr1.
G2_EXPR = (f"c0*0.09*(1000000000000*((1000000000000*{PI_EXPR})/(1000000000000*I1)))"
           "/(1000000000000*sqrt(I1-I2))")

SPEC_NAME = "qcr2000_spalart_2000"


def qcr2000_spec(c_cr1: float = C_CR1) -> CandidateSpec:
    spec = CandidateSpec(
        name=SPEC_NAME,
        bdelta=(Term("T2", G2_EXPR),),
        rsource=(),
        constants=(float(c_cr1),),
        gmax=EFFECTIVELY_UNBOUNDED,
        r_max_factor=5.0,       # no R channel: has no effect
    )
    spec.validate(search_policy=False)
    return spec


def deployed_expression(spec: CandidateSpec) -> str:
    """The exact string the OpenFOAM dictionary receives (constants substituted)."""
    node = expr.substitute_constants(expr.parse(spec.bdelta[0].expression), spec.constants)
    return expr.to_string(node)


# --------------------------------------------------------------------------- reference
_EPS = np.zeros((3, 3, 3))
for _i, _j, _k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    _EPS[_i, _j, _k] = 1.0
    _EPS[_i, _k, _j] = -1.0


def qcr2000_reference(grad_u, k, nut, frame_omega, c_cr1=C_CR1):
    """Independent NumPy QCR2000 on the SST stress; returns Delta R = R_QCR - R_SST.

    grad_u[..., i, j] = dU_i/dx_j in the solver's (possibly rotating) frame; frame_omega the frame
    angular velocity. QCR is evaluated with the inertial-frame velocity gradient
    dU_i/dx_j - eps_ijk Omega_k (the gradient of Omega x x added), which changes W only.
    Written from the published definition; uses nothing from tedp.
    """
    a = np.asarray(grad_u, dtype=float)
    a_in = a - np.einsum("ijk,...k->...ij", _EPS, np.asarray(frame_omega, dtype=float))
    s = 0.5 * (a + np.swapaxes(a, -1, -2))
    w = 0.5 * (a_in - np.swapaxes(a_in, -1, -2))            # W_ik = (dU_i/dx_k - dU_k/dx_i)/2
    eye = np.eye(3)
    tr_s = np.trace(s, axis1=-2, axis2=-1)[..., None, None]
    r_sst = (2.0 / 3.0) * k[..., None, None] * eye - nut[..., None, None] * (2.0 * s - (2.0 / 3.0) * tr_s * eye)
    tau = -r_sst                                               # tau = -<u'u'>
    mag = np.sqrt(np.einsum("...mn,...mn->...", a_in, a_in))
    o = np.where(mag[..., None, None] > 0.0,
                 2.0 * w / np.where(mag > 0.0, mag, 1.0)[..., None, None], 0.0)
    tau_qcr = tau - c_cr1 * (np.einsum("...ik,...jk->...ij", o, tau)
                             + np.einsum("...jk,...ik->...ij", o, tau))
    return -(tau_qcr - tau)


# --------------------------------------------------------------------------- spec path
def solver_variables(grad_u, k, nut, omega, omega_min, frame_omega):
    """Grammar variables and T2 exactly as kOmegaSSTBasis::updateCorrections builds them."""
    grad_of = np.swapaxes(np.asarray(grad_u, dtype=float), -1, -2)   # OpenFOAM grad(U)_ij = dU_j/dx_i
    omega_s = np.maximum(omega, omega_min)
    sym = 0.5 * (grad_of + np.swapaxes(grad_of, -1, -2))
    skew = 0.5 * (grad_of - np.swapaxes(grad_of, -1, -2))
    frame = np.einsum("ijk,...k->...ij", _EPS, np.asarray(frame_omega, dtype=float))
    s_hat = tensors.dev(sym) / omega_s[..., None, None]
    w_hat = (skew + frame) / omega_s[..., None, None]
    t, inv = tensors.integrity_basis(s_hat, w_hat)
    g2s = nut * 2.0 * np.einsum("...ij,...ij->...", sym, sym)
    dk = BETA_STAR * k * omega_s
    poe = np.minimum(g2s / np.maximum(dk, SMALL), 10.0)
    phi = dk / np.maximum(g2s + dk, SMALL)
    zeros = np.zeros_like(k)
    variables = {name: zeros for name in expr.VARIABLES}
    variables.update({"I1": inv[..., 0], "I2": inv[..., 1], "I3": inv[..., 2], "I4": inv[..., 3],
                      "I5": inv[..., 4], "PoE": poe, "phiDkPk": phi})
    return variables, t[..., 1, :, :], s_hat


def spec_stress(spec, grad_u, k, nut, omega, omega_min, frame_omega, deployed=True):
    """tau^Delta = 2 k g2 T2 (lambda = 1) from the spec, through tedp.expr.evaluate."""
    variables, t2, _ = solver_variables(grad_u, k, nut, omega, omega_min, frame_omega)
    if deployed:
        node = expr.parse(deployed_expression(spec))
        g2 = expr.evaluate(node, variables)
    else:
        g2 = expr.evaluate(expr.parse(spec.bdelta[0].expression), variables, spec.constants)
    g2 = np.clip(np.broadcast_to(g2, k.shape), -spec.gmax, spec.gmax)
    return 2.0 * k[..., None, None] * g2[..., None, None] * t2, variables, g2


# --------------------------------------------------------------------------- sampling
def sample_states(n, seed, *, solenoidal=True, trace_ratio=0.0):
    """Velocity-gradient states with SST nu_t, a frame rotation in 30 % and omega < omega_min in 10 %."""
    rng = np.random.default_rng(seed)
    omega = 10.0 ** rng.uniform(-1.0, 5.0, n)
    k = 10.0 ** rng.uniform(-8.0, 1.0, n)
    ratio = 10.0 ** rng.uniform(-4.0, 3.0, n)                 # |grad U|/omega
    a = rng.normal(size=(n, 3, 3))
    kind = rng.uniform(size=n)
    shear = kind < 0.10                                         # simple shear, random orientation
    if shear.any():
        q = np.stack([tensors.random_rotation(rng) for _ in range(int(shear.sum()))])
        e = np.zeros((int(shear.sum()), 3, 3))
        e[:, 0, 1] = 1.0
        a[shear] = q @ e @ np.swapaxes(q, -1, -2)
    strain = (kind >= 0.10) & (kind < 0.20)                     # irrotational strain: W = 0
    a[strain] = 0.5 * (a[strain] + np.swapaxes(a[strain], -1, -2))
    rotation = (kind >= 0.20) & (kind < 0.25)                   # rigid rotation: S = 0
    a[rotation] = 0.5 * (a[rotation] - np.swapaxes(a[rotation], -1, -2))
    if solenoidal:
        a -= np.trace(a, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3.0
    else:
        a -= np.trace(a, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3.0
        a += trace_ratio * rng.choice([-1.0, 1.0], n)[:, None, None] * np.eye(3) / np.sqrt(3.0)
    a /= np.sqrt(np.einsum("nij,nij->n", a, a))[:, None, None]
    grad_u = a * (ratio * omega)[:, None, None]
    f2 = rng.uniform(size=n)
    f2[rng.uniform(size=n) < 0.3] = 1.0
    f2[rng.uniform(size=n) < 0.1] = 0.0
    s = 0.5 * (grad_u + np.swapaxes(grad_u, -1, -2))
    s2 = 2.0 * np.einsum("nij,nij->n", s, s)
    nut = A1 * k / np.maximum(A1 * omega, f2 * np.sqrt(s2))     # OpenFOAM v2312 SST, b1 = 1
    frame = np.zeros((n, 3))
    rot = rng.uniform(size=n) < 0.3
    axis = rng.normal(size=(int(rot.sum()), 3))
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    frame[rot] = axis * (10.0 ** rng.uniform(-3.0, 0.5, int(rot.sum())) * omega[rot])[:, None]
    omega_min = np.zeros(n)
    floored = rng.uniform(size=n) < 0.1
    omega_min[floored] = omega[floored] * rng.uniform(1.0, 10.0, int(floored.sum()))
    return {"grad_u": grad_u, "k": k, "nut": nut, "omega": omega, "omega_min": omega_min,
            "frame_omega": frame, "f2": f2, "kind": kind}


def relative_difference(ours, ref):
    num = np.sqrt(np.einsum("nij,nij->n", ours - ref, ours - ref))
    den = np.sqrt(np.einsum("nij,nij->n", ref, ref))
    both_zero = (den == 0.0) & (num == 0.0)
    rel = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), np.where(both_zero, 0.0, np.inf))
    return rel


def compare(spec, states, deployed=True):
    ref = qcr2000_reference(states["grad_u"], states["k"], states["nut"], states["frame_omega"],
                            spec.constants[0])
    ours, variables, g2 = spec_stress(spec, states["grad_u"], states["k"], states["nut"],
                                      states["omega"], states["omega_min"], states["frame_omega"],
                                      deployed=deployed)
    rel = relative_difference(ours, ref)
    return ref, ours, rel, variables, g2


# --------------------------------------------------------------------------- C++ evaluator
CPP_DRIVER = r"""
#include "exprParser.H"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace tedp::expr;
int main(int argc, char* argv[])
{
    std::ifstream in(argv[1]);
    std::string source; std::getline(in, source);
    std::size_t n; in >> n;
    const auto ast = parse(source, false);
    std::vector<double> v(N_ALL_VARS, 0.0);
    for (std::size_t s = 0; s < n; ++s)
    {
        for (std::size_t i = 0; i < N_STATE_VARS; ++i) in >> v[i];
        std::printf("%.17g\n", ast->evalPoint(v.data()));
    }
    return 0;
}
"""


def cpp_evaluate(source: str, variables: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate the deployed expression with the solver's own parser (exprParser.C/exprNode.C)."""
    basis = REPO / "closures" / "kOmegaSSTBasis" / "basisExpr"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "driver.C").write_text(CPP_DRIVER)
        binary = tmp / "driver"
        subprocess.run(["g++", "-std=c++17", "-O2", f"-I{basis}", "-o", str(binary),
                        str(tmp / "driver.C"), str(basis / "exprNode.C"), str(basis / "exprParser.C")],
                       check=True)
        n = len(next(iter(variables.values())))
        table = np.stack([np.broadcast_to(variables[name], (n,)) for name in expr.VARIABLES], axis=1)
        with open(tmp / "job.txt", "w") as fh:
            fh.write(source + "\n" + f"{n}\n")
            np.savetxt(fh, table, fmt="%.17g")
        out = subprocess.run([str(binary), str(tmp / "job.txt")], check=True, capture_output=True,
                             text=True).stdout
    return np.array([float(x) for x in out.split()])


# --------------------------------------------------------------------------- checks
def simple_shear_check(spec):
    """U = gamma y: QCR must give <u'u'> > <w'w'> > <v'v'> (Spalart's calibration) and
    Delta<u'u'> = -Delta<v'v'> = 2 C_cr1 nu_t gamma."""
    gamma, k, nut, omega = 3.0, 0.5, 0.05, 2.0
    g = np.zeros((1, 3, 3))
    g[0, 0, 1] = gamma
    ours, _, _ = spec_stress(spec, g, np.array([k]), np.array([nut]), np.array([omega]),
                             np.array([0.0]), np.zeros((1, 3)))
    r_sst = (2.0 / 3.0) * k * np.eye(3)
    r_sst[0, 1] = r_sst[1, 0] = -nut * gamma
    r = r_sst + ours[0]
    expected = 2.0 * spec.constants[0] * nut * gamma
    return {
        "uu_minus_ww": float(r[0, 0] - r[2, 2]), "ww_minus_vv": float(r[2, 2] - r[1, 1]),
        "delta_uu": float(ours[0, 0, 0]), "delta_vv": float(ours[0, 1, 1]),
        "expected_delta_uu": expected,
        "passed": bool(r[0, 0] > r[2, 2] > r[1, 1]
                       and abs(ours[0, 0, 0] - expected) < 1e-12 * expected
                       and abs(ours[0, 1, 1] + expected) < 1e-12 * expected
                       and abs(ours[0, 2, 2]) < 1e-15 and abs(ours[0, 0, 1]) < 1e-15),
    }


def run(out_dir: Path, n: int = 20000, seed: int = 20261001) -> dict:
    spec = qcr2000_spec()
    assert from_json(to_json(spec)) == spec
    deployed = deployed_expression(spec)
    report: dict = {
        "spec": json.loads(to_json(spec)),
        "deployed_expression": deployed,
        "foam_coeffs": to_foam_coeffs(spec),
        "n_states": n, "seed": seed,
    }

    states = sample_states(n, seed)
    ref, ours, rel, variables, g2 = compare(spec, states, deployed=True)
    _, ours_c, rel_c, _, _ = compare(spec, states, deployed=False)
    ref_norm = np.sqrt(np.einsum("nij,nij->n", ref, ref))
    k_scaled = np.sqrt(np.einsum("nij,nij->n", ours - ref, ours - ref)) / states["k"]
    nonzero = ref_norm > 0.0
    pi_raw = 2.0 * states["nut"] * np.einsum("nij,nij->n", 0.5 * (states["grad_u"] + np.swapaxes(states["grad_u"], 1, 2)), 0.5 * (states["grad_u"] + np.swapaxes(states["grad_u"], 1, 2))) / (BETA_STAR * states["k"] * np.maximum(states["omega"], states["omega_min"]))
    worst = int(np.argmax(rel))
    report["solenoidal"] = {
        "max_relative_difference_frobenius": float(rel.max()),
        "max_relative_difference_symbolic_constants": float(rel_c.max()),
        "p99_relative_difference": float(np.quantile(rel, 0.99)),
        "max_difference_over_k": float(k_scaled.max()),
        "states_with_nonzero_qcr_stress": int(nonzero.sum()),
        "states_with_zero_qcr_stress_both_zero": int((~nonzero & (rel == 0.0)).sum()),
        "pi_range": [float(pi_raw.min()), float(pi_raw.max())],
        "states_above_poe_clip": int((pi_raw >= 10.0).sum()),
        "states_with_frame_rotation": int((np.abs(states["frame_omega"]).sum(axis=1) > 0).sum()),
        "states_with_omega_floor_active": int((states["omega_min"] > states["omega"]).sum()),
        "states_with_f2_below_one_and_limiter_active": int(((states["f2"] < 1.0) & (states["nut"] < states["k"] / states["omega"] * (1 - 1e-12))).sum()),
        "gradient_over_omega_range": [1e-4, 1e3],
        "worst_state": {"I1": float(variables["I1"][worst]), "I2": float(variables["I2"][worst]),
                        "PoE": float(variables["PoE"][worst]), "pi": float(pi_raw[worst])},
    }

    # zero-coefficient limit: the spec must give exactly zero stress (SST)
    sst = qcr2000_spec(0.0)
    zero, _, _ = spec_stress(sst, states["grad_u"], states["k"], states["nut"], states["omega"],
                             states["omega_min"], states["frame_omega"])
    report["c_cr1_zero_reduces_to_sst"] = {"max_abs_stress": float(np.abs(zero).max()),
                                           "passed": bool(np.all(zero == 0.0))}

    # C++ evaluator (the solver's parser) on the same states
    cpp = cpp_evaluate(deployed, variables)
    g2_py = np.broadcast_to(g2, (n,))
    cpp_rel = np.abs(cpp - g2_py) / np.maximum(np.abs(g2_py), 1e-300)
    _, t2, _ = solver_variables(states["grad_u"], states["k"], states["nut"], states["omega"],
                                   states["omega_min"], states["frame_omega"])
    ours_cpp = 2.0 * states["k"][:, None, None] * cpp[:, None, None] * t2
    rel_cpp = relative_difference(ours_cpp, ref)
    report["cpp_parser"] = {
        "max_relative_difference_g2_cpp_vs_python": float(cpp_rel[g2_py != 0].max()),
        "max_relative_difference_stress_cpp_vs_numpy_qcr": float(rel_cpp.max()),
    }

    # non-solenoidal gradients (cell-centred gradients are not exactly divergence-free)
    nonsol = {}
    for ratio in (1e-3, 1e-2, 1e-1):
        st = sample_states(n // 4, seed + 1, solenoidal=False, trace_ratio=ratio)
        _, _, r, _, _ = compare(spec, st)
        nonsol[f"{ratio:g}"] = float(r.max())
    report["non_solenoidal_max_relative_difference_by_trace_fraction"] = nonsol

    # traceless, symmetric, T2 : S = 0 (no production change)
    _, _, s_hat = solver_variables(states["grad_u"], states["k"], states["nut"], states["omega"],
                                      states["omega_min"], states["frame_omega"])
    trace = np.abs(np.trace(ours, axis1=1, axis2=2)) / np.maximum(ref_norm, 1e-300)
    work = np.abs(np.einsum("nij,nij->n", ours, s_hat)) / np.maximum(
        ref_norm * np.sqrt(np.einsum("nij,nij->n", s_hat, s_hat)), 1e-300)
    report["structure"] = {
        "max_relative_trace": float(trace[nonzero].max()),
        "max_relative_asymmetry": float((np.abs(ours - np.swapaxes(ours, 1, 2)).max(axis=(1, 2)) / np.maximum(ref_norm, 1e-300))[nonzero].max()),
        "max_relative_stress_work_on_S": float(work[nonzero].max()),
    }
    report["simple_shear"] = simple_shear_check(spec)

    passed = (report["solenoidal"]["max_relative_difference_frobenius"] < 1e-9
              and report["c_cr1_zero_reduces_to_sst"]["passed"]
              and report["cpp_parser"]["max_relative_difference_g2_cpp_vs_python"] < 1e-12
              and report["simple_shear"]["passed"])
    report["passed"] = bool(passed)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qcr2000_spec.json").write_text(to_json(spec) + "\n")
    (out_dir / "qcr2000_verification.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    report = run(args.out, args.n, args.seed)
    printable = {k: v for k, v in report.items() if k not in ("foam_coeffs", "spec")}
    print(json.dumps(printable, indent=1))


if __name__ == "__main__":
    main()
