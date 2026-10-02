"""Compare the C++ BSL-EARSM unit-test output with the independent NumPy reference.

Two commands:

  make-states   write the shared input states (shared_states.csv, next to this file by default):
                200 random 3D states and edge states (zero gradient, weak strain, pure rotation, pure strain,
                simple shear at the log-layer and homogeneous-shear equilibria, P2 ~ 0 on both sides,
                Kolmogorov limiter active, very large strain, rotating frame).
  compare       read the CSV written by the C++ unit test, recompute every state with reference.py and
                report the largest differences; exit status 1 if a tolerance is exceeded.

File format (shared with the C++ unit test tests/comparators/earsm/testEARSM): CSV, header row, comma
separated, %.17g, one row per state. Inputs (required): id, gxx..gzz (OpenFOAM grad(U), g_ij = dU_j/dx_i,
order XX XY XZ YX YY YZ ZX ZY ZZ), k, omega, nu; optional Ox, Oy, Oz (frameOmega, default 0). Outputs (each
optional, compared when present): tau, IIS, IIW, IV, N, beta1, beta3, beta4, beta6, beta9,
axx, axy, axz, ayy, ayz, azz (full anisotropy a = tau_ij/k - 2/3 delta_ij), nut,
nsxx, nsxy, nsxz, nsyy, nsyz, nszz (nonlinearStress, m2/s2). Unknown columns are ignored.

Usage:
  PY=python3
  $PY closures/comparators/earsm_reference/compare_cpp.py make-states
  $PY closures/comparators/earsm_reference/compare_cpp.py compare <cpp.csv> [--frame-mode relative|absolute|lrr|auto]
      [--include-t9] [--a1 1.245] [--cubic-a1 1.2] [--json report.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reference as ref  # noqa: E402

G_COLS = ["gxx", "gxy", "gxz", "gyx", "gyy", "gyz", "gzx", "gzy", "gzz"]
IN_COLS = ["id", *G_COLS, "k", "omega", "nu", "Ox", "Oy", "Oz"]
SYM = [("xx", 0, 0), ("xy", 0, 1), ("xz", 0, 2), ("yy", 1, 1), ("yz", 1, 2), ("zz", 2, 2)]
TOL = json.loads((HERE / "published_targets.json").read_text())["unit_comparison_tolerances"]


# ----------------------------------------------------------------------------------------------
# shared states
# ----------------------------------------------------------------------------------------------

def _grad_from_s_w(S, W, tau):
    """OpenFOAM g (g_ij = dU_j/dx_i) whose S, W (WJ convention, time scale tau) are the given ones."""
    L = (S + W) / tau
    return L.T


def make_states(seed: int = 20261002) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    def add(tag, g, k, omega, nu, frame=(0.0, 0.0, 0.0)):
        g = np.asarray(g, float).reshape(3, 3)
        rows.append({"id": len(rows), "tag": tag, **{c: g.flat[i] for i, c in enumerate(G_COLS)},
                     "k": float(k), "omega": float(omega), "nu": float(nu),
                     "Ox": float(frame[0]), "Oy": float(frame[1]), "Oz": float(frame[2])})

    for _ in range(200):
        g = rng.normal(size=(3, 3)) * 10.0 ** rng.uniform(-2, 4)
        g -= np.trace(g) * np.eye(3) / 3.0
        frame = rng.normal(size=3) * 10.0 ** rng.uniform(-2, 1) if rng.uniform() < 0.3 else np.zeros(3)
        add("random", g, 10.0 ** rng.uniform(-8, 0), 10.0 ** rng.uniform(0, 6),
            10.0 ** rng.uniform(-6, -4), frame)

    k, om, nu = 1.0, 100.0, 1e-5            # limiter inactive: tau = 1/(0.09*100)
    tau = 1.0 / (0.09 * om)
    add("zero_gradient", np.zeros((3, 3)), k, om, nu)
    for eps in (1e-8, 1e-5, 1e-3):
        g = rng.normal(size=(3, 3)); g -= np.trace(g) * np.eye(3) / 3.0
        add(f"weak_strain_{eps:g}", eps * om * g / np.linalg.norm(g), k, om, nu)
    w = rng.normal(size=3)
    add("pure_rotation", 50.0 * ref.frame_tensor(w), k, om, nu)
    g = rng.normal(size=(3, 3)); g = 0.5 * (g + g.T); g -= np.trace(g) * np.eye(3) / 3.0
    add("pure_strain", 30.0 * g, k, om, nu)
    for name, R in (("log_layer", 1.0), ("hom_shear_outer", ref.homogeneous_shear_pk_over_eps(0.0)),
                    ("hom_shear_inner", ref.homogeneous_shear_pk_over_eps(1.0))):
        cf = ref.shear_equilibrium_closed_form(R)
        S, W = ref.simple_shear(cf["s_tau"])
        add(f"shear_{name}", _grad_from_s_w(S, W, tau), k, om, nu)
    # P2 = 0 crossings, 2D states S = s(e1e2 + e2e1), W = w(e1e2 - e2e1): IIS = 2 s^2, IIW = -2 w^2
    for iis in (0.01, 0.3, 3.0, 30.0, 300.0):
        lo, hi = -10.0 * iis - 10.0, 0.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if ref.solve_n(iis, mid)[2] > 0:
                lo = mid
            else:
                hi = mid
        for side, iiw in (("p2pos", lo * (1 + 1e-9)), ("p2neg", hi * (1 - 1e-9))):
            s, ww = np.sqrt(iis / 2.0), np.sqrt(-iiw / 2.0)
            S, W = ref.simple_shear(0.0)
            S[0, 1] = S[1, 0] = s
            W[0, 1], W[1, 0] = ww, -ww
            add(f"p2zero_{side}_IIS{iis:g}", _grad_from_s_w(S, W, tau), k, om, nu)
    # Kolmogorov limiter active
    for i in range(5):
        g = rng.normal(size=(3, 3)) * 10.0 ** rng.uniform(0, 3); g -= np.trace(g) * np.eye(3) / 3.0
        add(f"limiter_{i}", g, 1e-10, 1e5, 1e-5)
    # very large strain
    for i in range(3):
        g = rng.normal(size=(3, 3)) * 1e4; g -= np.trace(g) * np.eye(3) / 3.0
        add(f"large_strain_{i}", g, 1.0, 1.0, 1e-5)
    # rotating frame: rotating-channel-like shear with Omega_z, both signs, and a 3D state
    for sgn in (1.0, -1.0):
        g = np.zeros((3, 3)); g[1, 0] = 100.0        # dUx/dy = 100
        add(f"rot_channel_{'pos' if sgn > 0 else 'neg'}", g, k, om, nu, (0.0, 0.0, sgn * 20.0))
    g = rng.normal(size=(3, 3)) * 50.0; g -= np.trace(g) * np.eye(3) / 3.0
    add("rot_3d", g, k, om, nu, (3.0, -7.0, 11.0))
    # 2D in-plane states (IV = 0) and an axisymmetric-strain state
    for i in range(4):
        g = np.zeros((3, 3)); g[:2, :2] = rng.normal(size=(2, 2)) * 40.0
        g[:2, :2] -= np.trace(g[:2, :2]) * np.eye(2) / 2.0
        add(f"plane_{i}", g, k, om, nu)
    add("axisymmetric_strain", np.diag([60.0, -30.0, -30.0]), k, om, nu)
    return rows


def write_states(path: Path) -> None:
    rows = make_states()
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(IN_COLS + ["tag"])
        for r in rows:
            w.writerow([r["id"]] + [f"{r[c]:.17g}" for c in IN_COLS[1:]] + [r["tag"]])
    print(f"wrote {len(rows)} states to {path}")


# ----------------------------------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------------------------------

def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open() as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    cols: dict[str, list] = {}
    for row in reader:
        for key, val in row.items():
            if key is None:
                continue
            cols.setdefault(key.strip(), []).append(val)
    out = {}
    for key, vals in cols.items():
        try:
            out[key] = np.array([float(v) for v in vals])
        except ValueError:
            out[key] = np.array(vals)
    return out


def reference_for(data: dict, coeffs: ref.Coefficients, frame_mode: str) -> dict:
    g = np.stack([data[c] for c in G_COLS], axis=-1).reshape(-1, 3, 3)
    frame = np.stack([data.get(c, np.zeros(len(g))) for c in ("Ox", "Oy", "Oz")], axis=-1)
    st = ref.earsm_state(g, data["k"], data["omega"], data["nu"], frame, coeffs, frame_mode)
    out = {"tau": st["tau"], "IIS": st["IIS"], "IIW": st["IIW"], "IV": st["IV"], "N": st["N"],
           "nut": st["nut"]}
    for i in (1, 3, 4, 6, 9):
        out[f"beta{i}"] = st["betas"][i]
    for name, i, j in SYM:
        out[f"a{name}"] = st["a"][:, i, j]
        out[f"ns{name}"] = st["nonlinear_stress"][:, i, j]
    return out


def compare(data: dict, refv: dict) -> dict:
    """Per-quantity maximum error and the state where it occurs."""
    k = data["k"]
    res = {}

    def record(name, err, tol):
        i = int(np.nanargmax(err)) if np.isfinite(err).any() else 0
        bad = ~np.isfinite(err)
        res[name] = {"max_error": float(np.nanmax(err)) if np.isfinite(err).any() else float("nan"),
                     "at_id": int(data["id"][i]) if "id" in data else i,
                     "non_finite": int(bad.sum()), "tolerance": tol,
                     "passed": bool(np.isfinite(err).all() and np.nanmax(err) <= tol)}

    def rel(c, r, floor):
        return np.abs(c - r) / np.maximum(np.abs(r), floor)

    if "tau" in data:
        record("tau_rel", rel(data["tau"], refv["tau"], 1e-300), TOL["tau_rel"])
    inv_scale = refv["IIS"] + np.abs(refv["IIW"])
    for name in ("IIS", "IIW"):
        if name in data:
            record(f"{name}_rel", rel(data[name], refv[name], 1e-12 * inv_scale + 1e-300), TOL["N_rel"])
    if "IV" in data:
        record("IV_rel", rel(data["IV"], refv["IV"], 1e-12 * inv_scale**1.5 + 1e-300), TOL["N_rel"])
    if "N" in data:
        record("N_rel", rel(data["N"], refv["N"], 1e-300), TOL["N_rel"])
    b1 = np.abs(refv["beta1"])
    for i in (1, 3, 4, 6, 9):
        key = f"beta{i}"
        if key in data:
            record(f"{key}_rel", rel(data[key], refv[key], 1e-12 * b1 + 1e-300), TOL["beta_rel"])
    if all(f"a{n}" in data for n, _, _ in SYM):
        diff = np.max([np.abs(data[f"a{n}"] - refv[f"a{n}"]) for n, _, _ in SYM], axis=0)
        scale = np.maximum(1.0, np.max([np.abs(refv[f"a{n}"]) for n, _, _ in SYM], axis=0))
        record("anisotropy_abs_over_max1", diff / scale, TOL["anisotropy_abs_over_max1"])
    if "nut" in data:
        record("nut_rel", rel(data["nut"], refv["nut"], 1e-300), TOL["nut_rel"])
    if all(f"ns{n}" in data for n, _, _ in SYM):
        diff = np.max([np.abs(data[f"ns{n}"] - refv[f"ns{n}"]) for n, _, _ in SYM], axis=0)
        record("nonlinear_stress_abs_over_k", diff / k, TOL["nonlinear_stress_abs_over_k"])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-states")
    m.add_argument("--out", type=Path, default=HERE / "shared_states.csv")
    c = sub.add_parser("compare")
    c.add_argument("cpp_csv", type=Path)
    c.add_argument("--frame-mode", default="auto", choices=["auto", "relative", "absolute", "lrr"])
    c.add_argument("--include-t9", action="store_true")
    c.add_argument("--a1", type=float, default=ref.A1_BSL_EARSM)
    c.add_argument("--cubic-a1", type=float, default=ref.A1_WALLIN_JOHANSSON)
    c.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "make-states":
        write_states(args.out)
        return 0

    data = read_csv(args.cpp_csv)
    missing = [col for col in ["k", "omega", "nu", *G_COLS] if col not in data]
    if missing:
        print(f"missing input columns: {missing}")
        return 2
    coeffs = ref.Coefficients(a1=args.a1, cubic_a1=args.cubic_a1, include_t9=args.include_t9)
    modes = ["relative", "absolute", "lrr"] if args.frame_mode == "auto" else [args.frame_mode]
    reports = {mode: compare(data, reference_for(data, coeffs, mode)) for mode in modes}
    # with frame rotation present the modes differ; pick the one that matches best
    def score(rep):
        return sum(0 if v["passed"] else 1 for v in rep.values()), max(
            (v["max_error"] / v["tolerance"] for v in rep.values() if np.isfinite(v["max_error"])), default=0)
    best = min(reports, key=lambda mo: score(reports[mo]))
    rep = reports[best]
    rotating = int(np.any([np.abs(data.get(cn, np.zeros(1))) > 0 for cn in ("Ox", "Oy", "Oz")], axis=0).sum())
    print(f"states: {len(data['k'])} (rotating frame: {rotating}); coefficients: A1 = {coeffs.a1}, "
          f"cubic A1 = {coeffs.cubic_a1}, T9 = {coeffs.include_t9}; frame mode: {best}"
          + (" (auto)" if args.frame_mode == "auto" else ""))
    width = max(len(k) for k in rep)
    for name, v in rep.items():
        print(f"  {name:<{width}}  max {v['max_error']:.3e}  (state {v['at_id']})  tol {v['tolerance']:.0e}  "
              f"{'PASS' if v['passed'] else 'FAIL'}" + (f"  non-finite {v['non_finite']}" if v["non_finite"] else ""))
    if args.frame_mode == "auto" and rotating:
        for mode, r in reports.items():
            if mode != best:
                worst = max(r.values(), key=lambda v: v["max_error"] / v["tolerance"])
                print(f"  (frame mode {mode}: {sum(not v['passed'] for v in r.values())} failing quantities, "
                      f"worst {worst['max_error']:.2e})")
    ok = all(v["passed"] for v in rep.values()) and len(rep) > 0
    if args.json:
        args.json.write_text(json.dumps({"input": str(args.cpp_csv), "frame_mode": best,
                                         "coefficients": coeffs.__dict__, "passed": ok,
                                         "states": int(len(data["k"])), "rotating_states": rotating,
                                         "results": rep, "all_modes": reports}, indent=2) + "\n")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
