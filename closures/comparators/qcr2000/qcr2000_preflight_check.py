"""Field-level preflight checks of the QCR2000 comparator on the square duct (no CFD; read-only).

Checks, on the final write time of a fully developed duct case (one streamwise cell layer, square
cross-section y, z in [-0.5, 0.5], streamwise x):

  bisector   in each of the four corners, the in-plane velocity component along the corner bisector,
             signed positive from the corner toward the duct centre, u_b, is negative at every
             bisector cell with 0.1 <= s/h <= 1.0 (s distance from the corner, h = 0.5 the
             half-width), and its peak |u_b| / u_bulk lies in [0.002, 0.063] (u_bulk of the
             reference npz; the DNS peak is 0.021 in every corner).
  vortices   the circulation of the streamwise vorticity omega_x = dW/dy - dV/dz over each of the
             eight triangles bounded by a wall, a corner bisector and a mid-plane has the DNS sign
             (the eight-vortex pattern): 8 of 8.
  sst_zero   the SST baseline's peak |u_b| is below 1e-3 of the comparator's.

Usage:
  python closures/comparators/qcr2000/qcr2000_preflight_check.py --case <QCR case dir> --time <t> \
      --sst <SST case dir> --sst-time <t> \
      --reference data/mcconkey/cache/holdout_duct_squareDuct_Re_2000.npz --out <json>
A case directory needs constant/polyMesh and <time>/U (fluidfoam reads ascii or binary). Passing the
reference npz itself as --case evaluates the DNS (the self-test that fixed DNS_SIGNS).
"""

from __future__ import annotations

import argparse
import json

import fluidfoam as ff
import numpy as np

#: DNS circulation signs (Pinelli et al. 2010 via McConkey et al., squareDuct_Re_2000), key
#: (sign y, sign z, nearer wall). Measured with this script's method on the DNS npz
#: (--case data/mcconkey/cache/holdout_duct_squareDuct_Re_2000.npz).
DNS_SIGNS = {"-1-1_z": 1, "-1-1_y": -1, "-1+1_z": -1, "-1+1_y": 1,
             "+1-1_z": -1, "+1-1_y": 1, "+1+1_z": 1, "+1+1_y": -1}


def cross_section(case: str, time: str):
    if case.endswith(".npz"):                       # reference npz: points (y, z), u (Ux, Uy, Uz)
        d = np.load(case)
        y, z, u = d["points"][:, 0], d["points"][:, 1], d["u"].T
    else:
        _, y, z = ff.readmesh(case, verbose=False)
        u = ff.readvector(case, time, "U", verbose=False)
    yr, zr = np.round(y, 10), np.round(z, 10)
    yy, zz = np.unique(yr), np.unique(zr)
    ny, nz = len(yy), len(zz)
    iy, iz = np.searchsorted(yy, yr), np.searchsorted(zz, zr)
    grids = []
    for comp in range(3):
        acc = np.zeros((ny, nz))
        cnt = np.zeros((ny, nz))
        np.add.at(acc, (iy, iz), u[comp])          # streamwise column average
        np.add.at(cnt, (iy, iz), 1.0)
        grids.append(acc / np.maximum(cnt, 1.0))
    if not (ny == nz and np.allclose(yy, zz, atol=1e-9)):
        raise ValueError("cross-section grid must be square and identical in y and z")
    return yy, grids


def bisector_profiles(yy, v, w):
    n = len(yy)
    out = {}
    for sy in (-1, 1):
        for sz in (-1, 1):
            idx = np.arange(n // 2)                 # from the corner toward the centre
            iy = idx if sy < 0 else n - 1 - idx
            iz = idx if sz < 0 else n - 1 - idx
            e = np.array([-sy, -sz]) / np.sqrt(2.0)  # unit vector corner -> centre
            ub = v[iy, iz] * e[0] + w[iy, iz] * e[1]
            s = np.hypot(yy[iy] - sy * 0.5, yy[iz] - sz * 0.5)
            out[f"{sy:+d}{sz:+d}"] = (s / 0.5, ub)
    return out


def triangle_circulation(yy, v, w):
    y2, z2 = np.meshgrid(yy, yy, indexing="ij")
    ox = np.gradient(w, yy, axis=0) - np.gradient(v, yy, axis=1)
    dy = np.gradient(yy)
    area = np.outer(dy, dy)
    out = {}
    for sy in (-1, 1):
        for sz in (-1, 1):
            q = (np.sign(y2) == sy) & (np.sign(z2) == sz)
            out[f"{sy:+d}{sz:+d}_z"] = float(np.sum((ox * area)[q & (np.abs(z2) > np.abs(y2))]))
            out[f"{sy:+d}{sz:+d}_y"] = float(np.sum((ox * area)[q & (np.abs(z2) < np.abs(y2))]))
    return out


def evaluate(case, time, u_bulk):
    yy, (u, v, w) = cross_section(case, time)
    prof = bisector_profiles(yy, v, w)
    corners = {}
    for key, (s, ub) in prof.items():
        band = (s >= 0.1) & (s <= 1.0)
        peak = int(np.argmax(np.abs(ub[band])))
        corners[key] = {"all_negative_0p1_to_1p0": bool(np.all(ub[band] < 0.0)),
                        "peak_over_ubulk": float(np.abs(ub[band]).max() / u_bulk),
                        "peak_sign": float(np.sign(ub[band][peak])),
                        "peak_s_over_h": float(s[band][peak])}
    circ = triangle_circulation(yy, v, w)
    signs = {k: int(np.sign(c)) for k, c in circ.items()}
    return {"corners": corners, "circulation": circ, "circulation_signs": signs,
            "matching_dns_signs": int(sum(signs[k] == DNS_SIGNS[k] for k in DNS_SIGNS)),
            "peak_over_ubulk_max": max(c["peak_over_ubulk"] for c in corners.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--time", required=True)
    ap.add_argument("--sst")
    ap.add_argument("--sst-time")
    ap.add_argument("--reference", default="data/mcconkey/cache/holdout_duct_squareDuct_Re_2000.npz")
    ap.add_argument("--out")
    args = ap.parse_args()
    u_bulk = float(np.load(args.reference)["u_bulk"])
    res = {"case": args.case, "time": args.time, "u_bulk": u_bulk, "qcr": evaluate(args.case, args.time, u_bulk)}
    q = res["qcr"]
    checks = {
        "bisector_sign": all(c["all_negative_0p1_to_1p0"] for c in q["corners"].values()),
        "bisector_peak_in_range": all(0.002 <= c["peak_over_ubulk"] <= 0.063 for c in q["corners"].values()),
        "eight_vortices": q["matching_dns_signs"] == 8,
    }
    if args.sst:
        res["sst"] = evaluate(args.sst, args.sst_time, u_bulk)
        checks["sst_zero"] = res["sst"]["peak_over_ubulk_max"] < 1e-3 * q["peak_over_ubulk_max"]
    res["checks"] = checks
    res["passed"] = all(checks.values())
    text = json.dumps(res, indent=1)
    if args.out:
        open(args.out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
