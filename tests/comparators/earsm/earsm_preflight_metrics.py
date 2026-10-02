"""Preflight metrics of the BSL-EARSM comparator (kOmegaEARSM) on imported case directories.

Post-processing only (no CFD). Two checks, with the gates of
closures/comparators/earsm_reference/published_targets.json:

  duct   corner-bisector secondary velocity of a fully developed square duct
         (Menter, Garbaruk and Egorov 2012, Fig. 3b, p. 97: BSL-EARSM v < 0 along y = z, i.e. flow
         into the corner; SST gives no secondary flow). For each of the four corners the in-plane
         velocity is interpolated along the bisector; v_n is the component normal to the adjacent
         wall, negative towards it, at the wall distance d, reported against d/(2h). Gate (sign and
         order, the repository's ducts are at lower Re than the published case): v_n < 0 for
         0.06 <= d/(2h) <= 0.35 in every corner, peak |v_n|/U_b in [0.004, 0.02] at
         0.05 <= d/(2h) <= 0.25; with --sst, SST max |in-plane U|/U_b < 1e-6. The sign band is
         narrower than published_targets.json (0.03-0.4) because the repository's DNS at Re_b 2000
         (Pinelli et al. 2010) itself has v ~ 0 at d/(2h) = 0.03 (corner vortex) and v > 0 beyond
         d/(2h) = 0.39; on the DNS: peak 0.0166 at d/(2h) = 0.19, v < 0 over 0.035-0.39.

  plate  log-layer equilibrium on the TMR flat plate near x = 0.97 (M12 p. 93: A1 = 1.245 matches
         the log layer; pp. 94-95: -uv = sqrt(cmu) k): cells with 50 <= y+ <= 0.1 delta+ (0.15 if
         fewer than three), R = 2/3 k I - nut twoSymm(grad U) + nonlinearStress. Gate: median -R_xy/k
         in [0.29, 0.32] (printed model 0.3056). Supporting (reported, not gated): median
         cmu_eff = nut 0.09 omega/k in [0.085, 0.100], a_xx = R_xx/k - 2/3 within 0.03 of 0.252,
         a_zz within 0.03 of 0. The slope of u+ against ln y+ over the same cells is reported
         (kappa_log_fit) for the EARSM/SST comparison only: the repository's SST run of this grid
         (137x97, nearest column x = 0.9505) gives 0.355 there, so it is not a test of 0.41.

Usage:
  python tests/comparators/earsm/earsm_preflight_metrics.py duct --case <model case dir> --time <final>
         [--sst <sst case dir> --sst-time <final>] --out <json>
  python tests/comparators/earsm/earsm_preflight_metrics.py plate --case <model case dir> --time <final>
         --out <json>
The JSON carries "passed" (the gate) and every measured number.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from fluidfoam import readmesh, readscalar, readsymmtensor, readtensor, readvector
from scipy.interpolate import LinearNDInterpolator

TARGETS = {
    'duct': {'negative_over': (0.06, 0.35), 'peak_abs_range': (0.004, 0.02), 'peak_location_range': (0.05, 0.25),
             'sst_abs_max': 1e-6},
    'plate': {'minus_uv_over_k': (0.29, 0.32), 'cmu_eff': (0.085, 0.100), 'a_xx': (0.222, 0.282),
              'a_zz': (-0.03, 0.03), 'x_station': 0.97},
}


def _time(case: Path, time: str) -> str:
    if (case / str(time)).is_dir():
        return str(time)
    times = sorted((p.name for p in case.iterdir() if p.is_dir() and re.fullmatch(r'[0-9.eE+-]+', p.name)
                    and p.name != '0'), key=float)
    if not times:
        raise FileNotFoundError(f'No time directory in {case}')
    return times[-1]


def _points_extent(case: Path):
    text = (case / 'constant/polyMesh/points').read_text(errors='replace')
    nums = re.findall(r'\(\s*([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*\)', text)
    pts = np.array(nums, dtype=float)
    return pts.min(axis=0), pts.max(axis=0)


def _ubar(case: Path):
    for name in ('system/fvOptions', 'constant/fvOptions', 'system/fvConstraints', 'constant/fvConstraints'):
        path = case / name
        if path.is_file():
            m = re.search(r'Ubar\s*\(\s*([-0-9.eE+]+)', path.read_text(errors='replace'))
            if m:
                return float(m.group(1)), name
    return None, None


def _widths(centres, lo, hi):
    """Cell widths of a tensor-product direction from the unique centres and the walls lo, hi."""
    c = np.unique(np.round(centres, 12))
    faces = np.concatenate([[lo], 0.5*(c[1:] + c[:-1]), [hi]])
    return c, np.diff(faces)


def _cross_section(case: Path, time: str):
    x, y, z = readmesh(str(case), verbose=False)
    U = np.asarray(readvector(str(case), time, 'U', verbose=False))
    lo, hi = _points_extent(case)
    # streamwise average onto the (y, z) cross-section
    key = np.round(y, 10) + 1j*np.round(z, 10)
    uniq, inv = np.unique(key, return_inverse=True)
    counts = np.bincount(inv)
    Uavg = np.vstack([np.bincount(inv, weights=U[i])/counts for i in range(3)])
    return uniq.real, uniq.imag, Uavg, lo, hi


def duct_metrics(case: Path, time: str, sst: Path | None, sst_time: str | None) -> dict:
    t = _time(case, time)
    yc, zc, U, lo, hi = _cross_section(case, t)
    ymid, zmid = 0.5*(lo[1] + hi[1]), 0.5*(lo[2] + hi[2])
    h = 0.5*(hi[1] - lo[1])
    if abs(0.5*(hi[2] - lo[2]) - h) > 1e-6*h:
        raise ValueError('Not a square duct cross-section')
    ub, ub_source = _ubar(case)
    if ub is None:
        ys, wy = _widths(yc, lo[1], hi[1])
        zs, wz = _widths(zc, lo[2], hi[2])
        iy = np.searchsorted(ys, np.round(yc, 12))
        iz = np.searchsorted(zs, np.round(zc, 12))
        area = wy[np.clip(iy, 0, len(wy)-1)]*wz[np.clip(iz, 0, len(wz)-1)]
        ub, ub_source = float(np.sum(U[0]*area)/np.sum(area)), 'area-weighted mean of U_x'
    interp = LinearNDInterpolator(np.column_stack([yc, zc]), np.column_stack([U[1], U[2]]))
    s = np.linspace(0.005, 0.5, 400)          # d/(2h)
    d = s*2*h
    corners = {}
    tgt = TARGETS['duct']
    ok = True
    for sy in (-1, 1):
        for sz in (-1, 1):
            py = ymid + sy*(h - d)
            pz = zmid + sz*(h - d)
            vw = interp(py, pz)
            vn = -sy*vw[:, 0]                  # normal to the y-wall, negative towards it
            valid = np.isfinite(vn)
            band = valid & (s > tgt['negative_over'][0]) & (s < tgt['negative_over'][1])
            negative = bool(band.any() and np.all(vn[band] < 0))
            k = int(np.nanargmax(np.where(valid, -vn, -np.inf)))
            peak = float(-vn[k]/ub)
            loc = float(s[k])
            corner_ok = (negative and tgt['peak_abs_range'][0] <= peak <= tgt['peak_abs_range'][1]
                         and tgt['peak_location_range'][0] <= loc <= tgt['peak_location_range'][1])
            ok &= corner_ok
            corners[f'{"y-" if sy < 0 else "y+"}{"z-" if sz < 0 else "z+"}'] = {
                'v_negative_over_band': negative, 'peak_abs_v_over_Ub': peak, 'peak_location_d_over_2h': loc,
                'min_v_over_Ub_in_band': float(np.nanmin(vn[band])/ub) if band.any() else None,
                'max_v_over_Ub_in_band': float(np.nanmax(vn[band])/ub) if band.any() else None,
                'passed': bool(corner_ok),
                'profile_d_over_2h': s[::20].round(4).tolist(),
                'profile_v_over_Ub': (vn[::20]/ub).round(6).tolist()}
    out = {'check': 'duct_corner_bisector', 'case': str(case), 'time': t, 'U_bulk': ub, 'U_bulk_source': ub_source,
           'half_width': h, 'corners': corners, 'targets': {k: list(v) if isinstance(v, tuple) else v
                                                            for k, v in tgt.items()},
           'source': 'closures/comparators/earsm_reference/published_targets.json (M12 Fig. 3b, p. 97)'}
    peaks = [c['peak_abs_v_over_Ub'] for c in corners.values()]
    out['peak_abs_v_over_Ub_mean'] = float(np.mean(peaks))
    out['peak_abs_v_over_Ub_corner_spread'] = float(np.ptp(peaks))
    if sst is not None:
        ts = _time(sst, sst_time or time)
        _, _, Us, _, _ = _cross_section(sst, ts)
        smax = float(np.max(np.hypot(Us[1], Us[2]))/ub)
        out['sst'] = {'case': str(sst), 'time': ts, 'max_inplane_over_Ub': smax,
                      'no_secondary_flow': smax < tgt['sst_abs_max']}
        ok &= smax < tgt['sst_abs_max']
    out['passed'] = bool(ok)
    return out


def _nu(case: Path) -> float:
    text = (case / 'constant/transportProperties').read_text(errors='replace')
    m = re.search(r'(?m)^\s*nu\s+(?:\[[^\]]*\]\s*)?([-0-9.eE+]+)\s*;', text)
    if not m:
        m = re.search(r'(?m)^\s*nu\s+nu\s+\[[^\]]*\]\s*([-0-9.eE+]+)\s*;', text)
    if not m:
        raise ValueError('nu not found in constant/transportProperties')
    return float(m.group(1))


def plate_metrics(case: Path, time: str) -> dict:
    t = _time(case, time)
    x, y, z = readmesh(str(case), verbose=False)
    U = np.asarray(readvector(str(case), t, 'U', verbose=False))
    k = np.asarray(readscalar(str(case), t, 'k', verbose=False))
    om = np.asarray(readscalar(str(case), t, 'omega', verbose=False))
    nut = np.asarray(readscalar(str(case), t, 'nut', verbose=False))
    ns_path = case / t / 'nonlinearStress'
    ns = np.asarray(readsymmtensor(str(case), t, 'nonlinearStress', verbose=False)) if ns_path.is_file() \
        else np.zeros((6, len(x)))
    nu = _nu(case)
    tgt = TARGETS['plate']
    xs = np.unique(np.round(x, 10))
    x0 = xs[np.argmin(abs(xs - tgt['x_station']))]
    col = np.where(abs(x - x0) < 1e-9 + 1e-9*abs(x0))[0]
    col = col[np.argsort(y[col])]
    yy, uu = y[col], U[0, col]
    if (case / t / 'grad(U)').is_file():
        g = np.asarray(readtensor(str(case), t, 'grad(U)', verbose=False))[:, col]
        dudy, dvdx, dudx = g[3], g[1], g[0]
        grad_source = 'grad(U) field'
    else:
        dudy, dvdx, dudx = np.gradient(uu, yy), np.zeros_like(uu), np.zeros_like(uu)
        grad_source = 'finite differences along the column (dV/dx = dU/dx = 0)'
    utau = math.sqrt(nu*uu[0]/yy[0])
    ue = float(np.max(uu))
    delta = float(np.interp(0.99*ue, uu[:np.argmax(uu)+1], yy[:np.argmax(uu)+1]))
    yplus = yy*utau/nu
    dplus = delta*utau/nu
    sel = (yplus >= 50) & (yplus <= 0.1*dplus)
    upper = 0.1
    if sel.sum() < 3:
        sel = (yplus >= 50) & (yplus <= 0.15*dplus)
        upper = 0.15
    kk, oo, nn = k[col], om[col], nut[col]
    rxy = -nn*(dudy + dvdx) + ns[1, col]
    muv = -rxy/kk
    cmu = nn*0.09*oo/kk
    axx = (-2*nn*dudx + ns[0, col])/kk
    azz = ns[5, col]/kk
    uplus = uu/utau
    slope = float(np.polyfit(np.log(yplus[sel]), uplus[sel], 1)[0]) if sel.sum() >= 2 else float('nan')
    med = lambda a: float(np.median(a[sel])) if sel.any() else float('nan')
    res = {'minus_uv_over_k': med(muv), 'cmu_eff': med(cmu), 'a_xx': med(axx), 'a_zz': med(azz)}
    inside = {q: bool(tgt[q][0] <= v <= tgt[q][1]) for q, v in res.items()}
    kappa_fit = 1/slope if slope and np.isfinite(slope) else float('nan')
    out = {'check': 'plate_log_layer', 'case': str(case), 'time': t, 'x_station': float(x0), 'nu': nu,
           'u_tau': utau, 'delta99': delta, 'delta_plus': dplus, 'log_range_yplus': [50, upper*dplus],
           'cells_in_range': int(sel.sum()), 'gradient_source': grad_source,
           'nonlinearStress_present': ns_path.is_file(), 'medians': res, 'within_target': inside,
           'kappa_log_fit': kappa_fit,
           'kappa_log_fit_note': 'u+ against ln y+ over the gate cells; repository SST on this grid: 0.355',
           'targets': {q: list(tgt[q]) for q in res}, 'gate': 'minus_uv_over_k',
           'per_cell': {'y_plus': yplus[sel].round(3).tolist(), 'minus_uv_over_k': muv[sel].round(5).tolist(),
                        'cmu_eff': cmu[sel].round(5).tolist()},
           'source': 'closures/comparators/earsm_reference/published_targets.json gate log_layer_equilibrium (M12 pp. 93-95)'}
    out['passed'] = bool(sel.sum() >= 2 and inside['minus_uv_over_k'])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('check', choices=('duct', 'plate'))
    ap.add_argument('--case', required=True, type=Path)
    ap.add_argument('--time', required=True)
    ap.add_argument('--sst', type=Path)
    ap.add_argument('--sst-time')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args(argv)
    if a.check == 'duct':
        out = duct_metrics(a.case, a.time, a.sst, a.sst_time)
    else:
        out = plate_metrics(a.case, a.time)
    text = json.dumps(out, indent=1, default=float)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
