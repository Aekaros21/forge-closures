"""A-priori f_r1 of SST-RC on a converged SST zero-pressure-gradient flat-plate solution.

Supports the flat-plate target of published_targets.json (f_r1 = 1, friction within 0.1 % of
SST). No solver is run: the converged k-omega SST fields of the TMR 2D flat plate shipped with
this repository (McConkey et al. data set, data/mcconkey/foam/komegasst/fp/flatplate/35500:
U, omega, nut on a 12138 x 385 rectilinear grid, U = 69.4 m/s, nu = 1.388e-5, plate 0 <= x <= 2)
are read, the velocity gradient and the steady Lagrangian derivative u.grad(S) are formed by
second-order finite differences on the recovered structured grid, and f_r1 is evaluated with
reference.py (inertial frame). Reported: the production-weighted mean of f_r1 - 1
(weight nu_t S^2 times cell area), overall and at the wall-cf stations.

Usage: python plate_apriori.py      -> plate_apriori.json (about 1 minute, 2 GB memory)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reference as ref  # noqa: E402

REPO = HERE.parents[3]   # repository root
CASE = REPO / 'data/mcconkey/foam/komegasst/fp/flatplate'
TIME = '35500'


def read_internal(path):
    b = Path(path).read_bytes()
    m = re.search(rb'internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(', b)
    kind, n, start = m.group(1).decode(), int(m.group(2)), m.end()
    body = b[start:b.find(b'\n)\n', start)]
    if kind == 'vector':
        return np.array(body.replace(b'(', b' ').replace(b')', b' ').split(), dtype=float).reshape(n, 3)
    return np.array(body.split(), dtype=float)


def cluster(v, rel=1e-6):
    s = np.unique(v)
    gid = np.concatenate([[0], np.cumsum(np.diff(s) > rel * np.maximum(np.abs(s[1:]), 1e-9))])
    idx = gid[np.searchsorted(s, v)]
    return idx, np.bincount(idx, weights=v) / np.bincount(idx)


def main():
    d = CASE / TIME
    C = read_internal(d / 'C')
    ix, x = cluster(C[:, 0])
    iy, y = cluster(C[:, 1])
    nx, ny = len(x), len(y)
    if nx * ny != len(C) or len(np.unique(ix * ny + iy)) != len(C):
        raise RuntimeError('the mesh is not rectilinear')

    def grid(a):
        g = np.empty((ny, nx)); g[iy, ix] = a
        return g

    U = read_internal(d / 'U')
    u, v = grid(U[:, 0]), grid(U[:, 1])
    w = grid(read_internal(d / 'omega'))
    nut = grid(read_internal(d / 'nut'))
    dudy, dudx = np.gradient(u, y, x)
    dvdy, dvdx = np.gradient(v, y, x)
    S11, S12, S22 = dudx, 0.5 * (dudy + dvdx), dvdy

    def conv(f):
        fy, fx = np.gradient(f, y, x)
        return u * fx + v * fy

    D11, D12, D22 = conv(S11), conv(S12), conv(S22)
    fr = np.empty((ny, nx)); Sm = np.empty((ny, nx)); rs = np.empty((ny, nx)); rt = np.empty((ny, nx))
    for j0 in range(0, ny, 40):
        j = slice(j0, min(ny, j0 + 40)); n = u[j].size; sh = u[j].shape
        A = np.zeros((n, 3, 3))
        A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] = dudx[j].ravel(), dudy[j].ravel(), dvdx[j].ravel(), dvdy[j].ravel()
        Dt = np.zeros((n, 3, 3))
        Dt[:, 0, 0], Dt[:, 1, 1] = D11[j].ravel(), D22[j].ravel()
        Dt[:, 0, 1] = Dt[:, 1, 0] = D12[j].ravel()
        r = ref.evaluate(A, Dt, np.zeros(3), w[j].ravel())
        fr[j], Sm[j], rs[j], rt[j] = (r.fr1.reshape(sh), r.S.reshape(sh), r.rStar.reshape(sh), r.rTilde.reshape(sh))
    Pk = nut * Sm ** 2 * np.outer(np.gradient(y), np.gradient(x))
    X = np.broadcast_to(x, (ny, nx))
    dev = fr - 1
    regions = {}
    for name, m in (('plate x > 0', X > 0), ('x >= 0.1', X >= 0.1), ('0.5 <= x <= 2', X >= 0.5)):
        regions[name] = dict(production_weighted_mean_fr1_minus_1=float((Pk * dev)[m].sum() / Pk[m].sum()),
                             production_weighted_mean_abs=float((Pk * abs(dev))[m].sum() / Pk[m].sum()))
    stations = []
    for xs in (0.002, 0.01, 0.05, 0.1, 0.25, 0.5, 0.97, 1.5, 1.9):
        i = int(np.argmin(abs(x - xs))); col = Pk[:, i]; sig = col > 1e-2 * col.max()
        stations.append(dict(x=float(x[i]), production_weighted_mean_fr1_minus_1=float((col * dev[:, i]).sum() / col.sum()),
                             max_abs_fr1_minus_1_where_P_above_1pct=float(abs(dev[sig, i]).max()),
                             r_star_range=[float(rs[sig, i].min()), float(rs[sig, i].max())],
                             r_tilde_range=[float(rt[sig, i].min()), float(rt[sig, i].max())]))
    out = dict(source=str(CASE.relative_to(REPO)) + '/' + TIME, grid=[ny, nx], model='reference.py, inertial frame, '
               'published constants', weight='nu_t S^2 x cell area', regions=regions, stations=stations)
    (HERE / 'plate_apriori.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(regions, indent=1))
    for s in stations:
        print(f"x = {s['x']:.4f}: <f_r1 - 1>_P = {s['production_weighted_mean_fr1_minus_1']:+.2e}, "
              f"max |f_r1 - 1| (P > 1 % max) = {s['max_abs_fr1_minus_1_where_P_above_1pct']:.2e}")


if __name__ == '__main__':
    main()
