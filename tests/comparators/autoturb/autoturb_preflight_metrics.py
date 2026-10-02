"""Preflight metrics for the AutoTurb comparator on the McConkey periodic hills (no CFD; reads fields).

For one hill case, given the final time directories of the harness's SST run and of the
kOmegaSSTAutoTurb run (same mesh), computes
  * the separation and reattachment positions on the lower wall, from the sign of the tangential
    velocity in the wall-adjacent cells (the sign of the wall shear stress to first order);
  * the error measures of AutoTurb's Table A.4 (arXiv:2410.10657v1, p. 23): the mean-squared error
    over all cells of the streamwise velocity U1 and of k against the projected DNS
    (data/mcconkey/cache/REF_<case>.npz), each divided by the same quantity for SST.
Writes a JSON record with the published values beside the measured ones and the gate verdict.

Gate (preflight P1, case_1p0): the separated region on the lower wall is shorter than SST's
(reattachment upstream of SST's), the published direction (preprint pp. 16 and 18, Figs. 10-11),
and MSE(U1) is below SST's. The paper reports no separation length; its magnitude is the
normalized MSE, recorded beside ours and not gated.

Usage:
  python autoturb_preflight_metrics.py --case case_1p0 --mesh <case dir with constant/polyMesh> \
      --sst <SST case dir> --sst-time <t> --model <AutoTurb case dir> --model-time <t> --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from fluidfoam import readscalar, readvector

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'evaluation'))
from tedp.foammesh import wall_geometry  # noqa: E402

# Table A.4 of the preprint: MSE of Model-LLMs divided by the MSE of SST
PUBLISHED = {
    'case_0p8': {'U1': 0.59732, 'k': 0.55390, 'tau12': 0.57899, 'table': 'A.4(a)'},
    'case_1p0': {'U1': 0.31749, 'k': 0.31281, 'tau12': 0.50574, 'table': 'A.4(b)'},
    'case_1p2': {'U1': 0.18771, 'k': 0.27540, 'tau12': 0.51952, 'table': 'A.4(c)'},
    'case_1p5': {'U1': 0.15356, 'k': 0.33078, 'tau12': 0.59956, 'table': 'A.4(d)'},
    'convdiv12600': {'U1': 0.17462, 'k': 0.76540, 'tau12': 0.87447, 'table': 'A.4(e)'},
    'cbfs13700': {'U1': 0.10612, 'k': 0.37198, 'tau12': 0.60913, 'table': 'A.4(f)'},
}


def crossings(x: np.ndarray, u: np.ndarray) -> tuple[list[float], list[float]]:
    """Linear-interpolated zero crossings of u(x): (+ to -) separations, (- to +) reattachments."""
    order = np.argsort(x)
    x, u = x[order], u[order]
    seps, reatts = [], []
    for i in range(len(x) - 1):
        a, b = u[i], u[i + 1]
        if a > 0 >= b or a >= 0 > b:
            seps.append(float(x[i] - a * (x[i + 1] - x[i]) / (b - a)))
        elif a < 0 <= b or a <= 0 < b:
            reatts.append(float(x[i] - a * (x[i + 1] - x[i]) / (b - a)))
    return seps, reatts


def wall_flow(mesh: Path, case: Path | None, time: str | None, patch: str, U=None) -> dict:
    geo = wall_geometry(mesh, patch)
    if U is None:
        U = readvector(str(case), time, 'U', verbose=False).T
    normal = geo.face_areas / np.linalg.norm(geo.face_areas, axis=1)[:, None]
    # streamwise tangent in the x-y plane, oriented with +x
    tangent = np.stack([-normal[:, 1], normal[:, 0], np.zeros(len(normal))], axis=1)
    tangent *= np.sign(tangent[:, 0])[:, None]
    ut = np.einsum('ij,ij->i', U[geo.owner_cells], tangent)
    x = geo.face_centres[:, 0]
    seps, reatts = crossings(x, ut)
    out = {'separations_x': seps, 'reattachments_x': reatts}
    if seps and reatts:
        xs = seps[0]
        xr = next((r for r in reatts if r > xs), None)
        out.update({'separation_x': xs, 'reattachment_x': xr,
                    'separated_length': None if xr is None else xr - xs})
    out['reversed_wall_fraction'] = float(np.mean(ut < 0))
    return out


def mse(case: Path, time: str, ref: dict) -> dict:
    U = readvector(str(case), time, 'U', verbose=False)
    k = readscalar(str(case), time, 'k', verbose=False)
    return {'U1': float(np.mean((U[0] - ref['REF_U_1']) ** 2)),
            'k': float(np.mean((k - ref['REF_k']) ** 2))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True)
    p.add_argument('--mesh', required=True, type=Path)
    p.add_argument('--sst', required=True, type=Path)
    p.add_argument('--sst-time', required=True)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--model-time', required=True)
    p.add_argument('--patch', default='bottomWall')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()

    ref = dict(np.load(REPO / 'data' / 'mcconkey' / 'cache' / f'REF_{a.case}.npz'))
    sst_m, mod_m = mse(a.sst, a.sst_time, ref), mse(a.model, a.model_time, ref)
    ratios = {q: mod_m[q] / sst_m[q] for q in ('U1', 'k')}
    record = {
        'case': a.case,
        'published_mse_ratio': PUBLISHED.get(a.case),
        'measured_mse_ratio': ratios,
        'mse_sst': sst_m, 'mse_model': mod_m,
        'note': 'Table A.4 normalizes each MSE by SST; the preprint does not state its SST run, '
                'interpolation or averaging weights beyond "averaged over all mesh points" (p. 10); '
                'the hills share the 99x149 grid (preprint Table 2, p. 13), the channel and the step do not.',
    }
    if a.case.startswith('case_'):
        sst_w = wall_flow(a.mesh, a.sst, a.sst_time, a.patch)
        mod_w = wall_flow(a.mesh, a.model, a.model_time, a.patch)
        ref_U = np.stack([ref['REF_U_1'], ref['REF_U_2'], ref['REF_U_3']], axis=1)
        record['wall'] = {'sst': sst_w, 'model': mod_w,
                          'reference_dns': wall_flow(a.mesh, None, None, a.patch, U=ref_U)}
        shorter = (sst_w.get('separated_length') is not None and mod_w.get('separated_length') is not None
                   and mod_w['separated_length'] < sst_w['separated_length'])
        upstream = (sst_w.get('reattachment_x') is not None and mod_w.get('reattachment_x') is not None
                    and mod_w['reattachment_x'] < sst_w['reattachment_x'])
        record['gate'] = {
            'separated_length_shorter_than_sst': bool(shorter),
            'reattachment_upstream_of_sst': bool(upstream),
            'mse_U1_below_sst': bool(ratios['U1'] < 1.0),
        }
        record['passed'] = all(record['gate'].values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
