"""Check SST-RC preflight results against published_targets.json (post-processing only, no CFD).

Inputs are the campaign evaluator results of the comparator and of SST for the same case
(``runs/experiments/<hash>/result.json``, with ``observables`` y, u, wall_friction_ratio for the
rotating channels and wall_x, wall_cf, drag_coefficient for the plate). A case directory
(``.../attempt-NNN/case``) or an experiment directory may be given instead: the result.json is
looked up in it and up to two levels above.

  python closures/comparators/sstrc_reference/check_targets.py MODEL SST [--case ID] [--out report.json]

Cases: rotchan_ro10, rotchan_ro50 (published Smirnov-Menter 2009 Fig. 1 targets), rotchan_ro00
(reduction to SST), tmr_plate_137x97 (f_r1 = 1: friction within 0.1 % of SST for x >= 0.1).
Prints a summary and a JSON report with "passed" (primary criteria) to --out or stdout.
Exit status 0 passed, 1 failed, 2 unusable input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TARGETS = HERE / 'published_targets.json'


def load_result(path):
    p = Path(path)
    cands = [p] if p.is_file() else [p / 'result.json', p.parent / 'result.json', p.parent.parent / 'result.json']
    for c in cands:
        if c.is_file():
            d = json.loads(c.read_text())
            if isinstance(d, dict) and isinstance(d.get('observables'), dict):
                return c, d
    raise FileNotFoundError(f'no evaluator result.json with observables at {path}')


def _inside(v, band):
    return v is not None and np.isfinite(v) and band[0] <= v <= band[1]


def velocity_max_case(y, u):
    """y/h of the velocity maximum: vertex maximum refined by the parabola through its neighbours."""
    k = int(np.argmax(u))
    if 0 < k < len(u) - 1:
        q = np.polyfit(y[k - 1:k + 2], u[k - 1:k + 2], 2)
        return float(-q[1] / (2 * q[0])), float(np.polyval(q, -q[1] / (2 * q[0])))
    return float(y[k]), float(u[k])


def check_channel(t, obs, sst_obs):
    y = np.asarray(obs['y'], float); u = np.asarray(obs['u'], float)
    o = np.argsort(y); y, u = y[o], u[o]
    ratio = float(obs['wall_friction_ratio'])
    ymax, umax = velocity_max_case(y, u)
    du = float(np.interp(0.5, y, u) - np.interp(-0.5, y, u))
    prof = t['published_profile']
    yH = np.asarray(prof['y_over_H']); up = np.asarray(prof['u_over_Um'])
    yy = 2 * yH - 1
    rms = float(np.sqrt(np.mean((np.interp(yy, y, u) - up) ** 2)))
    vm = t['velocity_maximum_y_over_H']
    band_ycase = [2 * vm['accept'][0] - 1, 2 * vm['accept'][1] - 1]
    checks = {
        'wall_friction_ratio': dict(measured=ratio, published=t['wall_shear_ratio_pressure_over_suction']['value'],
                                    accept=t['wall_shear_ratio_pressure_over_suction']['accept'], primary=True),
        'velocity_maximum_y_over_h_case': dict(measured=ymax, published=vm['case_coordinate']['value'],
                                               accept=band_ycase, primary=True),
        'velocity_asymmetry_dU_over_Um': dict(measured=du, published=t['velocity_asymmetry_dU']['value'],
                                              accept=t['velocity_asymmetry_dU']['accept'], primary=True),
        'u_max_over_Um': dict(measured=umax, published=t['u_max_over_Um']['value'],
                              accept=t['u_max_over_Um']['accept'], primary=False),
        'profile_rms_over_Um': dict(measured=rms, published=0.0, accept=[0.0, t['profile_rms_max']['value']],
                                    primary=False),
    }
    if sst_obs is not None:
        s = float(sst_obs['wall_friction_ratio'])
        checks['sst_control_wall_friction_ratio'] = dict(measured=s, published=1.0, accept=[0.999, 1.001], primary=False)
    return checks


def check_ro00(obs, sst_obs, tol=1e-4):
    out = {}
    for q in ('wall_cf_bottom', 'wall_cf_top', 'wall_friction_ratio'):
        m, s = float(obs[q]), float(sst_obs[q])
        out[f'{q}_relative_change'] = dict(measured=abs(m - s) / abs(s), published=0.0, accept=[0.0, tol], primary=True,
                                           model=m, sst=s)
    return out


def check_plate(t, obs, sst_obs):
    x = np.asarray(obs['wall_x'], float); cf = np.asarray(obs['wall_cf'], float)
    xs = np.asarray(sst_obs['wall_x'], float); cfs = np.asarray(sst_obs['wall_cf'], float)
    if len(x) != len(xs) or np.max(np.abs(x - xs)) > 1e-9:
        raise ValueError('model and SST plate stations differ')
    rel = np.abs(cf - cfs) / np.maximum(np.abs(cfs), t['scale_floor'])
    down = x >= t['x_min']
    i_down = int(np.argmax(np.where(down, rel, -1)))
    i_le = int(np.argmax(np.where(~down, rel, -1))) if (~down).any() else None
    out = {
        'wall_cf_max_relative_change_x_ge_0.1': dict(measured=float(rel[down].max()), at_x=float(x[i_down]),
                                                    published=0.0, accept=[0.0, t['relative_tolerance']], primary=True),
        'drag_coefficient_relative_change': dict(
            measured=abs(obs['drag_coefficient'] - sst_obs['drag_coefficient']) / abs(sst_obs['drag_coefficient']),
            published=0.0, accept=[0.0, t['relative_tolerance']], primary=True),
        'wall_cf_rms_relative_change': dict(
            measured=abs(obs['wall_cf_rms'] - sst_obs['wall_cf_rms']) / abs(sst_obs['wall_cf_rms']),
            published=0.0, accept=[0.0, t['relative_tolerance']], primary=False),   # includes the leading edge
    }
    if i_le is not None:
        out['wall_cf_max_relative_change_leading_edge_x_lt_0.1'] = dict(
            measured=float(rel[~down].max()), at_x=float(x[i_le]), published=0.0,
            accept=[0.0, t['leading_edge_relative_tolerance']], primary=False)
    return out


def check(model, sst=None, case=None, targets=TARGETS):
    T = json.loads(Path(targets).read_text())
    mfile, m = load_result(model)
    sfile, s = load_result(sst) if sst else (None, None)
    case = case or m.get('case_id')
    if s is not None and s.get('case_id') not in (None, case):
        raise ValueError(f"SST result is for {s.get('case_id')}, not {case}")
    obs, sobs = m['observables'], (s or {}).get('observables')
    chans = {v['case_id']: v for v in T['rotating_channel'].values()}
    if case in chans:
        checks = check_channel(chans[case], obs, sobs)
    elif case == 'rotchan_ro00':
        if sobs is None:
            raise ValueError('rotchan_ro00 needs the SST result')
        checks = check_ro00(obs, sobs)
    elif case == T['flat_plate']['case_id']:
        if sobs is None:
            raise ValueError('the plate check needs the SST result')
        checks = check_plate(T['flat_plate'], obs, sobs)
    else:
        raise ValueError(f'no SST-RC target for case {case}')
    for c in checks.values():
        c['passed'] = bool(_inside(c['measured'], c['accept']))
    primary = [c['passed'] for c in checks.values() if c['primary']]
    return dict(case=case, model_result=str(mfile), sst_result=str(sfile) if sfile else None,
                targets=str(Path(targets)), source=T['source']['reference'] + ', ' + T['source']['figure'],
                checks=checks, passed=all(primary), all_passed=all(c['passed'] for c in checks.values()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('model'); p.add_argument('sst', nargs='?')
    p.add_argument('--case'); p.add_argument('--out'); p.add_argument('--targets', default=str(TARGETS))
    a = p.parse_args(argv)
    try:
        rep = check(a.model, a.sst, a.case, a.targets)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps(dict(passed=None, error=str(exc))))
        return 2
    lines = [f"case {rep['case']}: {'PASS' if rep['passed'] else 'FAIL'} (primary criteria)"]
    for k, c in rep['checks'].items():
        lines.append(f"  {'P' if c['primary'] else 's'} {k:48s} {c['measured']:.6g} in [{c['accept'][0]:.6g}, "
                     f"{c['accept'][1]:.6g}] (published {c['published']:.6g}): {'ok' if c['passed'] else 'OUT'}")
    text = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + '\n')
        print('\n'.join(lines))
    else:
        print(text)
        print('\n'.join(lines), file=sys.stderr)
    return 0 if rep['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
