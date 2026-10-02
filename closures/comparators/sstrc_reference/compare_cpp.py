"""Compare the C++ SST-RC kernel output with the independent NumPy reference.

Input (the format shared with the C++ unit executable closures/comparators/sstrc_unit):
  * manufactured_states.txt  (this directory; written by analytic_cases.py)
        id A11..A33 D11 D12 D13 D22 D23 D33 Om1 Om2 Om3 omega
  * cpp_fr1.txt               (written by the C++ unit executable closures/comparators/sstrc_unit)
        '#' header/comment lines, then: id S W rStar rTilde fRotation fr1 fScaled

For every state the reference values (reference.evaluate, Smirnov & Menter 2009 Eqs. (1),
(4)-(11)) and, where they exist, the hand-derived closed forms (analytic_cases) are compared
column by column. A state passes when |cpp - ref| <= tol * max(1, |ref|) for every column
(default tol 1e-10; double-precision round-off is ~1e-15). fScaled must equal fr1 when the
C++ ran with Cscale = 1 (the published model has no such coefficient).

Usage:
  python compare_cpp.py <cpp_fr1.txt | directory containing it> [--states FILE] [--tol 1e-10]
                        [--json REPORT.json]
Exit status 0 when every state passes, 1 otherwise, 2 on unreadable input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analytic_cases as ac  # noqa: E402
import reference as ref  # noqa: E402

CPP_COLUMNS = ('S', 'W', 'rStar', 'rTilde', 'fRotation', 'fr1', 'fScaled')
COMPARED = ('S', 'W', 'rStar', 'rTilde', 'fRotation', 'fr1')
DEFAULT_TOL = 1e-10


def read_cpp(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'cpp_fr1.txt'
    rows = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        tok = line.split()
        if len(tok) != 1 + len(CPP_COLUMNS):
            raise ValueError(f'{path}:{n}: expected {1 + len(CPP_COLUMNS)} columns, got {len(tok)}')
        if tok[0] in rows:
            raise ValueError(f'{path}:{n}: duplicate id {tok[0]}')
        rows[tok[0]] = dict(zip(CPP_COLUMNS, (float(t) for t in tok[1:])))
    return path, rows


def reference_values(states_path=ac.STATES_FILE):
    states = ac.read_states(states_path)
    out = {}
    for sid, s in states.items():
        r = ref.evaluate(s['A'], s['DSDt'], s['Om'], s['omega']).as_dict()
        out[sid] = {c: float(r[c]) for c in COMPARED}
    return out


def closed_values():
    return {s.id: {c: float(s.closed[c]) for c in COMPARED} for s in ac.build_states() if s.closed}


def compare(cpp_path, states_path=ac.STATES_FILE, tol=DEFAULT_TOL):
    cpp_file, cpp = read_cpp(cpp_path)
    refv = reference_values(states_path)
    closed = closed_values() if Path(states_path).resolve() == ac.STATES_FILE.resolve() else {}
    missing = sorted(set(refv) - set(cpp))
    extra = sorted(set(cpp) - set(refv))
    max_abs = {c: 0.0 for c in COMPARED}
    max_rel = {c: 0.0 for c in COMPARED}
    worst = {c: None for c in COMPARED}
    failures = []
    per_state = {}
    for sid in sorted(set(refv) & set(cpp)):
        r, k = refv[sid], cpp[sid]
        rec = {}
        bad = []
        for c in COMPARED:
            d = abs(k[c] - r[c]) if np.isfinite(k[c]) and np.isfinite(r[c]) else np.inf
            scaled = d / max(1.0, abs(r[c])) if np.isfinite(d) else np.inf
            rec[c] = dict(cpp=k[c], ref=r[c], abs_diff=d)
            if c in closed.get(sid, {}):
                rec[c]['closed_form'] = closed[sid][c]
                rec[c]['abs_diff_closed'] = abs(k[c] - closed[sid][c])
            if d > max_abs[c]:
                max_abs[c] = d
                worst[c] = dict(id=sid, cpp=k[c], ref=r[c], abs_diff=d)
            max_rel[c] = max(max_rel[c], scaled)
            if not scaled <= tol:
                bad.append(c)
        if abs(k['fScaled'] - k['fr1']) > tol * max(1.0, abs(k['fr1'])):
            bad.append('fScaled!=fr1')
        per_state[sid] = rec
        if bad:
            failures.append(dict(id=sid, columns=bad))
    closed_max = {c: max((v[c]['abs_diff_closed'] for v in per_state.values() if 'abs_diff_closed' in v[c]),
                         default=None) for c in COMPARED}
    report = dict(
        cpp_file=str(cpp_file), states_file=str(states_path), tolerance=tol,
        criterion='|cpp - ref| <= tol * max(1, |ref|) for S, W, rStar, rTilde, fRotation, fr1; fScaled == fr1',
        n_states=len(refv), n_compared=len(per_state), missing=missing, extra=extra,
        max_abs_diff=max_abs, max_scaled_diff=max_rel, max_abs_diff_vs_closed_form=closed_max,
        worst=worst, failures=failures,
        passed=not failures and not missing and not extra,
        per_state=per_state,
    )
    return report


def write_reference_as_cpp(path, states_path=ac.STATES_FILE):
    """Write the reference values in the C++ output format (used by the tests of this tool)."""
    refv = reference_values(states_path)
    lines = ['# id S W rStar rTilde fRotation fr1 fScaled (reference.py, for self-test)']
    for sid, r in refv.items():
        vals = [r[c] for c in COMPARED] + [r['fr1']]
        lines.append(' '.join([sid] + [format(v, '.17g') for v in vals]))
    Path(path).write_text('\n'.join(lines) + '\n')
    return Path(path)


def _summary(rep):
    out = [f"cpp file   : {rep['cpp_file']}",
           f"states     : {rep['n_compared']}/{rep['n_states']} compared"
           + (f", missing {rep['missing']}" if rep['missing'] else '')
           + (f", extra {rep['extra']}" if rep['extra'] else ''),
           f"criterion  : {rep['criterion']} (tol {rep['tolerance']:g})",
           f"{'column':10s} {'max |cpp-ref|':>14s} {'max scaled':>12s} {'max |cpp-closed|':>17s}  worst state"]
    for c in COMPARED:
        w = rep['worst'][c]
        cm = rep['max_abs_diff_vs_closed_form'][c]
        out.append(f"{c:10s} {rep['max_abs_diff'][c]:14.3e} {rep['max_scaled_diff'][c]:12.3e} "
                   f"{(f'{cm:17.3e}' if cm is not None else ' ' * 17)}  {w['id'] if w else '-'}")
    if rep['failures']:
        out.append('failing states: ' + ', '.join(f"{f['id']}({'/'.join(f['columns'])})" for f in rep['failures'][:20])
                   + (' ...' if len(rep['failures']) > 20 else ''))
    out.append('RESULT     : ' + ('PASS' if rep['passed'] else 'FAIL'))
    return '\n'.join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('cpp', help='cpp_fr1.txt or a directory containing it')
    p.add_argument('--states', default=str(ac.STATES_FILE))
    p.add_argument('--tol', type=float, default=DEFAULT_TOL)
    p.add_argument('--json', help='write the full report (per-state values) to this JSON file')
    a = p.parse_args(argv)
    try:
        rep = compare(a.cpp, Path(a.states), a.tol)
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}')
        return 2
    print(_summary(rep))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rep, indent=1, default=float) + '\n')
    return 0 if rep['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
