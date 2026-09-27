#!/usr/bin/env python3
"""Score a closure run against the SST run of the same case, as the paper does.

    python reproduce/score_case.py <closure_case_dir> <sst_case_dir>

Both directories must come from reproduce/make_case.py and have been run with their Allrun scripts. The
errors against the reference data are computed by the paper's scorer (evaluation/forge/cases.py) at the
final time of each run. The case error is the weighted mean of the ratios of the closure's errors to those
of SST, with the metric weights of the paper (Sec. II A). The airfoil cases, which have no accuracy metric
in the search, are scored by their drag and lift errors with equal weights, as in the paper's tables. The
FAITH separation and reattachment errors are taken against the measured positions in reproduce/cases.json
(x/H = 0.38 and 1.78, as in the paper). The convergence admission of the paper's evaluator (Appendix B),
which needs several saved states per run, is not repeated here.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'evaluation'))


def used_metrics(case, rules):
    names = [n for n, r in rules.items() if r.get('kind', 'error') == 'error' and r.get('primary', True)]
    if not names and case['adapter'] == 'naca':
        names = ['drag_abs_error', 'lift_abs_error']
    return names


def reported_errors(case, result):
    """The scored errors; for the FAITH hill, the bubble errors against the measured positions of the case."""
    errors = dict(result['errors'])
    if case['adapter'] == 'faith' and 'separation_position_error' in errors:
        ref, obs = case['reference'], result.get('observables') or {}
        measured_sep, measured_reatt = ref['measured_separation_x_over_h'], ref['measured_reattachment_x_over_h']
        sep, reatt = obs.get('separation_x_over_h'), obs.get('reattachment_x_over_h')
        missed = measured_reatt - measured_sep
        if sep is None:
            errors['separation_position_error'] = errors['reattachment_position_error'] = missed
        else:
            errors['separation_position_error'] = abs(sep - measured_sep)
            errors['reattachment_position_error'] = abs(reatt - measured_reatt) if reatt is not None else missed
    return errors


def score(run, base):
    """The errors of a closure run and of the SST run of the same case, their ratios and the case error."""
    from forge.cases import score_case          # imported late: needs numpy and fluidfoam
    from forge.metrics import metric_rules
    run, base = Path(run).resolve(), Path(base).resolve()
    ids = {json.loads((d/'reproduce.json').read_text())['case_id'] for d in (run, base)}
    if len(ids) != 1:
        raise ValueError(f'the two runs are of different cases: {sorted(ids)}')
    cid = ids.pop()
    closure = json.loads((run/'reproduce.json').read_text())['closure']
    defs = json.loads((ROOT/'reproduce/cases.json').read_text())
    case = next(c for c in defs['cases'] if c['id'] == cid)
    a, b = (reported_errors(case, score_case(case, d)) for d in (run, base))
    rules = metric_rules(case, defs['contract'])
    used = used_metrics(case, rules)
    missing = [n for n in used if n not in a or n not in b]
    if missing:
        raise ValueError(f'errors missing from a run of {cid}: {", ".join(missing)}')
    rows, num, den = [], 0.0, 0.0
    for name, rule in rules.items():
        if rule.get('kind', 'error') != 'error' or name not in a or name not in b:
            continue
        ratio = a[name]/max(abs(b[name]), float(rule.get('floor', 1e-6)))
        w = (float(rule.get('weight', 1)) or 1.0) if name in used else 0.0
        num, den = num + w*ratio, den + w
        rows.append({'metric': name, 'closure': a[name], 'sst': b[name], 'ratio': ratio, 'weight': w})
    expected = json.loads((ROOT/'reproduce/expected.json').read_text()).get(cid, {}).get(closure)
    ranks = {d.name: json.loads((d/'reproduce.json').read_text()) for d in (run, base)}
    note = [f'{name} ran on {r["nprocs"]} ranks, the paper on {r["paper_nprocs"]}' for name, r in ranks.items()
            if r.get('nprocs', r.get('paper_nprocs')) != r.get('paper_nprocs')]
    return {'case': cid, 'closure': closure, 'metrics': rows, 'case_error': num/den, 'paper': expected, 'notes': note}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    try:
        r = score(sys.argv[1], sys.argv[2])
    except ValueError as e:
        sys.exit(str(e))
    print(f'{r["case"]}: {r["closure"]} against SST')
    print(f'{"metric":34s}{"closure":>13s}{"SST":>13s}{"ratio":>9s}  weight')
    for m in r['metrics']:
        print(f'{m["metric"]:34s}{m["closure"]:13.6g}{m["sst"]:13.6g}{m["ratio"]:9.4f}  {m["weight"]:g}')
    print(f'case error relative to SST: {r["case_error"]:.4f}')
    if r['paper'] is not None:
        print(f'paper:                      {r["paper"]:.4f}   (difference {r["case_error"] - r["paper"]:+.4f})')
    for n in r['notes']:
        print(f'note: {n}; the decomposition differs, so the errors can differ slightly')


if __name__ == '__main__':
    main()
