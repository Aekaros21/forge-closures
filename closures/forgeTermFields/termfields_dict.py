#!/usr/bin/env python3
"""Term dictionaries and checks for forgeTermFields (Fig. 15, where the terms act).

The rotation-limited correction (models/rotation_limited.json; label R1 in the records) has three terms,
each switched by one coefficient: the normal-stress term N by c0, the excess-production term P by c3
(theta only multiplies P) and the rotation term R by c7. The coefficient functions of N, P and R are
those of the full correction with the other two terms' coefficients set to zero; they add up to the
full ones exactly. They are the specs of the ablation variants RLN, RLP and RLR of the term ablation
(config/ablation_variants.json of campaign forge-v3-verify-007-ablation in the records archive; pass it
with --variants to check the match). Within R, the stress channel is its bdelta T1 function and the
source channel its rsource T1 function. The flow-state correction (models/flow_state.json, F01 in the
records) has the same three terms.

  dict   writes system/termFieldsDict for one model (rotation_limited, or flow_state), rendered
         from the canonical spec exactly as the solver's turbulenceProperties (tedp.spec.to_foam_coeffs)
  check  without CFD: (1) the term coefficient functions add up to the full ones on random states,
         (2) a NumPy twin of the per-cell algebra (closures/forgeTermFields/termAlgebra.H, the library's
         clamp, realizability factor and R bound) gives term contributions that sum to the full
         correction, (3) the compiled C++ (testTermAlgebra, the code forgeTermFields uses, with the
         dictionary parsed by the library grammar) agrees with the twin, and (4) the rendered full
         functions equal the model's dictionary models/<model>/turbulenceProperties, which a case reads.

Run from the repository root (no OpenFOAM case is needed; testTermAlgebra is built by
closures/forgeTermFields/Allwmake):
  python3 closures/forgeTermFields/termfields_dict.py dict --model rotation_limited --out <case>/system/termFieldsDict
  python3 closures/forgeTermFields/termfields_dict.py check --binary $FOAM_USER_APPBIN/testTermAlgebra --work <dir>
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # repository root
sys.path.insert(0, str(ROOT/'evaluation'))
from tedp import expr  # noqa: E402
from tedp.spec import canonical, from_json, spec_hash, to_json  # noqa: E402
from tedp.tensors import integrity_basis  # noqa: E402

SOURCES = {
    'rotation_limited': ROOT/'models/rotation_limited.json',
    'flow_state': ROOT/'models/flow_state.json',
}
VARIANTS = None   # config/ablation_variants.json of the records archive (--variants), if given
# term -> coefficients set to zero to isolate it (the ablation variants of Sec. IV of the paper)
TERMS = {'N': (3, 7), 'P': (0, 7), 'R': (0, 3)}
TERM_ORDER = ('N', 'P', 'R')
PAPER = {'N': 'normal-stress term', 'P': 'excess-production term', 'R': 'rotation term'}
SYM = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))   # OpenFOAM symmTensor order


def load(label):
    model = json.loads(SOURCES[label].read_text())
    spec = canonical(from_json(json.dumps(model['spec'])))
    # the model fingerprint of the records (freeze.json) is the spec hash
    model = {'name': model['name'], 'model_fingerprint': spec_hash(from_json(json.dumps(model['spec'])))}
    return model, json.loads(to_json(spec))


def zeroed(spec, zero):
    s = json.loads(json.dumps(spec))
    s['constants'] = [0.0 if i in zero else c for i, c in enumerate(s['constants'])]
    return s


def rendered(spec, section):
    """(tensor, literal expression) exactly as tedp.spec.to_foam_coeffs renders them."""
    return [(t, expr.to_string(expr.substitute_constants(expr.parse(e), spec['constants'])))
            for t, e in spec[section]]


def term_specs(label, spec):
    terms = {name: zeroed(spec, zero) for name, zero in TERMS.items()}
    if label == 'rotation_limited' and VARIANTS is not None:
        # identical to the ablation variants that ran in forge-v3-verify-007-ablation
        variants = {v['label']: v['spec'] for v in json.loads(Path(VARIANTS).read_text())['variants']}
        for name, var in (('N', 'RLN'), ('P', 'RLP'), ('R', 'RLR')):
            v = json.loads(to_json(canonical(from_json(json.dumps(variants[var])))))
            if (v['bdelta'], v['rsource'], v['constants']) != \
                    (terms[name]['bdelta'], terms[name]['rsource'], terms[name]['constants']):
                raise SystemExit(f'term {name} differs from ablation variant {var}')
    return terms


def foam_list(pairs, indent):
    pad = ' '*indent
    body = '\n'.join(f'{pad}    ({t} "{e}")' for t, e in pairs)
    return f'{pad}terms\n{pad}(\n{body}\n{pad});\n'


def write_dict(label, out):
    model, spec = load(label)
    terms = term_specs(label, spec)
    text = ['FoamFile { version 2.0; format ascii; class dictionary; object termFieldsDict; }',
            f'// Generated by closures/forgeTermFields/termfields_dict.py dict --model {label}; do not edit.',
            f'// {model["name"]}, model fingerprint {model["model_fingerprint"]}',
            '// Terms: ' + '; '.join(f'{n} = {PAPER[n]} (coefficients {TERMS[n]} set to zero)' for n in TERM_ORDER),
            f'label {label};', f'gMax {spec["gmax"]};', f'rMaxFactor {spec["r_max_factor"]};',
            f'termOrder ({" ".join(TERM_ORDER)});', 'full', '{']
    for section, key in (('bdelta', 'bDelta'), ('rsource', 'rSource')):
        text += [f'    {key}', '    {', foam_list(rendered(spec, section), 8).rstrip('\n'), '    }']
    text += ['}', 'terms', '{']
    for name in TERM_ORDER:
        text += [f'    {name}', '    {']
        for section, key in (('bdelta', 'bDelta'), ('rsource', 'rSource')):
            text += [f'        {key}', '        {', foam_list(rendered(terms[name], section), 12).rstrip('\n'),
                     '        }']
        text += ['    }']
    text += ['}', '']
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text('\n'.join(text))
    return model, spec, terms


# --------------------------------------------------------------------------------------------------
# checks

def random_states(rng, n, rotating_fraction=0.5):
    """Random normalised strain and rotation tensors, turbulence state and grammar inputs."""
    a = rng.normal(size=(n, 3, 3))
    s = 0.5*(a + np.swapaxes(a, 1, 2))
    s -= np.trace(s, axis1=1, axis2=2)[:, None, None]*np.eye(3)/3
    w = rng.normal(size=(n, 3, 3))
    w = 0.5*(w - np.swapaxes(w, 1, 2))
    # magnitudes from weak to strong, so that every limit is met somewhere
    s *= (10**rng.uniform(-2, 0.7, n))[:, None, None]/np.sqrt(np.einsum('nij,nij->n', s, s))[:, None, None]
    w *= (10**rng.uniform(-2, 0.7, n))[:, None, None]/np.sqrt(np.einsum('nij,nij->n', w, w) + 1e-30)[:, None, None]
    T, inv = integrity_basis(s, w)
    k = 10**rng.uniform(-4, 0, n)
    oms = 10**rng.uniform(0, 3, n)
    nut = k/oms*rng.uniform(0.1, 1.0, n)
    rot = rng.uniform(size=n) < rotating_fraction
    v = {name: np.zeros(n) for name in expr.VARIABLES}
    for i in range(5):
        v[f'I{i+1}'] = inv[:, i]
    v.update(Ret=10**rng.uniform(-1, 3, n), F1=rng.uniform(0, 1, n), PoE=rng.uniform(0, 10, n),
             Gp=rng.uniform(0, 1, n), Gk=rng.uniform(0, 1, n), Apk=rng.uniform(-1, 1, n),
             Psn=rng.uniform(-2, 2, n), Ksn=rng.uniform(-2, 2, n),
             Rf=np.where(rot, 10**rng.uniform(-4, 0, n), 0.0), Rw=rng.uniform(-1, 1, n))
    return {'S': s, 'W': w, 'T': T, 'k': k, 'oms': oms, 'nut': nut, 'Sd': s*oms[:, None, None], 'vars': v}


def coefficient_values(spec, section, v):
    return [expr.evaluate(expr.parse(e), v, spec['constants']) * np.ones_like(v['I1']) for _, e in spec[section]]


def in_triangle(b, tol=1e-6):
    ev = np.linalg.eigvalsh(b)
    return (ev[..., 0] >= -1/3 - tol) & (ev[..., -1] <= 2/3 + tol)


def twin(st, spec, terms, gmax, rmax, betastar=0.09, clip=True):
    """NumPy twin of termAlgebra.H, from the description of libkOmegaSSTBasis (Appendix A.2)."""
    v, k, oms, nut, Sd, T = st['vars'], st['k'], st['oms'], st['nut'], st['Sd'], st['T']
    tens = lambda section: [T[:, int(t[1:]) - 1] for t, _ in spec[section]]      # noqa: E731
    Tb, Tr = tens('bdelta'), tens('rsource')
    gF, hF = coefficient_values(spec, 'bdelta', v), coefficient_values(spec, 'rsource', v)
    gT = {n: coefficient_values(terms[n], 'bdelta', v) for n in TERM_ORDER}
    hT = {n: coefficient_values(terms[n], 'rsource', v) for n in TERM_ORDER}
    out = {'gF': gF, 'hF': hF, 'gT': gT, 'hT': hT}
    # full correction: clamp, realizability factor, stress
    gc = [np.clip(g, -gmax, gmax) for g in gF]
    sb = [np.where(np.abs(g) > gmax, c/np.where(g == 0, 1, g), 1.0) for g, c in zip(gF, gc)]
    b = sum(c[:, None, None]*t for c, t in zip(gc, Tb))
    bB = -(nut/np.maximum(k, 1e-300))[:, None, None]*Sd
    lam = np.ones_like(k)
    if clip:
        bad = ~in_triangle(bB + b)
        lo, hi = np.zeros_like(k), np.ones_like(k)
        for _ in range(6):
            mid = 0.5*(lo + hi)
            ok = in_triangle(bB + mid[:, None, None]*b)
            lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
        lam = np.where(bad, lo, 1.0)
        out['realizability'] = bad
    out['lambda'] = lam
    out['tau'] = 2*(k*lam)[:, None, None]*b
    # source: clamp, contraction, bound
    hc = [np.clip(h, -gmax, gmax) for h in hF]
    sr = [np.where(np.abs(h) > gmax, c/np.where(h == 0, 1, h), 1.0) for h, c in zip(hF, hc)]
    bR = sum(c[:, None, None]*t for c, t in zip(hc, Tr))
    Rraw = 2*k*np.einsum('nij,nij->n', bR, Sd)
    bnd = rmax*betastar*k*oms
    out['R'] = np.clip(Rraw, -bnd, bnd)
    sigma = np.where(np.abs(Rraw) > bnd, out['R']/np.where(Rraw == 0, 1, Rraw), 1.0)
    out['rBound'] = np.abs(Rraw) > bnd
    out['gClamp'] = np.any([np.abs(g) > gmax for g in gF], axis=0)
    out['hClamp'] = np.any([np.abs(h) > gmax for h in hF], axis=0)
    # term split
    out['tauT'], out['RT'] = {}, {}
    for n in TERM_ORDER:
        bj = sum((s*g)[:, None, None]*t for s, g, t in zip(sb, gT[n], Tb))
        out['tauT'][n] = 2*(k*lam)[:, None, None]*bj
        rj = sum((s*h)[:, None, None]*t for s, h, t in zip(sr, hT[n], Tr))
        out['RT'][n] = sigma*2*k*np.einsum('nij,nij->n', rj, Sd)
    return out


def write_states(path, st, spec, gmax, rmax, betastar, clip):
    v = st['vars']
    t1 = next((i for i, (t, _) in enumerate(spec['bdelta']) if t == 'T1'), -1)
    nb, nr = len(spec['bdelta']), len(spec['rsource'])
    rows = [f'{len(st["k"])} {len(expr.VARIABLES)} {nb} {nr} {len(TERM_ORDER)}',
            f'{gmax!r} {rmax!r} {betastar!r} {int(clip)} {t1}']
    comps = lambda m: [m[i, j] for i, j in SYM]                                   # noqa: E731
    for c in range(len(st['k'])):
        row = [v[name][c] for name in expr.VARIABLES] + [st['k'][c], st['oms'][c], st['nut'][c]] + comps(st['Sd'][c])
        for t, _ in spec['bdelta']:
            row += comps(st['T'][c, int(t[1:]) - 1])
        for t, _ in spec['rsource']:
            row += comps(st['T'][c, int(t[1:]) - 1])
        rows.append(' '.join(repr(float(x)) for x in row))
    Path(path).write_text('\n'.join(rows) + '\n')


def read_output(path, nb, nr, nterms):
    data = np.loadtxt(path, ndmin=2)
    out, i = {}, 0

    def take(m):
        nonlocal i
        x = data[:, i:i + m]
        i += m
        return x
    out['lambda'], out['realizability'], out['gClamp'], out['hClamp'], out['rBound'] = take(5).T
    out['tau'], out['R'], out['Gnl'] = take(6), take(1)[:, 0], take(1)[:, 0]
    out['gF'], out['hF'] = take(nb), take(nr)
    out['terms'] = []
    for _ in range(nterms):
        out['terms'].append({'g': take(nb), 'h': take(nr), 'tau': take(6), 'R': take(1)[:, 0],
                             'Gnl': take(1)[:, 0], 'g1': take(1)[:, 0]})
    assert i == data.shape[1]
    return out


def sym6(m):
    return np.stack([m[:, i, j] for i, j in SYM], axis=1)


def rel(a, b):
    scale = max(np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))/scale)


def case_terms(path):
    """(tensor, expression) lists of bDelta and rSource in a rendered turbulenceProperties."""
    text = Path(path).read_text()
    out = {}
    for key in ('bDelta', 'rSource'):
        block = re.search(key + r'\s*\{\s*terms\s*\((.*?)\);\s*\}', text, re.S).group(1)
        out[key] = sorted((t, e.replace(' ', '')) for t, e in re.findall(r'\((T\d+)\s+"([^"]*)"\)', block))
    return out


def check(label, binary, work, n, seed):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    dict_path = work/f'termFieldsDict.{label}'
    model, spec, terms = write_dict(label, dict_path)
    rng = np.random.default_rng(seed)
    report = {'model': label, 'name': model['name'], 'model_fingerprint': model['model_fingerprint'],
              'states': n, 'seed': seed, 'checks': {}}
    # (1) additivity of the coefficient functions
    st = random_states(rng, n)
    worst = 0.0
    for section in ('bdelta', 'rsource'):
        full = coefficient_values(spec, section, st['vars'])
        parts = [coefficient_values(terms[t], section, st['vars']) for t in TERM_ORDER]
        for i, f in enumerate(full):
            s = sum(p[i] for p in parts)
            worst = max(worst, float(np.max(np.abs(s - f)/np.maximum(1, np.abs(f)))))
    # the rotation term vanishes without frame rotation; N has no source; P and R no T3/T4
    v0 = {**st['vars'], 'Rf': np.zeros(n)}
    r0 = max(float(np.max(np.abs(x))) for sec in ('bdelta', 'rsource') for x in coefficient_values(terms['R'], sec, v0))
    n_src = max(float(np.max(np.abs(x))) for x in coefficient_values(terms['N'], 'rsource', st['vars']))
    pr_t34 = max(float(np.max(np.abs(x))) for t in ('P', 'R') for (name, _), x in
                 zip(terms[t]['bdelta'], coefficient_values(terms[t], 'bdelta', st['vars'])) if name in ('T3', 'T4'))
    report['checks']['coefficients_add_up'] = {'max_residual': worst, 'passed': worst < 1e-12}
    report['checks']['rotation_term_zero_without_frame_rotation'] = {'max_abs': r0, 'passed': r0 == 0.0}
    report['checks']['normal_stress_term_has_no_source'] = {'max_abs': n_src, 'passed': n_src == 0.0}
    report['checks']['T3_T4_only_in_normal_stress_term'] = {'max_abs': pr_t34, 'passed': pr_t34 == 0.0}
    # (2) and (3): split sums to the full correction; C++ equals the twin
    for tag, gmax, rmax in (('production_limits', spec['gmax'], spec['r_max_factor']),
                            ('tight_limits', 0.02, 0.05)):
        tw = twin(st, spec, terms, gmax, rmax)
        sum_tau = sum(tw['tauT'][t] for t in TERM_ORDER)
        sum_R = sum(tw['RT'][t] for t in TERM_ORDER)
        entry = {'gMax': gmax, 'rMaxFactor': rmax,
                 'cells_realizability_active': int(np.sum(tw['realizability'])),
                 'cells_stress_clamped': int(np.sum(tw['gClamp'])),
                 'cells_source_clamped': int(np.sum(tw['hClamp'])),
                 'cells_R_bound': int(np.sum(tw['rBound'])),
                 'twin_terms_sum_to_full_tau': rel(sum_tau, tw['tau']),
                 'twin_terms_sum_to_full_R': rel(sum_R, tw['R'])}
        states = work/f'states_{label}_{tag}.txt'
        result = work/f'cxx_{label}_{tag}.txt'
        write_states(states, st, spec, gmax, rmax, 0.09, True)
        run = subprocess.run([str(binary), str(dict_path), str(states), str(result)], capture_output=True, text=True)
        if run.returncode:
            raise SystemExit(f'testTermAlgebra failed: {run.stderr}')
        cx = read_output(result, len(spec['bdelta']), len(spec['rsource']), len(TERM_ORDER))
        cx_sum_tau = sum(t['tau'] for t in cx['terms'])
        cx_sum_R = sum(t['R'] for t in cx['terms'])
        entry.update({
            'cxx_coefficients_vs_python_spec': max(rel(cx['gF'][:, i], tw['gF'][i]) for i in range(len(tw['gF'])))
            if tw['gF'] else 0.0,
            'cxx_source_coefficients_vs_python_spec': max(rel(cx['hF'][:, i], tw['hF'][i]) for i in range(len(tw['hF'])))
            if tw['hF'] else 0.0,
            'cxx_term_coefficients_vs_python': max(rel(cx['terms'][j]['g'][:, i], tw['gT'][t][i])
                                                   for j, t in enumerate(TERM_ORDER) for i in range(len(tw['gF']))),
            'cxx_lambda_mismatches': int(np.sum(cx['lambda'] != tw['lambda'])),
            'cxx_realizability_flag_mismatches': int(np.sum(cx['realizability'].astype(bool) != tw['realizability'])),
            'cxx_full_tau_vs_twin': rel(cx['tau'], sym6(tw['tau'])),
            'cxx_full_R_vs_twin': rel(cx['R'], tw['R']),
            'cxx_terms_sum_to_full_tau': rel(cx_sum_tau, cx['tau']),
            'cxx_terms_sum_to_full_R': rel(cx_sum_R, cx['R']),
            'cxx_terms_sum_vs_twin_full_tau': rel(cx_sum_tau, sym6(tw['tau'])),
            'cxx_terms_sum_vs_twin_full_R': rel(cx_sum_R, tw['R']),
            'cxx_term_tau_vs_twin': max(rel(cx['terms'][j]['tau'], sym6(tw['tauT'][t])) for j, t in enumerate(TERM_ORDER)),
            'cxx_term_R_vs_twin': max(rel(cx['terms'][j]['R'], tw['RT'][t]) for j, t in enumerate(TERM_ORDER)),
        })
        tol = 1e-10
        entry['passed'] = (entry['twin_terms_sum_to_full_tau'] < tol and entry['twin_terms_sum_to_full_R'] < tol
                           and entry['cxx_terms_sum_to_full_tau'] < tol and entry['cxx_terms_sum_to_full_R'] < tol
                           and entry['cxx_terms_sum_vs_twin_full_tau'] < 1e-9 and entry['cxx_terms_sum_vs_twin_full_R'] < 1e-9
                           and entry['cxx_coefficients_vs_python_spec'] < 1e-12
                           and entry['cxx_term_coefficients_vs_python'] < 1e-12
                           and entry['cxx_lambda_mismatches'] <= max(1, n//10000)
                           and entry['cxx_term_tau_vs_twin'] < 1e-9 and entry['cxx_term_R_vs_twin'] < 1e-9)
        report['checks'][f'split_{tag}'] = entry
    # (4) the rendered full functions equal the model's dictionary (and any case under runs/ built with it)
    want = {'bDelta': sorted((t, e) for t, e in rendered(spec, 'bdelta')),
            'rSource': sorted((t, e) for t, e in rendered(spec, 'rsource'))}
    found = []
    candidates = [ROOT/'models'/label/'turbulenceProperties',
                  *sorted((ROOT/'runs').glob(f'*_{label}/constant/turbulenceProperties'))]
    for p in candidates:
        try:
            got = case_terms(p)
        except AttributeError:
            continue
        if got == {k: [(t, e.replace(' ', '')) for t, e in v] for k, v in want.items()}:
            found.append(str(p.relative_to(ROOT)))
    report['checks']['matches_a_solver_turbulenceProperties'] = {
        'local_matches': found[:3], 'count': len(found), 'passed': bool(found),
        'note': 'forgeTermFields also enforces this match against the case at run time (fatal otherwise)'}
    report['checks']['matches_the_ablation_variants'] = {
        'status': ('checked against ' + str(VARIANTS) if VARIANTS is not None and label == 'rotation_limited'
                   else 'not checked')}
    report['passed'] = all(c.get('passed', True) for c in report['checks'].values())
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('dict')
    d.add_argument('--model', choices=sorted(SOURCES), default='rotation_limited')
    d.add_argument('--out', required=True)
    c = sub.add_parser('check')
    c.add_argument('--binary', required=True, help='compiled testTermAlgebra')
    c.add_argument('--work', required=True)
    c.add_argument('--states', type=int, default=20000)
    c.add_argument('--seed', type=int, default=20261001)
    c.add_argument('--report', default=str(ROOT/'build/termfields_check.json'))
    for s in (d, c):
        s.add_argument('--variants', help='config/ablation_variants.json of the records archive (optional check)')
    a = ap.parse_args()
    global VARIANTS
    VARIANTS = a.variants
    if a.cmd == 'dict':
        write_dict(a.model, a.out)
        print(f'{a.out} written ({a.model})')
        return 0
    reports = [check(label, a.binary, a.work, a.states, a.seed) for label in ('rotation_limited', 'flow_state')]
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    record = {'purpose': 'forgeTermFields term split verified without CFD (Fig. 15)',
              'passed': all(r['passed'] for r in reports), 'models': reports}
    Path(a.report).write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
