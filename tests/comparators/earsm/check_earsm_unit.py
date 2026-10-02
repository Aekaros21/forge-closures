"""Unit qualification of the BSL-EARSM comparator (closures/comparators/earsm); no CFD.

Builds libkOmegaEARSM (class kOmegaEARSM) and the test programs tests/comparators/earsm/testEARSM
and testEARSMLoad in a scratch directory against OpenFOAM v2312, checks that the library loads and
registers kOmegaEARSM in the incompressible RAS table, runs

  (i)   homogeneous-shear equilibrium states, 2D closed forms, the 3D implicit-equation residual,
        P/eps consistency of the cubic and the OpenFOAM gradient convention
  (ii)  the closed-form root N of the cubic, Eqs. (6)-(8) of Menter, Garbaruk and Egorov (2012),
        on a sweep of the invariants including the branch boundary P2 = 0
  (iii) the divergence of the explicit stress of a manufactured field on in-memory box meshes
        (16^3, 32^3, 64^3 cells; no mesh files, no flow solve)

and compares (i) and (ii) with an independent Python implementation written from the paper
(numpy companion-matrix roots polished by Newton iteration in 50-digit decimal arithmetic).
Writes build/qualification/earsm/earsm_unit_tests.json and the logs next to it.

Usage: python tests/comparators/earsm/check_earsm_unit.py
"""
from __future__ import annotations

import decimal
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / 'closures'
OUT = REPO / 'build/qualification/earsm'   # the paper's run wrote its records here: see the records archive
BASHRC = '/usr/lib/openfoam/openfoam2312/etc/bashrc'

C1 = 1.8
C1P = 2.25 * (C1 - 1.0)
A1_PUB = 1.245
A1_WJ = 1.2
BETA_STAR = 0.09
KAPPA = 0.41


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: str, log: Path, cwd: Path | None = None) -> str:
    proc = subprocess.run(['bash', '--noprofile', '--norc', '-c', f'source {BASHRC} && {cmd}'],
                          cwd=cwd, capture_output=True, text=True)
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f'{cmd} failed, see {log}')
    return proc.stdout


def parse(text: str):
    results, sweep = {}, []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'RESULT':
            results[parts[1]] = float(parts[2])
        elif parts[0] == 'SWEEP':
            sweep.append([float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4]),
                          float(parts[5])])
    return results, sweep


# ------------------------------------------------------------------------------------------
# independent reference (written from the paper, not from the C++)
# ------------------------------------------------------------------------------------------
def n_reference(iis: float, iiw: float) -> float:
    """Largest real root of N^3 - C1' N^2 - (2.7 IIS + 2 IIW) N + 2 C1' IIW = 0 (Eq. 6)."""
    coeffs = [1.0, -C1P, -(2.7 * iis + 2.0 * iiw), 2.0 * C1P * iiw]
    roots = np.roots(coeffs)
    scale = max(1.0, float(np.abs(roots).max()))
    real = roots[np.abs(roots.imag) <= 1e-6 * scale].real
    x = decimal.Decimal(repr(float(real.max())))
    with decimal.localcontext() as ctx:
        ctx.prec = 50
        c = [decimal.Decimal(repr(float(v))) for v in coeffs]
        for _ in range(60):
            f = ((c[0] * x + c[1]) * x + c[2]) * x + c[3]
            df = (3 * c[0] * x + 2 * c[1]) * x + c[2]
            if df == 0:
                break
            step = f / df
            x -= step
            if abs(step) <= abs(x) * decimal.Decimal('1e-40'):
                break
        # the polished value must still be the largest real root
        return float(x)


def p2_reference(iis: float, iiw: float) -> float:
    p1 = C1P * (C1P ** 2 / 27 + 9 / 20 * iis - 2 / 3 * iiw)
    return p1 ** 2 - (C1P ** 2 / 9 + 9 / 10 * iis + 2 / 3 * iiw) ** 3


def shear_reference(sigma: float, a1: float) -> dict:
    """Simple shear, paper convention: S12 = W12 = sigma/2; Menter's beta1, beta4 (p. 92)."""
    s = 0.5 * sigma
    S = np.array([[0, s, 0], [s, 0, 0], [0, 0, 0]])
    W = np.array([[0, s, 0], [-s, 0, 0], [0, 0, 0]])
    I = np.eye(3)
    iis, iiw, iv = np.trace(S @ S), np.trace(W @ W), np.trace(S @ W @ W)
    n = n_reference(iis, iiw)
    q = (n * n - 2 * iiw) / a1
    q1 = q / 6 * (2 * n * n - iiw)
    b1, b3, b4, b6 = -n / q, -2 * iv / (n * q1), -1 / q, -n / q1
    T3 = W @ W - iiw / 3 * I
    T4 = S @ W - W @ S
    T6 = S @ W @ W + W @ W @ S - 2 / 3 * iv * I - iiw * S
    a = b1 * S + b3 * T3 + b4 * T4 + b6 * T6
    return dict(N=n, a=a, PoE=-float(np.sum(a * S)), CmuEff=-b1 / 2)


def shear_targets():
    g1 = 0.075 / BETA_STAR - 0.5 * KAPPA ** 2 / math.sqrt(BETA_STAR)
    g2 = 0.0828 / BETA_STAR - 0.856 * KAPPA ** 2 / math.sqrt(BETA_STAR)
    return {'logLayer': 1.0, 'homShearSet1': 0.075 / (g1 * BETA_STAR),
            'homShearSet2': 0.0828 / (g2 * BETA_STAR)}


def check_shear(res: dict) -> dict:
    out, worst = {}, 0.0
    for tag, a1 in (('pub', A1_PUB), ('wj', A1_WJ)):
        for name, target in shear_targets().items():
            sig = brentq(lambda x: shear_reference(x, a1)['PoE'] - target, 1e-3, 100.0, xtol=1e-15,
                         rtol=1e-15, maxiter=500)
            ref = shear_reference(sig, a1)
            key = f'shear.{tag}.{name}'
            pairs = {'sigma': sig, 'N': ref['N'], 'CmuEff': ref['CmuEff'], 'a11': ref['a'][0, 0],
                     'a22': ref['a'][1, 1], 'a33': ref['a'][2, 2], 'a12': ref['a'][0, 1],
                     'PoE': ref['PoE']}
            diffs = {q: abs(res[f'{key}.{q}'] - v) / max(abs(v), 1e-300) if abs(v) > 0 else
                     abs(res[f'{key}.{q}']) for q, v in pairs.items()}
            worst = max(worst, max(diffs.values()))
            out[key] = {'cpp': {q: res[f'{key}.{q}'] for q in pairs},
                        'python': pairs, 'relative_difference': diffs}
    return {'states': out, 'max_relative_difference': worst}


def check_sweep(sweep: list) -> dict:
    worst_rel, worst_res, branch_mismatch, below_c1p, largest_ok = 0.0, 0.0, 0, 0, True
    near_boundary = 0
    for iis, iiw, n, branch, residual in sweep:
        ref = n_reference(iis, iiw)
        worst_rel = max(worst_rel, abs(n - ref) / abs(ref))
        scale = max(abs(n) ** 3, C1P * n * n, (2.7 * iis + 2 * abs(iiw)) * n, 2 * C1P * abs(iiw), 1.0)
        worst_res = max(worst_res, abs(residual) / scale)
        p2 = p2_reference(iis, iiw)
        p1 = C1P * (C1P ** 2 / 27 + 9 / 20 * iis - 2 / 3 * iiw)
        if abs(p2) <= 1e-12 * p1 ** 2:
            near_boundary += 1
        elif (branch == 1) != (p2 < 0):
            branch_mismatch += 1
        if n < C1P * (1 - 1e-14):
            below_c1p += 1
        roots = np.roots([1.0, -C1P, -(2.7 * iis + 2.0 * iiw), 2.0 * C1P * iiw])
        real = roots[np.abs(roots.imag) <= 1e-6 * max(1.0, float(np.abs(roots).max()))].real
        if real.max() > n * (1 + 1e-9):
            largest_ok = False
    return {'points': len(sweep), 'points_within_1e-12_of_P2_zero': near_boundary,
            'max_relative_difference_to_largest_real_root': worst_rel,
            'max_scaled_cubic_residual': worst_res, 'branch_flag_mismatches': branch_mismatch,
            'points_with_N_below_C1prime': below_c1p, 'formula_returns_largest_real_root': largest_ok,
            'IIS_range': [min(s[0] for s in sweep), max(s[0] for s in sweep)],
            'IIW_range': [min(s[1] for s in sweep), max(s[1] for s in sweep)]}


def fixture(case: Path) -> None:
    """system/ dictionaries only (Time and the finite-volume schemes); the mesh is in memory."""
    head = 'FoamFile { version 2.0; format ascii; class dictionary; object %s; }\n'
    (case / 'system').mkdir(parents=True)
    (case / 'system/controlDict').write_text(head % 'controlDict' + (
        'application testEARSM; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n'
        'deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii;\n'
        'writePrecision 17; runTimeModifiable false;\n'))
    (case / 'system/fvSchemes').write_text(head % 'fvSchemes' + (
        'ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\n'
        'divSchemes { default Gauss linear; }\nlaplacianSchemes { default Gauss linear corrected; }\n'
        'interpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n'))
    (case / 'system/fvSolution').write_text(head % 'fvSolution' + 'solvers {}\n')


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='qualify_earsm_'))
    try:
        # the comparator alone, exactly as the harness builds it (evaluation/forge_deploy)
        (work / 'lib').mkdir()
        shutil.copytree(SRC / 'comparators/earsm', work / 'src')
        run(f'export FOAM_USER_LIBBIN={work}/lib && cd {work}/src && wmake libso',
            OUT / 'build_libkOmegaEARSM.log')
        lib = work / 'lib/libkOmegaEARSM.so'
        lib_sha = sha256(lib)

        # the test program against the same sources
        tree = work / 'tree'
        tests = REPO / 'tests/comparators/earsm'
        shutil.copytree(SRC / 'comparators/earsm', tree / 'closures/comparators/earsm')
        shutil.copytree(SRC / 'kOmegaSSTBasis/basisTensors', tree / 'closures/kOmegaSSTBasis/basisTensors')
        for name in ('testEARSM', 'testEARSMLoad'):
            shutil.copytree(tests / name, tree / f'tests/comparators/earsm/{name}',
                            ignore=shutil.ignore_patterns('linux64*', 'lnInclude'))
        (work / 'bin').mkdir()
        for name in ('testEARSM', 'testEARSMLoad'):
            run(f'export FOAM_USER_APPBIN={work}/bin && cd {tree}/tests/comparators/earsm/{name} && wmake',
                OUT / f'build_{name}.log')
        exe = work / 'bin/testEARSM'
        load_text = run(f'{work}/bin/testEARSMLoad {lib}', OUT / 'testEARSMLoad.log')
        load_ok = 'after load: 1' in load_text and 'registered before load: 0' in load_text

        shared = SRC / 'comparators/earsm_reference/shared_states.csv'
        states_rows = None
        if shared.is_file():
            st_text = run(f'{exe} -states {shared} -out {OUT / "cpp_states.csv"}',
                          OUT / 'testEARSM_states.log')
            states_rows = parse(st_text)[0].get('states.rows')

        algebra_text = run(f'{exe} -algebra', OUT / 'testEARSM_algebra.log')
        res, sweep = parse(algebra_text)

        case = work / 'fixture'
        fixture(case)
        div = {}
        for n in (16, 32, 64):
            text = run(f'{exe} -divergence -n {n} -case {case}', OUT / f'testEARSM_divergence_n{n}.log')
            div[n] = parse(text)[0]
    finally:
        shutil.rmtree(work, ignore_errors=True)

    shear = check_shear(res)
    sweep_check = check_sweep(sweep)

    ns = sorted(div)
    def order(key, a, b):
        return math.log(div[a][key] / div[b][key]) / math.log(b / a)
    divergence = {
        'grids': {str(n): {k: v for k, v in div[n].items()} for n in ns},
        'observed_order_rms_interior': {f'{a}-{b}': order('div.rmsErrorInterior', a, b)
                                        for a, b in zip(ns, ns[1:])},
        'observed_order_max_interior': {f'{a}-{b}': order('div.maxErrorInterior', a, b)
                                        for a, b in zip(ns, ns[1:])},
        'relative_rms_error_finest': div[ns[-1]]['div.rmsErrorInterior'] / div[ns[-1]]['div.rmsExactInterior'],
        'interior': 'cells at least two layers from the boundary',
    }

    checks = {
        'i_shear_cpp_vs_python_max_relative_difference': (shear['max_relative_difference'], 1e-10),
        'i_closed_form_2d_max_abs': (max(v for k, v in res.items() if '.closedForm.' in k), 1e-14),
        'i_implicit_equation_3d_any_N_max_relative_residual':
            (res['implicit3D.anyN.withT9.maxRelativeResidual'], 1e-13),
        'i_implicit_equation_3d_cubic_N_max_relative_residual':
            (res['implicit3D.cubicN.withT9.maxRelativeResidual'], 1e-13),
        'i_cubic_consistency_2d_A1_1.2_max_relative': (res['consistency2D.A1_1.2.maxRelative'], 1e-11),
        'i_gradient_convention_shear_max_abs': (res['convention.shear.maxAbsDiff'], 1e-14),
        'i_gradient_convention_random_max_abs': (res['convention.normalized.maxAbsDiff'], 1e-14),
        'ii_root_max_relative_difference': (sweep_check['max_relative_difference_to_largest_real_root'], 1e-12),
        'ii_scaled_cubic_residual': (sweep_check['max_scaled_cubic_residual'], 1e-13),
        'iii_reynolds_stress_decomposition_max_abs': (div[ns[-1]]['reynoldsStress.maxAbsDecompositionError'], 1e-15),
        'iii_basis_vs_integrity_basis_max_abs': (div[ns[-1]]['basis.maxAbsDifferenceToIntegrityBasis'], 1e-12),
    }
    passed = {k: bool(v <= tol) for k, (v, tol) in checks.items()}
    passed['ii_branch_flags_match_P2_sign'] = sweep_check['branch_flag_mismatches'] == 0
    passed['ii_formula_returns_largest_real_root'] = sweep_check['formula_returns_largest_real_root']
    passed['ii_N_not_below_C1prime'] = sweep_check['points_with_N_below_C1prime'] == 0
    passed['i_beta_denominators_positive'] = res['denominators.min'] > 0
    passed['iii_rms_order_finest_pair_at_least_1.8'] = list(divergence['observed_order_rms_interior'].values())[-1] >= 1.8
    passed['library_loads_and_registers_kOmegaEARSM'] = load_ok
    passed['iii_max_interior_error_decreasing'] = all(
        div[a]['div.maxErrorInterior'] > div[b]['div.maxErrorInterior'] for a, b in zip(ns, ns[1:]))

    log_layer = {
        'published_A1_1.245': {q: res[f'shear.pub.logLayer.{q}'] for q in
                               ('sigma', 'N', 'CmuEff', 'a11', 'a22', 'a33', 'a12', 'kappaEff')},
        'wallin_johansson_A1_1.2': {q: res[f'shear.wj.logLayer.{q}'] for q in
                                    ('sigma', 'N', 'CmuEff', 'a11', 'a22', 'a33', 'a12', 'kappaEff')},
        'target': 'betaStar = Cmu = 0.09 (log-layer calibration of the BSL omega equation with kappa = 0.41)',
        'note': ('N = C1prime + 9/4 = 4.05 exactly at P/eps = 1 with A1 = 1.2 (Eq. 5); with the published '
                 'A1 = 1.245 in Q and the cubic as printed, CmuEff = 0.0934 at P/eps = 1, i.e. an effective '
                 'von Karman constant 0.41 (CmuEff/0.09)^(3/4) = 0.421 (WJ value 0.400). The published '
                 'recalibration brings the log layer from 0.400 to 0.421 around the BSL value 0.41.'),
    }

    evidence = {
        'scope': 'BSL-EARSM comparator unit tests; no CFD, no mesh files, no flow solve',
        'reference': 'Menter, Garbaruk and Egorov (2012), Progress in Flight Physics 3, 89-104',
        'shared_states': ({'input': str(shared.relative_to(REPO)), 'output':
                           str((OUT / 'cpp_states.csv').relative_to(REPO)), 'rows': states_rows,
                           'compared_by': 'closures/comparators/earsm_reference (independent reference)'}
                          if states_rows is not None else 'shared_states.csv not present at run time'),
        'library': {'name': 'libkOmegaEARSM.so', 'sha256_local_build': lib_sha,
                    'openfoam': 'v2312 (/usr/lib/openfoam/openfoam2312)',
                    'build_log': str((OUT / 'build_libkOmegaEARSM.log').relative_to(REPO))},
        'source_sha256': {str(p.relative_to(REPO)): sha256(p) for p in
                          sorted((SRC / 'comparators/earsm').rglob('*')) if p.is_file()
                          and 'lnInclude' not in p.parts and 'linux64GccDPInt32Opt' not in p.parts},
        'test_source_sha256': {str(p.relative_to(REPO)): sha256(p) for p in
                               sorted((REPO / 'tests/comparators/earsm').rglob('*')) if p.is_file()
                               and 'linux64GccDPInt32Opt' not in p.parts and '__pycache__' not in p.parts},
        'integrity_basis_sha256': sha256(SRC / 'kOmegaSSTBasis/basisTensors/integrityBasis.H'),
        'i_homogeneous_shear': shear,
        'i_log_layer': log_layer,
        'i_other': {k: v for k, v in res.items() if not k.startswith('shear.')},
        'ii_root_sweep': sweep_check,
        'iii_divergence': divergence,
        'checks': {k: {'value': v, 'tolerance': tol} for k, (v, tol) in checks.items()},
        'passed': passed,
        'all_passed': all(passed.values()),
    }
    path = OUT / 'earsm_unit_tests.json'
    path.write_text(json.dumps(evidence, indent=1, default=float) + '\n')
    print(json.dumps({'all_passed': evidence['all_passed'], 'passed': passed,
                      'checks': evidence['checks'],
                      'orders': divergence['observed_order_rms_interior'],
                      'library_sha256': lib_sha}, indent=1, default=float))
    return 0 if evidence['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
