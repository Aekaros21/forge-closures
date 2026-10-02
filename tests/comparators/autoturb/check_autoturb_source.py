"""Unit check of the AutoTurb production correction (kOmegaSSTAutoTurb) on manufactured states.

No mesh and no CFD: the C++ test executable ``testAutoTurbSource`` runs ``autoTurb::evaluate``, the
one-point function that ``kOmegaSSTAutoTurb::updateR`` calls for every cell, on random velocity
gradients and turbulence states, and forms R and the omega source as the model does. This script
generates the states, builds and runs the executable against OpenFOAM v2312 in a scratch directory,
and compares its output with an independent NumPy implementation written from the publication:

    Zhang et al., AutoTurb, arXiv:2410.10657v1, Eq. (11), p. 15, with Eqs. (6)-(9), pp. 6-7:
        R = 2 k dU_i/dx_j [ (sin(lambda1) + 0.5) T1_ij + T2_ij ],
        T1 = S/omega, T2 = (S Omega - Omega S)/omega^2, lambda1 = S~_mn S~_nm,
        S = (L + L^T)/2 (trace-less), Omega = (L - L^T)/2, L_ij = dU_i/dx_j;
    omega equation: + gamma (P_k + R)/nu_t (Eq. (4); SpaRTA Eqs. (4)-(5)), gamma = F1 5/9 + (1-F1) 0.44.

The NumPy side works in the paper's index order (L_ij = dU_i/dx_j) and receives OpenFOAM's
gradU_ij = dU_j/dx_i only through the file, so a transposition error on either side is caught.

Usage:
    python tests/comparators/autoturb/check_autoturb_source.py --out <evidence.json> [--workdir DIR]
"""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
KERNEL = REPO / 'closures' / 'comparators' / 'autoturb' / 'autoTurbProduction.H'
APP = HERE / 'testAutoTurbSource'
FOAM_BASHRC = '/usr/lib/openfoam/openfoam2312/etc/bashrc'
PUBLISHED = (1.0, 0.5, 1.0)  # alpha1Sin, alpha1Const, alpha2 of Eq. (11)
GAMMA1, GAMMA2 = 5.0 / 9.0, 0.44
SEED = 20261001


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_states(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, slice]]:
    """Rows: L (paper order, 3x3), k, omega, nut, F1. Four groups of manufactured states."""
    groups = {}
    blocks = []

    def turbulence(n):
        k = 10.0 ** rng.uniform(-10, 1, n)
        omega = 10.0 ** rng.uniform(-3, 6, n)
        # nut = a1 k / max(a1 omega, F2 S): between the unlimited k/omega and a tenth of it
        nut = k / omega * rng.uniform(0.1, 1.0, n)
        f1 = rng.uniform(0.0, 1.0, n)
        return k, omega, nut, f1

    def scaled(a, omega):
        # velocity-gradient magnitude relative to omega spans 1e-3..20, so lambda1 spans ~1e-6..1e3
        mag = 10.0 ** rng.uniform(-3, 1.3, a.shape[0])
        return a * (mag * omega)[:, None, None]

    def add(name, L, k, omega, nut, f1):
        start = sum(b.shape[0] for b in blocks)
        rows = np.concatenate([L.reshape(-1, 9), k[:, None], omega[:, None], nut[:, None], f1[:, None]], axis=1)
        blocks.append(rows)
        groups[name] = slice(start, start + rows.shape[0])

    # 1. general 3D incompressible gradients (trace removed exactly in floating point is not
    #    possible; the residual trace is O(1e-16) of the gradient)
    n = 20000
    k, omega, nut, f1 = turbulence(n)
    a = rng.normal(size=(n, 3, 3))
    a -= np.trace(a, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3)
    add('3d_incompressible', scaled(a, omega), k, omega, nut, f1)

    # 2. two-dimensional (x-y) incompressible gradients, the setting of the publication
    n = 5000
    k, omega, nut, f1 = turbulence(n)
    a = np.zeros((n, 3, 3))
    a[:, 0, 0] = rng.normal(size=n)
    a[:, 1, 1] = -a[:, 0, 0]
    a[:, 0, 1] = rng.normal(size=n)
    a[:, 1, 0] = rng.normal(size=n)
    add('2d_incompressible', scaled(a, omega), k, omega, nut, f1)

    # 3. gradients with a nonzero trace (discretely non-solenoidal); the paper's S is trace-less
    n = 2000
    k, omega, nut, f1 = turbulence(n)
    a = rng.normal(size=(n, 3, 3))
    add('3d_with_trace', scaled(a, omega), k, omega, nut, f1)

    # 4. structured states: simple shear dU/dy = s with lambda1 = s^2/(2 omega^2) placed at the
    #    zeros of sin(lambda1) + 0.5 and elsewhere, pure rotation (R = 0), zero gradient (R = 0)
    lam = np.concatenate([
        np.array([0.0, 1e-8, 0.045, 0.5, 1.0, math.pi / 2, math.pi, 7 * math.pi / 6, 11 * math.pi / 6,
                  3 * math.pi / 2, 2 * math.pi + 7 * math.pi / 6, 50.0, 300.0]),
        np.linspace(0.0, 40.0, 401)])
    n = lam.size
    k = np.full(n, 0.01)
    omega = np.full(n, 30.0)
    nut = k / omega
    f1 = np.full(n, 0.5)
    s = np.sqrt(2.0 * lam) * omega
    L = np.zeros((n, 3, 3))
    L[:, 0, 1] = s  # dU/dy
    add('simple_shear', L, k, omega, nut, f1)
    n = 50
    k, omega, nut, f1 = turbulence(n)
    L = np.zeros((n, 3, 3))
    w = rng.normal(size=(n, 3)) * omega[:, None]
    L[:, 0, 1], L[:, 1, 0] = w[:, 0], -w[:, 0]
    L[:, 0, 2], L[:, 2, 0] = w[:, 1], -w[:, 1]
    L[:, 1, 2], L[:, 2, 1] = w[:, 2], -w[:, 2]
    add('pure_rotation', L, k, omega, nut, f1)
    n = 10
    k, omega, nut, f1 = turbulence(n)
    add('zero_gradient', np.zeros((n, 3, 3)), k, omega, nut, f1)
    return np.concatenate(blocks), groups


def reference(states: np.ndarray, a1s: float, a1c: float, a2: float) -> dict[str, np.ndarray]:
    """Independent NumPy evaluation in the paper's notation (L_ij = dU_i/dx_j)."""
    L = states[:, :9].reshape(-1, 3, 3)
    k, omega, nut, f1 = states[:, 9], states[:, 10], states[:, 11], states[:, 12]
    eye = np.eye(3)
    S = 0.5 * (L + np.swapaxes(L, 1, 2))
    S = S - np.trace(S, axis1=1, axis2=2)[:, None, None] / 3.0 * eye  # trace-less S (paper, p. 6)
    Om = 0.5 * (L - np.swapaxes(L, 1, 2))
    St = S / omega[:, None, None]
    Wt = Om / omega[:, None, None]
    lam1 = np.einsum('amn,anm->a', St, St)
    T1 = St
    T2 = np.einsum('aik,akj->aij', St, Wt) - np.einsum('aik,akj->aij', Wt, St)
    alpha1 = a1s * np.sin(lam1) + a1c
    t1 = np.einsum('aij,aij->a', T1, L)
    t2 = np.einsum('aij,aij->a', T2, L)
    bR = alpha1[:, None, None] * T1 + a2 * T2
    R = 2.0 * k * np.einsum('aij,aij->a', bR, L)
    gamma = f1 * GAMMA1 + (1.0 - f1) * GAMMA2
    omega_source = np.where(R == 0.0, 0.0, gamma * R / nut)
    # scale of the T2 work, for an absolute round-off test of the identity T2:S = 0
    t2_scale = omega * np.einsum('aij,aij->a', St, St) * np.sqrt(np.einsum('aij,aij->a', Wt, Wt))
    return {'lambda1': lam1, 'alpha1': alpha1, 't1Work': t1, 't2Work': t2, 'RbyK': R / k, 'R': R,
            'omegaSource': omega_source, 't2_scale': t2_scale}


def build(workdir: Path) -> tuple[Path, str]:
    src = workdir / 'app'
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(APP, src)
    # the app includes the kernel through ../../../../closures/comparators/autoturb: rebuild that layout
    kdir = workdir / 'closures' / 'comparators' / 'autoturb'
    kdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(KERNEL, kdir / KERNEL.name)
    nested = workdir / 'tests' / 'comparators' / 'autoturb'
    nested.mkdir(parents=True, exist_ok=True)
    target = nested / 'testAutoTurbSource'
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(src), str(target))
    bindir = workdir / 'bin'
    bindir.mkdir(exist_ok=True)
    log = workdir / 'build_test.log'
    cmd = (f'source {FOAM_BASHRC} && export FOAM_USER_APPBIN={bindir} && cd {target} && wmake')
    with log.open('w') as handle:
        subprocess.run(['bash', '--noprofile', '--norc', '-c', cmd], stdout=handle, stderr=subprocess.STDOUT,
                       check=True)
    return bindir / 'testAutoTurbSource', log.read_text()


def run(exe: Path, states_file: Path, coeffs: tuple[float, float, float]) -> np.ndarray:
    cmd = f'source {FOAM_BASHRC} && {exe} {states_file} ' + ' '.join(repr(c) for c in coeffs)
    out = subprocess.run(['bash', '--noprofile', '--norc', '-c', cmd], capture_output=True, text=True, check=True)
    return np.loadtxt(out.stdout.splitlines()) if out.stdout else np.zeros((0, 7))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--workdir')
    args = parser.parse_args()
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix='autoturb_unit_'))
    workdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    states, groups = make_states(rng)
    # file carries OpenFOAM order: gradU_ij = dU_j/dx_i = L_ji
    L = states[:, :9].reshape(-1, 3, 3)
    gradU = np.swapaxes(L, 1, 2).reshape(-1, 9)
    states_file = workdir / 'states.txt'
    with states_file.open('w') as handle:
        handle.write(f'{states.shape[0]}\n')
        np.savetxt(handle, np.concatenate([gradU, states[:, 9:]], axis=1), fmt='%.17g')

    exe, build_log = build(workdir)
    cols = ['lambda1', 'alpha1', 't1Work', 't2Work', 'RbyK', 'R', 'omegaSource']
    cpp = run(exe, states_file, PUBLISHED)
    ref = reference(states, *PUBLISHED)
    assert cpp.shape == (states.shape[0], 7), cpp.shape

    def rel(name, mask=None):
        a, b = cpp[:, cols.index(name)], ref[name]
        m = np.ones(a.size, bool) if mask is None else mask
        m &= b != 0.0
        d = np.abs(a[m] - b[m]) / np.abs(b[m])
        return float(d.max()) if d.size else 0.0, int(m.sum())

    results = {}
    nonzero_R = np.abs(ref['R']) > 0
    # cancellation: alpha1 = sin(lambda1) + 0.5 near zero makes the relative difference of R an
    # amplification of the last bit of lambda1 (condition number |lambda1 cos(lambda1)/alpha1|)
    cond = np.abs(ref['lambda1'] * np.cos(ref['lambda1'])) / np.maximum(np.abs(ref['alpha1']), 1e-300)
    well_conditioned = cond < 1e3
    for name in ('lambda1', 't1Work', 'RbyK', 'R', 'omegaSource'):
        mx, count = rel(name)
        mx_wc, count_wc = rel(name, well_conditioned.copy())
        results[name] = {'max_rel_diff_all_nonzero': mx, 'n_all_nonzero': count,
                         'max_rel_diff_well_conditioned': mx_wc, 'n_well_conditioned': count_wc}
    d_alpha = np.abs(cpp[:, 1] - ref['alpha1'])
    results['alpha1'] = {'max_abs_diff': float(d_alpha.max())}
    # R difference normalized by the magnitude of the T1 work (well defined for every state)
    scale_R = 2.0 * states[:, 9] * (np.abs(ref['t1Work']) * (np.abs(np.sin(ref['lambda1'])) + 0.5)
                                    + ref['t2_scale'])
    m = scale_R > 0
    results['R_normalized_by_term_scale'] = {
        'max_diff': float((np.abs(cpp[m, 5] - ref['R'][m]) / scale_R[m]).max()), 'n': int(m.sum())}
    # identity T2:grad(U) = 0, both implementations
    m = ref['t2_scale'] > 0
    results['t2Work_identity'] = {
        'max_abs_t2Work_over_scale_cpp': float((np.abs(cpp[m, 3]) / ref['t2_scale'][m]).max()),
        'max_abs_t2Work_over_scale_numpy': float((np.abs(ref['t2Work'][m]) / ref['t2_scale'][m]).max()),
        'n': int(m.sum())}
    # exact zeros where the publication gives R = 0
    zero_groups = {}
    for g in ('pure_rotation', 'zero_gradient'):
        sl = groups[g]
        zero_groups[g] = {'max_abs_R_cpp': float(np.abs(cpp[sl, 5]).max()),
                          'max_abs_omegaSource_cpp': float(np.abs(cpp[sl, 6]).max())}
    results['zero_states'] = zero_groups
    # simple shear: R = 2 k (sin(lambda1) + 0.5) s^2/(2 omega) analytically
    sl = groups['simple_shear']
    s = states[sl, 1]
    analytic = 2.0 * states[sl, 9] * (np.sin(s ** 2 / (2.0 * states[sl, 10] ** 2)) + 0.5) \
        * s ** 2 / (2.0 * states[sl, 10])
    scale = 2.0 * states[sl, 9] * 1.5 * s ** 2 / (2.0 * states[sl, 10])
    mm = scale > 0
    results['simple_shear_analytic'] = {
        'max_diff_over_scale': float((np.abs(cpp[sl, 5][mm] - analytic[mm]) / scale[mm]).max()),
        'n': int(mm.sum()),
        'note': 'R = 2k (sin(s^2/(2 omega^2)) + 0.5) s^2/(2 omega) for dU/dy = s; '
                'lambda1 placed at 0, 0.045 (log layer, beta*/2), pi, 7pi/6 and 11pi/6 (alpha1 = 0) and on a grid'}

    # zero coefficients: R and the omega source must be exactly zero (bitwise)
    cpp0 = run(exe, states_file, (0.0, 0.0, 0.0))
    results['zero_coefficients'] = {
        'all_R_exactly_zero': bool(np.all(cpp0[:, 5] == 0.0)),
        'all_omegaSource_exactly_zero': bool(np.all(cpp0[:, 6] == 0.0)),
        'n': int(cpp0.shape[0])}

    # per-group maxima of the relative difference of R
    per_group = {}
    for g, sl in groups.items():
        a, b = cpp[sl, 5], ref['R'][sl]
        mk = (b != 0.0) & well_conditioned[sl]
        per_group[g] = {'n': sl.stop - sl.start,
                        'max_rel_diff_R_well_conditioned': float((np.abs(a[mk] - b[mk]) / np.abs(b[mk])).max())
                        if mk.any() else None}
    results['per_group'] = per_group

    tol = 1e-10
    checks = {
        'lambda1_rel': results['lambda1']['max_rel_diff_all_nonzero'] < tol,
        'R_rel_well_conditioned': results['R']['max_rel_diff_well_conditioned'] < tol,
        'omegaSource_rel_well_conditioned': results['omegaSource']['max_rel_diff_well_conditioned'] < tol,
        'R_normalized_all_states': results['R_normalized_by_term_scale']['max_diff'] < tol,
        't2_identity': results['t2Work_identity']['max_abs_t2Work_over_scale_cpp'] < 1e-12,
        'zero_states_exact': all(v['max_abs_R_cpp'] == 0.0 or v['max_abs_R_cpp'] < 1e-300 for v in zero_groups.values()),
        'simple_shear_analytic': results['simple_shear_analytic']['max_diff_over_scale'] < tol,
        'zero_coefficients_bitwise': results['zero_coefficients']['all_R_exactly_zero']
        and results['zero_coefficients']['all_omegaSource_exactly_zero'],
    }
    payload = {
        'check': 'kOmegaSSTAutoTurb production correction vs independent NumPy (publication notation)',
        'source': 'Zhang et al., AutoTurb, arXiv:2410.10657v1 Eq. (11) p. 15; Eqs. (3)-(4), (6)-(9) pp. 6-7',
        'kernel': str(KERNEL.relative_to(REPO)), 'kernel_sha256': sha256(KERNEL),
        'test_app': str((APP / 'testAutoTurbSource.C').relative_to(REPO)),
        'test_app_sha256': sha256(APP / 'testAutoTurbSource.C'),
        'openfoam': 'v2312 (' + FOAM_BASHRC + ')',
        'seed': SEED, 'n_states': int(states.shape[0]),
        'groups': {g: sl.stop - sl.start for g, sl in groups.items()},
        'coefficients': dict(zip(('alpha1Sin', 'alpha1Const', 'alpha2'), PUBLISHED)),
        'tolerance_relative': tol,
        'well_conditioned_rule': 'condition number |lambda1 cos(lambda1)|/|alpha1| < 1e3 '
                                 '(excludes states within ~1e-3 of a zero of sin(lambda1)+0.5)',
        'n_ill_conditioned': int((~well_conditioned).sum()),
        'results': results, 'checks': checks, 'passed': all(checks.values()),
        'build_log_tail': build_log.splitlines()[-3:],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'passed': payload['passed'], 'checks': checks,
                      'R': results['R'], 'omegaSource': results['omegaSource'],
                      'R_normalized': results['R_normalized_by_term_scale'],
                      't2': results['t2Work_identity'], 'n_states': payload['n_states']}, indent=1))
    return 0 if payload['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
