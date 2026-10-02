"""Fields of the terms of the rotation-limited correction (Fig. 15, where the terms act), offline.

Run with ../.venv-figures/bin/python tools/term_fields.py [case ...] (from FORGE_PoF). No CFD is run and no case is
built: every case is a stored converged solution. Writes, per case, FORGE_PoF/field_cache/termfields/<case>/
(arrays.npz, termfields.json and the OpenFOAM fields of the terms with the mesh under foam/) and
evidence/term_fields.json (provenance and checks), and marks the TERMFIELD handoff ready.

Model. The rotation-limited correction (Appendix A.5) has three terms, each switched by one coefficient
(../v3verify/scripts/ablation_variants.py): the normal-stress term N (c0), the excess-production term P (c3) and the
rotation term R (c7). The coefficient functions of each term are those of the correction with the other two terms'
coefficients set to zero (the specs RLN, RLP, RLR of the ablation); they add up to the full functions exactly. Without
frame rotation the rotation-limited correction is the flow-state correction (evidence/nonrotating_equivalence.json),
so the stored flow-state solutions serve for the airfoil and the duct.

Two independent evaluations, compared cell by cell:
  1. NumPy (this file): basis tensors and invariants (tedp.tensors.integrity_basis), inputs F1, Re_t, Pi and the
     flow-state inputs as defined in libkOmegaSSTBasis, the coefficient functions with tedp.expr.evaluate on the
     frozen spec, and the split of the converged correction with the library's clamp, realizability factor and
     R bound (twin in ../v3verify/scripts/termfields_dict.py, itself checked against the compiled C++ on random
     states). The solver-written nonlinearStress checks the full stress.
  2. The post-processing utility ../v3verify/src/forgeTermFields (OpenFOAM v2312; it includes, never rebuilds, the
     library's basis-tensor and expression code). It supplies what depends on the discretisation and that the
     stored solutions do not hold: grad(U), grad(k), grad(omega), grad(p) with the case's schemes and the meshWave
     wall distance, and writes its own term fields, which must equal the NumPy ones.

Sources (read-only): ../archive/paper_pof/data/fields/<case>/<model>/ (hump R1, airfoil and duct flow-state),
../v3verify/results/*/pbs/<hash>/request.json (identities), ../v3verify/runs/experiments/<hash>/result.json.
The rotating channel (R1, Ro = 0.10) exists only on HX1; its fields are requested in the TERMFIELD handoff and read
from field_cache/termfields_inputs/rotchan_ro10/ when transferred.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from fluidfoam import readscalar, readsymmtensor, readtensor, readvector

P = Path(__file__).resolve().parents[1]
ROOT = P.parent
VER = ROOT/'v3verify'
ARCH = ROOT/'archive/paper_pof/data/fields'
CACHE = P/'field_cache/termfields'
INPUTS = P/'field_cache/termfields_inputs'
EVIDENCE = P/'evidence/term_fields.json'
HANDOFF = VER/'reports/overnight_20261001/handoff/termfield.json'
UTIL_SRC = VER/'src/forgeTermFields'
UTIL_BIN = UTIL_SRC/'bin'
BASHRC = Path('/usr/lib/openfoam/openfoam2312/etc/bashrc')
sys.path.insert(0, str(VER/'src'))
sys.path.insert(0, str(VER/'scripts'))
from tedp import expr  # noqa: E402
from tedp.tensors import integrity_basis  # noqa: E402
from tedp.holdout_cases import hump, tmr_naca, rotchan  # noqa: E402
import termfields_dict as tfd  # noqa: E402

SMALL = 1e-15          # OpenFOAM SMALL (double): kMin and omegaMin defaults
VSMALL = 1e-300
BETA_STAR, ALPHA_OMEGA2, C1 = 0.09, 0.856, 10.0
SYM = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

# The four panels. 'experiment' is the experiment whose stored fields are read; 'r1_experiment' the run of the
# rotation-limited correction itself in forge-v3-verify-007-ablation (identical equations without rotation).
CASES = {
    'nasa_hump_fine': dict(panel='a', model='R1', source=ARCH/'nasa_hump_fine/R1', time='8000', nu=hump.NU,
                           frame=(0.0, 0.0, 0.0),
                           experiment='bc7d8e3d4b256c1fab67f02c41dd49ebe2d0661d47a04e0d528228fde19bf7d7',
                           r1_experiment='bc7d8e3d4b256c1fab67f02c41dd49ebe2d0661d47a04e0d528228fde19bf7d7'),
    'naca0012_a015_225x65': dict(panel='b', model='F01', source=ARCH/'naca0012_a015_225x65/F01', time='80000',
                                 nu=tmr_naca.NU, frame=(0.0, 0.0, 0.0),
                                 experiment='313acd8282bdd1270468fac0478fb040e4d5dc86a43ad9cb0623ba72726999da',
                                 r1_experiment='acd7ea1167d09193ea88b340893f4dd7bc469706d8fe55799a6ee327c4b3321c'),
    'squareDuct_Re_2000': dict(panel='c', model='F01', source=ARCH/'squareDuct_Re_2000/F01', time='5000',
                               nu=None, frame=(0.0, 0.0, 0.0),
                               experiment='df04277bc27ab8ddb7a9cefd5f615032bff8f3993a2bd598c328d3ecd8f46102',
                               r1_experiment='0109663532de853615e87326fc77a0d2ad00f48f6e768e5e1ca9222202e490e5'),
    # Ro = 0.10: the HX1 run behind the 007 record if transferred, else its local reproduction (same model, case,
    # 40 000 iterations, one rank; errors equal to 1e-4 relative; 'unqualified' only under the later Uy/Uz
    # residual policy of the verification protocol)
    'rotchan_ro10': dict(panel='d', model='R1', time='40000', nu=None, frame=(0.0, 0.0, rotchan.omega_for(0.10)),
                         sources=[(INPUTS/'rotchan_ro10',
                                   '3cdbd6627280391d933974223980845dfecd1f475a336539f60b2e7a38956757'),
                                  (VER/'runs/experiments/e2ef3a9534bf12234a53d78341402432eb2e0ec2cdff5f77203acbd0bcaaa373'
                                   '/attempt-001/case',
                                   'e2ef3a9534bf12234a53d78341402432eb2e0ec2cdff5f77203acbd0bcaaa373')],
                         r1_experiment='3cdbd6627280391d933974223980845dfecd1f475a336539f60b2e7a38956757'),
    # optional: panel (b) on the fine mesh of Fig. 12 (flow-state run of forge-v3-verify-009-airfoil-fine), if transferred
    'naca0012_a015_449x129': dict(panel='b (fine mesh)', model='F01', source=INPUTS/'naca0012_a015_449x129',
                                  time='160000', nu=tmr_naca.NU, frame=(0.0, 0.0, 0.0),
                                  experiment='7ad70050f57f11f44d5c335fcb75101f15925375d5b13f86c6675be878c6d33c',
                                  r1_experiment=None),
}
DEFAULT_CASES = ('nasa_hump_fine', 'naca0012_a015_225x65', 'squareDuct_Re_2000', 'rotchan_ro10')
REQUIRED = ('U', 'k', 'omega', 'nut', 'p', 'nonlinearStress')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def duct_nu():
    from tedp.data import mcconkey
    text = (mcconkey.case_dir('squareDuct_Re_2000')/'constant/transportProperties').read_text()
    return float(text.split('\nnu')[1].split(';')[0].split()[-1])


def identity(key):
    """Experiment identity of the stored run, from the campaign's request journal."""
    hits = sorted(VER.glob(f'results/*/pbs/{key}/request.json'))
    record = {'experiment_hash': key, 'request': None, 'campaign': None}
    if hits:
        req = json.loads(hits[0].read_text())
        record.update(request=str(hits[0].relative_to(ROOT)), request_sha256=sha256(hits[0]),
                      campaign=hits[0].parts[-4], case_id=req['case']['id'],
                      end_time=req['protocol'].get('end_time'), ranks=req.get('ranks'))
    res = VER/f'runs/experiments/{key}/result.json'
    if res.exists():
        r = json.loads(res.read_text())
        record.update(result=str(res.relative_to(ROOT)), result_sha256=sha256(res), status=r.get('status'),
                      final_time=r.get('time'))
    return record


def ensure_utility():
    exe = UTIL_BIN/'forgeTermFields'
    sources = sorted(p for p in UTIL_SRC.rglob('*') if p.is_file() and p.suffix in ('.C', '.H')
                     and 'linux64' not in str(p) and 'testTermAlgebra' not in str(p))
    if exe.exists() and all(p.stat().st_mtime <= exe.stat().st_mtime for p in sources):
        return exe
    UTIL_BIN.mkdir(exist_ok=True)
    run = subprocess.run(['bash', '--noprofile', '--norc', '-c',
                          f'source {BASHRC} >/dev/null 2>&1; export FOAM_USER_APPBIN={UTIL_BIN}; wmake'],
                         cwd=UTIL_SRC, capture_output=True, text=True)
    if run.returncode or not exe.exists():
        raise SystemExit('forgeTermFields build failed:\n' + run.stdout[-3000:] + run.stderr[-3000:])
    return exe


def stage(case, cfg, spec_label):
    """Post-processing copy of the stored solution: mesh, system, the final-time fields, the model dictionaries."""
    src, t = cfg['source'], cfg['time']
    foam = CACHE/case/'foam'
    if foam.exists():
        shutil.rmtree(foam)
    (foam/'constant').mkdir(parents=True)
    shutil.copytree(src/'system', foam/'system')
    shutil.copytree(src/'constant/polyMesh', foam/'constant/polyMesh', symlinks=False)
    (foam/t).mkdir()
    for name in REQUIRED + ('wallShearStress',):
        if (src/t/name).exists():
            shutil.copy2(src/t/name, foam/t/name)
    # the model dictionaries the solver had (stored solutions keep only the mesh and the numerics)
    from tedp.casegen import render_turbulence_properties
    from tedp.spec import from_json
    _, spec = tfd.load(spec_label)
    text = render_turbulence_properties(from_json(json.dumps(spec)), mode='expressions')
    if any(cfg['frame']):
        text = text.replace('        gMax            ', '        frameOmega      (%r %r %r);\n        gMax            '
                            % tuple(cfg['frame']), 1)
    if (src/'constant/turbulenceProperties').exists():
        shutil.copy2(src/'constant/turbulenceProperties', foam/'constant/turbulenceProperties')
    else:
        (foam/'constant/turbulenceProperties').write_text(text)
    if (src/'constant/transportProperties').exists():
        shutil.copy2(src/'constant/transportProperties', foam/'constant/transportProperties')
    else:
        (foam/'constant/transportProperties').write_text(
            'FoamFile { version 2.0; format ascii; class dictionary; object transportProperties; }\n'
            f'transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {cfg["nu"]!r};\n')
    # the utility runs serially on the reconstructed fields; no function objects
    control = (foam/'system/controlDict').read_text()
    if 'functions' in control:
        control = control[:control.index('functions')] + '\n'
    (foam/'system/controlDict').write_text(control + '\nfunctions {}\n')
    tfd.write_dict(spec_label, foam/'system/termFieldsDict')
    return foam


def run_utility(foam, t):
    exe = ensure_utility()
    log = foam/'log.forgeTermFields'
    run = subprocess.run(['bash', '--noprofile', '--norc', '-c',
                          f'source {BASHRC} >/dev/null 2>&1; {exe} -time {t} -writeInputs'],
                         cwd=foam, capture_output=True, text=True, timeout=3600)
    log.write_text(run.stdout + run.stderr)
    if run.returncode:
        raise SystemExit(f'forgeTermFields failed in {foam}; see {log}')
    return json.loads((foam/f'postProcessing/forgeTermFields/{t}/summary.json').read_text())


def tensor33(a):
    """fluidfoam (9, n) row-major -> (n, 3, 3)."""
    return a.T.reshape(-1, 3, 3)


def sym33(a):
    out = np.zeros((a.shape[1], 3, 3))
    for c, (i, j) in enumerate(SYM):
        out[:, i, j] = out[:, j, i] = a[c]
    return out


def numpy_terms(foam, t, cfg, spec, terms):
    """The independent NumPy evaluation, from the stored fields and the discretised inputs."""
    rd = lambda f, n: f(str(foam), t, n, verbose=False)                                   # noqa: E731
    U, k, om, nut, p = rd(readvector, 'U'), rd(readscalar, 'k'), rd(readscalar, 'omega'), rd(readscalar, 'nut'), \
        rd(readscalar, 'p')
    ns = sym33(rd(readsymmtensor, 'nonlinearStress'))
    g = tensor33(rd(readtensor, 'tfGradU'))
    gk, gw, gp = rd(readvector, 'tfGradK').T, rd(readvector, 'tfGradOmega').T, rd(readvector, 'tfGradP').T
    y, C, V = rd(readscalar, 'tfWallDist'), rd(readvector, 'tfC').T, rd(readscalar, 'tfV')
    nu, Om = cfg['nu'], np.array(cfg['frame'], dtype=float)
    n = len(k)
    I = np.eye(3)
    symm = 0.5*(g + np.swapaxes(g, 1, 2))
    Sd = symm - np.trace(symm, axis1=1, axis2=2)[:, None, None]*I/3
    oms = np.maximum(om, SMALL)
    frame = np.array([[0.0, Om[2], -Om[1]], [-Om[2], 0.0, Om[0]], [Om[1], -Om[0], 0.0]])
    Sh = Sd/oms[:, None, None]
    Wh = (0.5*(g - np.swapaxes(g, 1, 2)) + frame)/oms[:, None, None]
    T, inv = integrity_basis(Sh, Wh)
    # F1 of kOmegaSSTBase (v2312)
    CD = 2*ALPHA_OMEGA2*np.einsum('ni,ni->n', gk, gw)/om
    arg1 = np.minimum(np.minimum(np.maximum(np.sqrt(k)/(BETA_STAR*om*y), 500*nu/(y**2*om)),
                                 4*ALPHA_OMEGA2*k/(np.maximum(CD, 1e-10)*y**2)), 10.0)
    F1 = np.tanh(arg1**4)
    Ret = k/(nu*oms)
    G2S = nut*2*np.einsum('nij,nij->n', symm, symm)
    PoE = np.minimum(G2S/np.maximum(BETA_STAR*k*oms, SMALL), 10.0)
    A = oms*np.sqrt(np.maximum(k, SMALL))
    a, gg = gp/A[:, None], gk/A[:, None]
    ma, mg = np.linalg.norm(a, axis=1), np.linalg.norm(gg, axis=1)
    curl = np.stack([g[:, 1, 2] - g[:, 2, 1], g[:, 2, 0] - g[:, 0, 2], g[:, 0, 1] - g[:, 1, 0]], axis=1)
    oh, zh = Om[None, :]/oms[:, None], curl/oms[:, None]
    mo, mz = np.linalg.norm(oh, axis=1), np.linalg.norm(zh, axis=1)
    eps = 1e-3
    v = {name: np.zeros(n) for name in expr.VARIABLES}
    for i in range(5):
        v[f'I{i+1}'] = inv[:, i]
    v.update(Ret=Ret, F1=F1, PoE=PoE, Gp=ma/(1 + ma), Gk=mg/(1 + mg),
             Apk=np.einsum('ni,ni->n', a, gg)/(ma*mg + eps),
             Psn=np.einsum('ni,nij,nj->n', a, Sh, a)/(np.einsum('ni,ni->n', a, a) + eps),
             Ksn=np.einsum('ni,nij,nj->n', gg, Sh, gg)/(np.einsum('ni,ni->n', gg, gg) + eps),
             Rf=mo, Rw=np.einsum('ni,ni->n', oh, zh)/(mo*mz + eps))
    st = {'T': T, 'k': k, 'oms': oms, 'nut': nut, 'Sd': Sd, 'vars': v}
    tw = tfd.twin(st, spec, terms, spec['gmax'], spec['r_max_factor'])
    G = nut*np.einsum('nij,nij->n', g, 2*Sd)          # gradU && devTwoSymm(gradU)
    Dk = BETA_STAR*k*om
    PkSST = np.minimum(G, C1*Dk)
    Gnl = -np.einsum('nij,nij->n', tw['tau'], Sd)
    dPk = np.minimum(G + Gnl, C1*Dk) + tw['R'] - PkSST
    t1 = next(i for i, (name, _) in enumerate(spec['bdelta']) if name == 'T1')
    out = {'C': C, 'V': V, 'U': U.T, 'k': k, 'omega': om, 'nut': nut, 'p': p, 'F1': F1, 'wall_distance': y,
           'G': G, 'Pk_SST': PkSST, 'Dk': Dk, 'Pi': PoE, 'Rf': v['Rf'], 'Rw': v['Rw'], 'Gk': v['Gk'], 'Gp': v['Gp'],
           'Ksn': v['Ksn'], 'lambda': tw['lambda'], 'tau': tw['tau'], 'R': tw['R'], 'Gnl': Gnl, 'DeltaPk': dPk,
           'nonlinearStress_solver': ns}
    # quantities of Appendix A.4-A.5 that the panels name (raw coefficient functions, before the clamp)
    r1 = next(i for i, (name, _) in enumerate(spec['rsource']) if name == 'T1')
    gd = lambda x, y: x*y/(y*y + expr.GUARD_EPS)                                          # guarded division
    c = spec['constants']
    vr = v['Rw']*v['Rf']
    sig = np.sqrt(np.maximum(v['I1'], 0.0))
    out.update(Ret=Ret, I1=v['I1'], I2=v['I2'],
               Phi=tw['hT']['P'][r1] - tw['gT']['P'][t1],                  # g1_P = theta Phi, h1_P = (1 + theta) Phi
               theta=np.tanh(abs(c[3])*c[4]*F1*v['Gk']),
               rr_Psi_lim=tw['gT']['R'][t1] - tw['hT']['R'][r1],         # g1_R = (1-chi) r_r Psi_lim, h1_R = -chi r_r Psi_lim
               chi=gd(v['Rf'], 1 - F1*v['Gk'] + v['Rf']),
               b_r=gd(vr*(sig + vr), v['I1'] + vr*vr))
    for name in tfd.TERM_ORDER:
        out[f'g1_{name}'] = tw['gT'][name][t1]
        out[f'h1_{name}'] = tw['hT'][name][r1]
    for name in tfd.TERM_ORDER:
        gj = np.clip(tw['gF'][t1], -spec['gmax'], spec['gmax'])
        share = np.where(np.abs(tw['gF'][t1]) > spec['gmax'], gj/np.where(tw['gF'][t1] == 0, 1, tw['gF'][t1]), 1.0)
        out[f'tau_{name}'] = tw['tauT'][name]
        out[f'R_{name}'] = tw['RT'][name]
        out[f'Gnl_{name}'] = -np.einsum('nij,nij->n', tw['tauT'][name], Sd)
        out[f'net_{name}'] = out[f'Gnl_{name}'] + out[f'R_{name}']
        out[f'dnut_{name}'] = -tw['lambda']*share*tw['gT'][name][t1]*k/oms
    return out


def rel_max(a, b):
    return float(np.max(np.abs(a - b))/max(np.max(np.abs(b)), VSMALL))


def rel_rms(a, b, V):
    w = V.reshape((-1,) + (1,)*(np.ndim(a) - 1))
    return float(np.sqrt(np.sum(w*(a - b)**2)/max(np.sum(w*b**2), VSMALL)))


def compare(foam, t, out):
    """NumPy against the utility (term fields) and against the solver (full stress)."""
    rd = lambda f, n: f(str(foam), t, n, verbose=False)                                   # noqa: E731
    V = out['V']
    checks = {'full_stress_vs_solver_nonlinearStress': {
        'max_relative': rel_max(out['tau'], out['nonlinearStress_solver']),
        'rms_relative': rel_rms(out['tau'], out['nonlinearStress_solver'], V)}}
    pairs = [('F1', 'tfF1', readscalar), ('Pk_SST', 'tfPkSST', readscalar), ('R', 'tfR', readscalar),
             ('lambda', 'tfLambda', readscalar), ('DeltaPk', 'tfDeltaPk', readscalar)]
    for name in tfd.TERM_ORDER:
        pairs += [(f'R_{name}', f'tfR{name}', readscalar), (f'net_{name}', f'tfNet{name}', readscalar),
                  (f'dnut_{name}', f'tfDnut{name}', readscalar)]
    worst = {}
    for mine, theirs, reader in pairs:
        worst[mine] = rel_max(out[mine], rd(reader, theirs)) if np.any(out[mine]) or np.any(rd(reader, theirs)) else 0.0
    for key in ['tau'] + [f'tau_{n}' for n in tfd.TERM_ORDER]:
        theirs = sym33(rd(readsymmtensor, 'tfTauDelta' + key[4:]))
        worst[key] = rel_max(out[key], theirs) if np.any(theirs) or np.any(out[key]) else 0.0
    checks['numpy_vs_utility_max_relative'] = worst
    tot = sum(out[f'tau_{n}'] for n in tfd.TERM_ORDER)
    checks['terms_sum_to_full'] = {'tau': rel_max(tot, out['tau']),
                                   'R': rel_max(sum(out[f'R_{n}'] for n in tfd.TERM_ORDER), out['R'])}
    checks['passed'] = (checks['full_stress_vs_solver_nonlinearStress']['rms_relative'] < 1e-3
                        and max(worst.values()) < 1e-6 and max(checks['terms_sum_to_full'].values()) < 1e-10)
    return checks


def budgets(case, out):
    """Volume integrals the prose and the tables may quote (shares of the SST production and nu_t changes)."""
    V, F1 = out['V'], out['F1']
    S = lambda x, m=None: float(np.sum((x*V)[m] if m is not None else x*V))             # noqa: E731
    Pk = S(out['Pk_SST'])
    b = {'integral_Pk_SST': Pk, 'integral_Dk': S(out['Dk'])}
    for name in tfd.TERM_ORDER:
        b[f'net_{name}_over_Pk_SST'] = S(out[f'net_{name}'])/Pk if Pk else None
    for label, m in (('F1_below_0.5', F1 < 0.5), ('F1_at_least_0.5', F1 >= 0.5)):
        pk = S(out['Pk_SST'], m)
        b[f'net_P_over_Pk_SST_{label}'] = S(out['net_P'], m)/pk if pk else None
        b[f'share_of_net_P_{label}'] = S(out['net_P'], m)/S(out['net_P']) if S(out['net_P']) else None
    if case == 'squareDuct_Re_2000':
        d = {n: out[f'tau_{n}'][:, 1, 1] - out[f'tau_{n}'][:, 2, 2] for n in tfd.TERM_ORDER}
        full = out['tau'][:, 1, 1] - out['tau'][:, 2, 2]
        b['cross_plane_difference_max_abs'] = float(np.max(np.abs(full)))
        b['cross_plane_difference_share_N_l2'] = float(np.sqrt(np.sum(V*d['N']**2)/max(np.sum(V*full**2), VSMALL)))
        b['cross_plane_difference_share_P_l2'] = float(np.sqrt(np.sum(V*d['P']**2)/max(np.sum(V*full**2), VSMALL)))
    if case == 'rotchan_ro10':
        yc = out['C'][:, 1]
        mid = 0.5*(yc.min() + yc.max())
        for label, m in (('lower_half', yc < mid), ('upper_half', yc >= mid)):
            b[f'dnut_R_over_nut_{label}'] = S(out['dnut_R'], m)/S(out['nut'], m)
            b[f'R_R_over_Pk_SST_{label}'] = S(out['R_R'], m)/S(out['Pk_SST'], m)
            b[f'Gnl_R_over_Pk_SST_{label}'] = S(out['Gnl_R'], m)/S(out['Pk_SST'], m)
            b[f'net_R_over_Pk_SST_{label}'] = S(out['net_R'], m)/S(out['Pk_SST'], m)
    return b


FIELD_DOC = {
    'C': 'cell centres [m], (n, 3)', 'V': 'cell volumes [m^3]', 'U': 'velocity (n, 3)', 'F1': 'SST blending F1',
    'wall_distance': 'meshWave wall distance y', 'Pk_SST': 'SST production min(G, 10 beta* k omega)',
    'Dk': 'SST dissipation beta* k omega', 'G': 'nu_t gradU : devTwoSymm(gradU)', 'Pi': 'Pi = min(G/(beta* k omega_s), 10)',
    'tau': 'full added stress tau^Delta (n, 3, 3) [m^2/s^2]', 'R': 'full explicit source R [m^2/s^3]',
    'Gnl': 'work of the added stress, -tau^Delta : S', 'DeltaPk': 'added k production incl. the SST limiter',
    'lambda': 'realizability factor', 'nonlinearStress_solver': 'tau^Delta written by the solver',
    'tau_N': 'normal-stress term stress (n, 3, 3)', 'tau_P': 'excess-production term, stress channel',
    'tau_R': 'rotation term, stress channel', 'R_N': 'normal-stress term source (zero)',
    'R_P': 'excess-production term, explicit source', 'R_R': 'rotation term, source channel',
    'Gnl_<j>': 'work of the stress of term j', 'net_<j>': 'Gnl_<j> + R_<j>, added production of term j before the limiter',
    'dnut_<j>': 'stress channel of term j as an eddy-viscosity change, -lambda g1_j k/omega_s',
    'g1_<j>': 'T1 stress coefficient function of term j (before the clamp)',
    'h1_<j>': 'T1 source coefficient function of term j (before the clamp)',
    'Phi': 'excess-production term Phi (Eq. A.18): h1_P - g1_P', 'theta': 'theta = tanh(|c3| c4 F1 G_k)',
    'rr_Psi_lim': 'r_r Psi_lim of the rotation term: g1_R - h1_R', 'chi': 'chi = R_f/(1 - F1 G_k + R_f)',
    'b_r': 'rotation ratio b_r (Eq. A.20)', 'Ret': 'Re_t = k/(nu omega_s)', 'I1': 'tr(S_hat^2)', 'I2': 'tr(W_hat^2)',
    'k': 'turbulent kinetic energy', 'omega': 'turbulence frequency', 'nut': 'eddy viscosity', 'p': 'kinematic pressure',
    'Rf': 'R_f', 'Rw': 'R_w', 'Gk': 'G_k', 'Gp': 'G_p', 'Ksn': 'K_sn',
    'wall_<patch>_Cf': 'wall face centres (nf, 3)',
    'wall_<patch>_wallShearStress': 'OpenFOAM wallShearStress on the wall faces (nf, 3); its streamwise component changes '
                                    'sign at separation and reattachment (negative for attached +x flow over a lower wall)',
}
UNITS = {'C': 'm', 'V': 'm^3', 'U': 'm/s', 'k': 'm^2/s^2', 'omega': '1/s', 'nut': 'm^2/s', 'p': 'm^2/s^2',
         'wall_distance': 'm', 'G': 'm^2/s^3', 'Pk_SST': 'm^2/s^3', 'Dk': 'm^2/s^3', 'tau': 'm^2/s^2',
         'R': 'm^2/s^3', 'Gnl': 'm^2/s^3', 'DeltaPk': 'm^2/s^3', 'nonlinearStress_solver': 'm^2/s^2',
         'tau_<j>': 'm^2/s^2', 'R_<j>': 'm^2/s^3', 'Gnl_<j>': 'm^2/s^3', 'net_<j>': 'm^2/s^3', 'dnut_<j>': 'm^2/s',
         'F1': '1', 'Pi': '1', 'lambda': '1', 'Phi': '1', 'theta': '1', 'rr_Psi_lim': '1', 'chi': '1', 'b_r': '1',
         'g1_<j>': '1', 'h1_<j>': '1', 'wall_<patch>_Cf': 'm', 'wall_<patch>_wallShearStress': 'm^2/s^2'}


def process(case, records):
    cfg = dict(CASES[case])
    if case == 'squareDuct_Re_2000':
        cfg['nu'] = duct_nu()
    for src, key in cfg.get('sources', []):          # first available source wins
        if all((src/cfg['time']/n).exists() for n in REQUIRED):
            cfg['source'], cfg['experiment'] = src, key
            break
    else:
        if 'sources' in cfg:
            cfg['source'], cfg['experiment'] = cfg['sources'][0]
    src, t = cfg['source'], cfg['time']
    missing = [n for n in REQUIRED if not (src/t/n).exists()] + [x for x in ('constant/polyMesh', 'system')
                                                                if not (src/x).exists()]
    if missing:
        records[case] = {'status': 'missing', 'source': str(src), 'missing': missing}
        print(f'{case}: missing {missing} in {src}')
        return False
    if cfg['nu'] is None:
        tp = (src/'constant/transportProperties').read_text()
        cfg['nu'] = float(tp.split('\nnu')[1].split(';')[0].split()[-1])
    model, spec = tfd.load(cfg['model'])
    terms = tfd.term_specs(cfg['model'], spec)
    foam = stage(case, cfg, cfg['model'])
    import re
    m = re.search(r'frameOmega\s*\(([^)]*)\)', (foam/'constant/turbulenceProperties').read_text())
    declared = [float(x) for x in m.group(1).split()] if m else [0.0, 0.0, 0.0]
    if max(abs(a - b) for a, b in zip(declared, cfg['frame'])) > 1e-12:
        raise SystemExit(f'{case}: frameOmega {declared} of the run differs from {cfg["frame"]}')
    summary = run_utility(foam, t)
    out = numpy_terms(foam, t, cfg, spec, terms)
    checks = compare(foam, t, out)
    checks['utility_summary'] = {k: summary[k] for k in ('consistency_vs_nonlinearStress', 'coefficient_additivity_residual',
                                                           'cells_realizability_active', 'cells_stress_coefficient_clamped',
                                                           'cells_source_coefficient_clamped', 'cells_R_bound_active')}
    arrays = {k: v for k, v in out.items()}
    import re
    from fluidfoam import readmesh
    walls = re.findall(r'\n\s*(\w+)\s*\{\s*type\s+wall;', (foam/'constant/polyMesh/boundary').read_text())
    for patch in walls:
        x, y, z = readmesh(str(foam), boundary=patch, verbose=False)
        arrays[f'wall_{patch}_Cf'] = np.stack([x, y, z], axis=1)
        if (foam/t/'wallShearStress').exists():
            arrays[f'wall_{patch}_wallShearStress'] = readvector(str(foam), t, 'wallShearStress', boundary=patch,
                                                                verbose=False).T
    np.savez_compressed(CACHE/case/'arrays.npz', **arrays)
    inputs = {name: {'path': str((src/t/name).relative_to(ROOT)), 'sha256': sha256(src/t/name)}
              for name in REQUIRED}
    mesh = {p.name: sha256(p) for p in sorted((src/'constant/polyMesh').iterdir()) if p.is_file()}
    rec = {'status': 'done', 'panel': cfg['panel'], 'case': case, 'model_of_fields': cfg['model'],
           'paper_model': 'rotation-limited correction' + ('' if cfg['model'] == 'R1' else
                          ' (stored flow-state solution; identical equations without frame rotation)'),
           'model_fingerprint': model['model_fingerprint'], 'experiment': identity(cfg['experiment']),
           'rotation_limited_experiment_hx1': cfg['r1_experiment'], 'final_time': t, 'nu': cfg['nu'],
           'frame_omega': list(cfg['frame']), 'source': str(src.relative_to(ROOT)), 'inputs': inputs,
           'mesh_sha256': mesh, 'terms': {n: {'paper': tfd.PAPER[n], 'zeroed_coefficients': list(tfd.TERMS[n])}
                                          for n in tfd.TERM_ORDER},
           'checks': checks, 'budgets': budgets(case, out),
           'outputs': {'arrays': str((CACHE/case/'arrays.npz').relative_to(P)),
                       'arrays_sha256': sha256(CACHE/case/'arrays.npz'),
                       'foam_case': str(foam.relative_to(P)),
                       'foam_fields': f'{foam.relative_to(P)}/{t}/tf*'},
           'fields': FIELD_DOC,
           'conventions': {'tensor_order': 'arrays are (n, 3, 3); OpenFOAM files store xx xy xz yy yz zz',
                           'duct': 'x streamwise; cross plane y-z; tau_yy - tau_zz from tau_N',
                           'rotating_channel': 'y wall-normal, frame rotation about +z; the -y wall is the pressure '
                                               '(destabilised) side (kOmegaSSTBasis.C)'},
           'created': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (CACHE/case/'termfields.json').write_text(json.dumps(rec, indent=1) + '\n')
    manifest = {'case': case, 'panel': cfg['panel'], 'source_experiment': cfg['experiment'],
                'rotation_limited_experiment_hx1': cfg['r1_experiment'], 'model_of_fields': rec['paper_model'],
                'arrays': {k: {'shape': list(np.shape(v)), 'meaning': FIELD_DOC.get(k, FIELD_DOC.get(
                    re.sub(r'_(N|P|R)$', '_<j>', k), FIELD_DOC.get(re.sub(r'^wall_\w+?_', 'wall_<patch>_', k), ''))),
                    'units': UNITS.get(k, UNITS.get(re.sub(r'_(N|P|R)$', '_<j>', k), ''))} for k, v in arrays.items()},
                'conventions': rec['conventions'], 'checks_passed': checks['passed'],
                'record': str((CACHE/case/'termfields.json').relative_to(P))}
    (CACHE/case/'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    records[case] = rec
    print(f'{case}: done; solver stress rms {checks["full_stress_vs_solver_nonlinearStress"]["rms_relative"]:.2e}, '
          f'NumPy vs utility max {max(checks["numpy_vs_utility_max_relative"].values()):.2e}, passed {checks["passed"]}')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cases', nargs='*', default=list(DEFAULT_CASES))
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    records = json.loads(EVIDENCE.read_text())['cases'] if EVIDENCE.exists() else {}
    for case in a.cases:
        process(case, records)
    evidence = {'purpose': 'Fig. 15, where the terms of the rotation-limited correction act (offline, stored fields)',
                'tool': 'tools/term_fields.py', 'utility': str(UTIL_SRC.relative_to(ROOT)),
                'algebra_check': '../v3verify/reports/termfields_check.json',
                'cases': records}
    EVIDENCE.write_text(json.dumps(evidence, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
