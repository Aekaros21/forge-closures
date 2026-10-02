"""Fixed C++ comparator models beside stock SST and expression specs (evaluation/forge_fixed).

The tests of the paper's verification driver (freeze-comparators, the comparator models file and the
report), which exercise scripts that are not part of this repository, are not included.
"""
import copy
import json
from pathlib import Path

import pytest

from forge.core import atomic_json, fingerprint, sha256
from forge.evaluator import Evaluator
from forge_fixed import (FixedModel, add_library, frame_omega, harness_sha256, prepare_fixed_case,
                         render_turbulence_properties)
from forge_fixed.evaluator import FixedModelMixin, _JOB, fixed_class, install_job_script
from tedp.casegen import render_turbulence_properties as render_stock
from tedp.spec import CandidateSpec, Term

REPO = Path(__file__).resolve().parents[2]

SRC_A = 'a'*64
SRC_B = 'b'*64
LIB_A = '1'*64
LIB_B = '2'*64


def _model(ras='kOmegaSSTRC', coeffs=None, source=SRC_A, name='sst_rc'):
    return FixedModel(name=name, ras_model=ras, library=f'lib{ras}.so', source_dir='src/comparators/sstrc',
                      coeffs={'cr1': 1.0, 'cr2': 2.0, 'cr3': 1.0} if coeffs is None else coeffs, source_sha256=source)


def _evaluator(cls, builds=None):
    """An evaluator with the identity attributes of a deployed campaign and no OpenFOAM."""
    ev = cls.__new__(cls)
    ev.root = REPO
    ev.config = {'campaign': 'test', 'backend': {'kind': 'pbs'}}
    ev.library_sha = 'f'*64
    ev.identity = {'simpleFoam_sha256': 'e'*64}
    ev.code = 'd'*64
    ev.env = {'WM_OPTIONS': 'linux64GccDPInt32Opt', 'WM_PROJECT_VERSION': 'v2312'}
    ev.runtime_identity = None
    ev._comparator_builds = builds
    return ev


BUILDS = {'kOmegaSSTRC': {'path': '/hx1/runtime/lib/libkOmegaSSTRC-111111111111.so', 'sha256': LIB_A,
                          'source_sha256': SRC_A, 'local_copy': None},
          'kOmegaSSTQCR': {'path': '/hx1/runtime/lib/libkOmegaSSTQCR-222222222222.so', 'sha256': LIB_B,
                           'source_sha256': SRC_B, 'local_copy': None}}
CASE = {'id': 'case', 'adapter': 'plate', 'adapter_config': {}, 'role': 'calibration'}
PROTOCOL = {'end_time': 100, 'nprocs': 4}


def _spec_model():
    return CandidateSpec('rotation_limited', rsource=(Term('T1', 'c0*tanh(I1+I2)'),), constants=(-0.02,))


# ---------------------------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------------------------
def test_identity_hashes_differ_between_sst_spec_and_fixed_models():
    plain = _evaluator(Evaluator)
    fixed = _evaluator(fixed_class(Evaluator), BUILDS)
    rc = _model()
    rc_other = _model(coeffs={'cr1': 1.0, 'cr2': 2.0, 'cr3': 1.5})
    qcr = _model('kOmegaSSTQCR', {'Ccr1': 0.3}, SRC_B, 'qcr2000')
    keys = {}
    for label, model in [('SST', None), ('spec', _spec_model()), ('rc', rc), ('rc_other', rc_other), ('qcr', qcr)]:
        pairing, identity, key = fixed.case_identity(model, CASE, PROTOCOL, 4)
        assert key == fingerprint(identity)
        keys[label] = (pairing, identity, key)
    assert len({k for _, _, k in keys.values()}) == 5
    # SST and expression identities are those of the unmodified evaluator, byte for byte (cached 001-007 runs).
    assert keys['SST'] == plain.case_identity(None, CASE, PROTOCOL, 4)
    assert keys['spec'] == plain.case_identity(_spec_model(), CASE, PROTOCOL, 4)
    # A fixed model is paired with exactly the SST experiment's pairing.
    assert all(keys[m][0] == keys['SST'][0] for m in ('rc', 'rc_other', 'qcr'))
    model = keys['rc'][1]['spec']
    assert model == {'kind': 'fixed_cpp_model', 'ras_model': 'kOmegaSSTRC', 'library': 'libkOmegaSSTRC.so',
                     'coeffs': {'cr1': 1.0, 'cr2': 2.0, 'cr3': 1.0}, 'source_sha256': SRC_A,
                     'library_sha256': LIB_A, 'harness_sha256': harness_sha256()}
    # The display name is not physics (as for expression specs); the coefficient order is irrelevant.
    renamed = _model(coeffs={'cr3': 1.0, 'cr2': 2.0, 'cr1': 1.0}, name='other_label')
    assert fixed.case_identity(renamed, CASE, PROTOCOL, 4)[2] == keys['rc'][2]


def test_fixed_identity_binds_source_and_built_library():
    fixed = _evaluator(fixed_class(Evaluator), copy.deepcopy(BUILDS))
    key = fixed.case_identity(_model(), CASE, PROTOCOL, 4)[2]
    rebuilt = _evaluator(fixed_class(Evaluator), {**BUILDS, 'kOmegaSSTRC': {**BUILDS['kOmegaSSTRC'], 'sha256': '3'*64}})
    assert rebuilt.case_identity(_model(), CASE, PROTOCOL, 4)[2] != key
    with pytest.raises(RuntimeError, match='other sources'):
        fixed.case_identity(_model(source='c'*64), CASE, PROTOCOL, 4)
    with pytest.raises(RuntimeError, match='no comparator build'):
        _evaluator(fixed_class(Evaluator), {}).case_identity(_model(), CASE, PROTOCOL, 4)


def test_worker_restores_the_bound_model_of_a_request():
    fixed = _evaluator(fixed_class(Evaluator), BUILDS)
    identity = fixed.case_identity(_model(), CASE, PROTOCOL, 4)[1]
    restored = FixedModel.from_identity(identity['spec'])
    assert restored.library_sha256 == LIB_A and restored.coeffs == _model().coeffs
    assert fixed.case_identity(restored, CASE, PROTOCOL, 4)[1] == identity
    with pytest.raises(RuntimeError, match='forge_fixed differs'):
        FixedModel.from_identity({**identity['spec'], 'harness_sha256': '0'*64})
    with pytest.raises(RuntimeError, match='not the requested'):
        _evaluator(fixed_class(Evaluator), {**BUILDS, 'kOmegaSSTRC': {**BUILDS['kOmegaSSTRC'], 'sha256': '3'*64}}) \
            .case_identity(restored, CASE, PROTOCOL, 4)


def test_model_validation():
    with pytest.raises(ValueError, match='reserved'):
        _model('kOmegaSST')
    with pytest.raises(ValueError, match='libkOmegaSSTRC.so'):
        FixedModel(name='x', ras_model='kOmegaSSTRC', library='libOther.so', source_dir='src/comparators/sstrc',
                   coeffs={}, source_sha256=SRC_A)
    with pytest.raises(ValueError, match='frameOmega'):
        _model(coeffs={'frameOmega': 1.0})
    with pytest.raises(ValueError, match='Unsupported'):
        _model(coeffs={'cr1': [1, 2]})
    with pytest.raises(ValueError, match='finite'):
        _model(coeffs={'cr1': float('nan')})
    with pytest.raises(ValueError, match='source_dir'):
        FixedModel(name='x', ras_model='kOmegaSSTRC', library='libkOmegaSSTRC.so', source_dir='../elsewhere',
                   coeffs={}, source_sha256=SRC_A)


def test_final_access_needs_the_frozen_fingerprint(tmp_path):
    manifest, contract = {'cases': []}, {'metrics': {}}
    record = {'manifest_fingerprint': fingerprint(manifest), 'contract_fingerprint': fingerprint(contract),
              'models': [{'label': 'SSTRC', 'model_fingerprint': _model().model_fingerprint()}]}
    atomic_json(tmp_path/'freeze.json', record)
    ev = _evaluator(fixed_class(Evaluator), BUILDS)
    ev.root, ev.manifest, ev.contract = tmp_path, manifest, contract
    ev.config = {'purpose': 'verification', 'backend': {'kind': 'pbs'},
                 'verification': {'freeze': {'path': 'freeze.json', 'sha256': sha256(tmp_path/'freeze.json')}}}
    assert ev.final_access(_model()) and ev.final_access(None)
    assert not ev.final_access(_model(coeffs={'cr1': 2.0}))


# ---------------------------------------------------------------------------------------------
# case: turbulenceProperties, controlDict libs, everything else the SST case
# ---------------------------------------------------------------------------------------------
def test_turbulence_properties_of_a_fixed_model():
    model = _model(coeffs={'cr3': 1.0, 'cr1': 1, 'curvatureCorrection': True, 'variant': 'standard'})
    text = render_turbulence_properties(model, (0, 0, 0.25))
    assert text == (
        'FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\n'
        'simulationType RAS;\n'
        'RAS\n{\n'
        '    RASModel        kOmegaSSTRC;\n'
        '    turbulence      on;\n'
        '    printCoeffs     off;\n'
        '    kOmegaSSTRCCoeffs\n    {\n'
        '        cr1             1;\n'
        '        cr3             1;\n'
        '        curvatureCorrection true;\n'
        '        variant         standard;\n'
        '        frameOmega      (0 0 0.25);\n'
        '    }\n}\n')
    assert text.startswith(render_stock(None).split('RAS\n{')[0])          # the SST header
    assert '0.10000000000000001' in render_turbulence_properties(_model(coeffs={'cr1': 0.1}))


def _stock_case(case, spec, work, *, end_time, library):
    """A stand-in for forge.cases.prepare_case(spec=None): the SST case's text files only."""
    assert spec is None
    work = Path(work)
    for d in ('system', 'constant', '0'):
        (work/d).mkdir(parents=True)
    control = ('application simpleFoam;\nendTime %d;\nfunctions\n{\n    yPlus1\n    {\n        type yPlus;\n'
               '        libs ("libfieldFunctionObjects.so");\n    }\n}\n' % end_time)
    control += '\nlibs ("' + str(library) + '");\n'
    (work/'system/controlDict').write_text(control)
    for name in ('fvSchemes', 'fvSolution', 'decomposeParDict'):
        (work/'system'/name).write_text(f'{name} of the SST case\n')
    (work/'constant/turbulenceProperties').write_text(render_stock(None))
    (work/'0/k').write_text('k\n')
    metadata = {'case_id': case['id'], 'spec': None, 'library': str(library),
                'numerics_sha256': {n: sha256(work/'system'/n) for n in ('controlDict', 'fvSchemes', 'fvSolution', 'decomposeParDict')},
                'turbulence_sha256': sha256(work/'constant/turbulenceProperties')}
    (work/'forge_case.json').write_text(json.dumps(metadata))
    return metadata


def _library(root, ras='kOmegaSSTRC'):
    lib = root/'runtime/lib'/'tmp.so'
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_bytes(b'compiled ' + ras.encode())
    digest = sha256(lib)
    frozen = lib.with_name(f'lib{ras}-{digest[:16]}.so')
    lib.rename(frozen)
    return frozen, digest


@pytest.mark.parametrize('adapter,config,omega', [('plate', {}, '(0 0 0)'),
                                                   ('rotation', {'ro': 0.1}, None)])
def test_prepared_fixed_case_is_the_sst_case_except_model_and_library(tmp_path, adapter, config, omega):
    from tedp.holdout_cases import rotchan
    case = {'id': 'c', 'adapter': adapter, 'adapter_config': config}
    basis = tmp_path/'runtime/lib/libFORGEv2Closure-0123456789ab.so'
    library, digest = _library(tmp_path)
    model = _model().bound(str(library), digest)
    sst = tmp_path/'runs/sst'
    _stock_case(case, None, sst, end_time=100, library=basis)
    work = tmp_path/'runs/fixed'
    metadata = prepare_fixed_case(case, model, work, end_time=100, library=basis, base_prepare=_stock_case, root=tmp_path)
    for name in ('system/fvSchemes', 'system/fvSolution', 'system/decomposeParDict', '0/k'):
        assert (work/name).read_bytes() == (sst/name).read_bytes()
    assert (work/'system/controlDict').read_text() == (sst/'system/controlDict').read_text().replace(
        f'libs ("{basis}");', f'libs ("{basis}" "{library}");')
    expected = omega or f'(0 0 {rotchan.omega_for(0.1)})'
    assert (work/'constant/turbulenceProperties').read_text() == render_turbulence_properties(model, frame_omega(case))
    assert f'frameOmega      {expected};' in (work/'constant/turbulenceProperties').read_text()
    assert metadata['fixed_model'] == model.identity() and metadata['comparator_library_sha256'] == digest
    assert metadata['turbulence_sha256'] == sha256(work/'constant/turbulenceProperties')
    assert metadata['numerics_sha256']['controlDict'] == sha256(work/'system/controlDict')
    assert json.loads((work/'forge_case.json').read_text()) == metadata


def test_prepared_fixed_case_refuses_a_changed_or_mutable_library(tmp_path):
    case = {'id': 'c', 'adapter': 'plate', 'adapter_config': {}}
    basis = tmp_path/'runtime/lib/libFORGEv2Closure-0123456789ab.so'
    library, digest = _library(tmp_path)
    library.write_bytes(b'rebuilt')
    with pytest.raises(RuntimeError, match='changed'):
        prepare_fixed_case(case, _model().bound(str(library), digest), tmp_path/'runs/a', end_time=10,
                           library=basis, base_prepare=_stock_case, root=tmp_path)
    mutable = tmp_path/'runtime/lib/libkOmegaSSTRC.so'
    mutable.write_bytes(b'x')
    with pytest.raises(ValueError, match='immutable'):
        prepare_fixed_case(case, _model().bound(str(mutable), sha256(mutable)), tmp_path/'runs/b', end_time=10,
                           library=basis, base_prepare=_stock_case, root=tmp_path)


def test_comparator_library_joins_only_the_top_level_libs_list():
    with pytest.raises(ValueError, match='top-level'):
        add_library('functions\n{\n    f\n    {\n        libs ("/x/basis.so");\n    }\n}\n', '/x/basis.so', '/x/c.so')
    with pytest.raises(ValueError, match='exactly once'):
        add_library('libs ("/x/basis.so");\nlibs ("/x/basis.so");\n', '/x/basis.so', '/x/c.so')
    assert add_library('a 1;\nlibs ("/x/basis.so");\n', '/x/basis.so', '/x/c.so') == 'a 1;\nlibs ("/x/basis.so" "/x/c.so");\n'


def test_case_preparation_hook_passes_sst_and_specs_through(monkeypatch):
    from forge import cases
    from forge_fixed import install_case_preparation
    calls = []
    original = lambda case, spec, work, *, end_time, library: calls.append(spec) or 'stock'
    monkeypatch.setattr(cases, 'prepare_case', original)
    install_case_preparation()
    assert cases.prepare_case is not original
    assert cases.prepare_case(CASE, None, 'w', end_time=1, library='l') == 'stock'
    spec = _spec_model()
    assert cases.prepare_case(CASE, spec, 'w', end_time=1, library='l') == 'stock'
    assert calls == [None, spec]


# ---------------------------------------------------------------------------------------------
# PBS job script and request
# ---------------------------------------------------------------------------------------------
def test_fixed_job_runs_the_fixed_worker_and_other_jobs_are_unchanged(monkeypatch):
    from forge import pbs
    monkeypatch.setattr(pbs, 'job_script', pbs.job_script)
    original = pbs.job_script
    install_job_script()
    args = ('/gpfs/home/x/FORGE_v2/verify', '/gpfs/home/x/FORGE_v2/verify/deployment/tasks/k/request.json')
    kwargs = dict(ranks=4, wall_s=3600, memory_gb=16, queue='hx', name='f2-abc')
    assert pbs.job_script(*args, **kwargs) == original(*args, **kwargs)
    _JOB.fixed = 1
    try:
        fixed = pbs.job_script(*args, **kwargs)
    finally:
        _JOB.fixed = 0
    assert fixed == original(*args, **kwargs).replace('-m forge.pbs_worker run', '-m forge_fixed.worker run')
    assert fixed != original(*args, **kwargs)


def test_scheduled_fixed_case_freezes_the_fixed_worker_request(monkeypatch, tmp_path):
    from forge import pbs
    monkeypatch.setattr(pbs, 'job_script', pbs.job_script)
    install_job_script()
    seen = {}

    class Base:
        backend_kind = 'pbs'

        def _scheduled_case(self, spec, cid, stage, case, protocol, ranks, identity, key, folder):
            seen[spec is None] = pbs.job_script('/r', '/r/deployment/tasks/k/request.json', ranks=1, wall_s=60,
                                                memory_gb=8, queue='hx', name='f2-k')
            return identity
    ev = FixedModelMixin.__new__(type('E', (FixedModelMixin, Base), {}))
    assert ev._scheduled_case(_model(), 'c', 's', {}, {}, 1, {'spec': 'x'}, 'k', tmp_path) == {'spec': 'x'}
    ev._scheduled_case(None, 'c', 's', {}, {}, 1, {'spec': None}, 'k', tmp_path)
    assert '-m forge_fixed.worker run' in seen[False] and '-m forge.pbs_worker run' in seen[True]
    assert not getattr(_JOB, 'fixed', 0)
