"""Comparator deployment (evaluation/forge_deploy): no OpenFOAM, no SSH.

The tests of the cluster controller scripts that drove the deployment, which are not part of this
repository, are not included.
"""
import json
from pathlib import Path
import stat
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]

import forge_deploy  # noqa: E402
import forge_fixed  # noqa: E402
from forge import runtime  # noqa: E402

CLS = 'kOmegaSSTDemo'
OPTIONS = b'EXE_INC = -I$(LIB_SRC)/finiteVolume/lnInclude\nLIB_LIBS = -lfiniteVolume\n'
SOURCES = {
    'Make/files': f'make{CLS}.C\n\nLIB = $(FOAM_USER_LIBBIN)/lib{CLS}\n',
    f'{CLS}.H': f'namespace Foam {{ namespace RASModels {{ TypeName("{CLS}"); }} }}\n',
    f'{CLS}.C': '// template definitions\n',
    f'make{CLS}.C': f'#include "turbulentTransportModels.H"\n#include "{CLS}.H"\nmakeRASModel({CLS});\n',
    'detail/helper.H': '// a further header inside the folder\n',
}
ARTEFACTS = ('lnInclude/kOmegaSSTDemo.H', 'Make/linux64GccDPInt32Opt/kOmegaSSTDemo.o',
             'Make/linux64GccDPInt32Opt/kOmegaSSTDemo.C.dep', 'log.wmake', 'kOmegaSSTDemo.dep', 'make.o')


def tree(root, name='demo', *, artefacts=True, files=None):
    (root/'src/kOmegaSSTBasis/Make').mkdir(parents=True, exist_ok=True)
    (root/'src/kOmegaSSTBasis/Make/options').write_bytes(OPTIONS)
    source = root/'src/comparators'/name
    for rel, text in (files or SOURCES).items():
        (source/rel).parent.mkdir(parents=True, exist_ok=True)
        (source/rel).write_text(text)
    (source/'Make/options').write_bytes(OPTIONS)
    if artefacts:
        for rel in ARTEFACTS:
            (source/rel).parent.mkdir(parents=True, exist_ok=True)
            (source/rel).write_text('build product')
    for package in forge_deploy.PACKAGES:
        (root/package).mkdir(parents=True, exist_ok=True)
        (root/package/'__init__.py').write_text('# package\n')
        (root/package/'__pycache__').mkdir(exist_ok=True)
        (root/package/'__pycache__/x.cpython-312.pyc').write_bytes(b'cache')
    return source


# ------------------------------------------------------------------------------------------------
# the staging manifest of a fixed model
# ------------------------------------------------------------------------------------------------
def test_staging_manifest_has_the_sources_and_no_build_artefacts(tmp_path):
    source = tree(tmp_path)
    files = forge_deploy.staging_manifest(tmp_path, ['demo'])
    expected = {f'src/comparators/demo/{rel}' for rel in [*SOURCES, 'Make/options']}
    assert expected <= set(files)
    staged = {f for f in files if f.startswith('src/comparators/')}
    assert staged == expected
    for rel in ARTEFACTS:
        assert f'src/comparators/demo/{rel}' not in files
    assert 'src/forge_fixed/__init__.py' in files and 'src/forge_deploy/__init__.py' in files
    assert not any('__pycache__' in f or f.endswith('.pyc') for f in files)
    # exactly the files of the source hash the handoff and the freeze record carry
    info = forge_deploy.check_tree(tmp_path, 'demo')
    assert set(info['sources']) == {f[len('src/comparators/demo/'):] for f in staged}
    assert info['source_sha256'] == forge_fixed.source_sha256(source)
    assert info['ras_model'] == CLS and info['library'] == f'lib{CLS}.so'
    assert set(info['excluded_artefacts']) == set(ARTEFACTS)


def test_staging_never_sends_the_physical_source(tmp_path):
    tree(tmp_path)
    for name in ('src/forge/runtime.py', 'src/tedp/spec.py', 'src/kOmegaSSTBasis/Make/options', 'pyproject.toml'):
        (tmp_path/name).parent.mkdir(parents=True, exist_ok=True)
        if not (tmp_path/name).exists():
            (tmp_path/name).write_text('x')
        with pytest.raises(ValueError, match='physical-source'):
            forge_deploy.staging_manifest(tmp_path, ['demo'], [name])
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/verify_hx1_build_comparators.sh').write_text('#!/bin/bash\n')
    files = forge_deploy.staging_manifest(tmp_path, ['demo'], ['scripts/verify_hx1_build_comparators.sh'])
    assert 'scripts/verify_hx1_build_comparators.sh' in files
    assert not any(forge_deploy.physical(f) for f in files)


@pytest.mark.parametrize('change, message', [
    (lambda s: (s/'stray.so').write_bytes(b'\x7fELF'), 'build products'),
    (lambda s: (s/'Make/options').write_bytes(OPTIONS + b'\n'), 'Make/options differs'),
    (lambda s: (s/'Make/files').write_text(f'make{CLS}.C\nLIB = $(FOAM_LIBBIN)/lib{CLS}\n'), 'last line'),
    (lambda s: (s/f'make{CLS}.C').write_text('// no registration\n'), 'makeRASModel'),
    (lambda s: (s/'Make/files').write_text('makekOmegaSST.C\n\nLIB = $(FOAM_USER_LIBBIN)/libkOmegaSST\n'), 'stock'),
])
def test_tree_checks_refuse_what_the_interface_forbids(tmp_path, change, message):
    source = tree(tmp_path, artefacts=False)
    change(source)
    with pytest.raises(ValueError, match=message):
        forge_deploy.check_tree(tmp_path, 'demo')


def test_folder_name_pattern(tmp_path):
    tree(tmp_path, 'Demo-1', artefacts=False)
    with pytest.raises(ValueError, match='a-z0-9_'):
        forge_deploy.check_tree(tmp_path, 'Demo-1')


# ------------------------------------------------------------------------------------------------
# the build (wmake replaced by a stub) and the records HARNESS reads
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'PROJECT_ROOT', tmp_path)
    tree(tmp_path)
    return tmp_path


def loaded(args, cwd, log, timeout, env):
    """The load check of the built library: (returncode, record) for a stub runner."""
    assert args[0] == sys.executable and 'ctypes.CDLL' in args[2] and Path(args[3]).name == f'lib{CLS}.so'
    return {'args': args, 'cwd': str(cwd), 'log': str(log), 'returncode': 0, 'timed_out': False}


def fake_wmake(seen):
    def runner(args, cwd, log, timeout, env):
        if args[0] == sys.executable:
            return loaded(args, cwd, log, timeout, env)
        assert args == ['wmake', 'libso']
        cwd = Path(cwd)
        seen['cwd'] = cwd
        seen['files'] = sorted(str(p.relative_to(cwd)) for p in cwd.rglob('*') if p.is_file())
        out = Path(env['FOAM_USER_LIBBIN'])
        (out/f'lib{CLS}.so').write_bytes(b'\x7fELF fake ' + CLS.encode() + b' ' + str(cwd).encode())
        Path(log).write_text('wmake libso: ok\n')
        return {'args': args, 'cwd': str(cwd), 'log': str(log), 'returncode': 0, 'timed_out': False}
    return runner


ENV = {'WM_PROJECT_VERSION': 'v2312', 'WM_OPTIONS': 'linux64GccDPInt32Opt'}


def test_build_freezes_a_hash_named_library_and_writes_the_section8_record(isolated):
    seen = {}
    source_hash = forge_fixed.source_sha256(isolated/'src/comparators/demo')
    record = forge_deploy.build_comparator(isolated, 'demo', expected_source_sha256=source_hash,
                                           env=ENV, runner=fake_wmake(seen))
    # wmake saw the sources only, in a private copy
    assert set(seen['files']) == set(SOURCES) | {'Make/options'}
    assert seen['cwd'].is_relative_to(isolated/'runtime/builds/comparators')
    frozen = Path(record['library'])
    assert frozen.parent == isolated/'runtime/lib'
    assert frozen.name == f'lib{CLS}-{record["library_sha256"][:16]}.so'
    assert stat.S_IMODE(frozen.stat().st_mode) == 0o444
    on_disk = json.loads((isolated/'runtime/comparators'/f'{CLS}.json').read_text())
    contract = {'schema_version': 1, 'ras_model': CLS, 'library': str(frozen),
                'library_sha256': runtime.sha256(frozen), 'source_sha256': source_hash,
                'source_dir': 'src/comparators/demo', 'openfoam_version': 'v2312',
                'wm_options': 'linux64GccDPInt32Opt', 'qualified_build': True}
    assert {k: on_disk[k] for k in contract} == contract
    assert forge_deploy.built_library(isolated, CLS) == (frozen, contract['library_sha256'])
    # HARNESS's worker-side reader finds the same library
    from forge_fixed.evaluator import node_builds
    assert node_builds(isolated)[CLS] == {'path': str(frozen), 'sha256': contract['library_sha256'],
                                          'source_sha256': source_hash, 'local_copy': None}
    described = forge_deploy.describe(isolated, [CLS])['comparators'][CLS]
    assert described['library'] == str(frozen) and described['source_sha256'] == source_hash


def test_build_refuses_a_tree_that_is_not_the_frozen_one(isolated):
    with pytest.raises(RuntimeError, match='differs from the frozen'):
        forge_deploy.build_comparator(isolated, 'demo', expected_source_sha256='0'*64, env=ENV, runner=fake_wmake({}))
    assert not (isolated/'runtime/lib').exists()


def test_build_failure_and_missing_typename_are_errors(isolated):
    def failing(args, cwd, log, timeout, env):
        return {'args': args, 'cwd': str(cwd), 'log': str(log), 'returncode': 2, 'timed_out': False}
    with pytest.raises(RuntimeError, match='build failed'):
        forge_deploy.build_comparator(isolated, 'demo', env=ENV, runner=failing)

    def unloadable(args, cwd, log, timeout, env):
        if args[0] == sys.executable:
            return {'args': args, 'cwd': str(cwd), 'log': str(log), 'returncode': 1, 'timed_out': False}
        return fake_wmake({})(args, cwd, log, timeout, env)
    with pytest.raises(RuntimeError, match='does not load'):
        forge_deploy.build_comparator(isolated, 'demo', env=ENV, runner=unloadable)
    assert not (isolated/'runtime/lib').exists()

    def anonymous(args, cwd, log, timeout, env):
        (Path(env['FOAM_USER_LIBBIN'])/f'lib{CLS}.so').write_bytes(b'\x7fELF other model')
        return {'args': args, 'cwd': str(cwd), 'log': str(log), 'returncode': 0, 'timed_out': False}
    with pytest.raises(RuntimeError, match='TypeName'):
        forge_deploy.build_comparator(isolated, 'demo', env=ENV, runner=anonymous)


def test_a_changed_library_is_refused(isolated):
    record = forge_deploy.build_comparator(isolated, 'demo', env=ENV, runner=fake_wmake({}))
    frozen = Path(record['library'])
    frozen.chmod(0o644)
    frozen.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='immutable record'):
        forge_deploy.built_library(isolated, CLS)
    with pytest.raises(RuntimeError, match='No built comparator'):
        forge_deploy.built_library(isolated, 'kOmegaSSTOther')


# ------------------------------------------------------------------------------------------------
# a fixed-model job loads the right library
# ------------------------------------------------------------------------------------------------
def test_fixed_model_job_runs_the_fixed_worker_and_its_case_loads_the_built_library(isolated, monkeypatch):
    from forge import pbs
    from forge_fixed import FixedModel, add_library, check_library
    from forge_fixed import evaluator as fixed_evaluator
    record = forge_deploy.build_comparator(isolated, 'demo', env=ENV, runner=fake_wmake({}))
    build = fixed_evaluator.node_builds(isolated)[CLS]
    model = FixedModel(name='demo', ras_model=CLS, library=f'lib{CLS}.so', source_dir='src/comparators/demo',
                       coeffs={'cr1': 1.0}, source_sha256=record['source_sha256']).bound(build['path'], build['sha256'])
    assert check_library(model, isolated) == Path(record['library'])
    basis = isolated/'runtime/lib/libFORGEv2Closure-0123456789abcdef.so'
    control = 'application simpleFoam;\nlibs ("' + str(basis) + '");\n'
    loaded = add_library(control, basis, check_library(model, isolated))
    assert f'libs ("{basis}" "{record["library"]}");' in loaded

    monkeypatch.setattr(pbs, 'job_script', pbs.job_script)        # restore after the test
    fixed_evaluator.install_job_script()
    remote = Path('/cluster/home/user/FORGE_v2/verify')
    request = remote/'deployment/tasks'/('a'*64)/'request.json'
    args = dict(ranks=4, wall_s=3600, memory_gb=16, queue='hx', name='f2-aaaaaaaaaaaa')
    stock = pbs.job_script(remote, request, **args)
    monkeypatch.setattr(fixed_evaluator._JOB, 'fixed', 1, raising=False)
    fixed = pbs.job_script(remote, request, **args)
    assert ' -m forge.pbs_worker run ' in stock and ' -m forge_fixed.worker run ' not in stock
    assert fixed == stock.replace(' -m forge.pbs_worker run ', ' -m forge_fixed.worker run ')
    assert f'-m forge_fixed.worker run {request}' in fixed
    assert 'export PYTHONPATH=' + str(remote/'src') in fixed     # forge_fixed and forge_deploy are staged there


def test_controller_record_verifies_the_retained_copies(tmp_path):
    lib = tmp_path/'runtime/hx1_verify/native'/f'comparator-{CLS}-x.so'
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b'library')
    entry = {'library_path': f'/cluster/home/user/FORGE_v2/verify/runtime/lib/lib{CLS}-0123456789abcdef.so',
             'library_sha256': runtime.sha256(lib), 'source_sha256': 'f'*64,
             'local_copy': str(lib.relative_to(tmp_path))}
    (tmp_path/forge_deploy.CONTROLLER_RECORD).write_text(json.dumps({CLS: entry}))
    assert forge_deploy.controller_record(tmp_path, CLS) == entry
    from forge_fixed.evaluator import controller_builds
    assert controller_builds(tmp_path, {})[CLS]['path'] == entry['library_path']
    lib.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='changed'):
        forge_deploy.controller_record(tmp_path)
