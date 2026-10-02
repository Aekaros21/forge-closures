"""Deployment of fixed C++ comparator closures to a FORGE root (the HX1 verification root).

Kept outside src/forge, src/tedp and src/kOmegaSSTBasis on purpose: those folders are the physical
source identity of every experiment (forge.core.source_identity), so changing them would invalidate
the cached SST baselines and corrections and make queued HX1 jobs refuse their requests. Nothing
here enters an experiment identity; the identity of a fixed model carries the built library's
SHA-256, which the forge_fixed worker checks against runtime/comparators/<ClassName>.json and the
file before the case runs.

The comparator library is built the way runtime.build_solver builds the basis library: a copy of
the source tree (build products excluded), the same pinned OpenFOAM v2312 environment
(runtime.foam_env: on HX1 the runtime/platform.json modules, WM_OPTIONS linux64GccDPInt32Opt),
`wmake libso` into a private output folder, then an immutable hash-named copy
runtime/lib/lib<ClassName>-<sha16>.so (mode 0444), which is on the LD_LIBRARY_PATH that
runtime.foam_env sets. Each build writes runtime/comparators/<ClassName>.json on the root where it
ran (the record the forge_fixed worker reads); the controller keeps runtime/hx1_verify/comparators.json
(handoff/harness_interface.md section 8). The case loads the library through its controlDict libs
list, by absolute path, beside the basis library (forge_fixed.add_library).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

from forge_fixed import source_files, source_sha256

COMPARATORS = 'src/comparators'
BASIS_OPTIONS = 'src/kOmegaSSTBasis/Make/options'
RECORDS = 'runtime/comparators'
CONTROLLER_RECORD = 'runtime/hx1_verify/comparators.json'
# The physical source identity (forge.core.source_identity); the comparator deployment never stages them.
PHYSICAL = ('src/forge/', 'src/tedp/', 'src/kOmegaSSTBasis/', 'requirements.lock', 'pyproject.toml')
# Packages the comparator jobs need besides the comparator sources (python only).
PACKAGES = ('src/forge_fixed', 'src/forge_deploy')
# Build products that forge_fixed.source_files would NOT exclude: their presence makes the
# source hash differ from the committed tree, so a tree holding them is refused.
FOREIGN = ('.so', '.a', '.pyc', '.orig', '.rej', '.swp')
DIR_PATTERN = re.compile(r'[a-z0-9_]+')
LIB_LINE = re.compile(r'LIB\s*=\s*\$\(FOAM_USER_LIBBIN\)/lib([A-Za-z][A-Za-z0-9_]*)')
STOCK_MODELS = frozenset('''kOmegaSST kOmegaSSTLM kOmegaSSTSAS kOmegaSSTDES kOmegaSSTDDES kOmegaSSTIDDES
    kOmega kOmega2006 kEpsilon realizableKE RNGkEpsilon LaunderSharmaKE SpalartAllmaras LRR SSG EBRSM
    kEpsilonPhitF LienCubicKE ShihQuadraticKE LienLeschziner qZeta v2f kkLOmega kEpsilonLopesdaCosta
    kL kOmegaSSTBasis'''.split())


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


# ------------------------------------------------------------------------------------------------
# Source tree checks and the staging manifest
# ------------------------------------------------------------------------------------------------
def class_name(source) -> str:
    """<ClassName> from the last non-empty line of Make/files, `LIB = $(FOAM_USER_LIBBIN)/lib<ClassName>`."""
    lines = [line.strip() for line in (Path(source)/'Make/files').read_text().splitlines() if line.strip()]
    match = LIB_LINE.fullmatch(lines[-1]) if lines else None
    if not match:
        raise ValueError(f'{source}/Make/files: the last line must be LIB = $(FOAM_USER_LIBBIN)/lib<ClassName>')
    return match.group(1)


def artefacts(source) -> list[str]:
    """Files in a comparator folder that are not sources: build products, logs, editor files."""
    source = Path(source)
    kept = set(source_files(source))
    return sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and p not in kept)


def check_tree(root, directory) -> dict:
    """The comparator folder src/comparators/<directory> as the interface requires (section 1-2)."""
    root = Path(root)
    if not DIR_PATTERN.fullmatch(directory):
        raise ValueError(f'Comparator folder name must match [a-z0-9_]+: {directory!r}')
    source = root/COMPARATORS/directory
    if not (source/'Make/files').is_file() or not (source/'Make/options').is_file():
        raise ValueError(f'{source}: Make/files and Make/options are required')
    cls = class_name(source)
    if cls in STOCK_MODELS:
        raise ValueError(f'{cls} is a stock OpenFOAM model name or the basis library; use a distinct name')
    if (source/'Make/options').read_bytes() != (root/BASIS_OPTIONS).read_bytes():
        raise ValueError(f'{source}/Make/options differs from {BASIS_OPTIONS}')
    files = source_files(source)
    foreign = [str(p.relative_to(source)) for p in files if p.suffix in FOREIGN or '__pycache__' in p.parts]
    if foreign:
        raise ValueError(f'{source} holds build products that would enter the source hash: {foreign[:5]}')
    listed = [line.strip() for line in (source/'Make/files').read_text().splitlines()
              if line.strip() and not line.strip().startswith('LIB')]
    registered = [name for name in listed if (source/name).is_file()
                  and re.search(r'makeRASModel\s*\(\s*' + re.escape(cls) + r'\s*\)', (source/name).read_text())]
    if not registered:
        raise ValueError(f'{source}: no Make/files source registers makeRASModel({cls})')
    manifest = {str(p.relative_to(source)): sha256(p) for p in files}
    return {'dir': directory, 'ras_model': cls, 'library': f'lib{cls}.so', 'source_dir': f'{COMPARATORS}/{directory}',
            'source_sha256': source_sha256(source), 'sources': manifest,
            'excluded_artefacts': artefacts(source), 'registration': registered}


def physical(relative) -> bool:
    relative = str(relative)
    return any(relative == p or relative.startswith(p) for p in PHYSICAL)


def staging_manifest(root, directories, extra=()) -> list[str]:
    """Root-relative files the comparator deployment stages: each comparator's sources (exactly the
    files of its source hash; lnInclude, Make/linux64*, *.o, *.dep, log.* never), the forge_fixed and
    forge_deploy packages (*.py), and `extra` (scripts, configuration, freeze record). Never a file of
    the physical source identity: the remote solver source must stay what the queued jobs froze."""
    root = Path(root)
    files = set()
    for directory in directories:
        check_tree(root, directory)
        source = root/COMPARATORS/directory
        files.update(str(p.relative_to(root)) for p in source_files(source))
    for package in PACKAGES:
        files.update(str(p.relative_to(root)) for p in sorted((root/package).glob('*.py')))
    for name in extra:
        if not (root/name).is_file():
            raise ValueError(f'Staged file is missing: {name}')
        files.add(str(Path(name)))
    bad = sorted(f for f in files if physical(f) or Path(f).is_absolute() or '..' in Path(f).parts)
    if bad:
        raise ValueError(f'Refusing to stage physical-source or external files: {bad[:5]}')
    return sorted(files)


# ------------------------------------------------------------------------------------------------
# Build (on the root that runs the jobs) and the runtime records
# ------------------------------------------------------------------------------------------------
def build_comparator(root, directory, *, expected_source_sha256=None, env=None, runner=None) -> dict:
    """Build one comparator library as runtime.build_solver builds the basis library; see the module doc."""
    from forge import runtime
    root = runtime._root(root)
    info = check_tree(root, directory)
    if expected_source_sha256 and info['source_sha256'] != expected_source_sha256:
        raise RuntimeError(f'{directory}: source_sha256 {info["source_sha256"]} differs from the frozen '
                           f'{expected_source_sha256}')
    cls, source = info['ras_model'], root/COMPARATORS/directory
    env = runtime.foam_env(root) if env is None else dict(env)
    runner = runtime.run_command if runner is None else runner
    build = root/'runtime/builds/comparators'/f'{cls}-{info["source_sha256"][:16]}-{time.time_ns()}'
    copied = build/'src'
    for relative in info['sources']:
        target = copied/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative, target)
    if source_sha256(copied) != info['source_sha256']:
        raise RuntimeError('Copied comparator sources differ from the checked tree')
    output = build/'output'
    output.mkdir()
    env['FOAM_USER_LIBBIN'] = str(output)
    run = runner(['wmake', 'libso'], copied, build/'build.log', 1800, env)
    if run['returncode'] != 0 or run['timed_out']:
        raise RuntimeError(f'Comparator build failed ({cls}); see {run["log"]}')
    lib = output/info['library']
    if not lib.is_file():
        raise RuntimeError(f'wmake produced no {info["library"]}; see {run["log"]}')
    if cls.encode() not in lib.read_bytes():
        raise RuntimeError(f'{lib} does not contain the TypeName {cls}')
    # wmake libso leaves undefined symbols to load time: load the library in the same environment
    # (OpenFOAM libraries on LD_LIBRARY_PATH) so that a missing symbol fails here, not in the first job.
    # The interpreter's own library path goes after OpenFOAM's (a module-built Python may need it to start).
    probe = {**env, 'LD_LIBRARY_PATH': ':'.join(x for x in (env.get('LD_LIBRARY_PATH', ''),
                                                            os.environ.get('LD_LIBRARY_PATH', '')) if x)}
    loaded = runner([sys.executable, '-c', 'import ctypes, os, sys; ctypes.CDLL(sys.argv[1], mode=os.RTLD_NOW); print("loaded")',
                     str(lib)], build, build/'load.log', 300, probe)
    if loaded['returncode'] != 0 or loaded['timed_out']:
        tail = Path(loaded['log']).read_text()[-600:] if Path(loaded['log']).is_file() else ''
        raise RuntimeError(f'{lib} does not load (undefined symbols?); see {loaded["log"]}: {tail}')
    digest = sha256(lib)
    frozen = root/'runtime/lib'/f'lib{cls}-{digest[:16]}.so'
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if frozen.exists() and sha256(frozen) != digest:
        raise RuntimeError('Immutable library hash collision')
    if not frozen.exists():
        shutil.copy2(lib, frozen)
        frozen.chmod(0o444)
    record = {'schema_version': 1, 'qualified_build': True, 'ras_model': cls, 'library_name': info['library'],
              'library': str(frozen), 'library_sha256': digest, 'source_dir': info['source_dir'],
              'source_sha256': info['source_sha256'], 'sources': info['sources'],
              'excluded_artefacts': info['excluded_artefacts'], 'registration': info['registration'],
              'type_name_in_binary': True,
              'make_options_sha256': sha256(root/BASIS_OPTIONS),
              'openfoam_version': env.get('WM_PROJECT_VERSION'), 'wm_options': env.get('WM_OPTIONS'),
              'platform_identity': runtime.platform_identity(root),
              'basis_build_manifest_sha256': (sha256(root/'runtime/build_manifest.json')
                                              if (root/'runtime/build_manifest.json').exists() else None),
              'build': run, 'load_check': loaded, 'build_tree': str(build), 'host': platform.node(),
              'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'isolation_root': str(root)}
    _atomic_json(root/RECORDS/f'{cls}.json', record)
    return record


def built_library(root, ras_model, expected_sha256=None) -> tuple[Path, str]:
    """Worker side: the immutable library of `ras_model` on this root (runtime/comparators/<ras_model>.json),
    verified as runtime.library_path verifies the basis library."""
    root = Path(root).resolve()
    path = root/RECORDS/f'{ras_model}.json'
    if not path.is_file():
        raise RuntimeError(f'No built comparator library for {ras_model} on {root}; build it first')
    entry = json.loads(path.read_text())
    library = Path(entry['library']).resolve()
    if entry.get('ras_model') != ras_model or not entry.get('qualified_build'):
        raise RuntimeError(f'{path} is not a qualified build record of {ras_model}')
    if not library.is_relative_to(root/'runtime/lib') or not library.is_file():
        raise RuntimeError(f'Comparator library is missing or outside runtime/lib: {library}')
    digest = sha256(library)
    if digest != entry['library_sha256'] or digest[:16] not in library.name:
        raise RuntimeError(f'Comparator library {library} does not match its immutable record')
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(f'Comparator library of {ras_model} is {digest}, the experiment froze {expected_sha256}')
    return library, digest


def describe(root, ras_models=None) -> dict:
    """The built libraries on this root, each verified (python -m forge_deploy describe; read by collect)."""
    root = Path(root).resolve()
    rows = {}
    for path in sorted((root/RECORDS).glob('*.json')):
        cls = path.stem
        if ras_models and cls not in ras_models:
            continue
        entry = json.loads(path.read_text())
        library, digest = built_library(root, cls)
        rows[cls] = {k: entry.get(k) for k in ('ras_model', 'library_name', 'source_dir', 'source_sha256',
                                               'openfoam_version', 'wm_options', 'built_utc', 'host')}
        rows[cls].update(library=str(library), library_sha256=digest, record=str(path.relative_to(root)),
                         record_sha256=sha256(path), build_log=entry['build']['log'])
    missing = sorted(set(ras_models or ()) - set(rows))
    if missing:
        raise RuntimeError(f'Not built on {root}: {missing}')
    return {'root': str(root), 'comparators': rows}


def controller_record(root, ras_model=None, *, path=None):
    """Controller side: runtime/hx1_verify/comparators.json, the flat map agreed with HARNESS
    {<ras_model>: {library_path (remote), library_sha256, source_sha256, local_copy, ...}}, every
    retained local copy verified. With `ras_model`, that model's entry."""
    root = Path(root).resolve()
    record = json.loads(Path(path or root/CONTROLLER_RECORD).read_text())
    for cls, entry in record.items():
        local = root/entry['local_copy']
        if not local.is_file() or sha256(local) != entry['library_sha256']:
            raise RuntimeError(f'Retained copy of the {cls} library changed: {local}')
    if ras_model is None:
        return record
    if ras_model not in record:
        raise RuntimeError(f'{ras_model} has not been built and collected; run the build step')
    return record[ras_model]
