"""Fixed C++ closures (comparators) beside stock SST and expression specs.

A fixed model is an OpenFOAM RAS model compiled from ``src/comparators/<dir>/`` into its own library
(``lib<RASModel>.so``) and configured by a ``<RASModel>Coeffs`` dictionary. It runs through the same
evaluator, experiment cache, admission rules, endpoint ladders and report as SST and the expression
models (reports/overnight_20261001/handoff/harness_interface.md).

This package is deliberately outside src/forge, src/tedp and src/kOmegaSSTBasis: those folders form the
physical source identity bound into every experiment, so editing them would turn every cached SST and
expression experiment into a cache miss. SST and expression identities are untouched; a fixed-model
experiment binds, instead, the model definition, the built library's sha256 and this package's own
sha256 (``harness_sha256``).

The case of a fixed model is the stock SST case of the same adapter, byte for byte, except
constant/turbulenceProperties (RASModel, coefficient dictionary) and the comparator library added to
the case's one top-level ``libs`` list in system/controlDict.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
KIND = 'fixed_cpp_model'
BUILD_PARTS = ('lnInclude', 'linux64GccDPInt32Opt')
# Physics-relevant modules of this package, bound into every fixed-model experiment identity.
HARNESS_FILES = ('__init__.py', 'evaluator.py', 'worker.py')

_WORD = re.compile(r'[A-Za-z][A-Za-z0-9_]*\Z')
_VALUE_WORD = re.compile(r'[A-Za-z0-9_.]+\Z')
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_SOURCE_DIR = re.compile(r'src/comparators/[a-z0-9_]+\Z')
# Stock OpenFOAM v2312 incompressible RAS models and the basis model: a comparator may not shadow one.
RESERVED_RAS_MODELS = frozenset({
    'kOmegaSST', 'kOmegaSSTLM', 'kOmegaSSTSAS', 'kOmega', 'kEpsilon', 'kEpsilonPhitF', 'realizableKE',
    'RNGkEpsilon', 'LaunderSharmaKE', 'SpalartAllmaras', 'LRR', 'SSG', 'EBRSM', 'LienCubicKE',
    'LienLeschziner', 'ShihQuadraticKE', 'kkLOmega', 'qZeta', 'v2f', 'LamBremhorstKE', 'kOmegaSSTBasis'})


# ---------------------------------------------------------------------------------------------
# hashes
# ---------------------------------------------------------------------------------------------
def _sha(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_files(directory) -> list[Path]:
    """The comparator's source files: build products and logs excluded (as runtime.build_solver)."""
    directory = Path(directory)
    return [p for p in sorted(directory.rglob('*')) if p.is_file()
            and not any(q in BUILD_PARTS for q in p.relative_to(directory).parts)
            and p.suffix not in ('.o', '.dep') and not p.name.startswith('log.')]


def source_manifest(directory) -> dict[str, str]:
    directory = Path(directory)
    return {str(p.relative_to(directory)): _sha(p) for p in source_files(directory)}


def source_sha256(directory) -> str:
    """SHA-256 of the sorted {relative path: file SHA-256} map, the basis library's scheme."""
    manifest = source_manifest(directory)
    if not manifest:
        raise ValueError(f'No comparator sources in {directory}')
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


_HARNESS = {}


def harness_sha256() -> str:
    """SHA-256 of this package's physics-relevant modules (case preparation, identity, worker)."""
    if 'sha' not in _HARNESS:
        files = {name: _sha(PACKAGE/name) for name in HARNESS_FILES}
        _HARNESS['sha'] = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return _HARNESS['sha']


def _fingerprint(value) -> str:
    from forge.core import fingerprint
    return fingerprint(value)


# ---------------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------------
def check_coeffs(coeffs) -> dict:
    if not isinstance(coeffs, dict):
        raise ValueError('coeffs must be a JSON object')
    for key, value in coeffs.items():
        if not isinstance(key, str) or not _WORD.match(key) or key == 'frameOmega':
            raise ValueError(f'Invalid coefficient name {key!r} (frameOmega is written by the harness)')
        format_value(value)
    return {key: coeffs[key] for key in sorted(coeffs)}


def format_value(value) -> str:
    """A coefficient as an OpenFOAM dictionary value: number, switch or word."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError('Coefficients must be finite')
        return f'{value:.17g}'
    if isinstance(value, str) and _VALUE_WORD.match(value):
        return value
    raise ValueError(f'Unsupported coefficient value {value!r}: number, boolean or word only')


class FixedModel:
    """One fixed C++ RAS model; ``library_path``/``library_sha256`` bind a built library."""
    __slots__ = ('name', 'ras_model', 'library', 'source_dir', 'coeffs', 'source_sha256',
                 'library_sha256', 'library_path')

    def __init__(self, *, name, ras_model, library, source_dir, coeffs, source_sha256,
                 library_sha256=None, library_path=None):
        if not isinstance(name, str) or not name:
            raise ValueError('A fixed model needs a name')
        if not isinstance(ras_model, str) or not _WORD.match(ras_model) or ras_model in RESERVED_RAS_MODELS:
            raise ValueError(f'Invalid or reserved RAS model name {ras_model!r}')
        if library != f'lib{ras_model}.so':
            raise ValueError(f'The library of {ras_model} must be lib{ras_model}.so, not {library!r}')
        if source_dir is not None and (not isinstance(source_dir, str) or not _SOURCE_DIR.match(source_dir)):
            raise ValueError(f'source_dir must be src/comparators/<dir>, not {source_dir!r}')
        if not isinstance(source_sha256, str) or not _HEX.match(source_sha256):
            raise ValueError('source_sha256 must be 64 lower-case hex digits')
        if library_sha256 is not None and (not isinstance(library_sha256, str) or not _HEX.match(library_sha256)):
            raise ValueError('library_sha256 must be 64 lower-case hex digits')
        self.name, self.ras_model, self.library, self.source_dir = name, ras_model, library, source_dir
        self.coeffs = check_coeffs(coeffs)
        self.source_sha256, self.library_sha256 = source_sha256, library_sha256
        self.library_path = None if library_path is None else str(library_path)

    @classmethod
    def from_entry(cls, entry: dict) -> 'FixedModel':
        """From a freeze-record or comparator-models entry {"name", "fixed": {...}}."""
        fixed = entry['fixed']
        unknown = set(fixed) - {'ras_model', 'library', 'source_dir', 'coeffs', 'source_sha256'}
        if unknown:
            raise ValueError(f'Unknown fixed-model fields {sorted(unknown)}')
        return cls(name=entry['name'], ras_model=fixed['ras_model'], library=fixed['library'],
                   source_dir=fixed['source_dir'], coeffs=fixed['coeffs'], source_sha256=fixed['source_sha256'])

    @classmethod
    def from_identity(cls, identity: dict) -> 'FixedModel':
        """The model of an experiment identity (a PBS request), bound to its library sha256."""
        expected = {'kind', 'ras_model', 'library', 'coeffs', 'source_sha256', 'library_sha256', 'harness_sha256'}
        if not isinstance(identity, dict) or set(identity) != expected or identity['kind'] != KIND:
            raise ValueError('Not a fixed-model experiment identity')
        if identity['harness_sha256'] != harness_sha256():
            raise RuntimeError('forge_fixed differs from the package that froze this request')
        return cls(name='pbs-comparator', ras_model=identity['ras_model'], library=identity['library'],
                   source_dir=None, coeffs=identity['coeffs'], source_sha256=identity['source_sha256'],
                   library_sha256=identity['library_sha256'])

    def entry(self) -> dict:
        return {'ras_model': self.ras_model, 'library': self.library, 'source_dir': self.source_dir,
                'coeffs': dict(self.coeffs), 'source_sha256': self.source_sha256}

    def definition(self) -> dict:
        """The frozen model (its fingerprint names it in a freeze record); the display name is excluded."""
        return {'kind': KIND, 'ras_model': self.ras_model, 'library': self.library,
                'coeffs': dict(self.coeffs), 'source_sha256': self.source_sha256}

    def model_fingerprint(self) -> str:
        return _fingerprint(self.definition())

    def bound(self, library_path, library_sha256) -> 'FixedModel':
        if self.library_sha256 is not None and library_sha256 != self.library_sha256:
            raise RuntimeError(f'{self.ras_model}: built library {library_sha256} is not the requested {self.library_sha256}')
        other = FixedModel(name=self.name, ras_model=self.ras_model, library=self.library, source_dir=self.source_dir,
                           coeffs=self.coeffs, source_sha256=self.source_sha256,
                           library_sha256=library_sha256, library_path=library_path)
        return other

    def identity(self) -> dict:
        """The model part of an experiment identity: definition, built library and harness."""
        if self.library_sha256 is None:
            raise RuntimeError(f'{self.ras_model}: no built library is bound (deploy and collect the comparator first)')
        return {**self.definition(), 'library_sha256': self.library_sha256, 'harness_sha256': harness_sha256()}

    def __bool__(self):
        return True

    def __repr__(self):
        return f'FixedModel({self.name!r}, {self.ras_model}, {len(self.coeffs)} coefficients)'


# ---------------------------------------------------------------------------------------------
# case preparation
# ---------------------------------------------------------------------------------------------
def frame_omega(case: dict) -> tuple:
    """The frame angular velocity of a case: the rotating channels' (as the basis library receives it)."""
    if case['adapter'] == 'rotation':
        from tedp.holdout_cases import rotchan
        return (0, 0, rotchan.omega_for(float(case['adapter_config']['ro'])))
    return (0, 0, 0)


def render_turbulence_properties(model: FixedModel, omega=(0, 0, 0)) -> str:
    from tedp.casegen import TURB_HEADER
    if len(omega) != 3 or not all(isinstance(c, (int, float)) and math.isfinite(c) for c in omega):
        raise ValueError('frameOmega must be three finite numbers')
    lines = ['RAS', '{', f'    RASModel        {model.ras_model};', '    turbulence      on;',
             '    printCoeffs     off;', f'    {model.ras_model}Coeffs', '    {']
    lines += [f'        {key:<15} {format_value(value)};' for key, value in model.coeffs.items()]
    lines += [f'        {"frameOmega":<15} ({" ".join(str(c) for c in omega)});', '    }', '}', '']
    return TURB_HEADER + '\n'.join(lines)


def _top_level(text: str, index: int) -> bool:
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), text)
    return masked[:index].count('{') == masked[:index].count('}')


def add_library(control_text: str, basis_library, comparator_library) -> str:
    """Add the comparator to the one top-level libs list that already loads the basis library."""
    quoted = '"' + str(basis_library) + '"'
    if control_text.count(quoted) != 1:
        raise ValueError('controlDict must name the basis library exactly once')
    index = control_text.index(quoted)
    start = control_text.rfind('\n', 0, index) + 1
    if not _top_level(control_text, index) or not re.match(r'\s*libs\s*\(', control_text[start:index]):
        raise ValueError('The basis library is not in a top-level libs list')
    return control_text[:index] + quoted + ' "' + str(comparator_library) + '"' + control_text[index+len(quoted):]


def check_library(model: FixedModel, root: Path = ROOT) -> Path:
    if model.library_path is None or model.library_sha256 is None:
        raise RuntimeError(f'{model.ras_model}: no built library is bound')
    path = Path(model.library_path).resolve()
    if not path.is_file() or not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f'{model.ras_model}: the comparator library must be a file inside the checkout: {path}')
    if not path.name.startswith(f'lib{model.ras_model}-') or model.library_sha256[:12] not in path.name:
        raise ValueError(f'{model.ras_model}: use the immutable hash-named library, not {path.name}')
    if _sha(path) != model.library_sha256:
        raise RuntimeError(f'{model.ras_model}: comparator library changed: {path}')
    return path


def prepare_fixed_case(case, model: FixedModel, work, *, end_time, library, base_prepare, root: Path = ROOT):
    """The stock SST case of ``case`` (``base_prepare`` with spec None), then the fixed model's
    turbulenceProperties and library. Everything else is the SST case byte for byte."""
    comparator = check_library(model, root)
    metadata = base_prepare(case, None, work, end_time=end_time, library=library)
    work = Path(work)
    turb = work/'constant/turbulenceProperties'
    if not re.search(r'(?m)^\s*RASModel\s+kOmegaSST\s*;', turb.read_text()):
        raise ValueError('The base case is not the stock SST case')
    omega = frame_omega(case)
    turb.write_text(render_turbulence_properties(model, omega))
    control = work/'system/controlDict'
    control.write_text(add_library(control.read_text(), Path(library).resolve(), comparator))
    metadata = {**metadata, 'spec': None, 'fixed_model': model.identity(), 'frame_omega': list(omega),
                'comparator_library': str(comparator), 'comparator_library_sha256': model.library_sha256,
                'turbulence_sha256': _sha(turb),
                'numerics_sha256': {**metadata['numerics_sha256'], 'controlDict': _sha(control)}}
    (work/'forge_case.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def install_case_preparation():
    """Route fixed models in forge.cases.prepare_case; SST and specs pass through unchanged."""
    from forge import cases
    if getattr(cases.prepare_case, '_forge_fixed', False):
        return
    original = cases.prepare_case

    def prepare_case(case, spec, work, *, end_time, library):
        if isinstance(spec, FixedModel):
            return prepare_fixed_case(case, spec, work, end_time=end_time, library=library, base_prepare=original)
        return original(case, spec, work, end_time=end_time, library=library)
    prepare_case._forge_fixed = True
    prepare_case._original = original
    cases.prepare_case = prepare_case
