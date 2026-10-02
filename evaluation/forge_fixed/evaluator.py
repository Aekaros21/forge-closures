"""Fixed models in the existing evaluators: identity, final-test access, binding and PBS job script.

``FixedModelMixin`` goes in front of an evaluator class (Evaluator, PBSDevelopmentEvaluator). For SST
(spec None) and expression specs every method defers to the evaluator unchanged, so their identities,
requests and job scripts are byte-identical to those of campaigns 001-007.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading

from . import FixedModel, _fingerprint, _sha, install_case_preparation

_JOB = threading.local()
STOCK_WORKER = ' -m forge.pbs_worker run '
FIXED_WORKER = ' -m forge_fixed.worker run '
DESCRIPTOR = 'runtime/hx1_verify/comparators.json'


def install_job_script():
    """A fixed-model job runs forge_fixed.worker; otherwise its job.pbs is pbs.job_script() unchanged."""
    from forge import pbs
    if getattr(pbs.job_script, '_forge_fixed', False):
        return
    original = pbs.job_script

    def job_script(*args, **kwargs):
        text = original(*args, **kwargs)
        if getattr(_JOB, 'fixed', 0):
            if text.count(STOCK_WORKER) != 1:
                raise RuntimeError('Unexpected PBS job script; cannot route the fixed-model worker')
            text = text.replace(STOCK_WORKER, FIXED_WORKER)
        return text
    job_script._forge_fixed = True
    job_script._original = original
    pbs.job_script = job_script


def controller_builds(root: Path, config: dict) -> dict:
    """Comparator builds collected from HX1 (HARNESS-DEPLOY's runtime/hx1_verify/comparators.json)."""
    path = Path(root)/config.get('backend', {}).get('comparator_descriptor', DESCRIPTOR)
    if not path.exists():
        return {}
    builds = {}
    for ras_model, entry in json.loads(path.read_text()).items():
        if isinstance(entry, dict) and 'library_sha256' in entry:
            builds[ras_model] = {'path': entry['library_path'], 'sha256': entry['library_sha256'],
                                 'source_sha256': entry['source_sha256'], 'local_copy': entry.get('local_copy')}
    return builds


def node_builds(root: Path) -> dict:
    """Comparator builds at this root (runtime/comparators/<ras_model>.json, one per library)."""
    builds = {}
    for path in sorted((Path(root)/'runtime/comparators').glob('*.json')):
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or 'ras_model' not in record or 'library' not in record:
            continue
        if record.get('qualified_build') is False or path.stem != record['ras_model']:
            continue
        builds[record['ras_model']] = {'path': record['library'], 'sha256': record['library_sha256'],
                                       'source_sha256': record['source_sha256'], 'local_copy': None}
    return builds


class FixedModelMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_case_preparation()
        install_job_script()
        self._comparator_builds = None

    # -- binding ---------------------------------------------------------------------------
    def comparator_builds(self) -> dict:
        if getattr(self, '_comparator_builds', None) is None:
            self._comparator_builds = (controller_builds(self.root, self.config) if self.backend_kind == 'pbs'
                                       else node_builds(self.root))
        return self._comparator_builds

    def bind_fixed(self, model: FixedModel) -> FixedModel:
        build = self.comparator_builds().get(model.ras_model)
        if build is None:
            raise RuntimeError(f'{model.ras_model}: no comparator build recorded (build it on HX1 and collect '
                               f'{DESCRIPTOR} first)')
        if build['source_sha256'] != model.source_sha256:
            raise RuntimeError(f'{model.ras_model}: the built library comes from other sources '
                               f'({build["source_sha256"][:12]}, frozen {model.source_sha256[:12]})')
        if build.get('local_copy'):
            copy = Path(self.root)/build['local_copy']
            if not copy.is_file() or _sha(copy) != build['sha256']:
                raise RuntimeError(f'{model.ras_model}: retained comparator library copy changed or missing: {copy}')
        if model.library_path is not None and model.library_sha256 == build['sha256'] and model.library_path == build['path']:
            return model
        return model.bound(build['path'], build['sha256'])

    # -- evaluator hooks -------------------------------------------------------------------
    def case_identity(self, spec, case, protocol, ranks):
        if not isinstance(spec, FixedModel):
            return super().case_identity(spec, case, protocol, ranks)
        pairing, _, _ = super().case_identity(None, case, protocol, ranks)
        identity = {**pairing, 'spec': self.bind_fixed(spec).identity()}
        return pairing, identity, _fingerprint(identity)

    def run_case(self, spec, cid, stage='development', **kwargs):
        if isinstance(spec, FixedModel):
            spec = self.bind_fixed(spec)
        return super().run_case(spec, cid, stage, **kwargs)

    def final_access(self, spec) -> bool:
        if not isinstance(spec, FixedModel):
            return super().final_access(spec)
        if not super().final_access(None):   # verification campaign, frozen record unchanged
            return False
        from forge.core import within
        record = json.loads(within(self.root, self.root/self.config['verification']['freeze']['path']).read_text())
        return spec.model_fingerprint() in {m['model_fingerprint'] for m in record['models']}

    def _scheduled_case(self, spec, *args, **kwargs):
        if not isinstance(spec, FixedModel):
            return super()._scheduled_case(spec, *args, **kwargs)
        previous = getattr(_JOB, 'fixed', 0)
        _JOB.fixed = previous + 1
        try:
            return super()._scheduled_case(spec, *args, **kwargs)
        finally:
            _JOB.fixed = previous


def fixed_class(base):
    """``base`` with fixed-model support, e.g. fixed_class(PBSDevelopmentEvaluator)."""
    return type('Fixed' + base.__name__, (FixedModelMixin, base), {})
