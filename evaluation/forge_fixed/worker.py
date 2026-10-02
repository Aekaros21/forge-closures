"""Run one frozen fixed-model request inside a PBS or harvested allocation.

The fixed-model twin of forge.pbs_worker ``run``: the same checks (physical source, case assets,
identity), the same receipt and the same local evaluator, with the comparator library resolved from
runtime/comparators/<ras_model>.json at this root and checked against the request's identity.

  python -m forge_fixed.worker run deployment/tasks/<key>/request.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time
import traceback

from forge.core import ROOT, atomic_json, fingerprint, sha256, source_identity, within


def run(path):
    from forge.catalog import validate_manifest
    from forge.evaluator import Evaluator
    from forge.pbs_worker import worker_config
    from forge_fixed import FixedModel
    from forge_fixed.evaluator import fixed_class
    path = within(ROOT, Path(path))
    request = json.loads(path.read_text())
    if not os.environ.get('PBS_JOBID'):
        raise RuntimeError('Physical worker requires a PBS compute allocation')
    if fingerprint(source_identity(ROOT, physical_only=True)) != request['source_hash']:
        raise RuntimeError('Worker source differs from the frozen request')
    started = time.time()
    receipt = {'pbs_job_id': os.environ['PBS_JOBID'], 'hostname': platform.node(),
               'started_unix': started, 'request_sha256': sha256(path), 'status': 'running'}
    atomic_json(path.parent/'receipt.json', receipt)
    try:
        model = FixedModel.from_identity(request['spec'])
        case = request['case']
        cid = case['id']
        manifest = {**request['manifest_metadata'], 'cases': [case]}
        if not validate_manifest(manifest, ROOT)['ready']:
            raise RuntimeError('Worker is missing a declared case asset')
        evaluator = fixed_class(Evaluator)(ROOT, manifest, request['contract'], worker_config(request))
        _, identity, key = evaluator.case_identity(model, case, evaluator.effective_protocol(case, request['stage']),
                                                   request['ranks'])
        if key != request['experiment_hash'] or identity != request['identity']:
            raise RuntimeError('Worker physical identity differs from the controller request')
        result = evaluator.run_case(model, cid, request['stage'])
        receipt.update(status='complete', result_status=result['status'], experiment_hash=key)
    except BaseException as exc:
        receipt.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(finished_unix=time.time(), worker_wall_s=time.time()-started)
        atomic_json(path.parent/'receipt.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run',))
    parser.add_argument('request')
    args = parser.parse_args()
    run(args.request)


if __name__ == '__main__':
    main()
