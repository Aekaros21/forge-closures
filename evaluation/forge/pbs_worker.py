"""Run one frozen evaluator request inside a PBS compute allocation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import time
import traceback

from .core import ROOT, atomic_json, fingerprint, sha256, source_identity, within


def describe(*, verify_qualification=False):
    from .runtime import foam_env, library_path, platform_identity
    import shutil
    env=foam_env(ROOT)
    library=library_path(ROOT)
    solver=Path(shutil.which('simpleFoam',path=env['PATH']))
    result={'root':str(ROOT),'physical_source':source_identity(ROOT,physical_only=True),
            'runtime_identity':platform_identity(ROOT),
            'environment':{k:env.get(k) for k in ('WM_OPTIONS','WM_PROJECT_VERSION','SCOTCH_PTHREAD_NUMBER')},
            'library':{'path':str(library),'sha256':sha256(library)},
            'solver':{'path':str(solver),'sha256':sha256(solver)}}
    if verify_qualification:
        export=json.loads((ROOT/'deployment/runtime_export.json').read_text())
        result['qualification_files']={name:sha256(within(ROOT,ROOT/name))
                                       for name in export['qualification_files']}
        if result['qualification_files']!=export['qualification_files']:
            raise RuntimeError('Native qualification files changed after export')
    return result


def restore_spec(definition):
    from tedp.spec import from_json
    # Names are excluded from physical identities; the parser still requires a
    # display name. Restoring it must not change the equation or its constants.
    return None if definition is None else from_json(json.dumps({**definition,'name':'pbs-candidate'}))


def worker_config(request):
    """The one-case evaluator configuration of a request; a final-test case keeps the
    controller's recorded authorisation (see pbs.final_access_grant)."""
    cid=request['case']['id']
    config={'campaign':'pbs-worker-'+request['experiment_hash'],
            'backend':{'kind':'local','ranks':request['ranks']},
            'budget':{'core_hours':None,'confirmation_reserve_core_hours':0},
            'numerics':{},'stages':{},'case_protocol_overrides':{cid:request['protocol']}}
    if request.get('final_access'):
        config['final_access_granted']=request['final_access']
    return config


def run(path):
    from .evaluator import Evaluator
    path=within(ROOT,Path(path))
    request=json.loads(path.read_text())
    if not os.environ.get('PBS_JOBID'):
        raise RuntimeError('Physical worker requires a PBS compute allocation')
    actual_source=source_identity(ROOT,physical_only=True)
    if fingerprint(actual_source)!=request['source_hash']:
        raise RuntimeError('Worker source differs from the frozen request')
    started=time.time()
    receipt={'pbs_job_id':os.environ['PBS_JOBID'],'hostname':platform.node(),
             'started_unix':started,'request_sha256':sha256(path),'status':'running'}
    atomic_json(path.parent/'receipt.json',receipt)
    try:
        case=request['case'];cid=case['id']
        manifest={**request['manifest_metadata'],'cases':[case]}
        from .catalog import validate_manifest
        if not validate_manifest(manifest,ROOT)['ready']:
            raise RuntimeError('Worker is missing a declared case asset')
        config=worker_config(request)
        evaluator=Evaluator(ROOT,manifest,request['contract'],config)
        spec=restore_spec(request['spec'])
        pairing,identity,key=evaluator.case_identity(spec,case,evaluator.effective_protocol(case,request['stage']),request['ranks'])
        if key!=request['experiment_hash'] or identity!=request['identity']:
            raise RuntimeError('Worker physical identity differs from the controller request')
        result=evaluator.run_case(spec,cid,request['stage'])
        receipt.update(status='complete',result_status=result['status'],experiment_hash=key)
    except BaseException as exc:
        receipt.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(finished_unix=time.time(),worker_wall_s=time.time()-started)
        atomic_json(path.parent/'receipt.json',receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('describe','verify','run'))
    parser.add_argument('request',nargs='?')
    args=parser.parse_args()
    if args.command in ('describe','verify'):
        print(json.dumps(describe(verify_qualification=args.command=='verify'),indent=2))
    else:run(args.request)


if __name__=='__main__':main()
