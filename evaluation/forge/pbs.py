"""PBS execution with durable submission records and verified local evidence."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import time

from .core import ROOT, atomic_json, file_lock, fingerprint, sha256, source_identity, within
from .evaluator import Evaluator


class SSHTransport:
    def __init__(self, options):
        self.options=options
        self.target=options['host']
        self.socket=Path(options['control_socket']).resolve()
        self.remote=Path(options['remote_root'])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*',self.target):
            raise ValueError('Invalid SSH host or alias')
        if not self.remote.is_absolute() or '..' in self.remote.parts or len(self.remote.parts)<5:
            raise ValueError('An isolated absolute remote repository path is required')

    def base(self):
        if not self.socket.is_socket():
            raise RuntimeError('HX1 shared SSH connection is unavailable; reopen the authenticated master')
        details=self.socket.stat()
        if details.st_uid!=os.getuid() or stat.S_IMODE(details.st_mode)&0o077:
            raise RuntimeError('SSH control socket is not private to this user')
        return ['ssh','-S',str(self.socket),'-o','BatchMode=yes','-o','ControlMaster=no',
                '-o','ProxyCommand=false','-o','ConnectTimeout=10']

    def call(self, arguments, *, input=None, timeout=60):
        result=subprocess.run([*self.base(),self.target,shlex.join(map(str,arguments))],
                              input=input,capture_output=True,text=True,timeout=timeout)
        if result.returncode:
            raise RuntimeError('HX1 command failed: '+result.stderr.strip()[-3000:])
        return result.stdout

    def dispatch(self, command, argument):
        text=self.call(['python3',self.remote/'scripts/pbs_dispatch.py',command,argument],timeout=90)
        return json.loads(text)

    def sync(self, source, destination, *, receive=False, exclude=()):
        args=['rsync','-a','--safe-links','--exclude=.lock',*('--exclude='+x for x in exclude),'-e',shlex.join(self.base())]
        if receive:args.extend([self.target+':'+str(source),str(destination)])
        else:args.extend([str(source),self.target+':'+str(destination)])
        result=subprocess.run(args,capture_output=True,text=True,timeout=1800)
        if result.returncode:
            raise RuntimeError('HX1 transfer failed: '+result.stderr.strip()[-3000:])


def wall_seconds(value):
    fields=str(value).split(':')
    if len(fields)!=3 or not all(re.fullmatch(r'\d+',x) for x in fields):
        raise ValueError('Invalid PBS wall-time accounting')
    h,m,s=map(int,fields)
    if m>=60 or s>=60:raise ValueError('Invalid PBS wall-time components')
    return h*3600+m*60+s


def allocated_cost(snapshot, reservation, expected_ranks):
    """Use allocated CPUs and elapsed wall time; CPU utilization is different."""
    job=snapshot.get('job') or {}
    used=job.get('resources_used',{})
    requested=job.get('Resource_List',{})
    if snapshot.get('available') and job.get('job_state') in ('F','C') and 'walltime' in used:
        cpus=int(requested.get('ncpus',expected_ranks))
        if cpus!=expected_ranks:
            raise RuntimeError('PBS allocated CPU count differs from the frozen request')
        return wall_seconds(used['walltime'])*cpus/3600,'PBS allocated ncpus × resources_used.walltime'
    return reservation,'Full reservation; final PBS accounting unavailable'


def job_script(remote, request_path, *, ranks, wall_s, memory_gb, queue, name):
    if not (1<=ranks<=64 and wall_s>0 and 1<=memory_gb<=450):
        raise ValueError('Invalid single-node PBS resources')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',queue) or not re.fullmatch(r'[A-Za-z0-9_-]{1,15}',name):
        raise ValueError('Invalid PBS queue or name')
    seconds=int(math.ceil(wall_s/60)*60)
    wall=f'{seconds//3600:02d}:{seconds%3600//60:02d}:00'
    remote=Path(remote);request_path=Path(request_path)
    return '\n'.join([
        '#!/bin/bash',f'#PBS -N {name}',f'#PBS -q {queue}',
        f'#PBS -l select=1:ncpus={ranks}:mpiprocs={ranks}:mem={memory_gb}gb',
        f'#PBS -l walltime={wall}','#PBS -j oe','set -eo pipefail',
        'cd '+shlex.quote(str(remote)),
        'exec > '+shlex.quote(str(request_path.parent/'worker.log'))+' 2>&1',
        'export PYTHONPATH='+shlex.quote(str(remote/'src')),
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'exec '+shlex.join([str(remote/'.venv/bin/python'),'-u','-m','forge.pbs_worker','run',str(request_path)]),
        ''])


def import_experiment(incoming, folder, remote_root, expected_identity, scheduler):
    """Preserve the original report and map only its evidence locations."""
    incoming,folder,remote_root=Path(incoming),Path(folder),Path(remote_root)
    # OpenFOAM's dynamicCode/lnInclude contains relative source-file links.
    # Materialize only files contained in the received experiment. Reject
    # directory links (including cycles), broken links and external targets
    # before reading reports or copying anything into the durable cache.
    materialized_links={}
    for path in incoming.rglob('*'):
        if path.is_symlink():
            try:
                target=path.resolve(strict=True)
            except (OSError,RuntimeError) as exc:
                raise RuntimeError('Transferred experiment contains an invalid symlink: '+str(path)) from exc
            if not target.is_relative_to(incoming.resolve()) or not target.is_file():
                raise RuntimeError('Transferred experiment symlink must target an internal regular file: '+str(path))
            materialized_links[str(path.relative_to(incoming))]={
                'target':str(path.readlink()),'sha256':sha256(target)}
    raw_path=incoming/'result.json'
    if not raw_path.is_file() or (incoming/'result.sha256').read_text().strip()!=sha256(raw_path):
        raise RuntimeError('Remote result checksum failed')
    raw=json.loads(raw_path.read_text())
    key=fingerprint(expected_identity)
    if raw.get('identity')!=expected_identity or raw.get('experiment_hash')!=key:
        raise RuntimeError('Remote result identity differs from the frozen request')
    remote_experiment=remote_root/'runs/experiments'/key
    if not raw.get('artifacts'):
        raise RuntimeError('Remote result has no physical evidence')
    for name,digest in raw['artifacts'].items():
        try:relative=Path(name).relative_to(remote_experiment)
        except ValueError:raise RuntimeError('Remote artifact is outside its experiment')
        local=within(incoming,incoming/relative)
        if not local.is_file() or sha256(local)!=digest:
            raise RuntimeError('Transferred artifact checksum failed: '+str(relative))
    def ignore(directory,names):
        return [x for x in names if x=='.lock' or (Path(directory)==incoming and x in ('result.json','result.sha256'))]
    shutil.copytree(incoming,folder,dirs_exist_ok=True,ignore=ignore)
    for relative,record in materialized_links.items():
        copied=folder/relative
        if copied.is_symlink() or sha256(copied)!=record['sha256']:
            raise RuntimeError('Materialized transfer link changed: '+relative)
    original=folder/'remote_result.json'
    original.write_bytes(raw_path.read_bytes())
    def mapped(value):
        if isinstance(value,dict):return {k:mapped(v) for k,v in value.items()}
        if isinstance(value,list):return [mapped(v) for v in value]
        if isinstance(value,str) and (value==str(remote_experiment) or value.startswith(str(remote_experiment)+'/')):
            return str(folder)+value[len(str(remote_experiment)):]
        return value
    result=mapped(raw)
    result['identity']=raw['identity']
    result['artifacts']={str(folder/Path(name).relative_to(remote_experiment)):digest
                         for name,digest in raw['artifacts'].items()}
    result['artifacts'][str(original)]=sha256(original)
    receipt=folder/'pbs_accounting.json';atomic_json(receipt,scheduler)
    result['artifacts'][str(receipt)]=sha256(receipt)
    result.update(remote_origin={'root':str(remote_root),'result_sha256':sha256(original)},
                  scheduler=scheduler,charged_core_hours=scheduler['charged_core_hours'],cache_hit=False)
    result['materialized_transfer_links']=materialized_links
    atomic_json(folder/'result.json',result)
    (folder/'result.sha256').write_text(sha256(folder/'result.json')+'\n')
    return result


def final_access_grant(evaluator,case,spec):
    """The controller's authorisation of a final-test case, carried in the frozen request.

    The worker on the compute node has only the one-case request, not the verification freeze
    record; it runs a final-test case only when the controller, which checked the record, says so."""
    if case.get('role') not in ('final','locked_final','test'):return None
    if not evaluator.final_access(spec):
        raise PermissionError('Discovery cannot evaluate final-test cases')
    freeze=evaluator.config['verification']['freeze']
    return {'campaign':evaluator.config['campaign'],'freeze':freeze['path'],'freeze_sha256':freeze['sha256']}


def build_request(evaluator,spec,cid,stage,*,remote_root,queue='hx',dispatcher_sha256=None):
    """The frozen PBS request and job script of one case evaluation, exactly as submitted.

    Shared by the live scheduler and the dry run so that a manifest produced offline is
    byte-for-byte what the controller would freeze for HX1."""
    case=evaluator.cases[cid]
    protocol=evaluator.effective_protocol(case,stage)
    ranks=min(evaluator.ranks,int(protocol.get('nprocs',evaluator.ranks)))
    if ranks<1:raise ValueError('Case MPI rank count must be positive')
    pairing,identity,key=evaluator.case_identity(spec,case,protocol,ranks)
    duration=float(protocol.get('job_wall_timeout_s',float(protocol['timeout_s'])+4200))+300
    duration=int(math.ceil(duration/60)*60)
    remote=Path(remote_root)
    remote_task=remote/'deployment/tasks'/key
    request={'experiment_hash':key,'identity':identity,'source_hash':evaluator.code,
             'manifest_metadata':{k:v for k,v in evaluator.manifest.items() if k!='cases'},
             'case':case,'protocol':protocol,'ranks':ranks,'contract':evaluator.contract,
             'spec':identity['spec'],'stage':stage,
             'dispatcher_sha256':dispatcher_sha256 or sha256(evaluator.root/'scripts/pbs_dispatch.py')}
    grant=final_access_grant(evaluator,case,spec)
    if grant:request['final_access']=grant
    script=job_script(remote,remote_task/'request.json',ranks=ranks,wall_s=duration,memory_gb=max(8,4*ranks),
                      queue=queue,name='f2-'+key[:12])
    return {'request':request,'script':script,'key':key,'identity':identity,'pairing':pairing,'ranks':ranks,
            'reservation_core_hours':duration*ranks/3600,'remote_task':str(remote_task)}


class PBSEvaluator(Evaluator):
    backend_kind='pbs'

    def __init__(self,root,manifest,contract,config):
        self.transport=SSHTransport(config['backend'])
        descriptor=within(Path(root),Path(root)/config['backend']['runtime_descriptor'])
        self.deployment=json.loads(descriptor.read_text())
        if self.deployment['root']!=str(self.transport.remote):
            raise ValueError('Remote runtime descriptor names a different deployment')
        if self.deployment['physical_source']!=source_identity(Path(root),physical_only=True):
            raise ValueError('HX1 deployment source is stale; restage and verify it before launch')
        super().__init__(root,manifest,contract,config)

    def _runtime_context(self):
        d=self.deployment
        library=within(self.root,self.root/d['library']['local_copy'])
        solver=within(self.root,self.root/d['solver']['local_copy'])
        if sha256(library)!=d['library']['sha256'] or sha256(solver)!=d['solver']['sha256']:
            raise RuntimeError('Retained native executable or library changed')
        return dict(d['environment']),library,solver,d['runtime_identity']

    def run_case(self,spec,cid,stage='development',*,protocol_override=None):
        if cid not in self.cases:raise ValueError('Unknown case: '+cid)
        case=self.cases[cid]
        if case['role'] in ('final','locked_final','test') and not self.final_access(spec):
            raise PermissionError('Discovery cannot evaluate final-test cases')
        protocol=self.effective_protocol(case,stage)
        if protocol_override:
            if spec is not None:raise ValueError('Numerical protocol selection must use SST alone')
            protocol.update(protocol_override)
        ranks=min(self.ranks,int(protocol.get('nprocs',self.ranks)))
        if ranks<1:raise ValueError('Case MPI rank count must be positive')
        pairing,identity,key=self.case_identity(spec,case,protocol,ranks)
        folder=within(self.root,self.experiments/key);folder.mkdir(parents=True,exist_ok=True)
        with file_lock(folder/'.lock'):
            path=folder/'result.json'
            if path.exists():
                if (folder/'result.sha256').read_text().strip()!=sha256(path):
                    raise RuntimeError('Cached imported result checksum failed')
                result=json.loads(path.read_text())
                if not self._cached_valid(result,identity):raise RuntimeError('Cached HX1 evidence changed')
                newly_charged=0.
                row=self.ledger.snapshot()['jobs'].get('pbs:'+key)
                if row and row['status']=='running':
                    newly_charged=result['scheduler']['charged_core_hours']
                    self.ledger.settle('pbs:'+key,newly_charged,result['status'],result['scheduler'])
                return {**result,'cache_hit':True,'charged_core_hours':newly_charged}
            return self._scheduled_case(spec,cid,stage,case,protocol,ranks,identity,key,folder)

    def _scheduled_case(self,spec,cid,stage,case,protocol,ranks,identity,key,folder):
        # The allocation includes interpreter/bootstrap time beyond the case timeout.
        duration=float(protocol.get('job_wall_timeout_s',float(protocol['timeout_s'])+4200))+300
        duration=int(math.ceil(duration/60)*60)
        reservation=duration*ranks/3600
        job_key='pbs:'+key
        task=self.campaign/'pbs'/key;task.mkdir(parents=True,exist_ok=True)
        remote_task=self.transport.remote/'deployment/tasks'/key
        request={'experiment_hash':key,'identity':identity,'source_hash':self.code,
                 'manifest_metadata':{k:v for k,v in self.manifest.items() if k!='cases'},
                 'case':case,'protocol':protocol,'ranks':ranks,'contract':self.contract,
                 'spec':identity['spec'],'stage':stage,
                 'dispatcher_sha256':sha256(self.root/'scripts/pbs_dispatch.py')}
        grant=final_access_grant(self,case,spec)
        if grant:request['final_access']=grant
        script=job_script(self.transport.remote,remote_task/'request.json',ranks=ranks,
                          wall_s=duration,memory_gb=max(8,4*ranks),
                          queue=self.config['backend'].get('queue','hx'),name='f2-'+key[:12])
        (task/'job.pbs').write_text(script)
        request['job_script_sha256']=sha256(task/'job.pbs')
        request_path=task/'request.json'
        if request_path.exists() and json.loads(request_path.read_text())!=request:
            raise RuntimeError('Frozen PBS request changed')
        atomic_json(request_path,request)
        resources=self.ledger.snapshot()
        if job_key not in resources['jobs']:
            self.ledger.reserve(job_key,reservation,{'case':cid,'experiment':key,'backend':'pbs'},
                                protected_reserve=float(self.config['budget'].get('confirmation_reserve_core_hours',0)))
        self.transport.call(['mkdir','-p',remote_task])
        self.transport.sync(str(task/'job.pbs'),remote_task/'job.pbs')
        self.transport.sync(str(request_path),remote_task/'request.json')
        submission=self.transport.dispatch('submit',remote_task/'request.json')
        atomic_json(task/'submission.json',submission)
        job_id=submission['job_id']
        print(f'HX1 {cid}: PBS {job_id}, {ranks} allocated cores',flush=True)
        last=None
        while True:
            snapshot=self.transport.dispatch('poll',job_id)
            atomic_json(task/'pbs_status.json',snapshot)
            state=(snapshot.get('job') or {}).get('job_state','unavailable')
            if state!=last:
                print(f'HX1 {cid}: {state}',flush=True);last=state
            if state in ('F','C','unavailable'):break
            time.sleep(float(self.config['backend'].get('poll_seconds',15)))
        # A finished job may have been purged from PBS history. Its durable worker
        # receipt is still mandatory; missing accounting charges the reservation.
        self.transport.sync(str(remote_task)+'/',str(task)+'/',receive=True)
        charged,basis=allocated_cost(snapshot,reservation,ranks)
        receipt_path=task/'receipt.json'
        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        scheduler={'job_id':job_id,'snapshot':snapshot,'receipt':receipt,
                   'charged_core_hours':charged,'charge_basis':basis,'reserved_core_hours':reservation}
        exit_status=(snapshot.get('job') or {}).get('Exit_status')
        if (not receipt or receipt.get('status')!='complete' or receipt.get('pbs_job_id')!=job_id
                or (exit_status is not None and int(exit_status)!=0)):
            self.ledger.settle(job_key,charged,'failed',scheduler)
            atomic_json(task/'failure.json',scheduler)
            raise RuntimeError(f'HX1 worker did not complete: {task}; inspect worker.log and PBS status')
        if receipt['request_sha256']!=sha256(request_path):
            raise RuntimeError('HX1 receipt belongs to a different request')
        incoming=self.root/'runtime/hx1/incoming'/key
        incoming.mkdir(parents=True,exist_ok=True)
        # Final-state-remote runs: mesh, fields and decomposed data stay on HX1 (their hashes
        # are in the result); everything the import verifies is transferred.
        slim=protocol.get('field_retention','all')=='final_state_remote'
        options={'exclude':('case/[0-9]*/','case/processor*/','case/constant/polyMesh')} if slim else {}
        self.transport.sync(str(self.transport.remote/'runs/experiments'/key)+'/',str(incoming)+'/',receive=True,**options)
        result=import_experiment(incoming,folder,self.transport.remote,identity,scheduler)
        self.ledger.settle(job_key,charged,result['status'],scheduler)
        return result


# The existing assessment adds independent SST endpoint sensitivity; preserve
# that behavior for the remote physical executor through normal method dispatch.
from .cli import DevelopmentEvaluator


class PBSDevelopmentEvaluator(DevelopmentEvaluator,PBSEvaluator):
    pass


def preflight(cfg,manifest,contract):
    from .catalog import validate_manifest
    from .core import verify_originals
    from .proposer import ClaudeProposer
    catalogue=validate_manifest(manifest,ROOT)
    evaluator=PBSDevelopmentEvaluator(ROOT,manifest,contract,cfg)
    descriptor=evaluator.deployment
    evidence=within(ROOT,ROOT/cfg['backend']['qualification_copy'])
    for relative,digest in descriptor['qualification_files'].items():
        path=within(evidence,evidence/relative)
        if not path.is_file() or sha256(path)!=digest:
            raise RuntimeError('Transferred native qualification evidence changed: '+relative)
    index=json.loads((evidence/'reports/runtime_qualification_index.json').read_text())
    index_matches=(index['minimum_code_qualification_passed'] and all(index['checks'].values())
                   and index['library_sha256']==evaluator.library_sha)
    for relative,digest in index['qualification_code_sha256'].items():
        index_matches=index_matches and sha256(ROOT/relative)==digest
    for item in index['reports'].values():
        index_matches=index_matches and sha256(evidence/item['path'])==item['sha256']
    for relative,digest in index['retained_settled_field_sha256'].items():
        index_matches=index_matches and sha256(evidence/relative)==digest
    live=json.loads(evaluator.transport.call([
        'env','PYTHONPATH='+str(evaluator.transport.remote/'src'),
        evaluator.transport.remote/'.venv/bin/python','-m','forge.pbs_worker','verify'],
        timeout=60))
    live_matches=all(live[k]==descriptor[k] for k in
                     ('root','physical_source','runtime_identity','environment','qualification_files'))
    live_matches=live_matches and all(live[k]['sha256']==descriptor[k]['sha256'] for k in ('library','solver'))
    originals=verify_originals(ROOT/'provenance/v1_before.json')
    integration_path=ROOT/'reports/hx1_transport_validation.json'
    integration=json.loads(integration_path.read_text()) if integration_path.exists() else {}
    integrated=(integration.get('passed') is True and integration.get('runtime_identity')==descriptor['runtime_identity']
                and integration.get('driver_sha256')==sha256(ROOT/'scripts/verify_hx1_transport.py'))
    if integrated:
        from .cfd_capability import _verify_experiment
        for key in integration['experiments']:
            record=_verify_experiment(ROOT,key)
            integrated=integrated and record['identity']['source']==evaluator.code
    provider=ClaudeProposer(ROOT/'reports/preflight_hx1',**cfg['proposer'])
    report={'backend':'pbs','ready_for_development_launch':bool(catalogue['ready'] and index_matches and live_matches and originals['unchanged'] and integrated),
            'catalogue':catalogue,'native_runtime_qualification':bool(index_matches),
            'live_deployment_matches':bool(live_matches),'transport_integration':bool(integrated),
            'library_sha256':evaluator.library_sha,'provider':provider.identity,'originals':originals,
            'scientific_release_ready':False,'core_hour_cap':cfg['budget']['core_hours'],
            'launch_scope':'HX1 qualification and discovery; required baselines and capability/protection gates still apply.'}
    atomic_json(ROOT/'reports/preflight_hx1.json',report)
    return report
