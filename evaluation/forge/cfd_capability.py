"""Prospective coupled-response diagnostics before a large symbolic search.

Witnesses may be different equations. Passing establishes a resolved admissible
mechanism response and sentinel preservation, not a universal successful model.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tedp.spec import CandidateSpec, Term, to_json, spec_hash
from .core import atomic_json, fingerprint, within, utc_now, sha256


def _verify_experiment(root: Path, identity: str) -> dict:
    folder = within(root, root/'runs/experiments'/identity)
    path = folder/'result.json'
    if not path.exists() or (folder/'result.sha256').read_text().strip() != sha256(path):
        raise ValueError('Qualification experiment missing or checksum changed')
    record = json.loads(path.read_text())
    if not record.get('artifacts'):
        raise ValueError('Qualification experiment has no retained evidence')
    for name, digest in record['artifacts'].items():
        p = within(root, Path(name))
        if not p.is_file() or sha256(p) != digest:
            raise ValueError('Qualification field or log evidence changed')
    return record


def normalized_axial_changes(first, second):
    """Pressure in a periodic incompressible duct is defined up to a constant."""
    import numpy as np
    a,b = np.array(first,dtype=float,copy=True),np.array(second,dtype=float,copy=True)
    offset = float(np.mean(b[:,3])-np.mean(a[:,3]))
    a[:,3] -= np.mean(a[:,3]); b[:,3] -= np.mean(b[:,3])
    velocity_scale = max(float(np.max(abs(b[:,0]))), 1e-12)
    scales = np.maximum(np.max(abs(b),axis=0),[velocity_scale]*3+[velocity_scale**2,1e-6,1e-6,1e-6])
    return np.max(abs(a-b),axis=0)/scales, offset


def axial_comparison(records: dict, *, tolerance: float = 1e-5) -> dict:
    """Compare complete cross-sectional fields, not just reference-error scores.

    Axial invariance is a fully developed RANS assumption, not a cross-section
    refinement study or evidence for developing/streamwise-unsteady duct flow.
    """
    import contextlib
    import io
    import numpy as np
    from fluidfoam import readmesh, readscalar, readvector
    from tedp.holdout_cases.duct_scoring import _column_average
    comparisons, reasons, offsets, geometry = {}, [], {}, {}
    for kind in ('baselines', 'case_results'):
        pair = list(records[kind].values())
        if len(pair) != 2 or not all(r.get('qualification', {}).get('qualified') for r in pair):
            reasons.append(f'{kind}: both axial meshes require qualified solutions')
            continue
        columns, axial_counts = [], []
        for record in pair:
            w = record['workdir']
            t = str(record['identity']['protocol']['end_time'])
            with contextlib.redirect_stdout(io.StringIO()):
                x, y, z = readmesh(w, verbose=False)
                values = np.column_stack([readvector(w,t,'U',verbose=False).T,
                    *[readscalar(w,t,n,verbose=False) for n in ('p','k','omega','nut')]])
            columns.append(_column_average(y,z,values))
            axial_counts.append(len(np.unique(np.round(x,9))))
        geometry[kind] = axial_counts
        if sorted(axial_counts) != [1,4]:
            reasons.append(f'{kind}: actual meshes must contain 1 and 4 axial cell columns')
        if columns[0][0].shape != columns[1][0].shape or not np.allclose(columns[0][0],columns[1][0],atol=1e-9,rtol=0):
            reasons.append(f'{kind}: cross-section coordinates changed')
            continue
        # Inactive components use physical velocity/pressure scales, never a
        # division by a roundoff-sized SST secondary velocity.
        differences, offsets[kind] = normalized_axial_changes(columns[0][1],columns[1][1])
        comparisons[kind] = dict(zip(('Ux','Uy','Uz','p','k','omega','nut'),map(float,differences)))
        if not np.isfinite(differences).all() or float(np.max(differences)) > tolerance:
            reasons.append(f'{kind}: axial field sensitivity exceeds the declared tolerance')
    return {'passed': not reasons and len(comparisons)==2, 'reasons': reasons,
            'relative_field_tolerance': tolerance, 'normalized_max_differences': comparisons,
            'removed_pressure_gauge_offsets': offsets, 'actual_axial_cell_counts': geometry,
            'normalization': 'Per-field maximum; velocity components share peak streamwise speed, '
                             'pressure uses its square after removing the arbitrary constant gauge; '
                             'positive turbulence floors are 1e-6.',
            'scope': '1 versus 4 axial cells; complete 96x96 cross-section and all velocity components. '
                     'No full 75-cell equivalence or cross-section grid convergence is claimed.'}


def run_axial_pilot(evaluator, output: Path) -> dict:
    """Qualify each reduced duct before its use in development discovery."""
    from copy import deepcopy
    output = within(evaluator.root, output)
    ducts = [c for c in evaluator.manifest['cases'] if c['adapter']=='duct'
             and c['adapter_config'].get('streamwise_cells') == 1]
    definition = {'evaluator': evaluator.protocol_hash, 'ducts': ducts,
                  'probe': json.loads(to_json(probes()['duct'][0])), 'tolerance': 1e-5,
                  'checker_sha256': sha256(Path(__file__))}
    identity = fingerprint(definition)
    if output.exists():
        old = json.loads(output.read_text())
        if old.get('identity') != identity:
            raise ValueError('Axial pilot definition changed; use a new campaign')
        if old.get('status') == 'complete':
            for key in old.get('experiments', []):
                _verify_experiment(evaluator.root, key)
            return old
    report = {'identity': identity, 'definition': definition, 'status': 'running',
              'passed': False, 'cases': {}, 'experiments': []}
    atomic_json(output, report)
    for case in ducts:
        clone = deepcopy(case)
        clone['id'] = case['id']+'__axial4'
        clone['adapter_config']['streamwise_cells'] = 4
        clone['mesh_id'] = 'fully_developed_96x96x4'
        clone['protocol'] = evaluator.effective_protocol(case, 'development')
        clone['protocol']['mesh_id'] = 'fully_developed_96x96x4'
        evaluator.cases[clone['id']] = clone
        ids = [case['id'], clone['id']]
        print(f"Axial qualification: {case['id']}, 1 versus 4 streamwise cells", flush=True)
        baseline = evaluator.qualify_baselines(ids, evaluator.config.get('baseline_endpoint_factors',[1,2,4]))
        if baseline['qualified']:
            result = evaluator(probes()['duct'][0], ids, 'development')
            check = axial_comparison(result)
            report['experiments'].extend(r['experiment_hash'] for group in ('baselines','case_results')
                                         for r in result[group].values() if 'experiment_hash' in r)
        else:
            check = {'passed': False, 'reasons': ['SST axial baseline qualification failed']}
            report['experiments'].extend(r['experiment_hash'] for r in baseline['cases'].values())
        report['cases'][case['id']] = check
        atomic_json(output, report)
    report.update(status='complete', passed=all(c['passed'] for c in report['cases'].values()))
    atomic_json(output, report)
    return report


def probes() -> dict[str, list[CandidateSpec]]:
    return {
        "duct": [CandidateSpec(f"equilibrium_stress_probe_{i}",
            bdelta=(Term("T2", "c0/max(1,c1*sqrt(max(I1*abs(I2),0)))"),),
            constants=(amplitude, 4.)) for i, amplitude in enumerate((.02, .005))],
        "rotation": [CandidateSpec(f"signed_rotation_source_probe_{i}",
            rsource=(Term("T1", "c0*tanh(c1*(I1+I2)/(I1+abs(I2)))/max(1,c1*I1)"),),
            constants=(amplitude, 4.)) for i, amplitude in enumerate((.02, -.02))],
    }


def resolved_response(mechanism: str, model: dict, baseline: dict) -> dict:
    """Reference-error improvement must exceed both temporal envelopes."""
    key = "secondary_velocity_rmse" if mechanism == "duct" else "wall_friction_asymmetry_error"
    reasons = []
    if not model.get("qualification", {}).get("qualified") or not baseline.get("qualification", {}).get("qualified"):
        reasons.append("Model and matched SST must both be numerically qualified")
    if model.get("pairing_hash") != baseline.get("pairing_hash"):
        reasons.append("Unmatched numerical or input definitions")
    old, new = baseline.get("errors", {}).get(key), model.get("errors", {}).get(key)
    ua, ub = model.get("uncertainty", {}).get(key), baseline.get("uncertainty", {}).get(key)
    if any(v is None for v in (old,new,ua,ub)):
        reasons.append("Missing mechanism error or temporal sensitivity")
        lower = None
    else:
        lower = old-new-ua-ub
        if lower <= 1e-10:
            reasons.append("No resolved improvement in the required mechanism")
    if mechanism == "duct":
        obs = model.get("observables", {})
        if obs.get("secondary_correlation", -1) <= 0:
            reasons.append("Secondary-flow direction does not agree with DNS")
        if not obs.get("secondary_orientation_resolved", False):
            reasons.append("Secondary-flow magnitude is unresolved")
    return {"passed": not reasons, "reasons": reasons, "metric": key,
            "lower_absolute_improvement": lower}


def run_cfd_pilot(evaluator, gate, output: Path) -> dict:
    output = within(evaluator.root, output)
    options = evaluator.config.get("capability", {})
    targets = options.get("targets", {"duct": "squareDuct_Re_1100", "rotation": "rotchan_ro10"})
    sentinels = options.get("protection_ids", ["tmr_bump_177x81", "naca0012_a000_225x65", "naca0012_a010_225x65"])
    declared = {"evaluator": evaluator.protocol_hash, "numerics": evaluator.qualified_numerics,
                "checker_sha256": sha256(Path(__file__)),
                "targets": targets, "sentinels": sentinels, "gate": gate.policy,
                "probes": {k:[json.loads(to_json(s)) for s in v] for k,v in probes().items()}}
    identity = fingerprint(declared)
    if output.exists():
        old = json.loads(output.read_text())
        if old.get("identity") != identity:
            raise ValueError("Capability protocol changed; preserve the previous report under a different campaign")
        if old.get("status") == "complete":
            for trial in old.get('trials', []):
                for key in [trial.get('experiment'), *trial.get('protection_experiments', {}).values()]:
                    if key:
                        _verify_experiment(evaluator.root, key)
            return old
    report = {"identity": identity, "definition": declared, "status": "running", "passed": False,
              "started_utc": utc_now(), "witnesses": {}, "trials": [],
              "scope": "Separate mechanism witnesses plus attached-flow protection; not one universal closure or a release"}
    atomic_json(output, report)
    # Each mechanism and its protection suite are independent CFD requests.
    # Try one probe per unresolved mechanism per round; the evaluator retains
    # the shared global CFD limit and physical experiment locks.
    choices = {mechanism: iter(probes()[mechanism]) for mechanism in targets}
    workers = 2 * len(targets) if evaluator.config.get('backend', {}).get('kind') == 'pbs' else 1
    while choices:
        batch = []
        for mechanism in list(choices):
            spec = next(choices[mechanism], None)
            if spec is None:
                del choices[mechanism]
                continue
            verdict = gate(spec)
            trial = {"mechanism": mechanism, "spec": json.loads(to_json(spec)),
                     "spec_hash": spec_hash(spec), "cheap_gate": verdict}
            report["trials"].append(trial)
            if verdict["passed"]:
                batch.append((mechanism, targets[mechanism], spec, trial))
        atomic_json(output, report)
        with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix='forge-capability') as pool:
            pending = {}
            for mechanism, cid, spec, trial in batch:
                print(f"CFD capability: {mechanism}, {spec.name}; target and protection concurrent", flush=True)
                pending[pool.submit(evaluator, spec, [cid], "development")] = (trial, cid, 'target')
                pending[pool.submit(evaluator, spec, list(sentinels), "protection")] = (trial, cid, 'protection')
            try:
                for future in as_completed(pending):
                    trial, cid, kind = pending[future]
                    result = future.result()
                    if kind == 'target':
                        trial["response"] = resolved_response(trial['mechanism'], result["case_results"][cid], result["baselines"][cid])
                        trial["experiment"] = result["case_results"][cid].get("experiment_hash")
                        trial["target_assessment"] = result["assessment"]
                    else:
                        trial["protection"] = result["assessment"]
                        trial["protection_experiments"] = {k:v.get("experiment_hash") for k,v in result["case_results"].items()}
                    atomic_json(output, report)
            finally:
                for future in pending:
                    future.cancel()
        for mechanism, cid, spec, trial in batch:
            if (trial['response']['passed'] and trial['target_assessment']['feasible']
                    and trial['protection']['feasible']):
                report['witnesses'][mechanism] = trial['spec_hash']
                del choices[mechanism]
        atomic_json(output, report)
    report.update(status="complete", passed=set(report["witnesses"]) == set(targets), finished_utc=utc_now())
    report["decision"] = ("Admissible coupled responses demonstrated; proceed with this algebraic hypothesis space"
        if report["passed"] else "Do not scale discovery: inspect missing response/protection witnesses and revise the capability experiment or representation")
    atomic_json(output, report)
    return report
