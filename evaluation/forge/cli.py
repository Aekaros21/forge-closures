"""Single isolated entrypoint: preflight, capability diagnostics and discovery."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import threading
from contextlib import contextmanager

from .core import ROOT, BudgetExhausted, atomic_json, file_lock, fingerprint, sha256, utc_now, verify_originals, within
from .evaluator import Evaluator as PhysicalEvaluator


def execution_workers(backend):
    kind = backend.get('kind')
    workers = backend.get('max_workers', 1)
    maximum = 256 if kind == 'pbs' else 1
    if kind not in ('local', 'pbs') or type(workers) is not int or not 1 <= workers <= maximum:
        raise ValueError('Use 1 local worker or 1–256 PBS case workers')
    return workers


def ordered_case_map(function, items, workers, *, completed=None):
    """Bound outstanding cases and return results in the declared input order.

    On failure, stop admitting new cases and drain the current workers so their
    evidence/accounting can finish. A killed controller recovers PBS jobs using
    the existing durable submission records and per-experiment locks.
    """
    items = list(items)
    if not items:
        return []
    results = [None] * len(items)
    next_index = 0
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='forge-case')
    pending = {}
    try:
        while next_index < min(workers, len(items)):
            pending[pool.submit(function, items[next_index])] = next_index
            next_index += 1
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Resolve all newly finished tasks before submitting replacements.
            for future in sorted(done, key=lambda f: pending[f]):
                index = pending.pop(future)
                results[index] = future.result()
                if completed is not None:
                    completed(items[index], results[index])
            while len(pending) < workers and next_index < len(items):
                pending[pool.submit(function, items[next_index])] = next_index
                next_index += 1
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
    return results


class CoreSlots:
    """Account actual requested ranks across concurrent case submissions."""
    def __init__(self, capacity):
        self.capacity, self.used = capacity, 0
        self.condition = threading.Condition()

    @contextmanager
    def acquire(self, ranks):
        if not 1 <= ranks <= self.capacity:
            raise ValueError('Case ranks exceed the active core allocation')
        with self.condition:
            self.condition.wait_for(lambda: self.used + ranks <= self.capacity)
            self.used += ranks
        try:
            yield
        finally:
            with self.condition:
                self.used -= ranks
                self.condition.notify_all()


class DevelopmentEvaluator(PhysicalEvaluator):
    """Add assessment-only endpoint sensitivity without changing CFD identities."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._case_slots = threading.BoundedSemaphore(self._case_workers())
        self._core_slots = CoreSlots(self.config['backend'].get('active_core_limit', 1024))
        self.protocol_hash = fingerprint({'physical_protocol':self.protocol_hash,
                                          'development_assessment_sha256':sha256(Path(__file__))})

    def run_case(self, *args, **kwargs):
        # Shared by every fit and qualification clone using this evaluator.
        # Nested per-fit case pools cannot multiply the campaign's PBS limit.
        with self._case_slots:
            cid = args[1] if len(args) > 1 else kwargs['cid']
            stage = args[2] if len(args) > 2 else kwargs.get('stage', 'development')
            protocol = self.effective_protocol(self.cases[cid], stage)
            protocol.update(kwargs.get('protocol_override') or {})
            ranks = min(self.ranks, int(protocol.get('nprocs', self.ranks)))
            with self._core_slots.acquire(ranks):
                return super().run_case(*args, **kwargs)

    def _case_workers(self):
        if self.backend_kind != 'pbs':
            return 1
        return execution_workers(self.config['backend'])

    def qualify_baselines(self, case_ids, factors=(1, 2, 4)):
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError('Qualification needs a nonempty unique case list')
        workers = self._case_workers()
        if workers == 1 or len(case_ids) == 1:
            return super().qualify_baselines(case_ids, factors)
        initial = copy.deepcopy(self.qualified_numerics)
        factors = tuple(factors)
        def qualify(cid):
            # Each case keeps its sequential SST endpoint ladder. Worker-local
            # certificates prevent concurrent writes to the shared numerics file.
            worker = copy.copy(self)
            worker.qualified_numerics = copy.deepcopy(initial)
            worker.numerics_path = self.campaign/'qualification_shards'/(fingerprint(cid)+'.json')
            report = super(DevelopmentEvaluator, worker).qualify_baselines([cid], factors)
            return report
        def retain(cid, report):
            # Only this coordinator writes the merged frozen certificates.
            if cid in report['numerics']:
                self.qualified_numerics[cid] = report['numerics'][cid]
                atomic_json(self.numerics_path, self.qualified_numerics)
        print(f'SST qualification: up to {workers} concurrent cases', flush=True)
        reports = ordered_case_map(qualify, case_ids, workers, completed=retain)
        return {'qualified': all(r['qualified'] for r in reports),
                'admitted': all(r['admitted'] for r in reports),
                'cases': {cid: r['cases'][cid] for cid, r in zip(case_ids, reports)},
                'numerics': self.qualified_numerics}

    def __call__(self, spec, case_ids, stage):
        from .admission import is_airfoil_control
        from .metrics import assess
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError('Evaluation needs a nonempty unique case list')
        workers = self._case_workers()
        if workers == 1 or len(case_ids) == 1:
            result = super().__call__(spec, case_ids, stage)
        else:
            # Physical execution already has per-experiment locks, independent
            # task directories, and a locked ledger. No numerics are selected here.
            physical = super().__call__
            reports = ordered_case_map(lambda cid: physical(spec, [cid], stage), case_ids, workers)
            status = ('failed' if any(r['status'] == 'failed' for r in reports) else
                      'complete' if all(r['status'] == 'complete' for r in reports) else 'unqualified')
            result = {'status': status,
                      'case_results': {cid: r['case_results'][cid] for cid, r in zip(case_ids, reports)},
                      'baselines': {cid: r['baselines'][cid] for cid, r in zip(case_ids, reports)},
                      'cost': {'core_hours': sum(r['cost']['core_hours'] for r in reports)},
                      'case_evaluations': sum(r['case_evaluations'] for r in reports)}
        for cid in case_ids:
            if not is_airfoil_control(self.cases[cid]):
                continue
            certificate = self.qualified_numerics.get(cid,{}).get('control_baseline_agreement') or {}
            if not certificate.get('passed'):
                continue
            stock = result['baselines'][cid]
            uncertainty = dict(stock.get('uncertainty',{}))
            evidence = {}
            for observable, metrics in {
                'drag_coefficient':('drag_coefficient','drag_abs_error'),
                'lift_coefficient':('lift_coefficient','lift_abs_error'),
                'wall_cf':('wall_cf',), 'wall_cp':('wall_cp',)}.items():
                change = certificate['changes'][observable]['absolute_difference']
                if not isinstance(change,(int,float)) or not math.isfinite(change) or change < 0:
                    raise RuntimeError('Invalid frozen baseline endpoint sensitivity')
                for metric in metrics:
                    previous = uncertainty.get(metric)
                    # A certificate cannot fill in a missing within-run history.
                    if isinstance(previous,(int,float)) and math.isfinite(previous) and previous >= 0:
                        uncertainty[metric] = max(previous,change)
                        evidence[metric] = {'within_run':previous,'between_endpoints':change,
                                            'used_bound':uncertainty[metric]}
            result['baselines'][cid] = {**stock,'uncertainty':uncertainty,
                'baseline_endpoint_sensitivity':{'experiments':certificate['experiments'],'bounds':evidence}}
        result['assessment'] = assess([self.cases[c] for c in case_ids],result['case_results'],
                                      result['baselines'],self.contract)
        return result


def configuration(path: Path) -> tuple[dict, dict, dict]:
    from .catalog import load_manifest
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if cfg.get("schema_version") not in (2, 3) or cfg.get("purpose") != "development":
        raise ValueError("An explicit development campaign configuration (schema 2 or 3) is required")
    v3 = cfg.get("schema_version") == 3
    if v3 != (cfg.get('search', {}).get('mode') == 'v3'):
        raise ValueError('schema_version 3 and search.mode v3 must be declared together')
    if not cfg.get("campaign") or any(x in cfg["campaign"] for x in ("/", "\\", "..")):
        raise ValueError("Unsafe campaign namespace")
    execution_workers(cfg['backend'])
    wall = cfg.get('wall_clock')
    if wall:
        from datetime import datetime
        start = datetime.fromisoformat(wall['started_utc']).timestamp()
        deadline = datetime.fromisoformat(wall['deadline_utc']).timestamp()
        reserve = wall['confirmation_reserve_hours'] * 3600
        maximum = wall.get('maximum_hours', 48)
        if (not isinstance(maximum, (int, float)) or not 0 < maximum <= 168 or
                deadline-start != maximum*3600 or deadline != wall['deadline_unix'] or
                not 0 < reserve < deadline-start or
                cfg['search'].get('search_deadline_unix') != deadline-reserve):
            raise ValueError('Wall-clock/search deadlines must match the declared maximum_hours '
                             '(up to 168 hours) and preserve the confirmation reserve')
        core_limit = cfg['backend'].get('active_core_limit', 1024)
        if type(core_limit) is not int or not 1 <= core_limit <= 8192:
            raise ValueError('HX1 active core allocation must be 1–8192 cores')
    manifest = load_manifest(within(ROOT, ROOT/cfg["manifest"]))
    contract = json.loads(within(ROOT, ROOT/cfg["contract"]).read_text())
    ids = {c["id"] for c in manifest["cases"] if c["role"] != "final"}
    for name in ("calibration_ids", "protection_ids"):
        if not set(cfg["search"][name]) <= ids:
            raise ValueError(f"{name} contains undeclared or final-test cases")
    roles = {c['id']: c['role'] for c in manifest['cases']}
    for key, role in (('calibration_ids','calibration'),('protection_ids','protection')):
        if any(roles[cid] != role for cid in cfg['search'][key]):
            raise ValueError(f'{key} violates frozen catalogue roles')
    from .search import SearchConfig
    options = dict(cfg['search'])
    if options.get('seeds_file') is not None:
        configured_seeds(options['seeds_file'])
    options.pop('seeds_file', None)
    for name in ('calibration_ids','protection_ids','tuning_protection_ids'):
        if name in options: options[name] = tuple(options[name])
    if v3:
        options.update(v3_search_options(cfg, manifest))
    SearchConfig(work_root=ROOT/'results'/cfg['campaign']/'search', protocol_hash='preflight-validation',
                 development_ids=tuple(c['id'] for c in manifest['cases'] if c['required'] and c['role']!='final'),
                 **options).validate()
    if cfg['backend']['kind'] != 'pbs' and max(cfg['search'].get('candidate_workers', 1), cfg['search'].get('trial_workers', 1)) != 1:
        raise ValueError('Concurrent closure fits require the PBS backend')
    return cfg, manifest, contract


def v3_search_options(cfg, manifest) -> dict:
    """Resolve the V3 configuration: evidence dossier, archived benchmark, interface and selection roles.

    Every check here fails explicitly: a dossier whose checksum changed, an undeclared or wrongly
    scoped selection case, a benchmark that is not the archived record, or a feature advertised
    before it exists in the grammar."""
    from tedp import expr
    from . import v3 as _v3
    search = cfg['search']
    roles = {c['id']: c['role'] for c in manifest['cases']}
    development = {c['id'] for c in manifest['cases'] if c['required'] and c['role'] != 'final'}
    for name in ('selection_constraint_ids', 'selection_accuracy_ids'):
        ids = search.get(name) or []
        if any(cid not in development or roles[cid] not in ('calibration', 'adaptive_validation') for cid in ids):
            raise ValueError(f'{name} must name required development cases (calibration or adaptive_validation), never protection or final-test cases')
    evidence = cfg.get('evidence') or {}
    dossier = within(ROOT, ROOT / evidence.get('dossier_path', ''))
    if not dossier.is_file():
        raise ValueError('V3 evidence dossier is missing: ' + str(dossier))
    if sha256(dossier) != evidence.get('dossier_sha256'):
        raise ValueError('V3 evidence dossier does not match the frozen dossier_sha256')
    dossier_data = json.loads(dossier.read_text())
    benchmark_path = within(ROOT, ROOT / evidence['benchmark_path'])
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark.get('dossier_version') != dossier_data.get('version'):
        raise ValueError('archived benchmark was built from a different dossier version')
    record = next((c for c in dossier_data['candidates'] if c['spec_hash'] == benchmark['spec_hash']), None)
    if record is None or record['id'] != benchmark['dossier_id'] or record['status'] != 'valid':
        raise ValueError('archived benchmark is not the valid incumbent of the dossier')
    allowed = tuple(search.get('allowed_variables') or expr.LEGACY_VARIABLES)
    unknown = set(allowed) - set(expr.CANDIDATE_VARIABLES)
    if unknown:
        raise ValueError('allowed_variables not implemented in the grammar: ' + ', '.join(sorted(unknown)))
    return {
        'mode': 'v3',
        'selection_constraint_ids': tuple(search.get('selection_constraint_ids') or ()),
        'selection_accuracy_ids': tuple(search.get('selection_accuracy_ids') or ()),
        'allowed_variables': allowed,
        'evidence': {'dossier_path': str(Path(evidence['dossier_path'])), 'dossier_sha256': evidence['dossier_sha256'],
                     'version': dossier_data['version'], 'trial_detail': evidence.get('trial_detail', 'summary'),
                     'benchmark_path': evidence['benchmark_path']},
        'benchmark': {k: benchmark[k] for k in ('source_campaign', 'checkpoint_sha256', 'checkpoint_generation', 'evaluator_protocol_fingerprint',
                                                 'search_protocol_hash', 'dossier_id', 'dossier_version', 'spec', 'spec_hash', 'struct_hash',
                                                 'name', 'objective', 'feasible', 'family_errors', 'case_errors', 'verdict_scope')},
        'interface': _v3.interface_description(allowed),
    }


def campaign_status(cfg):
    from .core import Ledger
    campaign = ROOT/'results'/cfg['campaign']
    choices = [campaign/'search/checkpoint.json', campaign/'summary.json', campaign/'prepared.json', campaign/'stage.json']
    report = next((json.loads(p.read_text()) for p in choices if p.exists()),
                  {'status':'not_launched','campaign':cfg['campaign']})
    result = {key:report[key] for key in ('status','generation','counts','stop_reason','campaign','phase','discovery_started') if key in report}
    if (campaign/'resources.json').exists():
        resources = Ledger(campaign/'resources.json',cfg['budget']['core_hours']).snapshot()
        result['resources'] = {key:resources[key] for key in ('spent','remaining')}
        result['active_or_unresolved_jobs'] = [r['identity'] for r in resources['jobs'].values() if r['status']=='running']
    if report.get('pid') and report.get('status') == 'qualifying':
        try: os.kill(report['pid'],0)
        except ProcessLookupError: result['status']='qualification_stopped'
    return result


def budgeted_evaluation(evaluator, search_config, confirmation_reserve=0.):
    """Reconcile physical spend even if interruption loses a callback return."""
    from .search import BudgetStop
    origin_path = search_config.work_root.parent/'search_resource_origin.json'
    if not origin_path.exists():
        atomic_json(origin_path, {'spent_before_search':evaluator.ledger.snapshot()['spent']})
    origin = json.loads(origin_path.read_text())['spent_before_search']
    checkpoint = search_config.work_root/'checkpoint.json'
    def evaluate(spec, requested, stage):
        resources = evaluator.ledger.snapshot()
        if resources['remaining'] is not None and resources['remaining'] <= confirmation_reserve:
            raise BudgetStop('Confirmation reserve reached')
        if search_config.max_core_hours is not None and resources['spent']-origin >= search_config.max_core_hours:
            raise BudgetStop('Authoritative discovery compute allocation reached')
        credited = json.loads(checkpoint.read_text())['counts']['core_hours'] if checkpoint.exists() else 0.
        try:
            result = evaluator(spec,list(requested),stage)
        except BudgetExhausted as exc:
            raise BudgetStop(str(exc)) from exc
        consumed = evaluator.ledger.snapshot()['spent']-origin
        if credited > consumed+1e-8:
            raise RuntimeError('Controller accounting exceeds its physical resource ledger')
        return {**result,'cost':{**result['cost'],'core_hours':max(0.,consumed-credited),
                                'scope':'New authoritative ledger charge, including any interrupted uncredited work'}}
    def concurrent(spec, requested, stage):
        # Individual callbacks report only their own new charges. The search
        # broker reconciles aggregate spend from the ledger under its state lock.
        resources = evaluator.ledger.snapshot()
        if resources['remaining'] is not None and resources['remaining'] <= confirmation_reserve:
            raise BudgetStop('Confirmation reserve reached')
        try:
            return evaluator(spec, list(requested), stage)
        except BudgetExhausted as exc:
            raise BudgetStop(str(exc)) from exc
    evaluate.concurrent = concurrent
    evaluate.authoritative_spend = lambda: evaluator.ledger.snapshot()['spent'] - origin
    return evaluate


def require_capability(future):
    report = future.result()
    if not report.get('passed'):
        raise RuntimeError('Coupled capability/protection pilot did not pass; model admission is blocked. See cfd_capability.json')
    return report


def configured_seeds(path):
    """Starting population from a frozen seeds file with specifications for every island."""
    from tedp.spec import from_json
    from .seeds import ISLANDS
    data = json.loads(within(ROOT, ROOT/path).read_text())
    if set(data) != set(ISLANDS) or not all(isinstance(group, list) and group for group in data.values()):
        raise ValueError('seeds_file must list at least one specification for every island')
    seeds = {island: [from_json(json.dumps(spec)) for spec in data[island]] for island in ISLANDS}
    for group in seeds.values():
        for spec in group:
            spec.validate(search_policy=True)
    return seeds


def require_discovery_result(result):
    """A paused proposer interrupts discovery; it never finishes the search."""
    if result.status == 'paused_llm':
        raise RuntimeError('Discovery paused by the proposer: ' + result.stop_reason +
                           '; summary and confirmation are withheld and a relaunch resumes discovery')
    return result


def capability_guarded_evaluation(evaluate, future):
    """Allow calibration while the pilot runs; full admission waits for its pass."""
    def guard(callback):
        def assessed(spec, requested, stage):
            if future.done():
                require_capability(future)
            result = callback(spec, requested, stage)
            if stage != 'calibration' or future.done():
                require_capability(future)
            return result
        return assessed
    wrapped = guard(evaluate)
    if hasattr(evaluate, 'concurrent'):
        wrapped.concurrent = guard(evaluate.concurrent)
    if hasattr(evaluate, 'authoritative_spend'):
        wrapped.authoritative_spend = evaluate.authoritative_spend
    return wrapped


def preflight(cfg, manifest, contract) -> dict:
    if cfg['backend']['kind'] == 'pbs':
        from .pbs import preflight as pbs_preflight
        return pbs_preflight(cfg, manifest, contract)
    from .catalog import validate_manifest
    from .proposer import ClaudeProposer
    from .runtime import foam_env, library_path
    catalogue = validate_manifest(manifest, ROOT)
    lib = library_path(ROOT)
    env = foam_env(ROOT)
    provider = ClaudeProposer(ROOT/"reports/preflight", **cfg["proposer"])
    original = verify_originals(ROOT/"provenance/v1_before.json")
    required = {
        "stress_exchange": ROOT/"reports/stress_exchange_regression.json",
        "compiled_basis": ROOT/"reports/compiled_basis_parity.json",
    }
    index_path = ROOT/"reports/runtime_qualification_index.json"
    index_valid = False
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index_valid = (index.get("minimum_code_qualification_passed") is True and
                       index.get("library_sha256") == sha256(lib) and
                       all(index.get("checks", {}).values()))
        for name, digest in index.get("qualification_code_sha256", {}).items():
            index_valid = index_valid and (ROOT/name).exists() and sha256(ROOT/name) == digest
        for entry in index.get("reports", {}).values():
            p = ROOT/entry["path"]
            index_valid = index_valid and p.exists() and sha256(p) == entry["sha256"]
        for name, digest in index.get("retained_settled_field_sha256", {}).items():
            p = ROOT/name
            index_valid = index_valid and p.exists() and sha256(p) == digest
    missing_checks = [name for name,path in required.items() if not path.exists()]
    failed_checks = [name for name,path in required.items()
                     if path.exists() and not json.loads(path.read_text()).get("passed", False)]
    if required["compiled_basis"].exists():
        basis = json.loads(required["compiled_basis"].read_text())
        if basis.get("basis_header_sha256") != sha256(ROOT/"src/kOmegaSSTBasis/basisTensors/integrityBasis.H"):
            failed_checks.append("compiled_basis_source_changed")
    partition_check = False
    partition_path = ROOT/'reports/decomposition_repeatability.json'
    if partition_path.exists():
        partition = json.loads(partition_path.read_text())
        native = Path(partition.get('scotch_library','/nonexistent'))
        partition_check = (partition.get('passed') is True and native.is_file() and
            partition.get('environment',{}).get('SCOTCH_PTHREAD_NUMBER') == '1' and
            sha256(native) == partition.get('scotch_library_sha256') and bool(partition.get('artifacts')))
        for name,digest in partition.get('artifacts',{}).items():
            p = within(ROOT,ROOT/name)
            partition_check = partition_check and p.is_file() and sha256(p)==digest
    checks = {"catalogue_available": catalogue["ready"], "original_tracked_files_unchanged": original["unchanged"],
              "qualification_artifacts_available": not missing_checks and not failed_checks,
              "loaded_library_and_settled_mpi_qualified": index_valid,
              "repeatable_partitioning": partition_check,
              "isolated_python": Path(sys.prefix).resolve() == ROOT/".venv"}
    report = {"created_utc": utc_now(), "ready_for_development_launch": all(checks.values()),
              "checks": checks, "catalogue": catalogue, "library": str(lib), "library_sha256": sha256(lib),
              "openfoam": env["WM_PROJECT_VERSION"], "provider": provider.identity,
              "provider_live_integration": ((ROOT/"reports/proposer_integration_current.json").exists() and
                  json.loads((ROOT/"reports/proposer_integration_current.json").read_text()).get('provider') == provider.identity),
              "missing_qualification": missing_checks, "failed_qualification": failed_checks, "originals": original,
              "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "fluidfoam", "cma", "pandas")},
              "scientific_release_ready": False,
              "launch_scope": "Development qualification and discovery; no new closure or universal superiority is claimed"}
    atomic_json(ROOT/"reports/preflight.json", report)
    return report


def _eligibility(manifest, baseline_report, contract) -> dict:
    from .metrics import metric_rules
    eligible = []
    for case in manifest["cases"]:
        record = baseline_report["cases"].get(case["id"])
        if not record or not record["qualification"]["qualified"]:
            continue
        for name, rule in metric_rules(case, contract).items():
            if not rule.get("primary", True) or rule.get("kind", "error") != "error":
                continue
            error, uncertainty = record["errors"].get(name), record["uncertainty"].get(name)
            if error is not None and uncertainty is not None and error > max(3*uncertainty,rule.get("floor",1e-6)):
                eligible.append(case["id"])
                break
    return {"eligible_cases": eligible, "selected_from": "SST only, before candidates",
            "baseline_identities": {c:r["experiment_hash"] for c,r in baseline_report["cases"].items()}}


def verify_control_baselines(evaluator):
    """Recheck both retained cold endpoints, including on a resumed launch."""
    from .admission import is_airfoil_control, baseline_agreement
    from .cfd_capability import _verify_experiment
    for case in evaluator.manifest['cases']:
        if not case.get('required') or not is_airfoil_control(case):
            continue
        selected = evaluator.qualified_numerics.get(case['id'],{})
        certificate = selected.get('control_baseline_agreement') or {}
        keys = certificate.get('experiments',[])
        if (not certificate.get('passed') or len(keys)!=2 or
                keys[-1] != selected.get('baseline_experiment')):
            raise RuntimeError(f"{case['id']}: missing frozen two-endpoint control evidence")
        records = [_verify_experiment(evaluator.root,key) for key in keys]
        rechecked = baseline_agreement(*records,case)
        if not rechecked['passed'] or rechecked != certificate:
            raise RuntimeError(f"{case['id']}: retained control baseline agreement changed")


def confirm_selected(evaluator, report, required_ids, output):
    """Cold repeat of the best fully admitted development model and matched SST.

    A frozen repeat ID in the stage protocol creates separate physical records.
    This is repeatability evidence on development cases, not a blind test set.
    """
    from tedp.spec import from_json
    from .search import _eligible, _member_key
    feasible = report.get('feasible', [])
    if not feasible:
        result = {'status':'skipped_no_feasible_model', 'passed':False,
                  'scope':'No candidate met the full development constraints'}
        atomic_json(output, result)
        return result
    repeat = evaluator.config.get('stages',{}).get('confirmation',{}).get('protocol',{}).get('confirmation_repeat_id')
    if not repeat:
        raise ValueError('Fresh confirmation needs a frozen, nonempty repeat ID')
    member = min(feasible, key=_member_key)
    choice = {'spec_hash':member['spec_hash'], 'spec':member['spec'],
              'case_ids':list(required_ids), 'repeat_id':repeat}
    choice_path = output.with_name('confirmation_selection.json')
    if choice_path.exists() and json.loads(choice_path.read_text()) != choice:
        raise RuntimeError('Frozen confirmation selection changed')
    atomic_json(choice_path, choice)
    result = evaluator(from_json(json.dumps(member['spec'])), list(required_ids), 'confirmation')
    report = {'status':'complete', 'passed':_eligible(result, required_ids), 'selection':choice,
              'scope':'Independent cold repeat on the same development cases; not independent generalization evidence',
              'evaluation':result}
    atomic_json(output, report)
    return report


def write_v3_manifest(campaign, cfg, search_cfg, evaluator, provider, protocol_parts):
    """Freeze the V3 evaluation contract: stage map, score definitions, checkpoint policy, budget,
    interface, evidence version and provenance. Immutable once written."""
    from .search import SearchController
    controller = SearchController.__new__(SearchController)
    controller.cfg = search_cfg
    probe = {'trials': search_cfg.tuning_trials}
    try:
        from tedp.spec import CandidateSpec, Term
        sample = CandidateSpec('budget_probe', rsource=(Term('T1', 'c0/max(1,c1*I1)+c2*Gp+c3*Rw+c4*Gk+c5*Psn+c6*Apk+c7*Rf'),),
                               constants=(0.1, 4., 0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
        probe = controller._budget_plan(sample)
    except Exception as exc:  # pragma: no cover - the plan is informational
        probe = {'error': str(exc)}
    manifest = {
        'campaign': cfg['campaign'], 'schema': 'forge-v3-manifest-1', 'created_utc': utc_now(),
        'stages': {
            'core_fitting': {'cases': list(search_cfg.calibration_ids), 'concurrent_protection': list(search_cfg.tuning_protection_ids),
                             'score': 'core: family mean of case normalized errors over the core fitting cases (CMA objective); '
                                      'feasibility requires every core and protection case admitted and inside its allowance'},
            'selection_checkpoint': {'policy': search_cfg.checkpoint_policy, 'max_per_candidate': search_cfg.max_checkpoints_per_candidate,
                                     'constraint_cases': list(search_cfg.selection_constraint_ids),
                                     'accuracy_cases': list(dict.fromkeys((*search_cfg.calibration_ids, *search_cfg.selection_accuracy_ids))),
                                     'score': 'augmented: family mean over calibration cases + selection accuracy cases, averaged within each family before '
                                              'the equal family weighting (the extra duct case joins the duct family); constraint cases are feasibility only; '
                                              'NACA sentinels remain protection only; finalists are ranked on this one fixed objective',
                                     'trigger': 'after the start trial and after every completed CMA batch, for the best feasible core trial that beats the '
                                                'best checkpoint-verified finalist (at most max_per_candidate checkpoints); a failed checkpoint demotes the trial '
                                                'to infeasible before CMA is updated; a candidate with no verified finalist gets no full suite'},
            'full_development': {'cases': list(dict.fromkeys((*search_cfg.development_ids, *search_cfg.protection_ids))),
                                 'score': 'official 9-family objective and allowances of forge.metrics.assess (unchanged); the only frontier score'}},
        'budget': {'case_cost_minutes': dict(search_cfg.case_cost_minutes) if isinstance(search_cfg.case_cost_minutes, dict) else {},
                   'calibration_budget_case_minutes': search_cfg.calibration_budget_case_minutes,
                   'example_plan_eight_constants': probe, 'trial_ceiling': search_cfg.tuning_trials,
                   'search_deadline_unix': search_cfg.search_deadline_unix, 'max_tokens': search_cfg.max_tokens},
        'tuning': {'step_policy': search_cfg.step_policy, 'min_trials': search_cfg.tuning_min_trials, 'patience': search_cfg.tuning_patience,
                   'starts': search_cfg.tuning_starts, 'mode': search_cfg.tuning_mode},
        'interface': dict(search_cfg.interface), 'allowed_variables': list(search_cfg.allowed_variables),
        'evidence': dict(search_cfg.evidence), 'benchmark': dict(search_cfg.benchmark), 'benchmark_reassessment': search_cfg.benchmark_reassessment,
        'proposer_identity': provider.identity, 'evaluator_protocol_fingerprint': evaluator.protocol_hash,
        'search_protocol_hash': search_cfg.protocol_hash, 'protocol_parts': {k: v for k, v in protocol_parts.items() if k != 'numerics'},
        'library_sha256': evaluator.library_sha, 'physical_source_hash': evaluator.code}
    path = campaign / 'search' / 'v3_manifest.json'
    if path.exists():
        previous = json.loads(path.read_text())
        frozen = {k: v for k, v in previous.items() if k != 'created_utc'}
        if frozen != {k: v for k, v in manifest.items() if k != 'created_utc'}:
            raise RuntimeError('Frozen V3 manifest changed; use a new campaign name')
        return previous
    atomic_json(path, manifest)
    return manifest


def launch(cfg, manifest, contract, *, prepare_only: bool = False) -> dict:
    from .capability import run_pilot
    from .evaluator import Evaluator
    from .gates import Gate
    from .proposer import ClaudeProposer
    from .search import SearchConfig, BudgetStop, run_search
    from .metrics import improvement_verdict
    print("FORGE v2: checking isolated runtime and grouped inputs", flush=True)
    readiness = preflight(cfg, manifest, contract)
    if not readiness["ready_for_development_launch"]:
        report_name='preflight_hx1.json' if cfg['backend']['kind']=='pbs' else 'preflight.json'
        raise RuntimeError('Preflight failed; see reports/'+report_name)
    campaign = ROOT/"results"/cfg["campaign"]
    with file_lock(campaign/".launch.lock", blocking=False):
        def phase(name):
            atomic_json(campaign/'stage.json',{'status':'qualifying','phase':name,'pid':os.getpid(),
                        'campaign':cfg['campaign'],'updated_utc':utc_now(),'discovery_started':False})
        phase('basis_and_feature_diagnostics')
        print("FORGE v2: running basis, feature and activation diagnostics", flush=True)
        pilot_path = ROOT/"reports"/cfg["campaign"]/"capability_pilot.json"
        pilot = run_pilot(ROOT, manifest, pilot_path, resume=True)
        if pilot["status"] != "complete":
            raise RuntimeError("Capability input diagnostics incomplete; see "+str(pilot_path))
        gates = Gate(**cfg["gates"], output=campaign/"gates")
        if cfg['backend']['kind'] == 'pbs':
            from .pbs import PBSDevelopmentEvaluator
            evaluator = PBSDevelopmentEvaluator(ROOT, manifest, contract, cfg)
        else:
            evaluator = DevelopmentEvaluator(ROOT, manifest, contract, cfg)
        from .cfd_capability import run_axial_pilot
        phase('duct_axial_qualification')
        axial = run_axial_pilot(evaluator, campaign/'duct_axial_qualification.json')
        if not axial['passed']:
            raise RuntimeError('Duct axial qualification failed; discovery has not started. See duct_axial_qualification.json')
        ids = [c["id"] for c in manifest["cases"] if c["required"] and c["role"] != "final"]
        phase('required_SST_qualification')
        baseline = evaluator.qualify_baselines(ids, cfg.get("baseline_endpoint_factors", [1]))
        atomic_json(campaign/"baseline_qualification.json", baseline)
        if not baseline["admitted"]:
            raise RuntimeError("Required SST baselines lack admission for their declared use; discovery has not started. See baseline_qualification.json")
        verify_control_baselines(evaluator)
        from .cfd_capability import run_cfd_pilot
        phase('coupled_capability_and_protection')
        overlap = cfg.get('capability', {}).get('overlap_calibration', False) and not prepare_only
        if not overlap:
            coupled = run_cfd_pilot(evaluator, gates, campaign/"cfd_capability.json")
            if not coupled["passed"]:
                raise RuntimeError("Coupled capability/protection pilot did not pass; discovery has not started. See cfd_capability.json")
        eligible = _eligibility(manifest, baseline, contract)
        eligibility_path = campaign/"improvement_eligibility.json"
        if eligibility_path.exists() and json.loads(eligibility_path.read_text()) != eligible:
            raise RuntimeError("Frozen SST eligibility changed")
        atomic_json(eligibility_path, eligible)
        if prepare_only:
            report = {"status": "baselines_admitted", "cases": ids,
                      'all_baselines_fully_qualified': baseline['qualified'],
                      "eligible_cases": eligible["eligible_cases"], "discovery_started": False}
            atomic_json(campaign/"prepared.json", report)
            return report
        search_options = dict(cfg["search"])
        v3 = cfg.get('schema_version') == 3
        proposer_options = dict(cfg["proposer"])
        if v3:
            v3_options = v3_search_options(cfg, manifest)
            search_options.update(v3_options)
            proposer_options.update(mode='v3', allowed_variables=v3_options['allowed_variables'], repo_root=ROOT)
        provider = ClaudeProposer(campaign, retry_deadline_unix=cfg['search'].get('search_deadline_unix'),
                                  **proposer_options)
        # A continuation campaign may start from an earlier campaign's tuned forms.
        seeds_file = search_options.pop("seeds_file", None)
        seeds = configured_seeds(seeds_file) if seeds_file is not None else None
        if v3:
            from .seeds import ISLANDS
            seeds = {island: [] for island in ISLANDS}   # fresh population: no preinstalled parents
        for key in ("calibration_ids", "protection_ids", "tuning_protection_ids"):
            if key not in search_options:
                continue
            search_options[key] = tuple(search_options[key])
        protocol_parts = {"evaluator": evaluator.protocol_hash, "numerics": evaluator.qualified_numerics,
                          "launcher_sha256": sha256(Path(__file__)),
                          "eligibility": eligible, "gates": cfg["gates"], "provider": provider.identity}
        if v3:
            from tedp.features import FEATURE_VERSION
            protocol_parts["feature_semantics"] = FEATURE_VERSION
            protocol_parts["evidence_version"] = search_options['evidence']['version']
        search_cfg = SearchConfig(work_root=campaign/"search", protocol_hash=fingerprint(protocol_parts),
            development_ids=tuple(ids), **search_options)
        if v3:
            write_v3_manifest(campaign, cfg, search_cfg, evaluator, provider, protocol_parts)
        evaluate = budgeted_evaluation(evaluator, search_cfg,
                    cfg['budget'].get('confirmation_reserve_core_hours',0))
        print(f"FORGE v2: launching {search_cfg.population_count}-population constrained discovery", flush=True)
        atomic_json(campaign/'stage.json',{'status':'discovery','phase':'search','pid':os.getpid(),
                    'campaign':cfg['campaign'],'updated_utc':utc_now(),'discovery_started':True})
        if overlap:
            # Pilot and seed coefficient fits share one evaluator and CFD ceiling.
            # Gate calls are serialized; no candidate can enter either archive
            # before the prospective capability and protection pilot passes.
            gate_lock = threading.Lock()
            def shared_gate(spec):
                with gate_lock:
                    return gates(spec)
            shared_gate.policy = gates.policy
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='forge-pilot') as pool:
                future = pool.submit(run_cfd_pilot, evaluator, shared_gate, campaign/'cfd_capability.json')
                evaluate = capability_guarded_evaluation(evaluate, future)
                print('FORGE v2: calibration fitting overlaps capability checks; model admission awaits their pass', flush=True)
                result = require_discovery_result(run_search(search_cfg, evaluate, provider, shared_gate, seeds))
                require_capability(future)
        else:
            result = require_discovery_result(run_search(search_cfg, evaluate, provider, gates, seeds))
        report = result.as_dict()
        report['resources'] = evaluator.ledger.snapshot()
        frozen_contract = {**contract, "improvement": {**contract["improvement"], "eligible_cases": eligible["eligible_cases"]}}
        report["minimum_benefit_verdicts"] = {m["spec_hash"]: improvement_verdict(m["assessment"], frozen_contract)
                                              for m in report["feasible"]}
        report["scientific_release_ready"] = False
        report["release_blockers"] = readiness["catalogue"]["release_blockers"]
        atomic_json(campaign/"summary.json", report)
        if cfg.get('confirmation', {}).get('enabled'):
            atomic_json(campaign/'stage.json', {'status':'confirming', 'phase':'cold_repeat_confirmation',
                'pid':os.getpid(), 'campaign':cfg['campaign'], 'updated_utc':utc_now(), 'discovery_started':True})
            print('FORGE v2: confirming selected model and matched SST with fresh cold runs', flush=True)
            report['confirmation'] = confirm_selected(evaluator, report, ids, campaign/'confirmation.json')
            if report['confirmation']['passed']:
                report['confirmation']['minimum_benefit_verdict'] = improvement_verdict(
                    report['confirmation']['evaluation']['assessment'], frozen_contract)
            report['resources'] = evaluator.ledger.snapshot()
            atomic_json(campaign/"summary.json", report)
        atomic_json(campaign/'stage.json', {'status':report['status'], 'phase':'finished',
            'campaign':cfg['campaign'], 'updated_utc':utc_now(), 'discovery_started':True})
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="FORGE v2 isolated development launcher")
    parser.add_argument("command", choices=("preflight", "pilot", "prepare", "launch", "status", "verify-v1"))
    parser.add_argument("--config", type=Path, default=ROOT/"config/campaign.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-v1":
            report = verify_originals(ROOT/"provenance/v1_before.json")
            atomic_json(ROOT/"reports/v1_preservation.json", report)
            print(json.dumps(report, indent=2))
            return 0 if report["unchanged"] else 1
        cfg, manifest, contract = configuration(args.config)
        if args.command == "preflight":
            report = preflight(cfg, manifest, contract)
            print(json.dumps(report, indent=2))
            return 0 if report["ready_for_development_launch"] else 1
        if args.command == "pilot":
            from .capability import run_pilot
            report = run_pilot(ROOT, manifest, ROOT/"reports/capability_pilot.json")
            print(json.dumps({"status":report["status"], "report":str(ROOT/"reports/capability_pilot.json"),
                              "cfd_capability":report["cfd_capability"]}, indent=2))
            return 0 if report["status"] == "complete" else 1
        if args.command == "status":
            report = campaign_status(cfg)
        else:
            report = launch(cfg, manifest, contract, prepare_only=args.command == "prepare")
        compact = {key: report[key] for key in ("status", "generation", "counts", "stop_reason", "campaign", "discovery_started", "phase", "resources", "active_or_unresolved_jobs") if key in report}
        if "feasible" in report: compact["feasible_candidates"] = len(report["feasible"])
        compact["results_directory"] = str(ROOT/"results"/cfg["campaign"])
        print(json.dumps(compact, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, BudgetExhausted) as exc:
        print(f"FORGE v2 stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
