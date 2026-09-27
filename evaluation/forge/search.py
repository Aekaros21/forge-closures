"""Persistent four-population discovery with progressive constrained evaluation.

The evaluator owns case preparation/CFD and must be idempotent for the same
model/cases/stage/protocol. The controller owns discovery, trial gates,
archives, budgets and replay. No historical runner, score or path is patched.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field, replace
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Mapping, Sequence
import fcntl
import hashlib
import json
import math
import os
import time
import uuid
import copy
import threading

from tedp.spec import CandidateSpec, from_json, spec_hash, struct_hash, to_json
from .core import BudgetExhausted
from .proposer import ProposalBatch, ProposalError
from .seeds import ISLANDS, ISLAND_PROFILES, DISCOVERY_ISLANDS, COMBINATION_ISLAND, default_seeds, seed_parameter_types
from .tuning import ParameterBound, assessment_key, default_bounds, gate_result, tune, validate_tuning_budget, cma_population_size

# V3 free-gate repair (see SearchController._repair_round): bounded, never fatal.
RESAMPLE_DRAWS = 8      # tuner redraws per gate-rejected CMA sample
BACKOFF_ROUNDS = 6      # halvings of a rejected start's signed amplitudes
REPAIR_TURNS = 1        # repair turns per proposing island and generation


class BudgetStop(RuntimeError):
    pass


class _ParallelCancelled(RuntimeError):
    pass


def _clean(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, Mapping): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x') as stream:
        json.dump(_clean(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


@dataclass(frozen=True)
class SearchConfig:
    work_root: Path
    protocol_hash: str
    calibration_ids: tuple[str, ...]
    protection_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    tuning_protection_ids: tuple[str, ...] = ()
    seed: int = 712
    generations: int = 20
    children_per_island: int = 4
    population_size: int = 8
    population_count: int = 4
    diagnostic_capacity: int = 40
    max_case_evaluations: int | None = 2000
    max_core_hours: float | None = 100.
    max_tokens: int = 500000
    # A predeclared upper allocation per case/call, checked before launch.
    # The evaluator/provider must also enforce its own execution limit.
    case_core_hour_reserve: float = 1.
    proposal_token_reserve: int = 16384
    tuning_trials: int = 72
    tuning_starts: int = 3
    tuning_mode: str = 'adaptive'
    parameter_bounds: dict | tuple = field(default_factory=dict)
    migration_every: int = 5
    plateau_generations: int = 6
    minimum_progress: float = 1e-4
    behavioral_resolution: float = .05
    evaluate_seeds: bool = True
    tune_seeds: bool = False
    candidate_workers: int = 1
    trial_workers: int = 1
    tuning_protection_mode: str = 'every_trial'
    tuning_protection_parallel: bool = False
    tuning_min_trials: int | None = None
    tuning_patience: int = 3
    promotion_forms: str = 'original_and_tuned'
    adaptive_trial_floor: bool = False
    search_deadline_unix: float | None = None
    # Explicitly retain completed/in-flight evaluation identities when an audited
    # administrative amendment changes only the campaign's wall-clock window.
    # The full controller identity still changes and requires a checked migration.
    evaluation_protocol_hash: str | None = None
    # ---- FORGE V3 (mode 'v3'); every default keeps the v2 behaviour ----
    mode: str = 'v2'
    selection_constraint_ids: tuple[str, ...] = ()
    selection_accuracy_ids: tuple[str, ...] = ()
    checkpoint_policy: str = 'batch_best_improvement'
    max_checkpoints_per_candidate: int = 6
    step_policy: str = 'legacy'
    allowed_variables: tuple[str, ...] = ()
    case_cost_minutes: dict | tuple = field(default_factory=dict)
    calibration_budget_case_minutes: float | None = None
    proposal_stagger_s: float = 0.
    evidence: dict | tuple = field(default_factory=dict)
    benchmark: dict | tuple = field(default_factory=dict)
    benchmark_reassessment: bool = False
    interface: dict | tuple = field(default_factory=dict)

    def validate(self):
        if self.mode not in ('v2', 'v3'):
            raise ValueError('mode must be v2 or v3')
        if self.mode == 'v3':
            from .tuning import STEP_POLICIES
            from tedp import expr as _expr
            for name in ('selection_constraint_ids', 'selection_accuracy_ids'):
                values = getattr(self, name)
                if len(set(values)) != len(values) or not set(values) <= set(self.development_ids):
                    raise ValueError(f'{name} must be unique declared development cases')
            if set(self.selection_accuracy_ids) & set(self.calibration_ids):
                raise ValueError('augmented accuracy cases add to the calibration cases; they never replace one')
            if set(self.selection_constraint_ids) & (set(self.calibration_ids) | set(self.selection_accuracy_ids)):
                raise ValueError('selection constraint cases must be distinct from the scored selection cases')
            if self.checkpoint_policy != 'batch_best_improvement':
                raise ValueError('checkpoint_policy must be batch_best_improvement')
            if type(self.max_checkpoints_per_candidate) is not int or self.max_checkpoints_per_candidate < 1:
                raise ValueError('max_checkpoints_per_candidate must be a positive integer')
            if self.step_policy not in STEP_POLICIES:
                raise ValueError('step_policy must be one of ' + ', '.join(STEP_POLICIES))
            if self.evaluate_seeds or self.tune_seeds:
                raise ValueError('V3 starts a fresh population: no seeds are evaluated or tuned')
            if self.migration_every:
                raise ValueError('V3 has no migration: the incumbent is never installed as a proposal template')
            if self.promotion_forms != 'selected':
                raise ValueError('V3 promotes the calibrated winner only (promotion_forms selected)')
            if not isinstance(self.evidence, Mapping) or not self.evidence.get('dossier_path') or not self.evidence.get('dossier_sha256'):
                raise ValueError('V3 needs the versioned evidence dossier (dossier_path and dossier_sha256)')
            if self.allowed_variables and not set(self.allowed_variables) <= set(_expr.CANDIDATE_VARIABLES):
                raise ValueError('allowed_variables must be candidate variables of the grammar')
            if not isinstance(self.proposal_stagger_s, (int, float)) or self.proposal_stagger_s < 0:
                raise ValueError('proposal_stagger_s must be nonnegative')
            if self.calibration_budget_case_minutes is not None and not (
                    isinstance(self.calibration_budget_case_minutes, (int, float)) and math.isfinite(self.calibration_budget_case_minutes)
                    and self.calibration_budget_case_minutes > 0):
                raise ValueError('calibration_budget_case_minutes must be positive and finite')
            if self.benchmark_reassessment and not (isinstance(self.benchmark, Mapping) and self.benchmark.get('spec')):
                raise ValueError('benchmark_reassessment needs the archived benchmark spec')
        if type(self.candidate_workers) is not int or not 1 <= self.candidate_workers <= 8:
            raise ValueError('candidate_workers must be an integer from 1 to 8')
        if type(self.trial_workers) is not int or not 1 <= self.trial_workers <= 16:
            raise ValueError('trial_workers must be an integer from 1 to 16')
        if self.tuning_protection_mode not in {'every_trial', 'selected'}:
            raise ValueError('tuning_protection_mode must be every_trial or selected')
        if type(self.tuning_protection_parallel) is not bool:
            raise ValueError('tuning_protection_parallel must be a boolean')
        if self.tuning_protection_parallel and self.tuning_protection_mode != 'every_trial':
            raise ValueError('parallel tuning protection requires every_trial mode')
        if self.tuning_min_trials is not None and (type(self.tuning_min_trials) is not int or
                not 1 <= self.tuning_min_trials <= self.tuning_trials or self.tuning_mode != 'adaptive'):
            raise ValueError('tuning_min_trials requires adaptive mode and must be within tuning_trials')
        if type(self.tuning_patience) is not int or self.tuning_patience < 1:
            raise ValueError('tuning_patience must be a positive integer')
        if self.promotion_forms not in {'original_and_tuned', 'selected'}:
            raise ValueError('promotion_forms must be original_and_tuned or selected')
        if self.search_deadline_unix is not None and (
                not math.isfinite(self.search_deadline_unix) or self.search_deadline_unix <= 0):
            raise ValueError('search_deadline_unix must be a positive UTC timestamp')
        if not self.protocol_hash or len(self.protocol_hash) < 8:
            raise ValueError('a frozen evaluator/manifest/contract protocol identity is required')
        if self.evaluation_protocol_hash is not None and (
                not isinstance(self.evaluation_protocol_hash, str) or len(self.evaluation_protocol_hash) < 8):
            raise ValueError('evaluation_protocol_hash must identify the unchanged evaluation protocol')
        for name in ('calibration_ids', 'development_ids'):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f'{name} must contain unique declared development cases')
        if len(set(self.protection_ids)) != len(self.protection_ids):
            raise ValueError('protection case IDs must be unique')
        if (len(set(self.tuning_protection_ids)) != len(self.tuning_protection_ids)
                or not set(self.tuning_protection_ids) <= set(self.protection_ids)):
            raise ValueError('tuning sentinels must be unique declared protection cases')
        if self.population_count not in (1, 4):
            raise ValueError('population_count must be 1 or 4 for the controlled population ablation')
        validate_tuning_budget(1, self.tuning_trials, self.tuning_starts, self.tuning_mode)
        required = set(self.development_ids) | set(self.protection_ids)
        if not set(self.calibration_ids) <= required:
            raise ValueError('calibration cases must be included in the full development/protection suite')
        for name in ('generations', 'children_per_island', 'population_size', 'diagnostic_capacity',
                     'tuning_starts', 'plateau_generations', 'proposal_token_reserve'):
            if getattr(self, name) < 1: raise ValueError(f'{name} must be positive')
        if self.max_case_evaluations is not None and (type(self.max_case_evaluations) is not int or self.max_case_evaluations < 1):
            raise ValueError('case-evaluation limit must be positive or None')
        if self.max_tokens < 0 or self.tuning_trials < 0 or self.migration_every < 0:
            raise ValueError('token/trial/migration limits must be nonnegative')
        if self.seed < 0 or self.minimum_progress < 0:
            raise ValueError('seed and progress threshold must be nonnegative')
        if not math.isfinite(self.behavioral_resolution) or self.behavioral_resolution <= 0:
            raise ValueError('behavioral response resolution must be finite and positive')
        if not ((self.max_core_hours is None or (math.isfinite(self.max_core_hours) and self.max_core_hours > 0)) and
                math.isfinite(self.case_core_hour_reserve) and self.case_core_hour_reserve > 0):
            raise ValueError('finite positive compute budget and reservation are required')


@dataclass
class SearchResult:
    status: str
    generation: int
    feasible: list[dict]
    diagnostic: list[dict]
    counts: dict
    stop_reason: str
    checkpoint: str
    populations: dict

    def as_dict(self):
        return _clean(asdict(self))


def _numeric(value, default=math.inf):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def _qualified(result, case_ids):
    """Require evidence admitted for each case's declared development use."""
    from .admission import admission
    if result.get('status') != 'complete': return False
    cases = result.get('case_results')
    if not isinstance(cases, Mapping) or any(case_id not in cases for case_id in case_ids): return False
    for case_id in case_ids:
        item = cases[case_id]
        if (not isinstance(item, Mapping) or item.get('status') not in ('complete','unqualified')
                or not admission(item)['admitted']): return False
    return True


def _eligible(result, case_ids):
    return assessment_key(result)[0] == 0 and _qualified(result, case_ids)


def _member_key(member):
    a = member['assessment']
    return (0 if member['feasible'] else 1,
            0. if member['feasible'] else max(0., _numeric(a.get('max_violation'))),
            _numeric(a.get('objective')), member['complexity'],
            _numeric(member.get('measured_case_core_hours')), member['spec_hash'])


def _dominates(left, right):
    la, ra = left['assessment'], right['assessment']
    lf, rf = la.get('family_errors', {}), ra.get('family_errors', {})
    keys = sorted(set(lf) | set(rf))
    lv = [_numeric(lf.get(key)) for key in keys] + [left['complexity'], _numeric(left.get('measured_case_core_hours'))]
    rv = [_numeric(rf.get(key)) for key in keys] + [right['complexity'], _numeric(right.get('measured_case_core_hours'))]
    if not keys:
        lv.insert(0, _numeric(la.get('objective'))); rv.insert(0, _numeric(ra.get('objective')))
    return all(a <= b for a, b in zip(lv, rv)) and any(a < b for a, b in zip(lv, rv))


def _select(members, capacity):
    unique = {member['spec_hash']: member for member in members}
    feasible = [m for m in unique.values() if m['feasible']]
    diagnostic = sorted((m for m in unique.values() if not m['feasible']), key=_member_key)
    selected = []
    while feasible and len(selected) < capacity:
        front = [m for m in feasible if not any(_dominates(other, m) for other in feasible if other is not m)]
        # Family errors already use fixed SST/physical-floor normalization.
        # Preserve distinct response signatures first; syntax alone is not
        # evidence of behavioral diversity. This is a coarse development-flow
        # response descriptor, not a claim of full-field independence.
        front.sort(key=_member_key)
        behavioral = set()
        structural = set()
        ordered = []
        for member in front:
            signature = tuple(sorted(member.get('behavior_signature', {}).items()))
            if signature not in behavioral:
                ordered.append(member); behavioral.add(signature); structural.add(member['struct_hash'])
        for member in front:
            if member not in ordered and member['struct_hash'] not in structural:
                ordered.append(member); structural.add(member['struct_hash'])
        ordered += [member for member in front if member not in ordered]
        selected.extend(ordered[:capacity - len(selected)])
        ids = {m['spec_hash'] for m in front}
        feasible = [m for m in feasible if m['spec_hash'] not in ids]
    selected.extend(diagnostic[:capacity - len(selected)])
    return selected


class SearchController:
    def __init__(self, config: SearchConfig, evaluate: Callable, proposer, gate: Callable,
                 seeds: Mapping[str, Sequence[CandidateSpec]] | None = None):
        config.validate()
        self.cfg, self.evaluate_callback, self.proposer, self.gate_callback = config, evaluate, proposer, gate
        self.root = Path(config.work_root).resolve()
        self.checkpoint = self.root / 'checkpoint.json'
        supplied = default_seeds() if seeds is None else seeds
        if set(supplied) != set(ISLANDS): raise ValueError('seeds must explicitly identify all four islands')
        self.seeds = {key: list(supplied[key]) for key in ISLANDS}
        for group in self.seeds.values():
            for spec in group: spec.validate(search_policy=True)
        if not hasattr(proposer, 'identity'):
            raise ValueError('proposer must expose a stable identity for checkpoint replay')
        cfg_payload = asdict(config)
        cfg_payload['work_root'] = str(self.root)
        sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ('search.py', 'tuning.py', 'proposer.py', 'seeds.py')
                   + (('v3.py', 'evidence.py') if config.mode == 'v3' else ())}
        self.identity = {'config': cfg_payload, 'proposer': proposer.identity, 'sources': sources,
            'seeds': {key: [json.loads(to_json(spec)) for spec in group] for key, group in self.seeds.items()}}
        self.fingerprint = _hash(self.identity)
        self.state = None
        self._state_lock = threading.RLock()
        self._evaluation_locks = {}
        self._gate_lock = threading.Lock()
        self._parallel_failure = None
        self._selection_results = {}

    def _population_key(self, island):
        return island if self.cfg.population_count == 4 else 'shared'

    def _population_capacity(self):
        # Match total retained population capacity between both arms.
        return self.cfg.population_size * (4 if self.cfg.population_count == 1 else 1)

    def save(self):
        with self._state_lock:
            _atomic(self.checkpoint, self.state)

    def event(self, kind, **details):
        with self._state_lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(_clean({'kind': kind, 'generation': self.state['generation'],
                    'time': time.time(), **details}), sort_keys=True, allow_nan=False) + '\n')

    def _load(self):
        if self.checkpoint.exists():
            self.state = json.loads(self.checkpoint.read_text())
            if self.state.get('fingerprint') != self.fingerprint:
                raise ValueError('resume identity differs: model policy, protocol, source, proposer, seeds or configuration changed')
            return
        self.state = {'schema': 1, 'fingerprint': self.fingerprint, 'identity': self.identity,
            'status': 'created', 'generation': 0, 'stop_reason': '', 'seed_index': 0,
            'counts': {'case_evaluations': 0, 'core_hours': 0., 'tokens': 0, 'gate_trials': 0},
            'evaluations': {}, 'calls': {}, 'parameter_types': {}, 'members': {},
            'populations': {key: [] for key in (ISLANDS if self.cfg.population_count == 4 else ('shared',))},
            'feasible': [], 'diagnostic': [],
            'pending_generation': None, 'completed_candidates': [], 'best_objective': None, 'plateau': 0}
        self.save()

    def _gate(self, spec):
        self.state['counts']['gate_trials'] += 1
        try:
            spec.validate(search_policy=True, allowed_variables=self.cfg.allowed_variables or None)
            result = gate_result(self.gate_callback(spec))
        except (ValueError, ArithmeticError) as exc:
            result = {'passed': False, 'reasons': [f'{type(exc).__name__}: {exc}']}
        self.event('gate', spec_hash=spec_hash(spec), result=result)
        if not result['passed']:
            failures = self.state.setdefault('recent_gate_rejections', [])
            failures.append({'spec': json.loads(to_json(spec)), 'reasons': result.get('reasons', [])})
            del failures[:-24]
        self.save()
        return result

    # ------------------------------------------------------------------ V3 free-gate repair
    def _gate_probe(self, spec):
        """The free gates of a proposed start, outside any candidate transaction (no rejection bookkeeping)."""
        self.state['counts']['gate_trials'] += 1
        try:
            spec.validate(search_policy=True, allowed_variables=self.cfg.allowed_variables or None)
            return gate_result(self.gate_callback(spec))
        except (ValueError, ArithmeticError) as exc:
            return {'passed': False, 'reasons': [f'{type(exc).__name__}: {exc}']}

    @staticmethod
    def _gate_score(result):
        checks = result.get('checks') if isinstance(result.get('checks'), Mapping) else {}
        failed = sum(1 for check in checks.values() if isinstance(check, Mapping) and check.get('passed', check.get('ok')) is False)
        tensor = checks.get('tensor') if isinstance(checks.get('tensor'), Mapping) else {}
        return (0 if result.get('passed') else 1, failed, _numeric(tensor.get('new_violation'), 0.), len(result.get('reasons') or []))

    def _backoff(self, spec, first, kinds, roles):
        """Free amplitude back-off of a start the gates reject: greedily halve the signed amplitude whose
        halving helps most, at most BACKOFF_ROUNDS times. Returns (spec, record) or (None, record)."""
        if not isinstance(first.get('checks'), Mapping):      # grammar/validation failures are not amplitude problems
            return None, {'eligible': False}
        used = {int(str(value).lstrip('c')) for value in spec.used_constants()}
        kinds = list(kinds or ['signed'] * len(spec.constants))
        signed = [i for i in range(len(spec.constants)) if i in used and i < len(kinds) and kinds[i] == 'signed' and spec.constants[i] != 0]
        preferred = [i for i in signed if roles and i < len(roles) and roles[i] == 'amplitude']
        movable = preferred or signed
        if not movable:
            return None, {'eligible': False}
        current, best, steps, probes = spec, first, [], 0
        for _ in range(BACKOFF_ROUNDS):
            trials = []
            for i in movable:
                constants = list(current.constants); constants[i] *= .5
                trial = replace(current, constants=tuple(constants))
                result = self._gate_probe(trial); probes += 1
                trials.append((self._gate_score(result), i, trial, result))
            _, i, trial, result = min(trials, key=lambda row: (row[0], row[1]))
            current, best = trial, result
            steps.append({'constant': f'c{i}', 'value': trial.constants[i], 'reasons': list(result.get('reasons') or [])[:3]})
            if result.get('passed'):
                return current, {'eligible': True, 'passed': True, 'probes': probes, 'steps': steps,
                                 'from_constants': list(spec.constants), 'to_constants': list(current.constants)}
        return None, {'eligible': True, 'passed': False, 'probes': probes, 'steps': steps, 'last_reasons': list(best.get('reasons') or [])[:3]}

    def _adopt_metadata(self, original, replacement, **notes):
        old, new = spec_hash(original), spec_hash(replacement)
        store = self.state.setdefault('candidate_metadata', {})
        store[new] = {**(store.get(old) or {}), **{k: v for k, v in notes.items() if v is not None}}
        if old in self.state['parameter_types']:
            self.state['parameter_types'].setdefault(new, self.state['parameter_types'][old])

    def _repair_round(self, generation, queue):
        """Before any CFD: rescue proposed starts the free gates reject. First a free amplitude back-off; then, for
        what it cannot rescue, ONE bounded repair turn in the proposing island's own conversation. Nothing here can
        stop the generation: every failure leaves the proposed candidate in the queue, where its own assessment
        records the gate rejection as evidence, exactly as before."""
        outcomes, failed = [], {}
        for position, item in enumerate(queue):
            spec = from_json(json.dumps(item['spec']))
            identity = spec_hash(spec)
            first = self._gate_probe(spec)
            row = {'island': item['island'], 'name': spec.name, 'spec_hash': identity, 'passed': bool(first.get('passed')),
                   'reasons': list(first.get('reasons') or [])[:3]}
            if not first.get('passed'):
                metadata = (self.state.get('candidate_metadata') or {}).get(identity) or {}
                rescued, record = self._backoff(spec, first, self.state['parameter_types'].get(identity), metadata.get('parameter_roles'))
                row['backoff'] = record
                if rescued is not None:
                    self._adopt_metadata(spec, rescued, start_backoff={k: record[k] for k in ('from_constants', 'to_constants', 'steps')})
                    queue[position] = {'island': item['island'], 'spec': json.loads(to_json(rescued))}
                    row.update(rescued=True, rescued_spec_hash=spec_hash(rescued))
                else:
                    failed.setdefault(item['island'], []).append((position, spec, first))
            outcomes.append(row)
        self.event('proposal_gate_round', generation=generation, outcomes=outcomes)
        self.save()
        if failed and hasattr(self.proposer, 'repair') and REPAIR_TURNS > 0:
            passing = [from_json(json.dumps(queue[i]['spec'])) for i, row in enumerate(outcomes) if row['passed'] or row.get('rescued')][:3]
            for island, rows in failed.items():
                try:
                    self._repair_island(generation, island, rows, passing, queue)
                except BudgetStop:
                    raise
                except Exception as exc:       # a repair is an optimisation; it must never pause discovery
                    self.event('proposal_repair', generation=generation, island=island, status='skipped',
                               reason=f'{type(exc).__name__}: {str(exc)[:240]}')
                    self.save()
        return queue

    def _repair_island(self, generation, island, rows, passing, queue):
        call_id, parent_call = f'g{generation:04d}-{island}-r1', f'g{generation:04d}-{island}'
        entry = self.state['calls'].get(call_id)
        recorded_usage = getattr(self.proposer, 'repair_usage', lambda _: 0)
        if entry and entry.get('status') != 'complete':
            if not recorded_usage(call_id):
                self.event('proposal_repair', generation=generation, island=island, status='skipped',
                           reason='an earlier repair attempt has no committed response; not repeated')
                return
            entry = None if entry.get('counted') else {'status': 'started', 'uncounted': True}
        uncounted = entry is None or bool(entry.get('uncounted'))
        if entry is None:
            reserve = int(self.cfg.proposal_token_reserve)
            if hasattr(self.proposer, 'repair_reserve_tokens'):
                reserve = max(reserve, self.proposer.repair_reserve_tokens(self._repair_message(rows, passing), parent_call))
            outstanding = sum(value.get('reserved_tokens', 0) for value in self.state['calls'].values() if value['status'] != 'complete')
            if self.state['counts']['tokens'] + outstanding + reserve > self.cfg.max_tokens:
                self.event('proposal_repair', generation=generation, island=island, status='skipped', reason='token budget')
                return
            self.state['calls'][call_id] = {'status': 'started', 'reserved_tokens': reserve, 'kind': 'repair'}
            self.save()
        batch, failure = None, None
        try:
            batch = self.proposer.repair(len(rows), self._repair_message(rows, passing), call_id, parent_call)
        except ProposalError as exc:
            failure = str(exc)[:240]
        usage = int(batch.usage_tokens) if batch is not None else int(recorded_usage(call_id))
        if uncounted:
            self.state['counts']['tokens'] += usage
        specs = list(batch.specs) if batch is not None else []
        if batch is not None:
            self.state['parameter_types'].update(batch.metadata.get('parameter_types', {}))
            if batch.metadata.get('candidate_metadata'):
                self.state.setdefault('candidate_metadata', {}).update(batch.metadata['candidate_metadata'])
        replaced = []
        for (position, original, _), spec in zip(rows, specs):
            result = self._gate_probe(spec)
            chosen, note = (spec, None) if result.get('passed') else (None, None)
            if chosen is None:
                metadata = (self.state.get('candidate_metadata') or {}).get(spec_hash(spec)) or {}
                chosen, record = self._backoff(spec, result, self.state['parameter_types'].get(spec_hash(spec)), metadata.get('parameter_roles'))
                note = {k: record[k] for k in ('from_constants', 'to_constants', 'steps')} if chosen is not None else None
            if chosen is None:
                replaced.append({'original': original.name, 'repair': spec.name, 'passed': False, 'reasons': list(result.get('reasons') or [])[:3]})
                continue
            self._adopt_metadata(spec, chosen, start_backoff=note,
                                 repaired_from={'name': original.name, 'spec_hash': spec_hash(original), 'call_id': call_id})
            queue[position] = {'island': island, 'spec': json.loads(to_json(chosen))}
            replaced.append({'original': original.name, 'repair': chosen.name, 'passed': True, 'spec_hash': spec_hash(chosen)})
        self.state['calls'][call_id] = {'status': 'complete', 'usage_tokens': usage, 'kind': 'repair',
                                        'specs': [json.loads(to_json(s)) for s in specs], 'failure': failure}
        self.event('proposal_repair', generation=generation, island=island, call_id=call_id,
                   status='complete' if batch is not None else 'no_response', usage_tokens=usage,
                   requested=len(rows), returned=len(specs), outcomes=replaced, failure=failure)
        self.save()

    @staticmethod
    def _repair_message(rows, passing):
        failures = [{'name': spec.name, 'gate_reasons': list(first.get('reasons') or [])[:4],
                     'tensor_check': {k: v for k, v in (((first.get('checks') or {}).get('tensor')) or {}).items()
                                      if k in ('new_violation', 'max_g', 'clamp_effect', 'r_clamp_fraction', 'recovery_residual')}}
                    for _, spec, first in rows]
        examples = [{'name': s.name, 'bdelta': [[term.tensor, term.expression] for term in s.bdelta],
                     'rsource': [[term.tensor, term.expression] for term in s.rsource], 'constants': list(s.constants)} for s in passing]
        return '\n'.join([
            'FORGE V3 REPAIR REQUEST',
            'The controller ran the free pre-CFD gates on your candidates at their proposed starting constants. The candidates below were '
            'rejected, and halving their signed amplitudes (up to %d times, the most helpful constant each time) did not make them pass. '
            'A rejected start receives no CFD at all.' % BACKOFF_ROUNDS,
            json.dumps({'rejected': failures}, indent=1),
            'Starts from this generation that PASSED the same gates (for calibration of what is admissible, not as templates):',
            json.dumps(examples, indent=1),
            'Return a JSON array of exactly %d corrected candidate objects in the same schema, one per rejected candidate and in the same order. '
            'Keep each physical hypothesis; change the functional form and/or the starting constants so that the free gates pass: realizability '
            '(Lumley triangle at both SST limiter extremes on every sampled state, including states with frame rotation), boundedness, clamp '
            'reliance below 1%%, channel Cf within 2%% of SST, homogeneous-shear safety. The tuner can raise an amplitude later if the gates '
            'allow it, so a conservative admissible start is worth more than an ambitious rejected one. No prose, no Markdown fences.' % len(rows)])

    def _sync_spend(self):
        source = getattr(self.evaluate_callback, 'authoritative_spend', None)
        if source is not None:
            spent = source()
            if (not isinstance(spent, (int, float)) or not math.isfinite(spent)
                    or spent < self.state['counts']['core_hours'] - 1e-8):
                raise RuntimeError('Controller accounting exceeds its physical resource ledger')
            self.state['counts']['core_hours'] = float(spent)

    def _check_deadline(self):
        if self.cfg.search_deadline_unix is not None and time.time() >= self.cfg.search_deadline_unix:
            raise BudgetStop('Search deadline reached; remaining wall time reserved for confirmation')

    def _evaluate(self, spec, case_ids, stage):
        case_ids = list(dict.fromkeys(case_ids))
        if not case_ids: raise ValueError('cannot evaluate an empty case set')
        key = _hash({'model': None if spec is None else spec_hash(spec), 'cases': case_ids,
                     'stage': stage, 'protocol': self.cfg.evaluation_protocol_hash or self.cfg.protocol_hash})
        with self._state_lock:
            key_lock = self._evaluation_locks.setdefault(key, threading.Lock())
        # Only identical evaluations serialize for their entire lifetime.
        # Reservations/checkpoints are brief transactions in the parent broker.
        with key_lock:
            path = self.root / 'evaluations' / f'{key}.json'
            with self._state_lock:
                if self._parallel_failure is not None:
                    raise _ParallelCancelled('Another fit stopped; current physical work has drained')
                entry = self.state['evaluations'].get(key)
                if path.exists():
                    payload = json.loads(path.read_text())
                    if payload.get('identity') != key: raise ValueError('evaluation cache identity mismatch')
                    result = payload['result']
                    observed = _hash(result)
                    if payload.get('result_sha256') != observed or (entry and entry.get('status') == 'complete'
                            and entry.get('result_sha256') != observed):
                        raise ValueError('evaluation cache checksum mismatch')
                    if entry is None or entry['status'] != 'complete':
                        self._commit_evaluation(key, case_ids, result)
                    return result
                if entry is not None and entry.get('status') == 'complete':
                    raise ValueError('completed evaluation artifact is missing; refusing an unaccounted rerun')
                self._check_deadline()
                self._sync_spend()
                if entry is None:
                    count = self.state['counts']
                    if self.cfg.max_case_evaluations is not None and count['case_evaluations'] + len(case_ids) > self.cfg.max_case_evaluations:
                        raise BudgetStop('case-evaluation budget would be exceeded')
                    reserve = len(case_ids) * self.cfg.case_core_hour_reserve
                    outstanding = sum(value.get('core_reserve', 0.) for value in self.state['evaluations'].values()
                                      if value['status'] == 'started')
                    if self.cfg.max_core_hours is not None and count['core_hours'] + outstanding + reserve > self.cfg.max_core_hours:
                        raise BudgetStop('compute reservation would exceed the campaign budget')
                    self.state['evaluations'][key] = {'status': 'started', 'case_ids': case_ids,
                                                     'stage': stage, 'core_reserve': reserve}
                    count['case_evaluations'] += len(case_ids)
                    self.save()
            callback = (getattr(self.evaluate_callback, 'concurrent', self.evaluate_callback)
                        if max(self.cfg.candidate_workers, self.cfg.trial_workers) > 1 or
                        self.cfg.tuning_protection_parallel else self.evaluate_callback)
            try:
                result = callback(spec, case_ids, stage)
            except BaseException as exc:
                with self._state_lock:
                    if self._parallel_failure is None:
                        self._parallel_failure = exc
                raise
            if not isinstance(result, dict) or result.get('status') not in {'complete', 'failed', 'unqualified'}:
                raise ValueError('evaluator returned an invalid top-level status')
            if not isinstance(result.get('assessment'), dict):
                raise ValueError('evaluator omitted assessment')
            cost = (result.get('cost') or {}).get('core_hours')
            if not isinstance(cost, (float, int)) or not math.isfinite(cost) or cost < 0:
                raise ValueError('evaluator must report finite nonnegative consumed core_hours')
            with self._state_lock:
                _atomic(path, {'identity': key, 'result': result, 'result_sha256': _hash(result)})
                self._commit_evaluation(key, case_ids, result)
            return result

    def _commit_evaluation(self, key, cases, result):
        existing = self.state['evaluations'].get(key)
        if existing is not None and existing['status'] == 'complete': return
        if existing is None: self.state['counts']['case_evaluations'] += len(cases)
        cost = float(result['cost']['core_hours'])
        if hasattr(self.evaluate_callback, 'authoritative_spend'):
            self._sync_spend()
        else:
            self.state['counts']['core_hours'] += cost
        self.state['evaluations'][key] = {'status': 'complete', 'case_ids': list(cases), 'core_hours': cost,
                                          'result_sha256': _hash(result)}
        self.event('evaluation', identity=key, status=result['status'], assessment=result['assessment'], core_hours=cost)
        self.save()
        if self.cfg.max_core_hours is not None and self.state['counts']['core_hours'] > self.cfg.max_core_hours:
            raise BudgetStop('actual evaluator cost exceeded its reservation; no more work will launch')

    def _bounds(self, spec):
        configured = self.cfg.parameter_bounds
        values = None
        if isinstance(configured, Mapping):
            for key in (spec_hash(spec), struct_hash(spec), spec.name):
                if key in configured:
                    values = configured[key]; break
            if values is None and configured and all(f'c{i}' in configured for i in range(len(spec.constants))):
                values = [configured[f'c{i}'] for i in range(len(spec.constants))]
        elif configured:
            values = configured
        if values is not None:
            return tuple(value if isinstance(value, ParameterBound) else ParameterBound(**value) for value in values)
        kinds = self.state['parameter_types'].get(spec_hash(spec)) or seed_parameter_types(spec)
        return default_bounds(spec, kinds)

    def _tuning_evaluate(self, spec):
        """Calibration accuracy ranked under calibration AND sentinel constraints."""
        sentinels = self.cfg.tuning_protection_ids or self.cfg.protection_ids
        protection = None
        if self.cfg.tuning_protection_parallel and sentinels:
            # Both groups share the evaluator's global case semaphore. Drain
            # already launched work even if one group raises during collection.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix='forge-fit-cases') as pool:
                calibration_future = pool.submit(self._evaluate, spec, self.cfg.calibration_ids, 'calibration')
                protection_future = pool.submit(self._evaluate, spec, sentinels, 'protection')
                calibration = calibration_future.result()
                protection = protection_future.result()
        else:
            calibration = self._evaluate(spec, self.cfg.calibration_ids, 'calibration')
        calibration_assessment = dict(calibration['assessment'])
        if protection is None and not _qualified(calibration, self.cfg.calibration_ids):
            # A partial or numerically failed calibration cannot supply a
            # qualified fitting objective. It is the sole CFD skip condition.
            return {**calibration, 'status': 'failed' if calibration['status'] == 'failed' else 'unqualified',
                    'assessment': {**calibration_assessment, 'feasible': False,
                        'reasons': list(calibration_assessment.get('reasons', [])) +
                                   ['tuning sentinels skipped: calibration is incomplete or unqualified']},
                    'tuning_protection': {'status': 'skipped_unqualified_calibration'}}
        if self.cfg.tuning_protection_mode == 'selected':
            return {**calibration, 'tuning_protection': {'status': 'deferred_to_selected_full_suite'}}
        if not sentinels:
            return {**calibration, 'tuning_protection': {'status': 'not_declared'}}
        if protection is None:
            protection = self._evaluate(spec, sentinels, 'protection')
        protection_assessment = protection['assessment']
        feasible = _eligible(calibration, self.cfg.calibration_ids) and _eligible(protection, sentinels)
        protection_status = ('failed' if protection['status'] == 'failed' else
                             'complete' if _qualified(protection, sentinels) else 'unqualified')
        status = ('failed' if 'failed' in (calibration['status'], protection_status) else
                  'complete' if _qualified(calibration, self.cfg.calibration_ids) and
                  protection_status == 'complete' else 'unqualified')
        violations = [_numeric(value.get('max_violation'))
                      for value in (calibration_assessment, protection_assessment)]
        maximum = max(0., *violations)
        reasons = [f'{stage}: {reason}' for stage, result in
                   (('calibration', calibration_assessment), ('tuning protection', protection_assessment))
                   for reason in result.get('reasons', [])]
        if not _qualified(protection, sentinels):
            reasons.append('tuning protection: incomplete or unqualified required-case evidence')
        if not _qualified(calibration, self.cfg.calibration_ids):
            reasons.append('calibration: incomplete or unqualified required-case evidence')
        return {'status': status,
                'case_results': {**calibration.get('case_results', {}), **protection.get('case_results', {})},
                'assessment': {**calibration_assessment, 'feasible': feasible,
                    'max_violation': maximum if math.isfinite(maximum) else None, 'reasons': reasons,
                    'cases': {**calibration_assessment.get('cases', {}), **protection_assessment.get('cases', {})},
                    'objective_scope': 'calibration accuracy; calibration and tuning protection are mandatory constraints'},
                'cost': {'core_hours': calibration['cost']['core_hours'] + protection['cost']['core_hours']},
                'tuning_protection': {'status': protection_status, 'case_ids': list(sentinels),
                                      'assessment': protection_assessment}}

    def _record_member(self, spec, island, result, origin, extra=None):
        required = list(dict.fromkeys((*self.cfg.development_ids, *self.cfg.protection_ids)))
        feasible = _eligible(result, required)
        assessment = dict(result.get('assessment') or {})
        if not feasible and assessment.get('feasible'):
            assessment.update(feasible=False, max_violation=None,
                              reasons=list(assessment.get('reasons', [])) + ['incomplete or unqualified required-case evidence'])
        results = result.get('case_results', {})
        costs = [results.get(cid, {}).get('core_hours') for cid in required]
        measured_cost = sum(costs) if all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in costs) else None
        signature = {name: int(round(value / self.cfg.behavioral_resolution))
                     for name, value in assessment.get('family_errors', {}).items()
                     if isinstance(value, (int, float)) and math.isfinite(value)}
        member = {'spec': json.loads(to_json(spec)), 'spec_hash': spec_hash(spec), 'struct_hash': struct_hash(spec),
            'island': island, 'origin': origin, 'generation': self.state['generation'], 'feasible': feasible,
            'assessment': assessment, 'complexity': spec.n_parameters(), 'ast_nodes': spec.ast_nodes(),
            'measured_case_core_hours': measured_cost,
            'cost_scope': 'sum of retained per-case core_hours on the full development suite; cache charge is not a model cost',
            'behavior_signature': signature, 'behavioral_resolution': self.cfg.behavioral_resolution,
            'case_statuses': {key: value.get('status') for key, value in result.get('case_results', {}).items()},
            'full_case_ids': required,
            'parameter_types': self.state['parameter_types'].get(spec_hash(spec))}
        if extra:
            member.update(_clean(extra))
        self._insert_member(member)
        self.event('candidate_assessed', spec_hash=member['spec_hash'], feasible=feasible, assessment=assessment, origin=origin)
        self.save()

    def _insert_member(self, member):
        self.state['members'][member['spec_hash']] = member
        population = self._population_key(member['island'])
        existing = [self.state['members'][identity] for identity in self.state['populations'][population]]
        self.state['populations'][population] = [m['spec_hash'] for m in _select([*existing, member], self._population_capacity())]
        all_members = list(self.state['members'].values())
        self.state['feasible'] = [m['spec_hash'] for m in _select([m for m in all_members if m['feasible']], self.cfg.population_size * 4)]
        self.state['diagnostic'] = [m['spec_hash'] for m in sorted((m for m in all_members if not m['feasible']), key=_member_key)[:self.cfg.diagnostic_capacity]]

    def _parallel_candidates(self, items):
        """Fit independently; publish members in the frozen input order.

        Worker checkpoints own fitting state/history. This parent alone owns
        shared evaluation reservations, charges and population selection.
        A failed worker stops new admissions and drains already-running calls.
        """
        workers = []
        seen = set(self.state['completed_candidates'])
        for spec, island, origin in items:
            transaction = _hash({'model': spec_hash(spec), 'island': island})
            if transaction not in seen:
                workers.append(_CandidateWorker(self, spec, island, origin, transaction))
                seen.add(transaction)
        # Offer one fit per island before the next child of each island.
        groups = [[w for w in workers if w.island == island] for island in ISLANDS]
        schedule = [group[i] for i in range(max(map(len, groups), default=0))
                    for group in groups if i < len(group)]
        pending, finished = {}, set()
        next_index = 0
        self._parallel_failure = None
        pool = ThreadPoolExecutor(max_workers=self.cfg.candidate_workers, thread_name_prefix='forge-fit')
        try:
            while pending or next_index < len(schedule):
                while (self._parallel_failure is None and len(pending) < self.cfg.candidate_workers
                       and next_index < len(schedule)):
                    worker = schedule[next_index]
                    pending[pool.submit(worker.execute)] = worker
                    next_index += 1
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    worker = pending.pop(future)
                    try:
                        future.result()
                        finished.add(worker.transaction)
                    except BaseException as exc:
                        with self._state_lock:
                            if self._parallel_failure is None:
                                self._parallel_failure = exc
        except BaseException as exc:
            with self._state_lock:
                if self._parallel_failure is None:
                    self._parallel_failure = exc
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            with self._state_lock:
                self._sync_spend()
                # Commit only the completed input prefix. Later finished fits
                # remain durable in their worker checkpoints for resumption.
                for worker in workers:
                    if worker.transaction not in finished:
                        break
                    for member in worker.state['members'].values():
                        self._insert_member(copy.deepcopy(member))
                    self.state['counts']['gate_trials'] += worker.state['counts']['gate_trials']
                    failures = self.state.setdefault('recent_gate_rejections', [])
                    failures.extend(worker.state.get('recent_gate_rejections', []))
                    del failures[:-24]
                    history = worker.root/'history.jsonl'
                    if history.exists():
                        with (self.root/'history.jsonl').open('a') as output:
                            output.write(history.read_text())
                    self.state['completed_candidates'].append(worker.transaction)
                    self.save()
                self.save()
        failure = self._parallel_failure
        self._parallel_failure = None
        if failure is not None:
            raise failure

    # ------------------------------------------------------------------ FORGE V3

    SELECTION_STAGE = 'selection'

    def _v3_case_minutes(self, case_ids):
        costs = self.cfg.case_cost_minutes if isinstance(self.cfg.case_cost_minutes, Mapping) else {}
        return sum(float(costs.get(cid, 0.) or 0.) for cid in case_ids)

    def _budget_plan(self, spec):
        """Explicit per-candidate CFD plan in measured case-minutes and estimated wall minutes.

        Redundant fitting trials pay for the selection checks: when the plan exceeds the
        declared calibration budget, the trial allocation is reduced (never below the adaptive
        floor). Missing measured costs count as zero and are reported as such."""
        costs = self.cfg.case_cost_minutes if isinstance(self.cfg.case_cost_minutes, Mapping) else {}
        calibration = tuple(self.cfg.calibration_ids)
        sentinels = tuple(self.cfg.tuning_protection_ids or ())
        accuracy_new = tuple(cid for cid in self.cfg.selection_accuracy_ids if cid not in calibration)
        constraints = tuple(self.cfg.selection_constraint_ids)
        required = list(dict.fromkeys((*self.cfg.development_ids, *self.cfg.protection_ids)))
        already = set(calibration) | set(sentinels) | set(accuracy_new) | set(constraints)
        suite_new = [cid for cid in required if cid not in already]
        n_active = max(1, len(spec.used_constants()))
        popsize = cma_population_size(n_active)
        floor = self.cfg.tuning_starts * (popsize + 2)
        trials = self.cfg.tuning_trials
        if self.cfg.adaptive_trial_floor:
            trials = max(trials, floor)
        per_trial = self._v3_case_minutes(calibration) + self._v3_case_minutes(sentinels)
        per_checkpoint = self._v3_case_minutes(accuracy_new) + self._v3_case_minutes(constraints)
        suite = self._v3_case_minutes(suite_new)
        checkpoints = self.cfg.max_checkpoints_per_candidate
        planned = trials * per_trial + checkpoints * per_checkpoint + suite
        reduced = False
        budget = self.cfg.calibration_budget_case_minutes
        if budget is not None and per_trial > 0 and planned > budget:
            affordable = int((budget - checkpoints * per_checkpoint - suite) // per_trial)
            trials = max(floor, min(trials, affordable))
            planned = trials * per_trial + checkpoints * per_checkpoint + suite
            reduced = True
        def wall(ids):
            return max((float(costs.get(cid, 0.) or 0.) for cid in ids), default=0.)
        batches = math.ceil(max(trials - 1, 0) / popsize) + 1
        wall_estimate = (batches * wall(calibration + sentinels) + checkpoints * wall(accuracy_new + constraints) + wall(suite_new))
        return {'trials': int(trials), 'trial_floor': int(floor), 'population_size': popsize, 'batches': batches,
                'per_trial_case_minutes': per_trial, 'per_checkpoint_case_minutes': per_checkpoint,
                'suite_case_minutes': suite, 'max_checkpoints': checkpoints,
                'planned_case_minutes': planned, 'budget_case_minutes': budget, 'trials_reduced_for_budget': reduced,
                'estimated_wall_minutes': wall_estimate, 'minimum_wall_minutes': (
                    (math.ceil(max(floor - 1, 0) / popsize) + 1) * wall(calibration + sentinels) + wall(accuracy_new + constraints) + wall(suite_new)),
                'unmeasured_cases': sorted(cid for cid in set(required) if cid not in costs)}

    def _components(self, result):
        """Every component metric with its qualification status, signed scalar changes and aligned
        signed residual bins of the retained surface/flow profiles (candidate minus SST, normalised
        by the SST profile RMS; 12 equal bins along the stored profile order). Full profiles stay in
        the evaluation payload on disk; coordinates are reporting aids, never closure inputs."""
        import numpy as np
        out = {}
        cases = result.get('case_results') or {}
        bases = result.get('baselines') or {}
        assessment = (result.get('assessment') or {}).get('cases') or {}
        for cid, case in cases.items():
            if not isinstance(case, Mapping):
                continue
            row = {'status': case.get('status'), 'admitted': (case.get('development_admission') or {}).get('admitted')}
            metrics = {}
            for name, metric in (((assessment.get(cid) or {}).get('metrics')) or {}).items():
                if not isinstance(metric, Mapping):
                    continue
                allowance = metric.get('allowance')
                upper = metric.get('upper_change', metric.get('change'))
                usage = (upper / allowance) if isinstance(upper, (int, float)) and isinstance(allowance, (int, float)) and allowance > 0 else None
                metrics[name] = {'cand': metric.get('candidate'), 'sst': metric.get('sst'), 'change': metric.get('change'),
                                 'allowance': allowance, 'usage': None if usage is None else round(usage, 4), 'passed': metric.get('passed')}
            row['metrics'] = metrics
            obs = case.get('observables') or {}
            bobs = (bases.get(cid) or {}).get('observables') or {}
            for name in ('lift_coefficient', 'drag_coefficient', 'wall_friction_ratio', 'secondary_correlation', 'secondary_magnitude_ratio'):
                a, b = obs.get(name), bobs.get(name)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b):
                    row.setdefault('signed_changes', {})[name] = round(a - b, 8)
            for name in ('wall_cf', 'wall_cp', 'u', 'uv'):
                a, b = obs.get(name), bobs.get(name)
                if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and len(a) >= 12:
                    try:
                        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
                    except (ValueError, TypeError):
                        continue
                    if not (np.isfinite(aa).all() and np.isfinite(bb).all()):
                        continue
                    scale = max(float(np.sqrt(np.mean(bb * bb))), 1e-12)
                    residual = (aa - bb) / scale
                    edges = np.linspace(0, len(residual), 13).astype(int)
                    row.setdefault('residual_bins', {})[name] = [round(float(np.mean(residual[edges[i]:edges[i + 1]])), 4) for i in range(12)]
                    coordinate = obs.get('wall_x') if name.startswith('wall') else obs.get('y')
                    if isinstance(coordinate, list) and len(coordinate) == len(a):
                        row.setdefault('bin_coordinate_ranges', {})[name] = [round(float(min(coordinate)), 4), round(float(max(coordinate)), 4)]
            if case.get('failure'):
                row['failure'] = {'type': (case['failure'] or {}).get('type'), 'message': str((case['failure'] or {}).get('message'))[:160]}
            qualification = case.get('qualification') or {}
            if isinstance(qualification, Mapping) and qualification.get('qualified') is False:
                row['qualification_reasons'] = [str(r)[:120] for r in (qualification.get('reasons') or [])][:4]
            out[cid] = row
        return out

    def _selection_checkpoint(self, candidate, core_result, record):
        """Provisional-winner checkpoint at the EXACT coefficient vector: the augmented accuracy set
        (calibration cases + hump + sd2000, cached calibration runs reused) and the development
        constraint cases (h0p5, h1p0, h20, h31; every required observable). Returned to the fit."""
        accuracy_ids = tuple(dict.fromkeys((*self.cfg.calibration_ids, *self.cfg.selection_accuracy_ids)))
        constraint_ids = tuple(self.cfg.selection_constraint_ids)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='forge-select') as pool:
            accuracy_future = pool.submit(self._evaluate, candidate, accuracy_ids, self.SELECTION_STAGE)
            constraint_future = pool.submit(self._evaluate, candidate, constraint_ids, self.SELECTION_STAGE) if constraint_ids else None
            accuracy = accuracy_future.result()
            constraints = constraint_future.result() if constraint_future is not None else None
        accuracy_assessment = accuracy.get('assessment') or {}
        constraint_assessment = (constraints or {}).get('assessment') or {}
        qualified = _qualified(accuracy, accuracy_ids) and (constraints is None or _qualified(constraints, constraint_ids))
        feasible = _eligible(accuracy, accuracy_ids) and (constraints is None or _eligible(constraints, constraint_ids))
        reasons = [f'augmented accuracy: {r}' for r in accuracy_assessment.get('reasons') or []]
        reasons += [f'development constraint: {r}' for r in constraint_assessment.get('reasons') or []]
        if not qualified:
            reasons.append('selection stage: incomplete or unqualified required-case evidence')
        violations = [_numeric(a.get('max_violation')) for a in (accuracy_assessment, constraint_assessment)]
        maximum = max(0., *violations)
        merged = {'status': 'failed' if 'failed' in (accuracy.get('status'), (constraints or {}).get('status')) else 'complete' if qualified else 'unqualified',
                  'case_results': {**accuracy.get('case_results', {}), **(constraints or {}).get('case_results', {})},
                  'baselines': {**accuracy.get('baselines', {}), **(constraints or {}).get('baselines', {})},
                  'assessment': {**accuracy_assessment, 'feasible': feasible,
                                 'max_violation': maximum if math.isfinite(maximum) else None, 'reasons': reasons,
                                 'cases': {**accuracy_assessment.get('cases', {}), **constraint_assessment.get('cases', {})},
                                 'objective_scope': 'augmented selection objective: calibration cases + selection accuracy cases, family-averaged; constraint cases are feasibility only'},
                  'cost': {'core_hours': accuracy['cost']['core_hours'] + ((constraints or {}).get('cost') or {}).get('core_hours', 0.)}}
        components = self._components(merged)
        verdict = {'feasible': bool(feasible), 'status': merged['status'],
                   'augmented_objective': _numeric(accuracy_assessment.get('objective'), None) if qualified else None,
                   'core_objective': _numeric((core_result or {}).get('assessment', {}).get('objective'), None) if isinstance(core_result, Mapping) else None,
                   'max_violation': maximum if math.isfinite(maximum) else None, 'reasons': reasons,
                   'accuracy_case_errors': {cid: _numeric(c.get('normalized_error'), None) for cid, c in (accuracy_assessment.get('cases') or {}).items()},
                   'accuracy_family_errors': accuracy_assessment.get('family_errors'),
                   'constraint_case_errors': {cid: _numeric(c.get('normalized_error'), None) for cid, c in (constraint_assessment.get('cases') or {}).items()},
                   'components': components}
        identity = spec_hash(candidate)
        self._selection_results[identity] = {'merged': merged, 'verdict': verdict}
        self.event('selection_checkpoint', spec_hash=identity, constants=list(candidate.constants),
                   verdict={k: v for k, v in verdict.items() if k != 'components'}, components=components)
        self.save()
        return verdict

    def _on_trial_v3(self, record):
        full = record.get('full_result')
        self.event('tuning_trial', spec_hash=record['spec_hash'], constants=record['constants'], gate=record['gate'],
                   rank=record['rank'], checkpoint=record.get('checkpoint'), demoted=bool(record.get('demoted_by_checkpoint')),
                   event=record.get('event'))
        if isinstance(full, Mapping) and record['gate'].get('passed'):
            self.event('trial_components', spec_hash=record['spec_hash'], constants=record['constants'],
                       status=full.get('status'), components=self._components(full))

    def _tuning_summary(self, fit, plan):
        checkpoints = [c for c in fit.checkpoints if 'skipped' not in c]
        return {'trials': len(fit.trials), 'evaluated': fit.evaluated, 'gate_rejected': fit.rejected,
                'allocated_trials': plan['trials'], 'trials_reduced_for_budget': plan['trials_reduced_for_budget'],
                'stop_reasons': fit.stop_reasons, 'step_policy': fit.step_policy, 'parameterisation': fit.parameterisation,
                'initial_steps': [{k: v for k, v in row.items() if k != 'law'} for start in fit.initial_steps[:1] for row in start['steps']],
                'effective_steps': fit.effective_steps, 'start_key': fit.start_key, 'best_core_key': fit.best_core_key,
                'tuning_improved_start': fit.tuning_improved_start,
                'checkpoints': [{k: v for k, v in c.items() if k != 'reasons'} | {'reasons': c.get('reasons', [])[:4]} for c in checkpoints],
                'checkpoints_skipped': len(fit.checkpoints) - len(checkpoints),
                'finalists': [{'spec_hash': f['spec_hash'], 'augmented_objective': f['augmented_objective'], 'core_key': f['core_key']} for f in fit.finalists],
                'selection': fit.selection}

    def _candidate_v3(self, spec, island, origin):
        transaction = _hash({'model': spec_hash(spec), 'island': island})
        if transaction in self.state['completed_candidates']:
            return
        self._check_deadline()
        identity = spec_hash(spec)
        metadata = (self.state.get('candidate_metadata') or {}).get(identity) or {}
        plan = self._budget_plan(spec)
        self.event('candidate_plan', spec_hash=identity, plan=plan)
        if self.cfg.search_deadline_unix is not None and plan['minimum_wall_minutes'] > 0:
            remaining = (self.cfg.search_deadline_unix - time.time()) / 60.
            if remaining < plan['minimum_wall_minutes']:
                raise BudgetStop(f'{remaining:.0f} min remain before the search deadline; a complete candidate assessment needs '
                                 f'at least {plan["minimum_wall_minutes"]:.0f} min (fit floor + one selection checkpoint + final suite)')
        validate_tuning_budget(len(spec.used_constants()), plan['trials'], self.cfg.tuning_starts, self.cfg.tuning_mode)
        if not self._gate(spec)['passed']:
            self.state['completed_candidates'].append(transaction); self.save(); return
        self._evaluate(spec, self.cfg.calibration_ids, 'calibration')
        roles = metadata.get('parameter_roles')
        if not roles or len(roles) != len(spec.constants):
            roles = None
        self._selection_results = {}
        required = list(dict.fromkeys((*self.cfg.development_ids, *self.cfg.protection_ids)))
        base_extra = {'parameter_roles': roles, 'hypothesis': metadata.get('hypothesis'),
                      'coupling': metadata.get('coupling'), 'parameter_types': self.state['parameter_types'].get(identity),
                      'reuse_of': metadata.get('reuse_of') if metadata.get('reuse_verified') else None,
                      'schema_warnings': metadata.get('schema_warnings'),
                      'start_backoff': metadata.get('start_backoff'), 'repaired_from': metadata.get('repaired_from')}
        if not spec.used_constants():
            # A closure with numeric literals only: nothing to fit, but the same staged verification applies.
            core = self._tuning_evaluate(spec)
            verdict = self._selection_checkpoint(spec, core, {}) if assessment_key(core)[0] == 0 else None
            selection = {'winner_spec_hash': identity if verdict and verdict['feasible'] else None,
                         'augmented_objective': (verdict or {}).get('augmented_objective'), 'finalists': int(bool(verdict and verdict['feasible'])),
                         'checkpoints': int(verdict is not None), 'rule': 'constant-free candidate: core feasibility then one selection checkpoint'}
            if verdict and verdict['feasible']:
                full = self._evaluate(spec, required, 'development')
                self._record_member(spec, island, full, origin + ':selected',
                                    extra={**base_extra, 'selection': selection, 'components': self._components(full),
                                           'augmented_objective': verdict['augmented_objective'], 'core_objective': verdict['core_objective'],
                                           'tuning': {'trials': 0, 'note': 'no free constants'}})
            else:
                stored = self._selection_results.get(identity)
                result = ({k: v for k, v in stored['merged'].items() if k != 'baselines'} if stored else
                          {'status': core.get('status'), 'case_results': {}, 'assessment': {**(core.get('assessment') or {}), 'feasible': False}})
                result['assessment'] = {**result['assessment'], 'feasible': False}
                self._record_member(spec, island, result, origin + (':selection_failed' if stored else ':no_feasible_core_point'),
                                    extra={**base_extra, 'selection': selection, 'verdict_scope': 'selection_stage_only' if stored else 'core_stage_only',
                                           'components': (stored or {}).get('verdict', {}).get('components'), 'tuning': {'trials': 0, 'note': 'no free constants'}})
            self.state['completed_candidates'].append(transaction)
            self.save()
            return
        fit = tune(spec, self._tuning_evaluate, self._gate, bounds=self._bounds(spec), trials=plan['trials'],
                   min_trials=self.cfg.tuning_min_trials if plan['trials'] else None, patience=self.cfg.tuning_patience,
                   starts=self.cfg.tuning_starts, mode=self.cfg.tuning_mode, trial_workers=self.cfg.trial_workers,
                   seed=int(_hash({'seed': self.cfg.seed, 'spec': identity})[:8], 16) % (2**31 - 1),
                   on_trial=self._on_trial_v3, step_policy=self.cfg.step_policy,
                   parameterisation='reused' if metadata.get('reuse_verified') else 'fresh',
                   parameter_roles=roles, checkpoint=self._selection_checkpoint,
                   max_checkpoints=self.cfg.max_checkpoints_per_candidate, resample_draws=RESAMPLE_DRAWS)
        summary = self._tuning_summary(fit, plan)
        self.event('tuning_completed', spec_hash=identity, mode=fit.mode, cma_updates=fit.cma_updates,
                   post_update_trials=fit.post_update_trials, evaluated=fit.evaluated, rejected=fit.rejected,
                   allocated_trials=plan['trials'], depth_decisions=fit.depth_decisions, stop_reasons=fit.stop_reasons,
                   resampled_gate_rejections=fit.resampled, v3=summary)
        extra = {**base_extra, 'tuning': summary, 'selection': fit.selection}
        if fit.best is None:
            # No checkpoint-verified feasible finalist: the candidate gets no full suite. Its selection-stage
            # evidence (or its core-stage evidence) is retained as a diagnostic member, never as a verdict.
            last = None
            for entry in reversed(fit.checkpoints):
                if entry.get('spec_hash') in self._selection_results:
                    last = entry; break
            if last is not None:
                stored = self._selection_results[last['spec_hash']]
                point = replace(spec, constants=tuple(last['constants']))
                result = {k: v for k, v in stored['merged'].items() if k != 'baselines'}
                result['assessment'] = {**result['assessment'], 'feasible': False}
                self._record_member(point, island, result, origin + ':selection_failed',
                                    extra={**extra, 'verdict_scope': 'selection_stage_only',
                                           'components': stored['verdict']['components'],
                                           'augmented_objective': stored['verdict']['augmented_objective'],
                                           'core_objective': stored['verdict']['core_objective']})
            else:
                best = fit.best_result if fit.best_result is not None else {}
                result = {'status': best.get('status', 'unqualified'), 'case_results': {},
                          'assessment': {**(best.get('assessment') or {}), 'feasible': False,
                                         'reasons': list((best.get('assessment') or {}).get('reasons', [])) + ['no feasible core tuning point']}}
                self._record_member(spec, island, result, origin + ':no_feasible_core_point',
                                    extra={**extra, 'verdict_scope': 'core_stage_only'})
        else:
            full = self._evaluate(fit.best, required, 'development')
            winner = fit.selection.get('winner_spec_hash')
            stored = self._selection_results.get(winner) or {}
            self._record_member(fit.best, island, full, origin + ':selected',
                                extra={**extra, 'components': self._components(full),
                                       'augmented_objective': fit.selection.get('augmented_objective'),
                                       'core_objective': (stored.get('verdict') or {}).get('core_objective')})
        self.state['completed_candidates'].append(transaction)
        self.save()

    def _v3_records(self):
        """Evidence records of every V3 member (the same schema as the v2 archive)."""
        from . import evidence
        terms = self.state.setdefault('v3_terms', {})
        term_ids = {text: key for key, text in terms.items()}
        records = []
        members = sorted(self.state['members'].values(), key=lambda m: (m.get('generation') or 0, m.get('island') or '', m['spec_hash']))
        for index, member in enumerate(members, 1):
            records.append(evidence.candidate_record_from_member(
                member, campaign=self.root.parent.name, index=index, terms=terms, term_ids=term_ids,
                hypothesis=member.get('hypothesis'), parameter_roles=member.get('parameter_roles'),
                tuning=self._tuning_record(member), selection=member.get('selection'),
                components=evidence.components_summary(member.get('components')), prefix='X'))
        return records

    @staticmethod
    def _tuning_record(member):
        t = member.get('tuning')
        if not isinstance(t, Mapping):
            return None
        start = t.get('start_key') or []
        best = t.get('best_core_key') or []
        checkpoints = t.get('checkpoints') or []
        broken = {}
        for c in checkpoints:
            if not c.get('feasible'):
                for reason in c.get('reasons') or []:
                    key = str(reason).split(':', 1)[-1].strip()[:60]
                    broken[key] = {'trials': broken.get(key, {}).get('trials', 0) + 1, 'max_usage': None}
        return {'trials': t.get('trials'), 'evaluated': t.get('evaluated'), 'gate_rejected': t.get('gate_rejected'),
                'feasible_points': None,
                'start_core_objective': start[2] if len(start) > 2 and start[0] == 0 else None,
                'start_feasible': (start[0] == 0) if start else None,
                'best_feasible_core_objective': best[2] if len(best) > 2 and best[0] == 0 else None,
                'best_feasible_constants': None, 'best_overall_core_objective': None,
                'tuning_improved_start': t.get('tuning_improved_start'),
                'better_but_infeasible': sum(1 for c in checkpoints if not c.get('feasible')),
                'constraints_broken_by_better_trials': broken, 'breach_detail_available': True, 'rows': []}

    def _benchmark_reassessment(self):
        """Evaluate the archived v2 incumbent once under the V3 protocol (a comparable benchmark
        line, never an active parent or population member)."""
        if not self.cfg.benchmark_reassessment or self.state.get('benchmark_v3'):
            return
        spec = from_json(json.dumps(self.cfg.benchmark['spec']))
        required = list(dict.fromkeys((*self.cfg.development_ids, *self.cfg.protection_ids)))
        full = self._evaluate(spec, required, 'development')
        assessment = full.get('assessment') or {}
        feasible = _eligible(full, required)
        self.state['benchmark_v3'] = {
            'spec_hash': spec_hash(spec), 'name': spec.name, 'feasible': feasible, 'status': full.get('status'),
            'objective': _numeric(assessment.get('objective'), None), 'family_errors': assessment.get('family_errors'),
            'case_errors': {cid: _numeric(c.get('normalized_error'), None) for cid, c in (assessment.get('cases') or {}).items() if isinstance(c, Mapping)},
            'reasons': [str(r)[:160] for r in assessment.get('reasons') or []][:8],
            'case_statuses': {cid: (c.get('status') if isinstance(c, Mapping) else None) for cid, c in (full.get('case_results') or {}).items()},
            'components': self._components(full), 'scope': 'archived v2 incumbent re-assessed under the V3 physical protocol; benchmark only'}
        self.event('benchmark_reassessed', **{k: v for k, v in self.state['benchmark_v3'].items() if k != 'components'})
        self.save()

    def _context_v3(self, island):
        from . import evidence
        records = self._v3_records()
        valid = [self.state['members'][h] for h in self.state['feasible'] if h in self.state['members']]
        valid = [m for m in valid if _numeric((m.get('assessment') or {}).get('objective')) < math.inf]
        best = min(valid, key=lambda m: m['assessment']['objective']) if valid else None
        archived = dict(self.cfg.benchmark) if isinstance(self.cfg.benchmark, Mapping) else {}
        archived.pop('spec', None)
        reassessed = self.state.get('benchmark_v3')
        frontier = {
            'best_v3_valid': None if best is None else {
                'name': best['spec'].get('name'), 'spec_hash': best['spec_hash'], 'generation': best.get('generation'),
                'objective': round(best['assessment']['objective'], 4),
                'family_errors': {k: round(v, 4) for k, v in (best['assessment'].get('family_errors') or {}).items() if _numeric(v) < math.inf}},
            'v3_valid_count': len(valid), 'v3_assessed_count': len(self.state['members']),
            'note': 'A V3 candidate sets a new frontier only when its full-development objective, under the V3 protocol, '
                    'beats the comparable benchmark line below; core and augmented scores are never frontier moves.'}
        benchmark = {'archived_v2_incumbent': archived,
                     'v3_reassessment': None if reassessed is None else {k: v for k, v in reassessed.items() if k not in ('components',)},
                     'comparable_benchmark_objective': (reassessed or {}).get('objective') if reassessed and reassessed.get('feasible') else archived.get('objective'),
                     'comparable_benchmark_scope': ('v3 protocol re-assessment' if reassessed and reassessed.get('feasible') else
                                                    'v2 protocol verdict (V3 re-assessment ' + ('not feasible' if reassessed else 'pending') + ')')}
        staging = {
            'core_fitting_cases': list(self.cfg.calibration_ids), 'concurrent_tuning_protection': list(self.cfg.tuning_protection_ids),
            'selection_constraint_cases': list(self.cfg.selection_constraint_ids), 'selection_accuracy_cases': list(self.cfg.selection_accuracy_ids),
            'full_development_cases': list(self.cfg.development_ids), 'protection_cases': list(self.cfg.protection_ids),
            'scores': {'core': 'family mean over the core fitting cases (CMA objective); constraints: calibration cases and tuning protection',
                       'augmented': 'family mean over core fitting cases + selection accuracy cases (extra duct case averaged inside the duct family); '
                                    'selection constraint cases are feasibility only; NACA protection only; ranks finalists',
                       'full_development': 'official 9-family objective on the full suite; the only frontier score'},
            'checkpoint_policy': self.cfg.checkpoint_policy, 'max_checkpoints_per_candidate': self.cfg.max_checkpoints_per_candidate,
            'tuning': {'trials': self.cfg.tuning_trials, 'min_trials': self.cfg.tuning_min_trials, 'patience': self.cfg.tuning_patience,
                       'starts': self.cfg.tuning_starts, 'step_policy': self.cfg.step_policy, 'mode': self.cfg.tuning_mode},
            'calibration_budget_case_minutes': self.cfg.calibration_budget_case_minutes,
            'case_cost_minutes': dict(self.cfg.case_cost_minutes) if isinstance(self.cfg.case_cost_minutes, Mapping) else {},
            'parameter_bounds': _clean(asdict(self.cfg)['parameter_bounds'])}
        interface = dict(self.cfg.interface)
        if not interface:
            from .v3 import interface_description
            interface = interface_description(self.cfg.allowed_variables)
        return {'mode': 'v3', 'protocol_hash': self.cfg.protocol_hash, 'island': island,
                'generation': self.state['generation'] + 1,
                'evidence': dict(self.cfg.evidence), 'v3_records': records, 'v3_terms': dict(self.state.get('v3_terms', {})),
                'frontier': frontier, 'benchmark': benchmark, 'interface': interface, 'staging': staging}

    def _candidate(self, spec, island, origin):
        if self.cfg.mode == 'v3':
            return self._candidate_v3(spec, island, origin)
        transaction = _hash({'model': spec_hash(spec), 'island': island})
        if transaction in self.state['completed_candidates']: return
        trial_budget = self.cfg.tuning_trials if origin != 'seed' or self.cfg.tune_seeds else 0
        self._check_deadline()
        if trial_budget and spec.used_constants() and self.cfg.adaptive_trial_floor:
            trial_budget = max(trial_budget, self.cfg.tuning_starts *
                               (cma_population_size(len(spec.used_constants())) + 2))
        validate_tuning_budget(len(spec.used_constants()), trial_budget,
                               self.cfg.tuning_starts, self.cfg.tuning_mode)
        if not self._gate(spec)['passed']:
            self.state['completed_candidates'].append(transaction); self.save(); return
        # Keep the original independent of the tuned candidate.
        self._evaluate(spec, self.cfg.calibration_ids, 'calibration')
        fit = tune(spec, self._tuning_evaluate,
                   self._gate, bounds=self._bounds(spec), trials=trial_budget,
                   min_trials=self.cfg.tuning_min_trials if trial_budget else None,
                   patience=self.cfg.tuning_patience,
                   starts=self.cfg.tuning_starts, mode=self.cfg.tuning_mode, trial_workers=self.cfg.trial_workers,
                   seed=int(_hash({'seed': self.cfg.seed, 'spec': spec_hash(spec)})[:8], 16) % (2**31 - 1),
                   on_trial=lambda record: self.event('tuning_trial', spec_hash=record['spec_hash'],
                       constants=record['constants'], gate=record['gate'], rank=record['rank']))
        self.event('tuning_completed', spec_hash=spec_hash(spec), mode=fit.mode,
                   cma_updates=fit.cma_updates, post_update_trials=fit.post_update_trials,
                   evaluated=fit.evaluated, rejected=fit.rejected, allocated_trials=trial_budget,
                   depth_decisions=fit.depth_decisions, stop_reasons=fit.stop_reasons)
        forms = [(spec, origin + ':original')]
        if fit.best is not None and spec_hash(fit.best) != spec_hash(spec):
            forms.append((fit.best, origin + ':tuned'))
        if self.cfg.promotion_forms == 'selected':
            forms = [(fit.best if fit.best is not None else spec, origin + ':selected')]
        required = list(dict.fromkeys((*self.cfg.development_ids, *self.cfg.protection_ids)))
        for candidate, label in forms:
            if self.cfg.protection_ids and self.cfg.promotion_forms != 'selected':
                sentinel = self._evaluate(candidate, self.cfg.protection_ids, 'protection')
                # A failed physical run cannot be repaired by averaging. A finite
                # constraint violation remains diagnostic, and full evaluation
                # is retained to learn the actual cross-family tradeoff.
                if sentinel['status'] == 'failed':
                    rejected = {'status': 'failed', 'assessment': sentinel['assessment'],
                                'case_results': sentinel.get('case_results', {})}
                    self._record_member(candidate, island, rejected, label)
                    continue
            full = self._evaluate(candidate, required, 'development')
            self._record_member(candidate, island, full, label)
        self.state['completed_candidates'].append(transaction)
        self.save()

    def _context(self, island):
        if self.cfg.mode == 'v3':
            return self._context_v3(island)

        def number(value):
            return round(float(value), 4) if isinstance(value, (int, float)) and math.isfinite(value) else None

        def summary(member):
            # A compact digest: every case error and each failed constraint,
            # without the full per-observable metric records.
            assessment = member.get('assessment') or {}
            cases = {cid: case for cid, case in (assessment.get('cases') or {}).items() if isinstance(case, Mapping)}
            failures = {}
            for cid, case in cases.items():
                entry = {}
                admission = case.get('candidate_admission')
                if isinstance(admission, Mapping) and admission.get('admitted') is False:
                    entry['admission'] = [str(reason)[:200] for reason in list(admission.get('reasons') or [])[:2]]
                for name, metric in (case.get('metrics') or {}).items():
                    if isinstance(metric, Mapping) and (metric.get('passed') is False or
                                                        _numeric(metric.get('violation'), 0.) > 0):
                        change, allowance = metric.get('upper_change', metric.get('change')), metric.get('allowance')
                        ratio = (abs(change)/allowance if isinstance(change, (int, float)) and
                                 isinstance(allowance, (int, float)) and allowance > 0 else None)
                        entry.setdefault('exceeded_change_over_allowance', {})[name] = number(ratio)
                if entry:
                    failures[cid] = entry
            return {'spec_hash': member['spec_hash'], 'spec': member['spec'], 'island': member.get('island'),
                    'origin': member.get('origin'), 'feasible': member['feasible'], 'complexity': member['complexity'],
                    'objective': number(assessment.get('objective')),
                    'max_violation': number(assessment.get('max_violation')),
                    'family_errors': {key: number(value) for key, value in (assessment.get('family_errors') or {}).items()},
                    'case_normalized_errors': {cid: number(case.get('normalized_error')) for cid, case in cases.items()},
                    'constraint_failures': failures,
                    'reasons': [str(reason)[:200] for reason in list(assessment.get('reasons') or [])[:12]],
                    'non_complete_case_statuses': {key: value for key, value in (member.get('case_statuses') or {}).items()
                                                   if value != 'complete'}}

        population = [self.state['members'][identity]
                      for identity in self.state['populations'][self._population_key(island)]]
        listed = {member['spec_hash'] for member in population}

        def examples(identities):
            return [summary(self.state['members'][identity]) if identity not in listed
                    else {'spec_hash': identity, 'listed_in': 'population'} for identity in identities[:8]]

        last = self.state['generation']
        def origin_generation(member):
            text = str(member.get('origin') or '')
            try: return int(text.split('generation_', 1)[1].split(':', 1)[0].split('.', 1)[0]) if text.startswith('generation_') else None
            except (IndexError, ValueError): return None
        recent = [m for m in self.state['members'].values()
                  if not m['feasible'] and (origin_generation(m) or -1) >= last - 1]
        recent.sort(key=lambda m: (m.get('island') != island, _numeric((m.get('assessment') or {}).get('objective'))))
        failure_feedback = {
            'recent_failed_candidates': [summary(m) for m in recent[:10]],
            'recent_failed_candidates_note': ('Every candidate that FAILED in the last two generations, whatever the size of '
                'its violation; this island\'s own failures are listed first. exceeded_change_over_allowance > 1 names each broken constraint.'),
            'tuning_limits_last_generation': self._tuning_limits(last)}
        discovery = island in DISCOVERY_ISLANDS
        own_feasible = [identity for identity in self.state['feasible']
                        if (self.state['members'].get(identity) or {}).get('island') == island]
        scoped = {'frontier': self._frontier(island if discovery else None),
                  'feasible_examples': examples(own_feasible if discovery else self.state['feasible'])}
        if not discovery:
            scoped['discovery_island_leaders'] = self._island_leaders()
        return {'protocol_hash': self.cfg.protocol_hash, 'profile': ISLAND_PROFILES[island],
            **{key: value for key, value in scoped.items() if key == 'frontier'},
            'failure_summary': self._failure_summary(), **failure_feedback,
            'shared_obligations': 'same grammar, all required cases, component constraints and admission for the declared use; island roles are guidance',
            'airfoil_control_scope': 'Required NACA controls constrain lift, drag, Cp and Cf before population admission and at confirmation. Scoped surface admission does not imply full-field convergence and contributes no accuracy/gain objective.',
            'objective_scale': '1.0 equals matched SST on the same cases and lower is better. Feasible requires every required case admitted and every preservation constraint met; exceeded_change_over_allowance above 1 names the constraints to repair.',
            'calibration_ids': list(self.cfg.calibration_ids), 'protection_ids': list(self.cfg.protection_ids),
            'development_ids': list(self.cfg.development_ids),
            'tuning_protection_ids': list(self.cfg.tuning_protection_ids or self.cfg.protection_ids),
            'population': [summary(member) for member in population],
            **{key: value for key, value in scoped.items() if key != 'frontier'},
            'diagnostic_examples': examples(self.state['diagnostic']),
            'tuning_trials': self.cfg.tuning_trials, 'tuning_starts': self.cfg.tuning_starts,
            'tuning_mode': self.cfg.tuning_mode,
            'tuning_protection_mode': self.cfg.tuning_protection_mode,
            'tuning_protection_parallel': self.cfg.tuning_protection_parallel,
            'tuning_min_trials': self.cfg.tuning_min_trials,
            'tuning_patience': self.cfg.tuning_patience,
            'promotion_forms': self.cfg.promotion_forms,
            'adaptive_trial_floor': self.cfg.adaptive_trial_floor,
            'parameter_bounds': _clean(asdict(self.cfg)['parameter_bounds'])}

    FRONTIER_MOVED, FRONTIER_UNIMPROVED = 0.85, 0.995

    def _frontier(self, island=None):
        """The feasible frontier a proposal must move: best valid objective, its family ratios, and
        where the remaining opportunity lies. Rendered at the top of the proposer prompt. With an island,
        the frontier is that island's own lineage and the campaign-wide best is a reference line only."""
        def number(value):
            return round(float(value), 4) if isinstance(value, (int, float)) and math.isfinite(value) else None
        members = [self.state['members'][identity] for identity in self.state['feasible']
                   if identity in self.state['members']]
        members = [m for m in members if number((m.get('assessment') or {}).get('objective')) is not None]
        reference = None
        if island is not None and members:
            best_all = min(members, key=lambda m: m['assessment']['objective'])
            reference = {'objective': number(best_all['assessment']['objective']),
                         'name': best_all['spec'].get('name'), 'island': best_all.get('island')}
            members = [m for m in members if m.get('island') == island]
        mandate = ('A successful proposal must plausibly move this ISLAND FRONTIER with a mechanism from this '
                   "island's area, not merely survive the gates or copy another island's structure."
                   if island is not None else
                   'A successful proposal must plausibly move this frontier, not merely survive the gates.')
        if not members:
            return {'scope': island, 'global_reference': reference,
                    'best_valid_objective': None, 'best_form': None, 'family_ratios_vs_sst': {},
                    'specialist_frontier': {}, 'protected_families': [],
                    'opportunities': ['No valid form exists yet in this scope; every family is open.'],
                    'mandate': mandate}
        best = min(members, key=lambda m: m['assessment']['objective'])
        ratios = {family: number(value) for family, value in
                  sorted((best['assessment'].get('family_errors') or {}).items(), key=lambda kv: _numeric(kv[1]))}
        cases = best['assessment'].get('cases') or {}
        protected = sorted({str(cases[cid].get('family')) for cid in self.cfg.protection_ids
                            if isinstance(cases.get(cid), Mapping) and cases[cid].get('family')})
        specialist = {}
        for member in members:
            for family, value in (member['assessment'].get('family_errors') or {}).items():
                if number(value) is not None and (family not in specialist or value < specialist[family]['ratio']):
                    specialist[family] = {'ratio': number(value), 'form': member['spec'].get('name')}
        open_families = {f: r for f, r in ratios.items() if r is not None and f not in protected}
        unimproved = [f for f, r in open_families.items() if r >= self.FRONTIER_UNIMPROVED]
        modest = [f for f, r in open_families.items() if self.FRONTIER_MOVED <= r < self.FRONTIER_UNIMPROVED]
        moved = [f for f, r in open_families.items() if r < self.FRONTIER_MOVED]
        opportunities = []
        if unimproved:
            opportunities.append('%s: unimproved on the best valid form (ratio >= %.3f).'
                                 % (', '.join(unimproved), self.FRONTIER_UNIMPROVED))
        for family in modest:
            special = specialist.get(family)
            if special and special['ratio'] is not None and special['ratio'] < ratios[family] - 0.01:
                opportunities.append('%s: gains remain modest on the best form (%.3f) while a valid specialist reached %.3f (%s); '
                                     'combining them without losing either is open.'
                                     % (family, ratios[family], special['ratio'], special['form']))
            else:
                opportunities.append('%s: gains remain modest (%.3f).' % (family, ratios[family]))
        if moved:
            opportunities.append('%s: already moved substantially (%s); further gains there are lower priority unless '
                                 'they come at zero cost elsewhere.'
                                 % (', '.join(moved), ', '.join('%.3f' % ratios[f] for f in moved)))
        return {'scope': island, 'global_reference': reference,
                'best_valid_objective': number(best['assessment']['objective']),
                'best_form': {'name': best['spec'].get('name'), 'island': best.get('island'),
                              'spec_hash': best['spec_hash']},
                'family_ratios_vs_sst': ratios, 'specialist_frontier': specialist,
                'protected_families': protected,
                'opportunities': opportunities, 'mandate': mandate}

    def _tuning_limits(self, generation):
        """Per candidate of `generation`: best feasible vs best overall tuning objective, and which constraints the
        better-but-infeasible tuning trials broke. Built from this campaign's retained trial evaluations written
        since the previous generation committed; payloads are read, never modified."""
        import re as _re
        def number(value):
            return round(float(value), 4) if isinstance(value, (int, float)) and math.isfinite(value) else None
        norm = lambda e: _re.sub(r'(\d+)\.0+(?![\d])', r'\1', _re.sub(r'[()\s"]', '', json.dumps(e)))
        structure = lambda spec: (tuple(sorted(norm(t) for t in (spec.get('rsource') or []))),
                                  tuple(sorted(norm(t) for t in (spec.get('bdelta') or []))))
        candidates = {}
        for member in self.state['members'].values():
            if str(member.get('origin') or '').startswith(f'generation_{generation}'):
                candidates[structure(member['spec'])] = member
        if not candidates:
            return {}
        start = 0.
        history = self.root / 'history.jsonl'
        if history.exists():
            for line in history.read_text().splitlines():
                try: record = json.loads(line)
                except ValueError: continue
                if record.get('kind') == 'generation_committed' and record.get('generation') == generation - 1:
                    start = float(record.get('time') or 0.)
        trials = {}
        for key, entry in self.state['evaluations'].items():
            path = self.root / 'evaluations' / f'{key}.json'
            if entry.get('status') != 'complete' or not path.exists() or path.stat().st_mtime < start:
                continue
            try: result = json.loads(path.read_text())['result']
            except (OSError, ValueError, KeyError): continue
            definitions = [r['identity']['spec'] for r in (result.get('case_results') or {}).values()
                           if isinstance(r, Mapping) and isinstance(r.get('identity'), Mapping) and r['identity'].get('spec')]
            if not definitions: continue
            shape = structure(definitions[0])
            if shape not in candidates: continue
            trial = trials.setdefault((shape, tuple(round(float(c), 9) for c in definitions[0].get('constants') or [])),
                                      {'objective': None, 'breaches': {}})
            assessment = result.get('assessment') or {}
            if 'case_0p8' in (assessment.get('cases') or {}) or entry['case_ids'] == list(self.cfg.calibration_ids):
                trial['objective'] = number(assessment.get('objective'))
            for cid, case in (assessment.get('cases') or {}).items():
                for name, metric in ((case or {}).get('metrics') or {}).items():
                    if isinstance(metric, Mapping) and _numeric(metric.get('violation'), 0.) > 0 and metric.get('allowance'):
                        trial['breaches'][f'{cid}/{name}'] = number(abs(_numeric(metric.get('change'), 0.)) / metric['allowance'])
        # Trial objectives and feasibility come from each candidate's own tuning log (it includes trials served
        # from caches that are not in this campaign's evaluations); evaluations supply the breach details.
        logs = {}
        for folder in ('candidate_jobs', 'previous_candidate_jobs'):
            for checkpoint in (self.root / folder).glob('*/checkpoint.json'):
                try: job = json.loads(checkpoint.read_text())
                except (OSError, ValueError): continue
                candidate = ((job.get('identity') or {}).get('candidate') or {})
                if not str(candidate.get('origin') or '').startswith(f'generation_{generation}'): continue
                shape = structure(candidate.get('spec') or {})
                if shape not in candidates: continue
                history_path = checkpoint.parent / 'history.jsonl'
                if not history_path.exists(): continue
                for line in history_path.read_text().splitlines():
                    if '"tuning_trial"' not in line: continue
                    try: record = json.loads(line)
                    except ValueError: continue
                    rank = record.get('rank') or []
                    if len(rank) < 3 or rank[2] is None: continue
                    logs.setdefault(shape, {})[tuple(round(float(c), 9) for c in record.get('constants') or [])] = rank
        limits = {}
        for shape, member in candidates.items():
            if shape in logs:
                rows = []
                for constants, rank in logs[shape].items():
                    detail = trials.get((shape, constants)) or {'breaches': {}}
                    rows.append({'objective': number(rank[2]), 'feasible': rank[0] == 0,
                                 'breaches': {} if rank[0] == 0 else detail['breaches']})
            else:
                rows = [dict(t, feasible=not t['breaches']) for (s_, _), t in trials.items()
                        if s_ == shape and t['objective'] is not None]
            if not rows: continue
            feasible = [t['objective'] for t in rows if t['feasible']]
            best_feasible = min(feasible) if feasible else None
            better = [t for t in rows if not t['feasible'] and (best_feasible is None or t['objective'] < best_feasible)]
            counts = {}
            for t in better:
                for name, ratio in t['breaches'].items():
                    item = counts.setdefault(name, {'trials': 0, 'max_change_over_allowance': 0.})
                    item['trials'] += 1; item['max_change_over_allowance'] = max(item['max_change_over_allowance'], ratio or 0.)
            limits[(member.get('spec') or {}).get('name') or member['spec_hash'][:12]] = {
                'island': member.get('island'), 'tuning_trials_seen': len(rows),
                'best_feasible_objective': best_feasible,
                'best_overall_objective': min(t['objective'] for t in rows),
                'better_but_infeasible_trials': len(better),
                'constraints_broken_by_better_trials': dict(sorted(counts.items(), key=lambda kv: -kv[1]['trials'])[:6])}
        return limits

    def _island_leaders(self):
        """Best feasible member of each discovery island, for the combination island."""
        def number(value):
            return round(float(value), 4) if isinstance(value, (int, float)) and math.isfinite(value) else None
        leaders = {}
        for identity in self.state['feasible']:
            member = self.state['members'].get(identity) or {}
            island, objective = member.get('island'), number((member.get('assessment') or {}).get('objective'))
            if island in DISCOVERY_ISLANDS and objective is not None and (
                    island not in leaders or objective < leaders[island]['objective']):
                leaders[island] = {'objective': objective, 'name': (member.get('spec') or {}).get('name'),
                                   'spec': member.get('spec'),
                                   'family_errors': {k: number(v) for k, v in
                                                     ((member.get('assessment') or {}).get('family_errors') or {}).items()}}
        return {island: leaders.get(island) for island in DISCOVERY_ISLANDS}

    def _failure_summary(self):
        """Recent gate rejections collapsed to one representative per failure class, with the
        feasible structures that passed using the same tensors as repair evidence."""
        def number(value):
            return round(float(value), 4) if isinstance(value, (int, float)) and math.isfinite(value) else None
        rejections = self.state.get('recent_gate_rejections', [])
        classes = {}
        rejected_terms = set()
        for rejection in rejections:
            spec = rejection.get('spec') or {}
            structure = _hash({'rsource': spec.get('rsource'), 'bdelta': spec.get('bdelta')})
            rejected_terms |= {term[0] for key in ('rsource', 'bdelta') for term in (spec.get(key) or [])}
            for reason in (rejection.get('reasons') or ['unspecified']):
                kind = str(reason).split(':', 1)[0].strip() or 'unspecified'
                entry = classes.setdefault(kind, {'rejections': 0, 'structures': {}, 'representative': None})
                entry['rejections'] += 1
                entry['structures'].setdefault(structure, spec.get('name'))
                if entry['representative'] is None:
                    entry['representative'] = {'spec': spec, 'reason': str(reason)[:200]}
        passed = []
        for identity in self.state['feasible']:
            member = self.state['members'].get(identity)
            spec = (member or {}).get('spec') or {}
            terms = {term[0] for key in ('rsource', 'bdelta') for term in (spec.get(key) or [])}
            if terms & rejected_terms:
                passed.append({'name': spec.get('name'), 'tensors': sorted(terms),
                               'objective': number((member.get('assessment') or {}).get('objective')),
                               'spec': spec})
        passed.sort(key=lambda item: _numeric(item['objective']))
        return {'classes': {kind: {'rejections': entry['rejections'],
                                   'distinct_structures': len(entry['structures']),
                                   'structure_names': sorted({name for name in entry['structures'].values() if name}),
                                   'representative': entry['representative']}
                            for kind, entry in classes.items()},
                'feasible_structures_sharing_rejected_tensors': passed[:6],
                'note': 'One representative per failure class; rejections counts how often the class recurred. '
                        'The feasible structures show normalizations that passed with the same tensors.'}

    def _proposals(self, island, generation, context):
        with self._state_lock:
            self._check_deadline()
            call_id = f'g{generation:04d}-{island}'
            entry = self.state['calls'].get(call_id)
            if entry and entry.get('status') == 'complete':
                if not entry['specs']:
                    raise ProposalError('recorded provider response has no valid candidates; reconcile before continuing')
                return [from_json(json.dumps(value)) for value in entry['specs']]
            seed = int(_hash({'seed': self.cfg.seed, 'generation': generation, 'island': island})[:8], 16)
            reserve = self.cfg.proposal_token_reserve
            if hasattr(self.proposer, 'reserve_tokens'):
                reserve = max(0, int(self.proposer.reserve_tokens(self.cfg.children_per_island, context, island, generation, seed)))
            if entry is None:
                outstanding = sum(value.get('reserved_tokens', 0) for value in self.state['calls'].values() if value['status'] != 'complete')
                if self.state['counts']['tokens'] + outstanding + reserve > self.cfg.max_tokens:
                    raise BudgetStop('proposal token reservation would exceed the campaign budget')
                self.state['calls'][call_id] = {'status': 'started', 'reserved_tokens': reserve, 'context': context}
                self.save()
        batch = self.proposer.propose(self.cfg.children_per_island, context, island, generation, seed, call_id)
        with self._state_lock:
            if not isinstance(batch, ProposalBatch) or batch.usage_tokens < 0:
                raise ValueError('proposer must return ProposalBatch with accountable usage')
            self.state['counts']['tokens'] += int(batch.usage_tokens)
            self.state['calls'][call_id] = {'status': 'complete', 'usage_tokens': int(batch.usage_tokens),
                'specs': [json.loads(to_json(spec)) for spec in batch.specs], 'metadata': batch.metadata}
            self.state['parameter_types'].update(batch.metadata.get('parameter_types', {}))
            if batch.metadata.get('candidate_metadata'):
                # V3: roles, hypothesis, coupling and verified reuse claims per proposed spec_hash.
                self.state.setdefault('candidate_metadata', {}).update(batch.metadata['candidate_metadata'])
            self.event('proposal', call_id=call_id, usage_tokens=batch.usage_tokens, count=len(batch.specs), metadata=batch.metadata)
            self.save()
            if self.state['counts']['tokens'] > self.cfg.max_tokens:
                raise BudgetStop('actual provider usage exceeded its reservation; no more work will launch')
            if not batch.specs:
                raise ProposalError('provider returned no valid candidates; discovery is paused, not completed')
            return batch.specs

    def _migrate(self):
        if self.cfg.population_count == 1:
            return
        old = {key: list(self.state['populations'][key]) for key in ISLANDS}
        # Discovery islands keep their own lineages; the combination island receives each discovery leader.
        candidates = [self.state['members'][identity] for identity in old[COMBINATION_ISLAND]]
        for island in DISCOVERY_ISLANDS:
            if old[island] and old[island][0] not in old[COMBINATION_ISLAND]:
                candidates.append(self.state['members'][old[island][0]])
        self.state['populations'][COMBINATION_ISLAND] = [m['spec_hash'] for m in _select(candidates, self.cfg.population_size)]
        self.event('migration', populations=self.state['populations'])

    def result(self):
        counts = {**self.state['counts'],
            'reserved_tokens_unresolved': sum(entry.get('reserved_tokens', 0) for entry in self.state['calls'].values()
                                              if entry['status'] != 'complete'),
            'reserved_core_hours_unresolved': sum(entry.get('core_reserve', 0.) for entry in self.state['evaluations'].values()
                                                  if entry['status'] == 'started')}
        return SearchResult(self.state['status'], self.state['generation'],
            [self.state['members'][key] for key in self.state['feasible']],
            [self.state['members'][key] for key in self.state['diagnostic']],
            counts, self.state['stop_reason'], str(self.checkpoint), dict(self.state['populations']))

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / '.search.lock').open('a+') as lock:
            try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc: raise RuntimeError('another search controller holds this workspace') from exc
            self._load()
            if self.state['status'] in {'complete', 'stopped_no_feasible', 'stopped_plateau', 'paused_budget'}:
                return self.result()
            self.state.update(status='running', stop_reason=''); self.save()
            try:
                self._sync_spend()
                if self.cfg.mode == 'v3':
                    self._benchmark_reassessment()
                seed_list = [(island, spec) for island in ISLANDS for spec in self.seeds[island]] if self.cfg.evaluate_seeds else []
                if self.cfg.candidate_workers > 1:
                    self._parallel_candidates([(spec, island, 'seed')
                                               for island, spec in seed_list[self.state['seed_index']:]])
                    self.state['seed_index'] = len(seed_list); self.save()
                while self.state['seed_index'] < len(seed_list):
                    island, spec = seed_list[self.state['seed_index']]
                    self._candidate(spec, island, 'seed')
                    self.state['seed_index'] += 1; self.save()
                for generation in range(self.state['generation'] + 1, self.cfg.generations + 1):
                    self._check_deadline()
                    pending = self.state['pending_generation']
                    if pending is None:
                        # Freeze all island contexts at the generation boundary.
                        pending = {'generation': generation, 'contexts': {i: self._context(i) for i in ISLANDS},
                                   'queue': None, 'index': 0}
                        self.state['pending_generation'] = pending; self.save()
                    if pending['queue'] is None:
                        queue = []
                        with ThreadPoolExecutor(max_workers=min(4, self.cfg.candidate_workers),
                                                thread_name_prefix='forge-proposal') as proposal_pool:
                            futures = {}
                            for index, island in enumerate(ISLANDS):
                                futures[island] = proposal_pool.submit(self._proposals, island, generation,
                                                                       pending['contexts'][island])
                                if index == 0 and self.cfg.proposal_stagger_s and len(ISLANDS) > 1:
                                    # Let the first call establish the shared prompt-prefix cache
                                    # before the identical-evidence calls of the other workers.
                                    time.sleep(self.cfg.proposal_stagger_s)
                            for island in ISLANDS:
                                for spec in futures[island].result():
                                    queue.append({'island': island, 'spec': json.loads(to_json(spec))})
                        if self.cfg.mode == 'v3':
                            queue = self._repair_round(generation, queue)
                        pending['queue'] = queue; self.save()
                    if self.cfg.candidate_workers > 1:
                        self._parallel_candidates([(from_json(json.dumps(item['spec'])), item['island'],
                            f'generation_{generation}') for item in pending['queue'][pending['index']:]])
                        pending['index'] = len(pending['queue']); self.save()
                    while pending['index'] < len(pending['queue']):
                        item = pending['queue'][pending['index']]
                        self._candidate(from_json(json.dumps(item['spec'])), item['island'], f'generation_{generation}')
                        pending['index'] += 1; self.save()
                    self.state['generation'] = generation
                    self.state['pending_generation'] = None
                    best = min((_numeric(self.state['members'][key]['assessment'].get('objective'))
                                for key in self.state['feasible']), default=math.inf)
                    previous = self.state['best_objective']
                    if math.isfinite(best) and (previous is None or previous - best >= self.cfg.minimum_progress):
                        self.state['best_objective'] = best; self.state['plateau'] = 0
                    else: self.state['plateau'] += 1
                    if self.cfg.migration_every and generation % self.cfg.migration_every == 0: self._migrate()
                    self.event('generation_committed', best_objective=best, feasible_count=len(self.state['feasible']))
                    self.save()
                    if self.state['plateau'] >= self.cfg.plateau_generations:
                        self.state.update(status='stopped_plateau' if self.state['feasible'] else 'stopped_no_feasible',
                                          stop_reason='predeclared feasible-objective plateau reached')
                        break
                else:
                    self.state.update(status='complete' if self.state['feasible'] else 'stopped_no_feasible',
                        stop_reason='requested generations completed; candidates remain admitted for development only' if self.state['feasible']
                        else 'requested generations completed without a full-suite feasible candidate')
            except (BudgetStop, BudgetExhausted) as exc:
                self.state.update(status='paused_budget', stop_reason=str(exc))
            except ProposalError as exc:
                self.state.update(status='paused_llm', stop_reason=str(exc))
            except BaseException:
                self.state.update(status='interrupted', stop_reason='controller interrupted; committed evaluations retained')
                self.save(); raise
            self.save()
            result = self.result()
            _atomic(self.root / 'search_result.json', result.as_dict())
            return result


class _CandidateWorker(SearchController):
    """One independently checkpointed fit; shared CFD is brokered by its parent."""
    def __init__(self, parent, spec, island, origin, transaction):
        self.parent = parent
        self.spec, self.island, self.origin, self.transaction = spec, island, origin, transaction
        cfg = replace(parent.cfg, work_root=parent.root/'candidate_jobs'/transaction)
        super().__init__(cfg, parent.evaluate_callback, parent.proposer, self._serialized_gate, parent.seeds)
        self.identity['candidate'] = {'spec': json.loads(to_json(spec)), 'island':island, 'origin':origin,
                                     'parameter_types':parent.state['parameter_types'].get(spec_hash(spec))}
        self.fingerprint = _hash(self.identity)
        self._load()
        self.state['parameter_types'] = copy.deepcopy(parent.state['parameter_types'])
        self.state['generation'] = parent.state['generation']

    def _serialized_gate(self, spec):
        with self.parent._gate_lock:
            with self.parent._state_lock:
                if self.parent._parallel_failure is not None:
                    raise _ParallelCancelled('Another fit stopped; no further gate trials admitted')
            return self.parent.gate_callback(spec)

    def _evaluate(self, spec, case_ids, stage):
        return self.parent._evaluate(spec, case_ids, stage)

    def execute(self):
        try:
            self.state['status'] = 'running'; self.save()
            self._candidate(self.spec, self.island, self.origin)
            self.state['status'] = 'complete'; self.save()
        except BaseException as exc:
            with self.parent._state_lock:
                if self.parent._parallel_failure is None:
                    self.parent._parallel_failure = exc
            self.state['status'] = 'interrupted'; self.save()
            raise


def run_search(config: SearchConfig, evaluate: Callable, proposer, gate: Callable,
               seeds: Mapping[str, Sequence[CandidateSpec]] | None = None) -> SearchResult:
    return SearchController(config, evaluate, proposer, gate, seeds).run()


SearchCfg = SearchConfig
