"""Typed bounded multistart coefficient fitting with a gate before every trial.

CMA-ES is replayed from a deterministic random stream; callers persist/cache
individual evaluations, so an interrupted fit can replay without repeated CFD.
The original candidate is never replaced in-place.

FORGE V3 additions (all optional, legacy behaviour is the default):
  * step policies: the initial CMA step of a signed amplitude is a declared
    fraction of its magnitude (with a nonzero floor) and that of a positive scale
    a declared number of decades, instead of one full amplitude / 0.4 decades;
    ``parameterisation`` distinguishes a fresh vector from a genuinely reused one;
  * parameter roles (amplitude, activation_scale, normalisation_scale, coupled)
    are recorded with the fit and select the step law together with the type;
  * a provisional-winner ``checkpoint`` hook verifies the best feasible trial at
    its EXACT coefficient vector on additional cases before it can become a
    finalist; a failed checkpoint demotes the trial (CMA sees the constraint) and
    the calibrated winner is the finalist with the best augmented objective;
  * trial records keep every component metric of the evaluation in compact form
    (never the multi-megabyte case payloads).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor
import math
from typing import Callable, Mapping, Sequence

import numpy as np
from tedp.spec import CandidateSpec, spec_hash


@dataclass(frozen=True)
class ParameterBound:
    kind: str = 'signed'
    lower: float = -4.0
    upper: float = 4.0

    def __post_init__(self):
        if self.kind not in {'signed', 'positive'}:
            raise ValueError('parameter kind must be signed or positive')
        if not (math.isfinite(self.lower) and math.isfinite(self.upper) and self.lower < self.upper):
            raise ValueError('parameter bounds must be finite and ordered')
        if self.kind == 'positive' and self.lower <= 0:
            raise ValueError('positive parameters require a positive lower bound')

    def encode(self, value: float) -> float:
        value = float(np.clip(value, self.lower, self.upper))
        if self.kind == 'positive':
            return math.log(value / self.lower) / math.log(self.upper / self.lower)
        return (value - self.lower) / (self.upper - self.lower)

    def decode(self, value: float) -> float:
        z = float(np.clip(value, 0., 1.))
        if self.kind == 'positive':
            return self.lower * (self.upper / self.lower) ** z
        return self.lower + z * (self.upper - self.lower)

    def decades(self) -> float:
        """Width of a positive bound in decades (the unit of its normalized coordinate)."""
        return math.log10(self.upper / self.lower)


PARAMETER_ROLES = ('amplitude', 'activation_scale', 'normalisation_scale', 'coupled', 'unspecified')

#: Initial CMA step laws. ``legacy`` is the v2 behaviour (one full amplitude per signed
#: constant, 0.4 decades per positive scale). ``fresh`` is for a newly proposed vector,
#: ``reused`` for a vector the proposer declares as reused from an archived, already
#: useful point; the choice is explicit metadata, never inferred from matching numbers.
STEP_POLICIES = {
    'legacy': {'signed_fraction': 1.0, 'signed_floor': 0.02, 'signed_cap_fraction_of_range': 1.0, 'positive_decades': 0.4},
    'fresh': {'signed_fraction': 0.25, 'signed_floor': 0.02, 'signed_cap_fraction_of_range': 0.125, 'positive_decades': 0.15},
    'reused': {'signed_fraction': 0.10, 'signed_floor': 0.02, 'signed_cap_fraction_of_range': 0.125, 'positive_decades': 0.05},
}


def initial_steps(bounds: Sequence[ParameterBound], active: Sequence[int], center: Sequence[float],
                  policy: str = 'legacy', roles: Sequence[str] | None = None) -> list[dict]:
    """Per-active-parameter initial CMA std in normalized coordinates, with its natural-unit meaning."""
    if policy not in STEP_POLICIES:
        raise ValueError('step policy must be one of ' + ', '.join(STEP_POLICIES))
    law = STEP_POLICIES[policy]
    out = []
    for index, z in zip(active, center):
        bound = bounds[index]
        role = (roles[index] if roles is not None and index < len(roles) else None) or 'unspecified'
        value = bound.decode(float(z))
        if bound.kind == 'signed':
            natural = max(law['signed_fraction'] * abs(value), law['signed_floor'])
            std = min(natural / (bound.upper - bound.lower), law['signed_cap_fraction_of_range'])
            out.append({'index': index, 'kind': 'signed', 'role': role, 'value': value,
                        'std_normalized': std, 'std_natural': std * (bound.upper - bound.lower),
                        'law': f"{law['signed_fraction']:.2f}*|value| floored at {law['signed_floor']}"})
        else:
            std = law['positive_decades'] / bound.decades()
            out.append({'index': index, 'kind': 'positive', 'role': role, 'value': value,
                        'std_normalized': std, 'std_decades': law['positive_decades'],
                        'law': f"{law['positive_decades']:.2f} decades"})
    return out


def default_bounds(spec: CandidateSpec, parameter_types: Sequence[str] | None = None) -> tuple[ParameterBound, ...]:
    """Unknown amplitudes are signed; positive scales must be declared explicitly."""
    kinds = list(parameter_types) if parameter_types is not None else ['signed'] * len(spec.constants)
    if len(kinds) != len(spec.constants):
        raise ValueError('one parameter type is required for every stored coefficient')
    return tuple(ParameterBound('positive', 1e-3, 100.) if kind == 'positive'
                 else ParameterBound(kind, -4., 4.) for kind in kinds)


def gate_result(value) -> dict:
    if isinstance(value, bool):
        return {'passed': value, 'reasons': [] if value else ['gate rejected']}
    if not isinstance(value, Mapping) or not isinstance(value.get('passed'), (bool, np.bool_)):
        raise ValueError('gate must return a bool or a mapping with boolean passed')
    return {**dict(value), 'passed': bool(value['passed']), 'reasons': list(value.get('reasons', []))}


def assessment_key(result: Mapping) -> tuple:
    """Feasibility dominates accuracy; missing/invalid assessments cannot win."""
    assessment = result.get('assessment') or {}
    status = result.get('status')
    objective = assessment.get('objective')
    violation = assessment.get('max_violation')
    objective = float(objective) if isinstance(objective, (int, float)) and math.isfinite(objective) else math.inf
    violation = float(violation) if isinstance(violation, (int, float)) and math.isfinite(violation) else math.inf
    feasible = status == 'complete' and assessment.get('feasible') is True and violation <= 0. and math.isfinite(objective)
    qualified = status == 'complete'
    return (0 if feasible else 1 if qualified else 2,
            0. if feasible else max(0., violation), objective)


def compact_result(result: Mapping | None) -> dict | None:
    """Retain the assessment (every component metric) and statuses, never case payloads."""
    if result is None:
        return None
    kept = {key: value for key, value in result.items() if key not in ('case_results', 'baselines')}
    cases = result.get('case_results')
    if isinstance(cases, Mapping):
        kept['case_statuses'] = {cid: (case.get('status') if isinstance(case, Mapping) else None) for cid, case in cases.items()}
    return kept


def cma_population_size(dimensions: int) -> int:
    if dimensions < 1:
        raise ValueError('CMA needs at least one active parameter')
    return max(4, 4 + int(3 * math.log(dimensions)))


def validate_tuning_budget(dimensions: int, trials: int, starts: int, mode: str = 'adaptive') -> int:
    """Require an update and a subsequent draw per start for adaptive fitting.

    The center consumes one trial, the first complete population supplies an
    update, and at least one further trial observes the adapted distribution.
    Explicit smoke mode supports bounded integration tests without claiming
    an adaptive fit. Zero trials means tuning was disabled.
    """
    if mode not in {'adaptive', 'smoke'}:
        raise ValueError('tuning mode must be adaptive or smoke')
    if trials < 0 or starts < 1:
        raise ValueError('trials must be nonnegative and starts positive')
    if not dimensions or not trials:
        return 0
    minimum = starts * (cma_population_size(dimensions) + 2)
    if mode == 'adaptive' and trials < minimum:
        raise ValueError(f'Adaptive tuning of {dimensions} active parameters across {starts} starts '
                         f'needs at least {minimum} trials; received {trials}. '
                         'Use explicit smoke mode only for an integration/sampling pilot.')
    return minimum


@dataclass
class TuningResult:
    best: CandidateSpec | None
    best_result: dict | None
    trials: list[dict]
    evaluated: int
    rejected: int
    mode: str = 'adaptive'
    cma_updates: list[int] = field(default_factory=list)
    post_update_trials: list[int] = field(default_factory=list)
    depth_decisions: list[dict] = field(default_factory=list)
    stop_reasons: list[str] = field(default_factory=list)
    # V3 records
    step_policy: str = 'legacy'
    parameterisation: str = 'fresh'
    initial_steps: list[dict] = field(default_factory=list)
    effective_steps: list[dict] = field(default_factory=list)
    start_key: list | None = None
    best_core_key: list | None = None
    tuning_improved_start: bool | None = None
    finalists: list[dict] = field(default_factory=list)
    checkpoints: list[dict] = field(default_factory=list)
    selection: dict = field(default_factory=dict)
    resampled: int = 0
    resample_rejections: list[dict] = field(default_factory=list)


def tune(
    spec: CandidateSpec,
    evaluate: Callable[[CandidateSpec], dict],
    gate: Callable[[CandidateSpec], dict],
    *, bounds: Sequence[ParameterBound] | None = None,
    trials: int = 72,
    starts: int = 3,
    seed: int = 0,
    mode: str = 'adaptive',
    on_trial: Callable[[dict], None] | None = None,
    trial_workers: int = 1,
    min_trials: int | None = None,
    patience: int = 3,
    step_policy: str = 'legacy',
    parameterisation: str = 'fresh',
    parameter_roles: Sequence[str] | None = None,
    checkpoint: Callable[[CandidateSpec, dict, dict], dict] | None = None,
    max_checkpoints: int = 0,
    keep_full_results: bool = False,
    resample_draws: int = 0,
) -> TuningResult:
    """Fit by feasible-first rank CMA-ES with typed sign exploration.

    The number of gate trials is bounded, including rejected vectors. A
    monotone rank transform feeds CMA-ES; no scalar penalty can promote an
    infeasible model over a feasible one. Final full-suite promotion belongs
    to the controller, not to this calibration-only fit.

    With a ``checkpoint`` hook (V3), a trial that becomes the best feasible core point
    is verified at its exact vector before it can be a finalist; the winner is the
    finalist with the lowest augmented objective (``checkpoint`` returns
    ``{'feasible', 'augmented_objective', 'assessment', 'status', 'reasons'}``).

    ``resample_draws`` > 0 handles the free gate as a sampling constraint: a CMA draw the gate
    rejects is redrawn from the same distribution (at most that many times per slot), so rejected
    draws are recorded but do not consume the candidate's bounded CFD trial slots.
    """
    if type(resample_draws) is not int or not 0 <= resample_draws <= 32:
        raise ValueError('resample_draws must be an integer from 0 to 32')
    if type(trial_workers) is not int or not 1 <= trial_workers <= 16:
        raise ValueError('trial_workers must be an integer from 1 to 16')
    if min_trials is not None and (type(min_trials) is not int or
            not 1 <= min_trials <= trials or mode != 'adaptive'):
        raise ValueError('min_trials requires adaptive mode and must be within trials')
    if type(patience) is not int or patience < 1:
        raise ValueError('patience must be a positive integer')
    if step_policy not in STEP_POLICIES:
        raise ValueError('step_policy must be one of ' + ', '.join(STEP_POLICIES))
    if parameterisation not in ('fresh', 'reused'):
        raise ValueError('parameterisation must be fresh or reused')
    if checkpoint is not None and (type(max_checkpoints) is not int or max_checkpoints < 1):
        raise ValueError('a checkpoint hook needs a positive max_checkpoints budget')
    if parameter_roles is not None:
        if len(parameter_roles) != len(spec.constants) or any(r not in PARAMETER_ROLES for r in parameter_roles):
            raise ValueError('one declared parameter role per stored coefficient is required')
    spec.validate(search_policy=True)
    active = sorted(spec.used_constants())
    validate_tuning_budget(len(active), trials, starts, mode)
    if not spec.constants or trials == 0:
        return TuningResult(None, None, [], 0, 0, mode='disabled', step_policy=step_policy, parameterisation=parameterisation)
    bounds = tuple(bounds or default_bounds(spec))
    if len(bounds) != len(spec.constants):
        raise ValueError('parameter bounds do not match candidate constants')
    if not active:
        return TuningResult(None, None, [], 0, 0, mode='disabled', step_policy=step_policy, parameterisation=parameterisation)
    from cma import CMAEvolutionStrategy
    rng = np.random.RandomState(seed)
    origin = np.array([bounds[i].encode(spec.constants[i]) for i in active])
    centers = [origin]
    if starts > 1:
        centers.append(np.array([1.-v if bounds[i].kind == 'signed' else v
                                 for i, v in zip(active, origin)]))
    while len(centers) < starts:
        centers.append(rng.uniform(.1, .9, len(active)))
    records: list[dict] = []
    best_spec = None
    best_result = None
    best_key = (math.inf, math.inf, math.inf)
    anchor = origin.copy()
    evaluated = rejected = 0
    resampled, resample_rejections = 0, []
    finalists: list[dict] = []
    checkpoints: list[dict] = []
    verified: set[str] = set()
    start_key = None

    def prepare(vector, *, original=False):
        constants = list(spec.constants)
        for index, z in zip(active, vector):
            # Preserve an admissible original exactly; encode/decode roundoff
            # must not invent a second physical experiment for its center.
            if not original or not bounds[index].lower <= constants[index] <= bounds[index].upper:
                constants[index] = bounds[index].decode(float(z))
        candidate = replace(spec, constants=tuple(constants))
        try:
            candidate.validate(search_policy=True)
            passed = gate_result(gate(candidate))
        except (ValueError, ArithmeticError) as exc:
            passed = {'passed': False, 'reasons': [f'{type(exc).__name__}: {exc}']}
        return candidate, constants, passed

    def record(prepared, result, start_index):
        nonlocal best_spec, best_result, best_key, evaluated, rejected, anchor
        candidate, constants, passed = prepared
        vector = np.array([bounds[i].encode(constants[i]) for i in active])
        if passed['passed']:
            evaluated += 1
            key = assessment_key(result)
            if key < best_key:
                best_spec, best_result, best_key = candidate, (result if keep_full_results else compact_result(result)), key
                anchor = vector
        else:
            rejected += 1
            key = (3, math.inf, math.inf)
        stored = result if keep_full_results else compact_result(result)
        item = {'index': len(records), 'start': start_index,
                'spec_hash': spec_hash(candidate), 'constants': constants,
                'gate': passed, 'result': stored, 'rank': list(key),
                'bounds': [asdict(bound) for bound in bounds], 'candidate': candidate}
        records.append(item)
        if on_trial:
            on_trial({**item, 'full_result': result})
        return key

    def best_finalist():
        eligible = [f for f in finalists if f['feasible'] and f['augmented_objective'] is not None]
        return min(eligible, key=lambda f: (f['augmented_objective'], f['core_key'])) if eligible else None

    def verify(keys, offset):
        """Provisional-winner checkpoint on the best unverified feasible trial of a batch."""
        if checkpoint is None:
            return keys
        batch = [(keys[j], records[offset + j]) for j in range(len(keys))]
        candidates = [(key, item) for key, item in batch if key[0] == 0 and item['spec_hash'] not in verified]
        if not candidates:
            return keys
        key, item = min(candidates, key=lambda pair: pair[0])
        incumbent = best_finalist()
        if incumbent is not None and not (key < tuple(incumbent['core_key'])):
            return keys
        if len(checkpoints) >= max_checkpoints:
            checkpoints.append({'spec_hash': item['spec_hash'], 'skipped': 'checkpoint budget exhausted'})
            return keys
        verified.add(item['spec_hash'])
        verdict = checkpoint(item['candidate'], item['result'], item)
        entry = {'spec_hash': item['spec_hash'], 'constants': item['constants'], 'core_key': list(key),
                 'feasible': bool(verdict.get('feasible')), 'status': verdict.get('status'),
                 'augmented_objective': verdict.get('augmented_objective'),
                 'reasons': list(verdict.get('reasons') or []), 'trial_index': item['index']}
        checkpoints.append(entry)
        item['checkpoint'] = {k: v for k, v in entry.items() if k != 'constants'}
        if entry['feasible'] and entry['augmented_objective'] is not None:
            finalists.append(entry)
        else:
            violation = verdict.get('max_violation')
            violation = float(violation) if isinstance(violation, (int, float)) and math.isfinite(violation) and violation > 0 else 1.0
            demoted = (1, violation, key[2])
            item['rank'] = list(demoted)
            item['demoted_by_checkpoint'] = True
            position = [j for j in range(len(keys)) if records[offset + j] is item][0]
            keys = list(keys)
            keys[position] = demoted
            if on_trial:
                on_trial({**item, 'full_result': None, 'event': 'checkpoint_demotion'})
        return keys

    def resample(es, vectors, count):
        """Redraw gate-rejected CMA samples from the same distribution (bounded per slot)."""
        nonlocal resampled
        vectors, prepared = list(vectors), []
        for j in range(count):
            item, draws = prepare(vectors[j]), 0
            while not item[2]['passed'] and draws < resample_draws:
                draws += 1
                if len(resample_rejections) < 64:
                    resample_rejections.append({'constants': item[1], 'reasons': list(item[2].get('reasons') or [])[:3]})
                vectors[j] = es.ask(1)[0]
                item = prepare(vectors[j])
            resampled += draws
            prepared.append(item)
        return vectors, prepared

    def trials_batch(vectors, start_index, *, original=False, prepared=None):
        # Gate and checkpoint on the coordinator. Only physical evaluations
        # overlap. Ordered consumption preserves CMA updates and tie-breaking.
        if prepared is None:
            prepared = [prepare(v, original=original) for v in vectors]
        offset = len(records)
        if trial_workers == 1:
            keys = [record(p, evaluate(p[0]) if p[2]['passed'] else None, start_index)
                    for p in prepared]
        else:
            with ThreadPoolExecutor(max_workers=trial_workers, thread_name_prefix='forge-coeff') as pool:
                pending = [pool.submit(evaluate, p[0]) if p[2]['passed'] else None for p in prepared]
                try:
                    keys = [record(p, f.result() if f is not None else None, start_index)
                            for p, f in zip(prepared, pending)]
                finally:
                    for future in pending:
                        if future is not None:
                            future.cancel()
        return verify(keys, offset)

    starts = min(starts, trials)
    allocations = [trials // starts + (index < trials % starts) for index in range(starts)]
    minimum = max(min_trials or trials, starts * (cma_population_size(len(active)) + 2))
    floors = [minimum // starts + (index < minimum % starts) for index in range(starts)]
    decisions, stop_reasons = [], []
    updates, after_update = [0] * starts, [0] * starts
    steps_initial, steps_effective = [], []
    policy = step_policy if parameterisation == 'fresh' or step_policy == 'legacy' else 'reused'
    for start_index, allowance in enumerate(allocations):
        center = centers[start_index]
        progress_key = trials_batch([center], start_index, original=start_index == 0)[0]
        if start_index == 0:
            start_key = list(progress_key)
        stale = 0
        stop_reason = 'trial_cap'
        remaining = allowance - 1
        if remaining <= 0:
            stop_reasons.append(stop_reason)
            continue
        local_rng = np.random.RandomState(seed + 104729 * (start_index + 1))
        population = cma_population_size(len(active))

        def distribution(mean, scale, restart):
            # Signed amplitudes step about a declared fraction of their own magnitude with a
            # nonzero floor; positive scales keep a logarithmic step of a declared number of
            # decades. The legacy policy is one full amplitude / 0.4 decades.
            table = initial_steps(bounds, active, mean, policy, parameter_roles)
            stds = [row['std_normalized'] for row in table]
            if restart == 0:
                steps_initial.append({'start': start_index, 'policy': policy, 'steps': table})
            return CMAEvolutionStrategy([float(z) for z in mean], scale, {
                'bounds': [0., 1.], 'popsize': population, 'seed': seed + start_index + 1 + 7919 * restart,
                'randn': local_rng.randn, 'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
                'maxiter': max(1, allowance), 'CMA_stds': stds,
                # pycma can apply a scalar decoding update before initializing its
                # one-dimensional scaling vector. Bounds still constrain all draws.
                **({'maxstd': float('inf'), 'maxstd_boundrange': float('inf')}
                   if len(active) == 1 else {}),
            })

        restarts = 0
        es = distribution(center, 1., restarts)
        while remaining > 0:
            vectors = es.ask()
            count = min(remaining, len(vectors))
            if updates[start_index]:
                after_update[start_index] += count
            batch_prepared = None
            if resample_draws > 0:
                vectors, batch_prepared = resample(es, vectors, count)
            keys = trials_batch(vectors[:count], start_index, prepared=batch_prepared)
            remaining -= count
            if count == len(vectors):
                order = sorted(range(count), key=lambda index: keys[index])
                ranks = [0.] * count
                previous = None
                rank = 0
                for index in order:
                    if previous is not None and keys[index] != previous:
                        rank += 1
                    ranks[index] = float(rank)
                    previous = keys[index]
                es.tell(vectors, ranks)
                updates[start_index] += 1
                if all(key[0] == 3 for key in keys):
                    # No admissible draw: restart at the best admissible vector with half the step.
                    restarts += 1
                    es = distribution(anchor, max(es.sigma * .5, 1e-3), restarts)
                elif 2 * sum(1 for key in keys if key[0] == 3) >= count:
                    es.sigma = max(es.sigma * .5, .01)
            if checkpoint is not None and len(checkpoints) >= max_checkpoints and best_finalist() is not None:
                # The selection budget is spent: further trials could no longer become finalists.
                stop_reason = 'checkpoint_budget'
                decisions.append({'start': start_index, 'trials': allowance - remaining, 'action': 'stop_checkpoint_budget'})
                break
            if min_trials is not None:
                # Count meaningful progress against the last significant gain,
                # allowing small improvements to accumulate. Infeasible fits
                # earn depth by repairing constraints, not by lowering error
                # while leaving the violated airfoil constraint unchanged.
                current = min([progress_key, *keys])
                progress = current[0] < 2 and (
                    current[0] < progress_key[0] or
                    (current[0] == progress_key[0] == 0 and current[2] < progress_key[2] - 1e-4) or
                    (current[0] == progress_key[0] == 1 and current[1] < progress_key[1] - 1e-3))
                stale = 0 if progress else stale + 1
                if progress:
                    progress_key = current
                used = allowance - remaining
                stop = used >= floors[start_index] and stale >= patience
                decisions.append({'start': start_index, 'trials': used,
                    'significant_progress': progress, 'stagnant_batches': stale,
                    'best_rank': list(best_key), 'action': 'stop_stagnant' if stop else
                    'trial_cap' if remaining == 0 else 'continue'})
                if stop:
                    stop_reason = 'stagnation'
                    break
        try:
            sigma = float(es.sigma)
            scaled = list(np.asarray(es.sigma_vec.scaling if hasattr(es.sigma_vec, 'scaling') else es.sigma_vec, dtype=float).ravel())
        except Exception:  # pragma: no cover - pycma internals
            sigma, scaled = None, []
        steps_effective.append({'start': start_index, 'sigma': sigma, 'sigma_vec': scaled, 'restarts': restarts})
        stop_reasons.append(stop_reason)
    selection = {}
    if checkpoint is not None:
        winner = best_finalist()
        if winner is not None:
            best_spec = next(r['candidate'] for r in records if r['spec_hash'] == winner['spec_hash'])
            best_result = next(r['result'] for r in records if r['spec_hash'] == winner['spec_hash'])
            selection = {'winner_spec_hash': winner['spec_hash'], 'augmented_objective': winner['augmented_objective'],
                         'core_key': winner['core_key'], 'finalists': len([f for f in finalists if f['feasible']]),
                         'checkpoints': len([c for c in checkpoints if 'skipped' not in c]),
                         'rule': 'lowest augmented objective among checkpoint-verified feasible finalists'}
        else:
            best_spec, best_result = None, None
            selection = {'winner_spec_hash': None, 'finalists': 0,
                         'checkpoints': len([c for c in checkpoints if 'skipped' not in c]),
                         'rule': 'no checkpoint-verified feasible finalist; no calibrated winner'}
    improved = None
    if start_key is not None and best_key[0] == 0:
        improved = bool(tuple(start_key)[0] != 0 or best_key[2] < tuple(start_key)[2] - 1e-6)
    for item in records:
        item.pop('candidate', None)
    return TuningResult(best_spec, best_result, records, evaluated, rejected,
                        mode=mode, cma_updates=updates, post_update_trials=after_update,
                        depth_decisions=decisions, stop_reasons=stop_reasons,
                        step_policy=policy, parameterisation=parameterisation,
                        initial_steps=steps_initial, effective_steps=steps_effective,
                        start_key=start_key, best_core_key=list(best_key) if best_key[0] < 3 else None,
                        tuning_improved_start=improved, finalists=finalists, checkpoints=checkpoints,
                        selection=selection, resampled=resampled, resample_rejections=resample_rejections)
