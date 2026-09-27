"""Numerical qualification from retained OpenFOAM fields and full solver history.

Observable accuracy is intentionally absent: a low prediction error cannot
rescue missing numerical evidence. End markers never establish convergence.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import re
import numpy as np

NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
TIME = re.compile(rf'^Time = ({NUMBER})\s*$', re.MULTILINE)
RES = re.compile(rf'Solving for (\w+), Initial residual = ({NUMBER})')
CONT = re.compile(rf'time step continuity errors : sum local = ({NUMBER}), global = ({NUMBER}), cumulative = ({NUMBER})')
FIELDS = ('U', 'p', 'k', 'omega', 'nut')
DEFAULT_TOLS = {'U': 1e-4, 'k': 1e-3, 'omega': 1e-3, 'nut': 1e-3}
DEFAULT_FLOORS = {'U': 1e-12, 'p': 1e-12, 'k': 1e-16, 'omega': 1e-12, 'nut': 1e-16}


def _snapshots(case):
    found = []
    for p in case.iterdir():
        if p.is_dir():
            try:
                t = float(p.name)
            except ValueError:
                continue
            if math.isfinite(t) and all((p / f).is_file() for f in FIELDS):
                found.append((t, p.name))
    return sorted(found)


def _read_fields(case, time, cell_count=None):
    from fluidfoam import readscalar, readvector
    fields = {}
    for field in FIELDS:
        text = (case / str(time) / field).read_text(errors='replace')
        # Some valid compact ASCII dictionaries are misread by fluidfoam's
        # uniform-field parser. Parse internalField only, never boundary values.
        text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
        uniform = re.search(r'internalField\s+uniform\s+([^;]+);', text)
        nonuniform = re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s+(\d+)\s*\((.*?)\)\s*;', text, re.S)
        if uniform:
            values = np.array([float(x) for x in re.findall(NUMBER, uniform.group(1))])
            expected = 3 if field == 'U' else 1
            if len(values) != expected:
                raise ValueError(f'Invalid uniform {field} internalField')
            fields[field] = values.reshape(expected, 1) if field == 'U' else values
        elif nonuniform and re.search(r'format\s+ascii\s*;', text):
            n = int(nonuniform.group(2))
            values = np.array([float(x) for x in re.findall(NUMBER, nonuniform.group(3))])
            expected = n * (3 if field == 'U' else 1)
            if n < 1 or values.size != expected:
                raise ValueError(f'Invalid nonuniform {field} internalField size')
            fields[field] = values.reshape(n, 3).T if field == 'U' else values
        else:
            reader = readvector if field == 'U' else readscalar
            fields[field] = np.asarray(reader(str(case), time, field, verbose=False), dtype=float)
    sizes = [v.shape[-1] for v in fields.values()]
    n = cell_count if cell_count is not None else max(sizes)
    for field, values in fields.items():
        if values.shape[-1] == 1 and n > 1:
            values = np.repeat(values, n, axis=-1)
        if values.shape[-1] != n:
            raise ValueError(f'Inconsistent cell count for {field}')
        fields[field] = values.ravel()
    return fields


def _roundoff_residual_evidence(component, rule, snapshots, prior, final, continuity):
    mapping = {'Ux': ('U', 0), 'Uy': ('U', 1), 'Uz': ('U', 2), 'p': ('p', None)}
    if component not in mapping:
        raise ValueError(f'Unsupported inactive residual component: {component}')
    bounds = {k: float(rule[k]) for k in ('max_abs', 'max_abs_change', 'max_local_continuity')}
    if any(not math.isfinite(v) or v <= 0 for v in bounds.values()) or not rule.get('rationale', '').strip():
        raise ValueError('Residual physical-floor policy requires positive finite bounds and an explicit rationale')
    if snapshots is None or prior is None:
        return {'exempted': False, 'reason': 'Missing adjacent or long-span field history', **bounds}
    field, index = mapping[component]
    def values(row):
        return row[field] if index is None else row[field].reshape(3, -1)[index]
    arrays = [values(row) for row in snapshots]
    pairs = list(zip(arrays[:-1], arrays[1:])) + [(values(prior), values(final))]
    magnitude = max(float(np.max(np.abs(x))) for x in arrays + [values(prior)])
    change = max(float(np.max(np.abs(b-a))) for a,b in pairs)
    finite = all(np.isfinite(x).all() for x in arrays + [values(prior)])
    admitted = finite and magnitude <= bounds['max_abs'] and change <= bounds['max_abs_change'] and continuity <= bounds['max_local_continuity']
    return {'exempted': bool(admitted), 'max_observed_abs': magnitude,
            'max_observed_abs_change': change, 'max_observed_local_continuity': continuity,
            'rationale': rule['rationale'], **bounds}


def _change(a, b, floor):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('inf')
    return float(np.sqrt(np.mean((b-a)**2)) / max(float(np.sqrt(np.mean(b*b))), floor))


def _finite_json(value):
    """Keep failed evidence serializable without inventing finite measurements."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    return value


def _log_diagnostics(text):
    matches = list(TIME.finditer(text))
    blocks = [(float(m.group(1)), text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)])
              for i, m in enumerate(matches)]
    streak = longest = 0
    history = []
    for t, block in blocks:
        bounded = bool(re.search(r'\bbounding (?:k|omega),', block))
        streak = streak + 1 if bounded else 0
        longest = max(longest, streak)
        residuals = {k: float(v) for k, v in RES.findall(block)}
        continuity = [tuple(map(float, c)) for c in CONT.findall(block)]
        history.append({'time': t, 'residuals': residuals, 'continuity': continuity, 'bounded': bounded})
    fatal = [x for x in ('Foam::sigFpe', 'FOAM FATAL', 'job aborted', 'Segmentation fault') if x in text]
    nonfinite_log = bool(re.search(r'\b(?:nan|[-+]?inf)\b', text, re.I))
    return history, longest, fatal, nonfinite_log


def qualify_run(case_dir, protocol) -> dict:
    case = Path(case_dir)
    checks = {}
    reasons = []
    metrics = {}
    def check(name, value, explanation):
        checks[name] = bool(value)
        if not value:
            reasons.append(explanation)
    try:
        target = float(protocol['end_time'])
        if target <= 0 or not math.isfinite(target):
            raise ValueError('end_time must be finite and positive')
        span = float(protocol.get('long_span_steps', max(20, target/4)))
        count = int(protocol.get('adjacent_window', 5))
        dt = float(protocol.get('snapshot_step', 1))
        if span <= 0 or count < 4 or dt <= 0:
            raise ValueError('long_span_steps>0, adjacent_window>=4 and snapshot_step>0 required')
        execution = protocol.get('execution')
        if execution is None and (case/'execution.json').is_file():
            execution = json.loads((case/'execution.json').read_text())
        check('successful_execution', isinstance(execution, dict) and execution.get('returncode') == 0
              and not execution.get('timed_out', False), 'Successful solver execution record is missing or failed')
        log = case / protocol.get('log_file', 'log.simpleFoam')
        if not log.resolve().is_relative_to(case.resolve()):
            raise ValueError('Solver log must be inside its case')
        text = log.read_text(errors='replace')
        history, bounded, fatal, nonfinite = _log_diagnostics(text)
        check('solver_history', bool(history), 'No complete physical iteration records')
        check('trajectory_finite', not fatal and not nonfinite, f'Fatal/nonfinite solver history: {fatal}')
        check('bounding_admitted', bounded <= int(protocol.get('max_consecutive_bounded', 50)),
              f'Sustained turbulence bounding: {bounded} consecutive iterations')
        metrics['max_consecutive_bounded'] = bounded
        times = _snapshots(case)
        check('retained_fields', bool(times), 'Missing complete U/p/k/omega/nut snapshots')
        if not times or not history:
            raise ValueError('Insufficient solver or retained field history')
        final_time, final_name = times[-1]
        tolerance_time = 1e-8 * max(1, target)
        check('target_reached', abs(final_time-target) <= tolerance_time and
              abs(history[-1]['time']-target) <= tolerance_time,
              'Written and logged endpoint do not both match required end_time')
        final = _read_fields(case, final_name)
        check('finite_fields', all(v.size > 0 and np.isfinite(v).all() for v in final.values()),
              'Final fields contain missing or nonfinite values')
        check('positive_turbulence', bool(np.all(final['k'] > 0) and np.all(final['omega'] > 0)
              and np.all(final['nut'] >= 0)), 'Final k/omega must be positive and nut nonnegative')
        metrics['final_time'] = final_time
        metrics['field_ranges'] = {f: [float(np.min(v)), float(np.max(v))] for f,v in final.items()}
        residuals = history[-1]['residuals']
        required = protocol.get('required_residuals', ['Ux', 'Uy', 'p', 'k', 'omega'])
        residual_tol = float(protocol.get('residual_tolerance', 1e-6))
        check('residuals_present', all(f in residuals for f in required), 'Required final residuals missing')
        metrics['final_residuals'] = residuals
        recent = history[-int(protocol.get('continuity_window', 20)):]
        continuities = [c for row in recent for c in row['continuity']]
        check('continuity_present', len(continuities) >= len(recent), 'Continuity history missing')
        max_local = max((abs(c[0]) for c in continuities), default=float('inf'))
        check('conservation', max_local <= float(protocol.get('continuity_tolerance', 1e-6)),
              'Local continuity threshold unmet')
        metrics['max_recent_local_continuity'] = max_local
        earlier = [(t,n) for t,n in times if final_time-t >= span-tolerance_time]
        check('long_span_available', bool(earlier), 'Required long-span snapshot is missing')
        tols = DEFAULT_TOLS | protocol.get('field_relative_tolerances', {})
        floors = DEFAULT_FLOORS | protocol.get('field_scale_floors', {})
        if any(not math.isfinite(v) or v < 0 for v in tols.values()) or any(not math.isfinite(v) or v <= 0 for v in floors.values()):
            raise ValueError('Field tolerances/floors must be finite and nonnegative/positive')
        prior = snapshots = None
        if earlier:
            t0, n0 = earlier[-1]
            prior = _read_fields(case, n0, cell_count=final['k'].size)
            drift = {f: _change(prior[f], final[f], floors[f]) for f in tols}
            metrics['long_span'] = {'times': [t0, final_time], 'relative_changes': drift}
            check('long_span_converged', all(drift[f] <= tols[f] for f in tols), 'Long-span field drift threshold unmet')
        lookup = {round(t, 8): n for t,n in times}
        wanted = [round(final_time - i*dt, 8) for i in range(count, -1, -1)]
        available = all(t in lookup for t in wanted)
        check('adjacent_history_available', available, 'Consecutive final field window is missing')
        if available:
            snapshots = [_read_fields(case, lookup[t], cell_count=final['k'].size) for t in wanted]
            finite_history = all(np.isfinite(v).all() for row in snapshots for v in row.values())
            positive_history = all(np.all(row['k'] > 0) and np.all(row['omega'] > 0) and np.all(row['nut'] >= 0) for row in snapshots)
            check('adjacent_fields_admitted', finite_history and positive_history, 'Consecutive fields are nonfinite or nonpositive')
            lag1 = {f:max(_change(a[f],b[f],floors[f]) for a,b in zip(snapshots[:-1],snapshots[1:])) for f in tols}
            lag2 = {f:max(_change(a[f],b[f],floors[f]) for a,b in zip(snapshots[:-2],snapshots[2:])) for f in tols}
            # A prospective, per-field cycle amplitude policy; legacy defaults remain.
            overrides = protocol.get('two_cycle_relative_tolerances', {})
            if (not isinstance(overrides, dict) or set(overrides) - set(tols)
                    or any(not math.isfinite(v) or not 0 < v <= tols[f]
                           for f, v in overrides.items())):
                raise ValueError('Cycle tolerances must name monitored fields and be finite, positive and no larger than field tolerances')
            cycle_tols = {f: overrides.get(f, 1e-6) for f in tols}
            pattern = {f: lag1[f] > 0 and lag2[f] <= min(1e-8, lag1[f]*.01) for f in tols}
            cycle = {f: pattern[f] and lag1[f] > cycle_tols[f] for f in tols}
            metrics['adjacent'] = {'times': wanted, 'max_relative_change': lag1,
                'max_lag2_relative_change': lag2, 'two_cycle': cycle,
                'two_cycle_pattern': pattern, 'two_cycle_relative_tolerances': cycle_tols}
            check('adjacent_converged', all(lag1[f] <= tols[f] for f in tols), 'Consecutive field drift threshold unmet')
            check('no_resolved_two_cycle', not any(cycle.values()), 'Resolved alternating field state detected')
        residual_policy = protocol.get('residual_component_policy', {})
        exemptions = {component: _roundoff_residual_evidence(component, rule, snapshots, prior, final, max_local)
                      for component, rule in residual_policy.items()}
        metrics['residual_physical_floor_evidence'] = exemptions
        check('residuals_converged', bool(residuals) and all(math.isfinite(v) and
              (v <= residual_tol or exemptions.get(component, {}).get('exempted', False))
              for component, v in residuals.items()), 'Final normalized initial residual threshold unmet without an admissible predeclared physical-floor exemption')
        required_obs = protocol.get('required_observables', {})
        if required_obs:
            obs_path = case / protocol.get('observables_file', 'observables.json')
            if not obs_path.resolve().is_relative_to(case.resolve()):
                raise ValueError('Observable history must be inside its case')
            obs = json.loads(obs_path.read_text())['records']
            records = {round(float(r['time']),8):r['values'] for r in obs}
            obs_times = wanted + ([round(earlier[-1][0],8)] if earlier else [])
            for name, rule in required_obs.items():
                complete = all(t in records and name in records[t] for t in obs_times) and bool(earlier)
                check(f'observable_{name}_history', complete, f'Missing {name} long/consecutive history')
                if complete:
                    arr = [np.asarray(records[t][name],dtype=float).ravel() for t in obs_times]
                    last = np.asarray(records[round(final_time,8)][name],dtype=float).ravel()
                    floor = float(rule.get('scale_floor',1e-12))
                    tol = float(rule.get('relative_tolerance',1e-3))
                    if floor <= 0 or tol < 0 or not math.isfinite(floor+tol):
                        raise ValueError(f'Invalid observable tolerance for {name}')
                    changes = [_change(a,last,floor) for a in arr]
                    metrics.setdefault('observables',{})[name] = {'max_relative_drift':max(changes)}
                    check(f'observable_{name}_converged', all(v <= tol for v in changes), f'{name} drift threshold unmet')
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        check('evidence_complete', False, f'Qualification evidence error: {exc}')
    qualified = bool(checks) and all(checks.values())
    return {'qualified': qualified, 'status': 'numerically_qualified' if qualified else 'unqualified',
            'reasons': reasons, 'checks': checks, 'metrics': _finite_json(metrics),
            'scope': 'Numerical qualification only; physical accuracy and protection are separate decisions.'}
