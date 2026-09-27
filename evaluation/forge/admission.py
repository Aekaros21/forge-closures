"""Admission for a declared development use; full-field convergence is separate.

Airfoil controls preserve surface quantities. They cannot supply calibration
accuracy, a minimum-benefit claim or full-field/release qualification.
"""
from __future__ import annotations

import math
import numpy as np
from .core import fingerprint

CONTROL_NAMES = ('drag_coefficient', 'lift_coefficient', 'wall_cf', 'wall_cp')
REQUIRED_CHECKS = ('successful_execution', 'target_reached', 'solver_history',
    'trajectory_finite', 'residuals_present', 'residuals_converged',
    'continuity_present', 'conservation', 'retained_fields', 'finite_fields',
    'positive_turbulence', 'adjacent_history_available', 'adjacent_fields_admitted',
    'adjacent_converged', 'long_span_available', 'no_resolved_two_cycle',
    'bounding_admitted', 'independent_surface_force_parity')


def is_airfoil_control(case):
    return (case.get('role') == 'protection' and case.get('adapter') == 'naca'
            and case.get('admission_policy', {}).get('scope') == 'aerodynamic_control')


def _difference(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if not a.size or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Missing, nonfinite or incompatible surface data')
    return float(np.max(np.abs(a-b)))


def admission(record, case=None):
    case = case if case is not None else record.get('identity', {}).get('case', {})
    qualified = record.get('qualification', {}).get('qualified') is True
    if not is_airfoil_control(case):
        return {'admitted': qualified, 'scope': 'full_numerical_qualification',
                'fully_qualified': qualified,
                'reasons': [] if qualified else ['Full numerical qualification required']}
    policy = case['admission_policy']
    reasons, changes = [], {}
    if record.get('status') not in ('complete', 'unqualified') or record.get('failure'):
        reasons.append('Failed or incomplete execution cannot supply an aerodynamic control')
    checks = record.get('qualification', {}).get('checks', {})
    for name in REQUIRED_CHECKS:
        if checks.get(name) is not True:
            reasons.append(f'{name}: required numerical integrity check failed or missing')
    # The sole scoped exception concerns whole-field long-span stationarity.
    # No crash, active residual, conservation, adjacent or future check is waived.
    for name, passed in checks.items():
        if name != 'long_span_converged' and passed is not True:
            reasons.append(f'{name}: numerical check failed')
    history = record.get('history', [])
    try:
        times = [float(s['time']) for s in history]
        end = float(record['identity']['protocol']['end_time'])
        if (len(times) < 7 or any(not math.isfinite(t) for t in times)
                or times != sorted(set(times)) or times[-1] != end
                or times[-1]-times[0] < max(float(policy['minimum_span_steps']),
                                           float(policy['minimum_span_fraction'])*end)):
            raise ValueError('Insufficient retained surface-history span')
        limits = policy['max_absolute_drift']
        for name in CONTROL_NAMES:
            limit = float(limits[name])
            if not math.isfinite(limit) or limit <= 0:
                raise ValueError('Invalid surface-stationarity limit')
            terminal = record['observables'][name]
            if _difference(history[-1]['observables'][name], terminal) > 1e-14:
                raise ValueError('Terminal observable disagrees with retained history')
            change = max(_difference(s['observables'][name], terminal) for s in history)
            changes[name] = {'max_absolute_drift': change, 'limit': limit,
                             'passed': change <= limit}
            if change > limit:
                reasons.append(f'{name}: surface history is not stationary')
        for coordinate in ('wall_x', 'wall_y'):
            terminal = record['observables'][coordinate]
            if max(_difference(s['observables'][coordinate], terminal) for s in history) > 1e-12:
                reasons.append(f'{coordinate}: surface sampling coordinates changed')
        if not all(s['observables'].get('force_history_parity_passed') is True for s in history):
            reasons.append('Independent force integration/history parity missing')
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        reasons.append(str(exc))
    return {'admitted': not reasons, 'scope': 'aerodynamic_control',
            'fully_qualified': qualified, 'policy_hash': fingerprint(policy),
            'changes': changes, 'reasons': reasons,
            'limitations': 'Surface protection only; unresolved full-field drift is retained. '
                           'Not calibration accuracy, a gain claim, or release qualification.'}


def baseline_agreement(first, second, case):
    """A control baseline needs two independent cold endpoints, selected on SST."""
    reasons, changes = [], {}
    if not is_airfoil_control(case):
        return {'passed': False, 'reasons': ['Not a declared airfoil control']}
    for record in (first, second):
        if not admission(record, case)['admitted']:
            reasons.append('Both SST endpoints need surface-control admission')
        if record.get('identity', {}).get('spec') is not None:
            reasons.append('Endpoint selection is reserved for SST')
    try:
        a, b = first['identity'], second['identity']
        for key in ('case', 'library', 'solver', 'source', 'ranks', 'initialization', 'environment'):
            if a[key] != b[key]: reasons.append(f'Endpoint pair differs in {key}')
        pa = {k:v for k,v in a['protocol'].items() if k not in ('end_time','long_span_steps')}
        pb = {k:v for k,v in b['protocol'].items() if k not in ('end_time','long_span_steps')}
        if pa != pb: reasons.append('Endpoint pair has different numerical settings')
        if b['protocol']['end_time'] < 2*a['protocol']['end_time']:
            reasons.append('Longer cold endpoint must be at least twice the shorter one')
        if first['experiment_hash'] == second['experiment_hash']:
            reasons.append('Two distinct cold experiments are required')
        for coordinate in ('wall_x','wall_y'):
            if _difference(first['observables'][coordinate],second['observables'][coordinate]) > 1e-12:
                reasons.append('Endpoint pair has different surface coordinates')
        for name in CONTROL_NAMES:
            difference = _difference(first['observables'][name], second['observables'][name])
            limit = case['admission_policy']['max_absolute_drift'][name]
            changes[name] = {'absolute_difference': difference, 'limit': limit,
                             'passed': difference <= limit}
            if difference > limit: reasons.append(f'{name}: independent endpoints do not agree')
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append(str(exc))
    return {'passed': not reasons, 'reasons': reasons, 'changes': changes,
            'experiments': [r.get('experiment_hash') for r in (first,second)],
            'scope': 'Matched SST surface quantities across two cold run lengths; '
                     'not full-field or discretization convergence.'}
