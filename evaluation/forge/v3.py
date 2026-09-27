"""FORGE V3 proposal contract: the substantive instruction, the generated interface and the
candidate schema. Shared by the proposer (prompt rendering, response parsing) and the
controller (context construction), so the rendered prompt and the parsed candidates agree.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

from tedp import expr, features
from tedp.spec import CandidateSpec, from_json, spec_hash, struct_hash

from .tuning import PARAMETER_ROLES

#: The exact substantive proposer instruction of the V3 specification (section 4), verbatim.
V3_INSTRUCTION = '''Design a complete algebraic correction to base k-omega SST using the available physical state and implemented modelling interface. The supplied archive is experimental evidence, not an architecture you must preserve. There is no designated parent.

Consider what the measured successes and failures imply for the complete response the closure should produce. You may reuse, reformulate, combine, simplify or discard previous mechanisms. Do not automatically retain the incumbent's terms, constants, channel allocation or activation logic. Refinement is allowed when you independently judge it appropriate; it is not required.

Aim to outperform the best valid model under the declared objective and all preservation constraints. Neither novelty alone nor mere survival is sufficient. Propose a coherent complete formulation with starting coefficients suitable for the bounded tuning budget.

Previous causal explanations are hypotheses unless supported by the supplied comparisons. A failed coefficient set is not a proof against a tensor or mechanism. A protected-case failure is not proof that the correction must be switched off there. Use measured changes relative to the reference, not just pass/fail labels, when available.

Use only the implemented inputs, operators and output channels. One formulation and coefficient vector must apply across all cases. No case identifiers, filenames, coordinates, mesh/decomposition features, reference-solution values, angle-of-attack labels or case-level selectors may enter the closure.

State a concise physical hypothesis, the role of each parameter and any deliberate coupling. Do not claim untested CFD outcomes, universal generalisation, guaranteed preservation, or that the current plateau is a proven physical limit. Return exactly the requested candidate objects in the validated schema.'''

VARIABLE_DEFINITIONS = {
    'I1': 'tr(Shat^2) >= 0, Shat = dev(symm(grad U))/omega_s (dimensionless strain)',
    'I2': 'tr(What^2) <= 0, What = (skew(grad U) + frame rotation)/omega_s (absolute dimensionless rotation; contains the frame rotation)',
    'I3': 'tr(Shat^3)', 'I4': 'tr(What^2 Shat)', 'I5': 'tr(What^2 Shat^2)',
    'Ret': 'k/(nu omega_s), turbulence Reynolds number',
    'F1': 'SST blending function (1 near walls, 0 in the free shear region)',
    'PoE': 'min(P_k/(beta* k omega), 10) with P_k = nut 2|S|^2 >= 0 (production over dissipation, clamped)',
    'Gp': '|a|/(1+|a|) with a = grad(p)/(omega_s sqrt(k_s)): resolved pressure-gradient strength, [0,1). p is the kinematic '
          'pressure of the solved field; the uniform driving force of streamwise-periodic cases is not part of it',
    'Gk': '|g|/(1+|g|) with g = grad(k)/(omega_s sqrt(k_s)): turbulent-energy-gradient strength, [0,1)',
    'Apk': '(a.g)/(|a||g| + 1e-3): alignment of the pressure gradient with the k gradient, [-1,1], 0 when either vanishes',
    'Psn': '(a.Shat.a)/(|a|^2 + 1e-3): dimensionless normal strain rate along the pressure-gradient direction (>0 stretching, <0 compression)',
    'Ksn': '(g.Shat.g)/(|g|^2 + 1e-3): dimensionless normal strain rate along the k-gradient direction',
    'Rf': '|Omega|/omega_s: system (frame) rotation strength; exactly 0 in inertial cases',
    'Rw': '(Omega . curl U)/(|Omega||curl U| + 1e-3): signed alignment of the frame rotation with the RELATIVE vorticity, [-1,1]; '
          '+1 stabilised (suction) side, -1 destabilised (pressure) side of a spanwise-rotating channel; 0 in inertial cases',
}

CANDIDATE_SCHEMA = {
    'name': 'short_name',
    'hypothesis': 'concise physical hypothesis: what complete response the closure produces and why',
    'bdelta': [['T2', 'expression']], 'rsource': [['T1', 'expression']],
    'constants': [0.1, 4.0], 'parameter_types': ['signed', 'positive'],
    'parameter_roles': ['amplitude', 'activation_scale'],
    'coupling': 'optional: which constants are deliberately shared between mechanisms and why (omit if none)',
    'reuse_of': 'optional: archive id (e.g. C131) whose exact constants you deliberately reuse as an already useful parameterisation; omit for a fresh vector',
    'rationale': 'role of each parameter and the expected trade-offs across the development families',
}


def interface_description(allowed_variables) -> dict:
    """Deterministic description of the implemented modelling interface for the prompt."""
    allowed = list(allowed_variables) if allowed_variables else list(expr.CANDIDATE_VARIABLES)
    return {
        'grammar': 'expr := term (+|- term)*; term := unary (*|/ unary)*; unary := - unary | primary; '
                   'primary := NUMBER | VARIABLE | c0..c7 | tanh(x) | exp(x) | sqrt(x) | abs(x) | min(a,b) | max(a,b) | (expr). '
                   'Guarded semantics identical in the solver and the cheap evaluator: a/b -> a*b/(b*b+1e-12), exp(x) -> exp(min(x,50)), sqrt(x) -> sqrt(max(x,0)).',
        'variables': {name: VARIABLE_DEFINITIONS[name] for name in allowed if name in VARIABLE_DEFINITIONS},
        'variables_not_available': [name for name in expr.CANDIDATE_VARIABLES if name not in allowed],
        'tensors': 'T1..T10: Pope integrity basis of (Shat, What): T1=S, T2=SW-WS, T3=dev(S^2), T4=dev(W^2), T5=WS^2-S^2W, T6=dev(W^2S+SW^2), '
                   'T7=WSW^2-W^2SW, T8=SWS^2-S^2WS, T9=dev(W^2S^2+S^2W^2), T10=WS^2W^2-W^2S^2W. Solver convention: W = skew(grad U) with '
                   'OpenFOAM grad(U)_ij = dU_j/dx_i, i.e. minus Pope\'s W; T2,T5,T7,T8,T10 are odd in W.',
        'channels': 'bDelta = sum_n g_n(state) T_n enters the Reynolds stress as nonlinearStress = 2 k bDelta (momentum and production); '
                    'rSource: R = 2 k (sum_m h_m(state) T_m) : grad U is added to the k equation and gamma(F1) R/nut to the omega equation. '
                    'Zero terms recover stock SST exactly (verified). No other channel exists in this release (no a1 limiter, no omega-specific channel, no new transport equation).',
        'clamps': 'every g_n and h_m is clamped to |g| <= gMax = 10; |R| <= rMaxFactor beta* k omega with rMaxFactor = 5; total b is clipped to the Lumley triangle; '
                  'bDelta and R are ramped over 200 iterations and relaxed (0.5). Material reliance on any clamp is rejected by the cheap gates.',
        'constants': 'c0..c7 (at most 8), each typed signed (bounds [-4,4]) or positive (bounds [1e-3,100], tuned in log coordinates). '
                     'Numeric literals other than 0 and 1 count as fitted complexity; computing numbers from numeric-only arithmetic or repeating an additive atom to '
                     'spell an integer coefficient is rejected. Constants must be contiguous c0..c(n-1) and each used at least once.',
        'parameter_roles': f"one role per constant from {list(PARAMETER_ROLES[:-1])}: amplitude (signed, steps about 25% of its value), "
                           'activation_scale / normalisation_scale (positive, steps of 0.15 decades), coupled (deliberately shared between mechanisms; explain in "coupling"). '
                           'A fresh vector is tuned with these steps; a vector declared as reuse_of an archive id (exact constants) is tuned with tighter steps.',
        'cheap_gates': 'before any CFD: grammar/legality, boundedness on 8192 sampled solver-consistent states (|bDelta| <= 1.65, clamp reliance <= 1%), '
                       'realizability (Lumley triangle at both SST limiter extremes), stock-SST recovery as Shat,What -> 0, a 1-D channel at Re_tau 5200 '
                       '(Cf within 2% of SST, sane log law) and HOM23 homogeneous-shear safety. The sampled states include the V3 features drawn consistently with the tensors.',
        'schema': CANDIDATE_SCHEMA,
        'hard_rules': ['one formulation and one coefficient vector for all cases', 'no case identifiers, coordinates, filenames, mesh/decomposition features, '
                       'reference values, angle-of-attack labels or case-level selectors', 'return a JSON array of exactly the requested number of objects, no prose, no Markdown fences',
                       'do not propose alternative clamp values (gMax, rMaxFactor are protocol-fixed)'],
    }


def render_v3_prompt(context: Mapping, dossier_text: str, n: int, island: str, generation: int, seed: int) -> str:
    """The complete prompt. The evidence part is identical for every island; island-specific
    material (an orchestration label only) comes last so a shared prompt prefix can be cached."""
    ctx = dict(context)
    parts = []
    w = parts.append
    w('FORGE V3 PROPOSAL REQUEST')
    w('')
    w(V3_INSTRUCTION)
    w('')
    w('=== IMPLEMENTED MODELLING INTERFACE (generated from the code; the only inputs, operators and channels that exist) ===')
    w(json.dumps(ctx.get('interface') or {}, indent=1, sort_keys=True))
    w('')
    w('=== EVALUATION CONTRACT, STAGES AND BUDGET (frozen for this campaign) ===')
    w(json.dumps(ctx.get('staging') or {}, indent=1, sort_keys=True))
    w('')
    w('=== CURRENT STATE (full-development objective: lower is better, 1.0 = SST) ===')
    w(json.dumps(ctx.get('frontier') or {}, indent=1, sort_keys=True))
    w(json.dumps(ctx.get('benchmark') or {}, indent=1, sort_keys=True))
    w('')
    w(dossier_text.rstrip('\n'))
    w('')
    w('=== REQUEST ===')
    w(json.dumps({'requested': n, 'orchestration_island': island, 'generation': generation, 'experiment_seed': seed,
                  'note': 'The island label is an orchestration/provenance id only; it restricts nothing. All four workers received the same evidence. '
                          'Return exactly the requested number of complete candidate objects in the schema above.'}, sort_keys=True))
    return '\n'.join(parts)


def instruction_sha256() -> str:
    return hashlib.sha256(V3_INSTRUCTION.encode()).hexdigest()


def parse_candidate_entry(entry: Mapping, allowed_variables=None) -> tuple[CandidateSpec, dict]:
    """Validate one returned candidate object; returns (spec, metadata). Raises ValueError."""
    if not isinstance(entry, Mapping):
        raise ValueError('candidate must be an object')
    spec = from_json(json.dumps({'name': entry.get('name'), 'bdelta': entry.get('bdelta', []),
                                 'rsource': entry.get('rsource', []), 'constants': entry.get('constants', [])}))
    spec.validate(search_policy=True, allowed_variables=allowed_variables)
    kinds = entry.get('parameter_types', ['signed'] * len(spec.constants))
    if len(kinds) != len(spec.constants) or any(kind not in {'signed', 'positive'} for kind in kinds):
        raise ValueError('invalid coefficient types')
    if any(kind == 'positive' and value <= 0 for kind, value in zip(kinds, spec.constants)):
        raise ValueError('positive scale has a nonpositive starting value')
    roles = entry.get('parameter_roles')
    warnings = []
    if roles is None:
        roles = ['unspecified'] * len(spec.constants)
        if spec.constants:
            warnings.append('parameter_roles missing; recorded as unspecified')
    else:
        if isinstance(roles, Mapping):
            roles = [roles.get(f'c{i}', roles.get(i, 'unspecified')) for i in range(len(spec.constants))]
        roles = [(r.get('role') if isinstance(r, Mapping) else r) for r in roles]
        roles = [str(r).strip().lower().replace(' ', '_').replace('-', '_') for r in roles]
        alias = {'normalization_scale': 'normalisation_scale', 'scale': 'normalisation_scale', 'activation': 'activation_scale',
                 'amp': 'amplitude', 'shared': 'coupled'}
        roles = [alias.get(r, r) for r in roles]
        if len(roles) != len(spec.constants):
            warnings.append('parameter_roles length mismatch; recorded as unspecified')
            roles = ['unspecified'] * len(spec.constants)
        elif any(r not in PARAMETER_ROLES for r in roles):
            warnings.append('unknown parameter role(s) ' + ','.join(sorted({r for r in roles if r not in PARAMETER_ROLES})))
            roles = [r if r in PARAMETER_ROLES else 'unspecified' for r in roles]
    reuse_of = entry.get('reuse_of')
    metadata = {
        'parameter_types': list(kinds), 'parameter_roles': roles,
        'hypothesis': str(entry.get('hypothesis') or '')[:1200] or None,
        'rationale': str(entry.get('rationale') or '')[:1200] or None,
        'coupling': str(entry.get('coupling') or '')[:600] or None,
        'reuse_of': str(reuse_of)[:64] if reuse_of else None,
        'legacy_role': entry.get('role'),
        'schema_warnings': warnings,
        'struct_hash': struct_hash(spec),
    }
    return spec, metadata


def verify_reuse(metadata: dict, archive_lookup) -> dict:
    """A reused parameterisation must name an archive record with the SAME structure and constants;
    nothing is inferred from matching numbers alone."""
    claim = metadata.get('reuse_of')
    metadata['reuse_verified'] = False
    if not claim:
        return metadata
    record = archive_lookup(claim)
    if record and record.get('struct_hash') == metadata['struct_hash'] and \
            [float(c) for c in record.get('constants') or []] == [float(c) for c in metadata.get('constants') or []]:
        metadata['reuse_verified'] = True
    else:
        metadata['schema_warnings'] = list(metadata.get('schema_warnings') or []) + [
            f'reuse_of {claim} does not match an archive record with identical structure and constants; treated as fresh']
    return metadata


def estimate_tokens(text: str, bytes_per_token: float) -> int:
    return int(math.ceil(len(text.encode()) / bytes_per_token))
