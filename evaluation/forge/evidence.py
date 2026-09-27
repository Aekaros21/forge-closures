"""FORGE V3 evidence dossier: the complete, deduplicated development record.

Built deterministically from authoritative campaign records (checkpoints, per-candidate
tuning logs, proposal records, handoff records, retained evaluation payloads) and from
the V3 controller's own state. No LLM summariser is involved.

Identity rules (learned the hard way in v2): a candidate is identified by its spec_hash
(equation + exact constants), a structure by its struct_hash; tuning trials are joined
to their candidate through the candidate job's frozen identity, never through a name or
a constants tuple. Verdicts: the LATEST campaign that holds a spec_hash carries the
current verdict (re-assessments under corrected policies supersede older ones); every
older verdict is retained as superseded provenance.

Three kinds of statements are kept apart in the rendering:
  OBSERVATIONS  measured results of identified (equation, constants, protocol) points;
  COMPARISONS   what differs between tested candidates (structure and constants), with
                the number of simultaneous changes stated;
  HYPOTHESES    proposer rationales, verbatim excerpts, never modelling rules.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from tedp import expr
from tedp.spec import CandidateSpec, canonical, from_json, spec_hash, struct_hash, to_json

SCHEMA = 'forge-v3-evidence-1'
FAMILIES = ('square_duct', 'nasa_hump', 'rotating_channel', 'mcconkey_pehill', 'mcconkey_convdiv',
            'mcconkey_cbfs', 'mcconkey_bump', 'tmr_bump', 'tmr_plate', 'airfoil_naca0012')
FAMILY_SHORT = {'square_duct': 'duct', 'nasa_hump': 'hump', 'rotating_channel': 'rot', 'mcconkey_pehill': 'hill',
                'mcconkey_convdiv': 'cdiv', 'mcconkey_cbfs': 'cbfs', 'mcconkey_bump': 'bump', 'tmr_bump': 'tbump',
                'tmr_plate': 'plate', 'airfoil_naca0012': 'naca'}
CASE_SHORT = {'naca0012_a010_225x65': 'na010', 'naca0012_a000_225x65': 'na000', 'tmr_bump_177x81': 'tbump',
              'tmr_plate_137x97': 'tplate', 'nasa_hump_fine': 'hump', 'convdiv12600': 'cd126', 'convdiv20580': 'cd206',
              'cbfs13700': 'cbfs', 'squareDuct_Re_1100': 'sd1100', 'squareDuct_Re_2000': 'sd2000',
              'squareDuct_Re_3500': 'sd3500', 'rotchan_ro00': 'ro00', 'rotchan_ro10': 'ro10', 'rotchan_ro50': 'ro50',
              'case_0p5': 'h0p5', 'case_0p8': 'h0p8', 'case_1p0': 'h1p0', 'case_1p2': 'h1p2', 'case_1p5': 'h1p5',
              'h20': 'h20', 'h26': 'h26', 'h31': 'h31', 'h38': 'h38', 'h42': 'h42'}
CASE_ORDER = ('case_0p5', 'case_0p8', 'case_1p0', 'case_1p2', 'case_1p5', 'convdiv12600', 'convdiv20580', 'cbfs13700',
              'h20', 'h26', 'h31', 'h38', 'h42', 'squareDuct_Re_1100', 'squareDuct_Re_2000', 'squareDuct_Re_3500',
              'rotchan_ro00', 'rotchan_ro10', 'rotchan_ro50', 'nasa_hump_fine', 'tmr_plate_137x97', 'tmr_bump_177x81',
              'naca0012_a000_225x65', 'naca0012_a010_225x65')
METRIC_SHORT = {'skin_friction_rmse': 'Cf', 'velocity_rmse': 'U', 'turbulent_k_rmse': 'k', 'shear_stress_rmse': 'uv',
                'wall_cp': 'Cp', 'wall_cf': 'Cfwall', 'lift_abs_error': 'CLerr', 'drag_abs_error': 'CDerr',
                'lift_coefficient': 'CL', 'drag_coefficient': 'CD', 'secondary_velocity_rmse': 'Usec',
                'wall_friction_asymmetry_error': 'Cfasym', 'composite': 'comp'}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _num(value, digits=4):
    return round(float(value), digits) if _finite(value) else None


def _short_hash(value: str) -> str:
    return value[:10]


def canonical_terms(spec: dict) -> tuple:
    """Order-free structural signature of a spec's terms (channel, tensor, canonical expression)."""
    out = []
    for channel in ('bdelta', 'rsource'):
        for tensor, text in spec.get(channel) or []:
            try:
                node = expr.parse(text)
                out.append((channel, tensor, expr.to_string(node)))
            except expr.GrammarError:
                out.append((channel, tensor, text))
    return tuple(sorted(out))


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


# --------------------------------------------------------------------------- loading

def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _generation_of(origin: str):
    match = re.match(r'generation_(\d+)', str(origin or ''))
    return int(match.group(1)) if match else None


def _era_label(campaign: str, origin: str) -> tuple:
    """Sort key and human label of the generation a member was proposed in (v2 numbering)."""
    number = int(campaign.rsplit('-', 1)[-1])
    gen = _generation_of(origin)
    if number <= 9:
        return ((0, 0), 'v2 seeds (campaigns 006-009)')
    if number == 10:
        return ((1, gen or 0), f'v2 campaign 010 gen {gen}' if gen else 'v2 campaign 010 seeds')
    if number == 11:
        return ((2, gen or 0), f'v2 campaign 011 gen {gen}' if gen else 'v2 campaign 011 seeds (010 gen-4 forms at partial constants)')
    if number == 12:
        return ((3, 0), 'v2 campaign 011 gen 3 (judged as 012 seeds)')
    if gen is not None:
        return ((4, gen), f'v2 gen {gen} (campaigns 013-020 numbering)')
    return ((5, 0), f'v2 {campaign} {origin}')


def load_v2_campaigns(root: Path, pattern: str = 'forge-v2-hx1-*') -> list[dict]:
    """Every v2 campaign directory with a search checkpoint, oldest first."""
    campaigns = []
    for folder in sorted((root / 'results').glob(pattern)):
        checkpoint = folder / 'search/checkpoint.json'
        state = _read_json(checkpoint)
        if not isinstance(state, dict) or 'members' not in state:
            continue
        protocol = _read_json(folder / 'protocol.json') or {}
        handoff = _read_json(folder / 'handoff.json') or {}
        config = (protocol.get('definition') or {}).get('config') or {}
        campaigns.append({
            'campaign': folder.name, 'folder': folder, 'state': state,
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'protocol_fingerprint': protocol.get('fingerprint'),
            'physical_source_hash': (protocol.get('definition') or {}).get('source_hash'),
            'library_sha256': (protocol.get('definition') or {}).get('library_sha256'),
            'proposer': config.get('proposer'), 'search_config': config.get('search'),
            'case_protocol_overrides_hash': content_hash(config.get('case_protocol_overrides')),
            'handoff': handoff, 'generation': state.get('generation'), 'best_objective': state.get('best_objective'),
        })
    return campaigns


def _proposal_entries(folder: Path) -> list[dict]:
    """Every parsed candidate object the proposers returned in this campaign, with its call id."""
    from .proposer import ClaudeProposer
    out = []
    for record in sorted(folder.glob('proposals/g*.json')):
        data = _read_json(record)
        if not isinstance(data, dict):
            continue
        try:
            envelope = ClaudeProposer.envelope(data.get('stdout') or '')
            result = envelope.get('result') if isinstance(envelope, dict) else None
            entries = json.loads(result) if isinstance(result, str) else result
        except (ValueError, TypeError, AttributeError):
            continue
        for entry in entries or []:
            if isinstance(entry, dict) and (entry.get('bdelta') or entry.get('rsource')):
                out.append({'call_id': data.get('call_id') or record.stem, 'entry': entry,
                            'prompt_sha256': hashlib.sha256((data.get('prompt') or '').encode()).hexdigest()})
    return out


def _candidate_jobs(folder: Path) -> Iterable[tuple[dict, list[dict]]]:
    for sub in ('search/candidate_jobs', 'search/previous_candidate_jobs'):
        for job in sorted((folder / sub).glob('*')):
            checkpoint = _read_json(job / 'checkpoint.json')
            identity = ((checkpoint or {}).get('identity') or {}).get('candidate')
            if not identity or not (job / 'history.jsonl').exists():
                continue
            events = []
            for line in (job / 'history.jsonl').read_text().splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
            yield identity, events


def _member_breaches(assessment: dict) -> list[dict]:
    """Every metric whose (change + sensitivity bound) exceeded its allowance, as a usage ratio."""
    out = []
    for cid, case in (assessment.get('cases') or {}).items():
        for name, metric in ((case or {}).get('metrics') or {}).items():
            if not isinstance(metric, dict):
                continue
            allowance = metric.get('allowance')
            upper = metric.get('upper_change', metric.get('change'))
            usage = (upper / allowance) if _finite(upper) and _finite(allowance) and allowance > 0 else None
            if metric.get('passed') is False or (_finite(metric.get('violation')) and metric['violation'] > 0):
                out.append({'case': cid, 'metric': name, 'usage': _num(usage, 3),
                            'candidate': _num(metric.get('candidate'), 6), 'sst': _num(metric.get('sst'), 6)})
    return out


def _member_margins(assessment: dict) -> list[dict]:
    """Allowance usage of every PASSED preservation metric: how close a valid form sits to a wall."""
    out = []
    for cid, case in (assessment.get('cases') or {}).items():
        for name, metric in ((case or {}).get('metrics') or {}).items():
            if not isinstance(metric, dict) or metric.get('passed') is not True:
                continue
            allowance = metric.get('allowance')
            upper = metric.get('upper_change', metric.get('change'))
            if _finite(upper) and _finite(allowance) and allowance > 0:
                out.append({'case': cid, 'metric': name, 'usage': _num(upper / allowance, 3)})
    return out


def _failure_categories(member: dict) -> dict:
    """Distinct failure categories of an assessed member (never merged into one label)."""
    assessment = member.get('assessment') or {}
    reasons = [str(r) for r in assessment.get('reasons') or []]
    statuses = member.get('case_statuses') or {}
    categories = {'preservation': [], 'numerical_qualification': [], 'infrastructure_or_solver_failure': [],
                  'incomplete_evidence': [], 'other': []}
    for reason in reasons:
        if 'preservation exceeded' in reason:
            categories['preservation'].append(reason.split(':')[0])
        elif 'admission' in reason:
            categories['numerical_qualification'].append(reason.split(':')[0])
        elif 'incomplete' in reason or 'missing' in reason:
            categories['incomplete_evidence'].append(reason[:80])
        else:
            categories['other'].append(reason[:120])
    for cid, status in statuses.items():
        if status == 'failed':
            categories['infrastructure_or_solver_failure'].append(cid)
    return {key: sorted(set(value)) for key, value in categories.items() if value}


def _trial_rows(events: list[dict]) -> list[dict]:
    rows, seen = [], set()
    for event in events:
        if event.get('kind') != 'tuning_trial' or event.get('spec_hash') in seen:
            continue
        seen.add(event.get('spec_hash'))
        rank = event.get('rank') or []
        gate = event.get('gate') or {}
        rows.append({'spec_hash': event.get('spec_hash'), 'constants': [_num(c, 6) for c in event.get('constants') or []],
                     'class': int(rank[0]) if rank and _finite(rank[0]) else 3,
                     'violation': _num(rank[1]) if len(rank) > 1 else None,
                     'objective': _num(rank[2]) if len(rank) > 2 else None,
                     'gate_passed': bool(gate.get('passed', True)) if isinstance(gate, dict) else True,
                     'gate_reasons': [str(r)[:120] for r in (gate.get('reasons') or [])][:3] if isinstance(gate, dict) else []})
    return rows


def _trial_breaches(folder: Path, state: dict, calibration_ids) -> dict:
    """(spec_hash of a tuning point) -> breach list, from retained evaluation payloads (if any)."""
    out = {}
    evaluations = folder / 'search/evaluations'
    if not evaluations.exists():
        return out
    for key, entry in (state.get('evaluations') or {}).items():
        if entry.get('status') != 'complete':
            continue
        path = evaluations / f'{key}.json'
        payload = _read_json(path)
        result = (payload or {}).get('result') if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            continue
        definitions = [r['identity']['spec'] for r in (result.get('case_results') or {}).values()
                       if isinstance(r, dict) and isinstance(r.get('identity'), dict) and r['identity'].get('spec')]
        if not definitions:
            continue
        try:
            identity = spec_hash(from_json(json.dumps({**definitions[0], 'name': 'x'})))
        except (ValueError, KeyError, TypeError):
            continue
        assessment = result.get('assessment') or {}
        breaches = _member_breaches(assessment)
        stage = entry.get('stage')
        record = out.setdefault(identity, {'stages': {}, 'breaches': []})
        record['stages'][stage] = {'status': result.get('status'), 'objective': _num(assessment.get('objective')),
                                   'case_errors': {cid: _num(c.get('normalized_error')) for cid, c in (assessment.get('cases') or {}).items()
                                                   if isinstance(c, dict)}}
        for breach in breaches:
            if breach not in record['breaches']:
                record['breaches'].append(breach)
    return out


# --------------------------------------------------------------------------- building

def build_v2_dossier(root: Path, *, manifest: dict, contract: dict) -> dict:
    """Assemble the complete v2 development record from the read-only campaign directories."""
    campaigns = load_v2_campaigns(root)
    if not campaigns:
        raise ValueError(f'no v2 campaign checkpoints under {root}')
    first, latest = {}, {}
    verdict_history = defaultdict(list)
    for campaign in campaigns:
        for identity, member in campaign['state'].get('members', {}).items():
            first.setdefault(identity, (campaign, member))
            latest[identity] = (campaign, member)
            assessment = member.get('assessment') or {}
            verdict_history[identity].append({'campaign': campaign['campaign'], 'feasible': bool(member.get('feasible')),
                                              'objective': _num(assessment.get('objective')),
                                              'protocol_fingerprint': campaign['protocol_fingerprint']})
    # proposer rationales and legacy role labels, joined by exact structure
    hypotheses = {}
    for campaign in campaigns:
        for item in _proposal_entries(campaign['folder']):
            entry = item['entry']
            try:
                spec = from_json(json.dumps({'name': entry.get('name', 'x'), 'bdelta': entry.get('bdelta', []),
                                             'rsource': entry.get('rsource', []), 'constants': entry.get('constants', [])}))
            except (ValueError, KeyError, TypeError):
                continue
            key = (campaign['campaign'], struct_hash(spec))
            hypotheses.setdefault(key, {'name': entry.get('name'), 'legacy_role': entry.get('role'),
                                        'rationale': str(entry.get('rationale') or '')[:600], 'call_id': item['call_id'],
                                        'parameter_types': entry.get('parameter_types')})
    # tuning logs joined by the candidate job identity (campaign, structure, island)
    trials = {}
    for campaign in campaigns:
        breaches = _trial_breaches(campaign['folder'], campaign['state'],
                                   (campaign.get('search_config') or {}).get('calibration_ids'))
        for identity, events in _candidate_jobs(campaign['folder']):
            try:
                spec = from_json(json.dumps(identity['spec']))
            except (ValueError, KeyError, TypeError):
                continue
            rows = _trial_rows(events)
            if not rows:
                continue
            for row in rows:
                detail = breaches.get(row['spec_hash'])
                if detail:
                    row['breaches'] = detail['breaches']
                    row['stages'] = detail['stages']
            key = (campaign['campaign'], struct_hash(spec), identity.get('island'))
            if key not in trials or len(rows) > len(trials[key]['rows']):
                trials[key] = {'rows': rows, 'origin': identity.get('origin'), 'start_constants': list(spec.constants)}
    terms, term_ids = {}, {}
    def term_id(text):
        if text not in term_ids:
            term_ids[text] = f'E{len(term_ids) + 1}'
            terms[term_ids[text]] = text
        return term_ids[text]
    candidates = []
    ordering = sorted(first, key=lambda h: (_era_label(first[h][0]['campaign'], first[h][1].get('origin'))[0],
                                            first[h][1].get('island') or '', first[h][1]['spec'].get('name') or ''))
    for index, identity in enumerate(ordering, 1):
        campaign0, member0 = first[identity]
        campaign1, member1 = latest[identity]
        spec = member1['spec']
        assessment = member1.get('assessment') or {}
        era_key, era = _era_label(campaign0['campaign'], member0.get('origin'))
        hypothesis = hypotheses.get((campaign0['campaign'], member1['struct_hash']))
        if hypothesis is None:
            hypothesis = next((h for (c, s), h in hypotheses.items() if s == member1['struct_hash']), None)
        trial_key = next((k for k in ((campaign0['campaign'], member1['struct_hash'], member0.get('island')),) if k in trials), None)
        if trial_key is None:
            trial_key = next((k for k in trials if k[1] == member1['struct_hash']), None)
        tuning = summarize_trials(trials[trial_key]['rows'], trials[trial_key]['start_constants']) if trial_key else None
        history = verdict_history[identity]
        superseded = [h for h in history[:-1] if (h['feasible'], h['objective']) != (history[-1]['feasible'], history[-1]['objective'])]
        candidates.append({
            'id': f'C{index:03d}', 'name': spec.get('name'), 'spec_hash': identity, 'struct_hash': member1['struct_hash'],
            'era': era, 'era_key': list(era_key), 'campaign_first': campaign0['campaign'], 'campaign_verdict': campaign1['campaign'],
            'island': member0.get('island'), 'legacy_origin': member0.get('origin'),
            'legacy_role': (hypothesis or {}).get('legacy_role'),
            'status': 'valid' if member1.get('feasible') else 'not_valid',
            'objective': _num(assessment.get('objective')),
            'family_errors': {f: _num(v) for f, v in (assessment.get('family_errors') or {}).items()},
            'case_errors': {cid: _num(c.get('normalized_error')) for cid, c in (assessment.get('cases') or {}).items() if isinstance(c, dict)},
            'breaches': _member_breaches(assessment), 'margins': _member_margins(assessment) if member1.get('feasible') else [],
            'failure_categories': _failure_categories(member1),
            'reasons': [str(r)[:160] for r in (assessment.get('reasons') or [])][:8],
            'case_statuses': {cid: s for cid, s in (member1.get('case_statuses') or {}).items() if s != 'complete'},
            'equations': {channel: [[tensor, term_id(text)] for tensor, text in (spec.get(channel) or [])]
                          for channel in ('bdelta', 'rsource')},
            'constants': [_num(c, 6) for c in spec.get('constants') or []],
            'parameter_types': (hypothesis or {}).get('parameter_types'),
            'complexity': {'n_parameters': member1.get('complexity'), 'ast_nodes': member1.get('ast_nodes')},
            'hypothesis': (hypothesis or {}).get('rationale') or None,
            'tuning': tuning,
            'verdict_provenance': {'current': history[-1], 'superseded': superseded},
            'profiles_retained': False,
        })
    comparisons = build_comparisons(candidates, terms)
    dossier = {
        'schema': SCHEMA, 'source': 'v2 campaign records', 'campaigns': [
            {k: v for k, v in c.items() if k not in ('state', 'folder', 'handoff')} | {
                'members': len(c['state'].get('members', {})), 'feasible': len(c['state'].get('feasible', [])),
                'counts': c['state'].get('counts'), 'handoff_reason': (c['handoff'] or {}).get('reason'),
                'handoff_changes': (c['handoff'] or {}).get('changes')} for c in campaigns],
        'terms': terms, 'candidates': candidates, 'comparisons': comparisons,
        'gate_rejections': gate_rejection_classes(campaigns),
        'overview': build_overview(candidates, campaigns, manifest, contract),
    }
    dossier['version'] = content_hash({k: v for k, v in dossier.items() if k != 'version'})[:16]
    return dossier


def summarize_trials(rows: list[dict], start_constants: list) -> dict:
    """Per-candidate tuning summary; the rows themselves stay in the detail record."""
    evaluated = [r for r in rows if r['objective'] is not None]
    feasible = [r for r in evaluated if r['class'] == 0]
    start = next((r for r in rows if [_num(c, 6) for c in start_constants] == r['constants']), None)
    best_feasible = min(feasible, key=lambda r: r['objective']) if feasible else None
    best_overall = min(evaluated, key=lambda r: r['objective']) if evaluated else None
    better = [r for r in evaluated if r['class'] != 0 and (best_feasible is None or r['objective'] < best_feasible['objective'])]
    counts = Counter()
    worst = {}
    detail_available = any('breaches' in r for r in rows)
    for r in better:
        for b in r.get('breaches') or []:
            key = f"{CASE_SHORT.get(b['case'], b['case'])}/{METRIC_SHORT.get(b['metric'], b['metric'])}"
            counts[key] += 1
            if b.get('usage') is not None:
                worst[key] = max(worst.get(key, 0.0), b['usage'])
    return {
        'trials': len(rows), 'evaluated': len(evaluated), 'gate_rejected': len(rows) - len(evaluated),
        'feasible_points': len(feasible),
        'start_core_objective': start['objective'] if start else None,
        'start_feasible': (start['class'] == 0) if start else None,
        'best_feasible_core_objective': best_feasible['objective'] if best_feasible else None,
        'best_feasible_constants': best_feasible['constants'] if best_feasible else None,
        'best_overall_core_objective': best_overall['objective'] if best_overall else None,
        'tuning_improved_start': (best_feasible is not None and start is not None and start['class'] == 0
                                  and best_feasible['objective'] < start['objective'] - 1e-6),
        'better_but_infeasible': len(better),
        'constraints_broken_by_better_trials': {k: {'trials': v, 'max_usage': worst.get(k)} for k, v in counts.most_common(6)},
        'breach_detail_available': detail_available,
        'rows': rows,
    }


def build_comparisons(candidates: list[dict], terms: dict) -> list[dict]:
    """For each candidate, its nearest earlier archived candidate by shared terms, and what differs."""
    out = []
    by_id = {c['id']: c for c in candidates}
    for index, candidate in enumerate(candidates):
        mine = {(ch, t, e) for ch in ('bdelta', 'rsource') for t, e in candidate['equations'][ch]}
        best, score = None, -1.0
        for other in candidates[:index]:
            theirs = {(ch, t, e) for ch in ('bdelta', 'rsource') for t, e in other['equations'][ch]}
            union = mine | theirs
            if not union:
                continue
            jaccard = len(mine & theirs) / len(union)
            if jaccard > score:
                best, score = other, jaccard
        if best is None or score <= 0.0:
            continue
        theirs = {(ch, t, e) for ch in ('bdelta', 'rsource') for t, e in best['equations'][ch]}
        added = sorted(f'{ch}:{t}:{e}' for ch, t, e in mine - theirs)
        removed = sorted(f'{ch}:{t}:{e}' for ch, t, e in theirs - mine)
        n = max(len(candidate['constants']), len(best['constants']))
        changed = [f'c{i}' for i in range(n) if i >= len(candidate['constants']) or i >= len(best['constants'])
                   or candidate['constants'][i] != best['constants'][i]]
        deltas = {}
        for family in FAMILIES:
            a, b = candidate['family_errors'].get(family), best['family_errors'].get(family)
            if a is not None and b is not None and abs(a - b) >= 0.002:
                deltas[FAMILY_SHORT[family]] = _num(a - b, 3)
        out.append({'candidate': candidate['id'], 'reference': best['id'], 'shared_term_fraction': _num(score, 2),
                    'terms_added': added, 'terms_removed': removed, 'constants_changed': changed,
                    'simultaneous_changes': len(added) + len(removed) + (1 if changed else 0),
                    'objective_delta': (_num(candidate['objective'] - best['objective'], 4)
                                        if candidate['objective'] is not None and best['objective'] is not None else None),
                    'family_deltas': deltas, 'status_pair': [candidate['status'], best['status']]})
    return out


def gate_rejection_classes(campaigns: list[dict]) -> dict:
    classes = {}
    for campaign in campaigns:
        for rejection in campaign['state'].get('recent_gate_rejections') or []:
            spec = rejection.get('spec') or {}
            for reason in rejection.get('reasons') or ['unspecified']:
                kind = str(reason).split(':', 1)[0].strip() or 'unspecified'
                entry = classes.setdefault(kind, {'rejections': 0, 'names': set(), 'representative': None})
                entry['rejections'] += 1
                if spec.get('name'):
                    entry['names'].add(spec['name'])
                if entry['representative'] is None:
                    entry['representative'] = {'name': spec.get('name'), 'reason': str(reason)[:160],
                                               'bdelta': spec.get('bdelta'), 'rsource': spec.get('rsource'),
                                               'constants': spec.get('constants')}
    return {kind: {'rejections': e['rejections'], 'distinct_names': len(e['names']), 'representative': e['representative']}
            for kind, e in classes.items()}


def build_overview(candidates: list[dict], campaigns: list[dict], manifest: dict, contract: dict) -> dict:
    valid = [c for c in candidates if c['status'] == 'valid' and c['objective'] is not None]
    incumbent = min(valid, key=lambda c: c['objective']) if valid else None
    family_best = {}
    for family in FAMILIES:
        scored = [c for c in candidates if c['family_errors'].get(family) is not None]
        if not scored:
            continue
        best_valid = min((c for c in scored if c['status'] == 'valid'), key=lambda c: c['family_errors'][family], default=None)
        best_any = min(scored, key=lambda c: c['family_errors'][family])
        family_best[family] = {
            'best_valid': {'id': best_valid['id'], 'ratio': best_valid['family_errors'][family]} if best_valid else None,
            'best_any': {'id': best_any['id'], 'ratio': best_any['family_errors'][family], 'status': best_any['status']}}
    walls = Counter()
    categories = Counter()
    for c in candidates:
        if c['status'] == 'valid':
            continue
        for b in c['breaches']:
            walls[f"{CASE_SHORT.get(b['case'], b['case'])}/{METRIC_SHORT.get(b['metric'], b['metric'])}"] += 1
        for kind, items in c['failure_categories'].items():
            categories[kind] += 1
    tuning_walls = Counter()
    for c in candidates:
        for key, item in ((c.get('tuning') or {}).get('constraints_broken_by_better_trials') or {}).items():
            tuning_walls[key] += item['trials']
    protocol_changes = []
    for campaign in campaigns:
        handoff = campaign.get('handoff') or {}
        if handoff.get('reason'):
            protocol_changes.append({'campaign': campaign['campaign'], 'from': handoff.get('from_campaign'),
                                     'reason': str(handoff['reason'])[:400], 'changes': list((handoff.get('changes') or {}).keys())})
    superseded = [{'id': c['id'], 'name': c['name'], 'superseded': c['verdict_provenance']['superseded'],
                   'current': c['verdict_provenance']['current']} for c in candidates if c['verdict_provenance']['superseded']]
    cases = {c['id']: c for c in manifest['cases']}
    scored_families = sorted({c['family'] for c in cases.values() if c['role'] != 'protection' or c['adapter'] in ('plate', 'bump')})
    protection_only = sorted(c['id'] for c in cases.values() if c.get('admission_policy', {}).get('scope') == 'aerodynamic_control')
    return {
        'incumbent': None if incumbent is None else {
            'id': incumbent['id'], 'name': incumbent['name'], 'objective': incumbent['objective'],
            'family_errors': incumbent['family_errors'], 'case_errors': incumbent['case_errors'],
            'constants': incumbent['constants'], 'equations': incumbent['equations'],
            'tightest_margins': sorted(incumbent['margins'], key=lambda m: -(m['usage'] or 0))[:10],
            'campaign_verdict': incumbent['campaign_verdict'], 'era': incumbent['era']},
        'valid_count': len(valid), 'candidate_count': len(candidates),
        'family_best': family_best,
        'suite_constraint_walls': dict(walls.most_common(12)),
        'failure_categories': dict(categories),
        'tuning_constraint_walls': dict(tuning_walls.most_common(10)),
        'protocol_changes': protocol_changes,
        'superseded_verdicts': superseded,
        'scoring': {'objective': 'mean over the 9 scored families of (mean over physical groups of the mean case normalized error)',
                    'case_normalized_error': 'weighted mean over the primary metrics of candidate_error/SST_error (weights: U 0.5, Cf 0.3, uv 0.2; duct Usec 0.5; rotation Cfasym 0.5; TMR plate/bump Cf only)',
                    'validity': 'every required case admitted for its declared use and every required metric holds change + sensitivity bound <= allowance (defaults: allowance_rel 0.02 of SST error; wall_cp abs 0.005; lift abs 0.001 + 2%; drag abs 1e-5 + 2%)',
                    'protection_only_cases': protection_only,
                    'scored_families': scored_families},
        'data_limits': [
            'Per-case surface/profile data of v2 runs are not retained in the dossier (the v2 records kept scalar metrics); '
            'V3 runs record aligned signed residual bins prospectively.',
            'Per-trial breach details exist only where evaluation payloads are still local (campaigns 017-020); '
            'older trials carry class/objective/constants only.',
            'Tuning objectives are 4-case core calibration scores (case_0p8, convdiv12600, squareDuct_Re_1100, rotchan_ro10) '
            'and are NOT comparable with full-development objectives.',
            'Campaigns before 015 assessed tmr_bump under a two-cycle policy later corrected; superseded verdicts are listed.',
            'Since campaign 016 the NACA a010 sentinel runs on 8 ranks (different experiment identities, same physics contract).'],
    }


# --------------------------------------------------------------------------- V3 records

def components_summary(components: dict | None, *, limit: int = 6) -> str:
    """Compact fixed-bin failure information of one assessed V3 point (full records stay on disk)."""
    if not components:
        return ''
    rows = []
    for cid, row in components.items():
        metrics = row.get('metrics') or {}
        flagged = {n: m for n, m in metrics.items() if m.get('passed') is False}
        tight = sorted(((n, m) for n, m in metrics.items() if m.get('usage') is not None), key=lambda kv: -kv[1]['usage'])[:2]
        text = []
        for n, m in list(flagged.items()) + [kv for kv in tight if kv[0] not in flagged]:
            usage = f"{m['usage']*100:.0f}%" if m.get('usage') is not None else 'n/a'
            text.append(f"{METRIC_SHORT.get(n, n)} {usage} (cand {m.get('cand'):.3g} vs sst {m.get('sst'):.3g})" if _finite(m.get('cand')) and _finite(m.get('sst'))
                        else f"{METRIC_SHORT.get(n, n)} {usage}")
        for name, delta in (row.get('signed_changes') or {}).items():
            text.append(f"d{METRIC_SHORT.get(name, name)} {delta:+.3g}")
        for name, bins in (row.get('residual_bins') or {}).items():
            if flagged or row.get('status') != 'complete':
                text.append(f"{METRIC_SHORT.get(name, name)} residual bins(cand-sst)/rms(sst): [" + ' '.join(f'{b:+.2f}' for b in bins) + ']')
        if row.get('status') not in (None, 'complete'):
            text.append(f"status {row['status']}" + (f" ({row['failure']['type']})" if row.get('failure') else ''))
        if row.get('admitted') is False:
            text.append('not admitted')
        if text:
            rows.append(f"{CASE_SHORT.get(cid, cid)}: " + '; '.join(text))
    return ' | '.join(rows[:limit]) + (f" | (+{len(rows) - limit} more cases in the retained record)" if len(rows) > limit else '')


def candidate_record_from_member(member: dict, *, campaign: str, index: int, terms: dict, term_ids: dict,
                                 hypothesis: str | None = None, parameter_roles=None, tuning=None,
                                 selection=None, components=None, prefix: str = 'E') -> dict:
    """A V3 member in the same schema as the v2 archive records."""
    spec = member['spec']
    assessment = member.get('assessment') or {}
    def term_id(text):
        if text not in term_ids:
            term_ids[text] = f'{prefix}{len(term_ids) + 1}'
            terms[term_ids[text]] = text
        return term_ids[text]
    generation = _generation_of(member.get('origin')) or member.get('generation')
    return {
        'id': f'V{index:03d}', 'name': spec.get('name'), 'spec_hash': member['spec_hash'], 'struct_hash': member['struct_hash'],
        'era': f'V3 gen {generation}', 'era_key': [9, generation or 0], 'campaign_first': campaign, 'campaign_verdict': campaign,
        'island': member.get('island'), 'legacy_origin': member.get('origin'), 'legacy_role': None,
        'status': 'valid' if member.get('feasible') else ('selection_stage_only' if member.get('verdict_scope') == 'selection_stage_only' else 'not_valid'),
        'objective': _num(assessment.get('objective')) if member.get('verdict_scope') != 'selection_stage_only' else None,
        'family_errors': {f: _num(v) for f, v in (assessment.get('family_errors') or {}).items()},
        'case_errors': {cid: _num(c.get('normalized_error')) for cid, c in (assessment.get('cases') or {}).items() if isinstance(c, dict)},
        'breaches': _member_breaches(assessment), 'margins': _member_margins(assessment) if member.get('feasible') else [],
        'failure_categories': _failure_categories(member),
        'reasons': [str(r)[:160] for r in (assessment.get('reasons') or [])][:8],
        'case_statuses': {cid: s for cid, s in (member.get('case_statuses') or {}).items() if s != 'complete'},
        'equations': {channel: [[tensor, term_id(text)] for tensor, text in (spec.get(channel) or [])] for channel in ('bdelta', 'rsource')},
        'constants': [_num(c, 6) for c in spec.get('constants') or []],
        'parameter_types': member.get('parameter_types'), 'parameter_roles': parameter_roles,
        'complexity': {'n_parameters': member.get('complexity'), 'ast_nodes': member.get('ast_nodes')},
        'hypothesis': hypothesis, 'tuning': tuning, 'selection': selection, 'components': components,
        'verdict_provenance': {'current': {'campaign': campaign, 'feasible': bool(member.get('feasible')),
                                           'objective': _num(assessment.get('objective'))}, 'superseded': []},
        'profiles_retained': bool(components),
    }


# --------------------------------------------------------------------------- rendering

def _fmt(value, digits=3):
    if value is None:
        return '-'
    return f'{value:.{digits}f}'.replace('0.', '.', 1) if abs(value) < 1 else f'{value:.{digits}f}'


def _fmt_constants(values):
    return '[' + ','.join(f'{v:g}' for v in values) + ']'


def render_candidate_line(c: dict) -> str:
    fam = ' '.join(f"{FAMILY_SHORT[f]} {_fmt(c['family_errors'][f])}" for f in FAMILIES if f in c['family_errors'])
    cases = ' '.join(f"{CASE_SHORT.get(cid, cid)} {_fmt(v)}" for cid, v in sorted(c['case_errors'].items(), key=lambda kv: CASE_ORDER.index(kv[0]) if kv[0] in CASE_ORDER else 99) if v is not None)
    if c['status'] == 'valid':
        verdict = f"VALID obj {c['objective']:.4f}"
        margins = sorted(c.get('margins') or [], key=lambda m: -(m['usage'] or 0))[:3]
        if margins:
            verdict += ' tightest ' + ', '.join(f"{CASE_SHORT.get(m['case'], m['case'])}/{METRIC_SHORT.get(m['metric'], m['metric'])} {m['usage']*100:.0f}%" for m in margins if m['usage'] is not None)
    elif c['status'] == 'selection_stage_only':
        verdict = 'NOT VALID (failed the selection-stage checks; no full suite)'
    else:
        verdict = f"NOT VALID obj {c['objective']:.4f}" if c['objective'] is not None else 'NOT VALID'
        breaches = c.get('breaches') or []
        if breaches:
            verdict += ' breaches ' + ', '.join(f"{CASE_SHORT.get(b['case'], b['case'])}/{METRIC_SHORT.get(b['metric'], b['metric'])} {b['usage']*100:.0f}%" if b['usage'] is not None else f"{CASE_SHORT.get(b['case'], b['case'])}/{METRIC_SHORT.get(b['metric'], b['metric'])}" for b in breaches[:8])
        other = {k: v for k, v in (c.get('failure_categories') or {}).items() if k != 'preservation'}
        if other:
            verdict += ' ' + '; '.join(f"{k}: {','.join(CASE_SHORT.get(x, x) for x in v)}" for k, v in other.items())
    equations = ' '.join(f"{ch[0].upper()}{t}:{e}" for ch in ('bdelta', 'rsource') for t, e in c['equations'][ch]) or 'no terms'
    types = ''.join((p or 'signed')[0] for p in (c.get('parameter_types') or []))
    line = (f"{c['id']} {c['name']} | {c['era']} island {c['island']}"
            + (f" ({c['legacy_role']} slot)" if c.get('legacy_role') else '')
            + f" | {verdict} | fam: {fam} | cases: {cases} | c={_fmt_constants(c['constants'])}"
            + (f" types={types}" if types else '') + f" | eq: {equations} | ast {c['complexity'].get('ast_nodes')}")
    t = c.get('tuning')
    if t:
        line += (f" | tuning: {t['trials']} trials, start core {_fmt(t['start_core_objective'], 4)}"
                 f"{'' if t['start_feasible'] in (None, True) else ' (start infeasible)'}, best feasible core {_fmt(t['best_feasible_core_objective'], 4)}"
                 f"{' at ' + _fmt_constants(t['best_feasible_constants']) if t.get('tuning_improved_start') and t.get('best_feasible_constants') else ''}"
                 f", best overall core {_fmt(t['best_overall_core_objective'], 4)}, {t['better_but_infeasible']} better-but-infeasible")
        walls = t.get('constraints_broken_by_better_trials') or {}
        if walls:
            line += ' broke ' + ', '.join(f"{k} x{v['trials']}" + (f" (max {v['max_usage']*100:.0f}%)" if v.get('max_usage') else '') for k, v in walls.items())
        elif t['better_but_infeasible'] and not t.get('breach_detail_available'):
            line += ' (breach detail not retained)'
    if c['verdict_provenance'].get('superseded'):
        old = c['verdict_provenance']['superseded'][0]
        line += f" | superseded verdict: {'valid' if old['feasible'] else 'not valid'} {_fmt(old['objective'], 4)} in {old['campaign']}"
    if c.get('selection'):
        s = c['selection']
        if isinstance(s, dict):
            line += (f" | selection: augmented {_fmt(s.get('augmented_objective'), 4)} from {s.get('finalists', 0)} verified finalists / "
                     f"{s.get('checkpoints', 0)} checkpoints" + (f" ({s.get('note')})" if s.get('note') else ''))
        else:
            line += f" | selection: {s}"
    if c.get('parameter_roles'):
        line += ' | roles ' + ','.join(str(r)[:4] for r in c['parameter_roles'])
    return line


def render_dossier(dossier: dict, *, v3_candidates: list[dict] | None = None, v3_terms: dict | None = None,
                   trial_detail: str = 'summary', max_hypothesis_chars: int = 320) -> str:
    """Deterministic text rendering delivered to every proposer (identical for all islands)."""
    o = dossier['overview']
    lines = []
    w = lines.append
    w(f"=== FORGE V3 EVIDENCE DOSSIER version {dossier['version']} (schema {dossier['schema']}) ===")
    w("Every number is a measured record. Labels: VALID = full development suite complete, admitted and inside every allowance; "
      "NOT VALID = at least one breach/admission failure (breaches given as (change + sensitivity bound)/allowance in %, so >100% fails). "
      "Objectives are full-development objectives unless labelled 'core' (4-case tuning score) or 'augmented' (V3 selection score); "
      "these three scales are never comparable with each other. Family ratios and case errors are candidate/SST error ratios (1.000 = SST).")
    w("")
    w("--- OVERVIEW (deterministic, from records) ---")
    inc = o.get('incumbent')
    if inc:
        w(f"Archived v2 incumbent (benchmark and fallback, NOT a parent): {inc['id']} {inc['name']} objective {inc['objective']:.4f} "
          f"({inc['era']}, verdict from {inc['campaign_verdict']}). Family ratios: "
          + ' '.join(f"{FAMILY_SHORT[f]} {_fmt(inc['family_errors'][f])}" for f in FAMILIES if f in inc['family_errors']))
        w("  Per-case errors: " + ' '.join(f"{CASE_SHORT.get(cid, cid)} {_fmt(v)}" for cid, v in sorted(inc['case_errors'].items(), key=lambda kv: CASE_ORDER.index(kv[0]) if kv[0] in CASE_ORDER else 99) if v is not None))
        w("  Tightest preservation margins (allowance usage): " + ', '.join(
            f"{CASE_SHORT.get(m['case'], m['case'])}/{METRIC_SHORT.get(m['metric'], m['metric'])} {m['usage']*100:.0f}%" for m in inc['tightest_margins'] if m['usage'] is not None))
        w(f"  Constants c={_fmt_constants(inc['constants'])}; equations " + ' '.join(f"{ch[0].upper()}{t}:{e}" for ch in ('bdelta', 'rsource') for t, e in inc['equations'][ch]))
    w(f"Archive size: {o['candidate_count']} distinct assessed candidates, {o['valid_count']} valid.")
    w("Best ratio per family (valid | any candidate): " + '; '.join(
        f"{FAMILY_SHORT[f]} {_fmt(v['best_valid']['ratio']) if v.get('best_valid') else '-'} ({v['best_valid']['id'] if v.get('best_valid') else '-'}) | "
        f"{_fmt(v['best_any']['ratio'])} ({v['best_any']['id']}{'' if v['best_any']['status'] == 'valid' else ', not valid'})"
        for f, v in o['family_best'].items()))
    w("Recurring suite constraint walls (case/metric: count of NOT VALID candidates breaching it): " + ', '.join(f"{k} {v}" for k, v in o['suite_constraint_walls'].items()))
    w("Failure categories among NOT VALID candidates: " + ', '.join(f"{k} {v}" for k, v in o['failure_categories'].items()))
    if o.get('tuning_constraint_walls'):
        w("Tuning-stage walls (constraints broken by better-scoring infeasible trials, where retained): " + ', '.join(f"{k} {v}" for k, v in o['tuning_constraint_walls'].items()))
    w("Scoring contract: " + o['scoring']['objective'] + '. Case error: ' + o['scoring']['case_normalized_error'] + '. Validity: ' + o['scoring']['validity']
      + f". Protection-only (no accuracy credit): {', '.join(CASE_SHORT.get(c, c) for c in o['scoring']['protection_only_cases'])}.")
    if o.get('protocol_changes'):
        w("Protocol history (campaign: reason):")
        for item in o['protocol_changes']:
            w(f"  {item['campaign']} (from {item['from']}): {item['reason']}")
    if o.get('superseded_verdicts'):
        w("Corrected verdicts (the current verdict supersedes the older one): " + '; '.join(
            f"{s['id']} {s['name']}: " + ', '.join(f"{'valid' if x['feasible'] else 'not valid'} {_fmt(x['objective'], 4)} in {x['campaign']}" for x in s['superseded'])
            + f" -> now {'valid' if s['current']['feasible'] else 'not valid'} {_fmt(s['current']['objective'], 4)} ({s['current']['campaign']})" for s in o['superseded_verdicts']))
    w("Data limits: " + ' '.join(o['data_limits']))
    w("")
    w("--- TERM TABLE (each distinct coefficient expression once; candidates reference them as B<tensor>:E<n> for the bDelta channel and R<tensor>:E<n> for the R-source channel) ---")
    all_terms = dict(dossier['terms'])
    if v3_terms:
        all_terms.update(v3_terms)
    for key in sorted(all_terms, key=lambda k: (k[0], int(k[1:]))):
        w(f"{key} = {all_terms[key]}")
    w("")
    w("--- OBSERVATIONS: every assessed candidate (v2 archive, chronological), then V3 runs ---")
    for c in dossier['candidates']:
        w(render_candidate_line(c))
    if v3_candidates:
        w("--- V3 RUNS (this campaign, same protocol as the V3 benchmark re-assessment) ---")
        for c in v3_candidates:
            w(render_candidate_line(c))
            if c.get('components'):
                w("   components: " + c['components'])
    else:
        w("--- V3 RUNS: none yet ---")
    if trial_detail == 'full':
        w("")
        w("--- TUNING TRIALS (class 0 feasible, 1 breach, 2 not qualified, 3 gate-rejected; core objective; constants) ---")
        for c in dossier['candidates'] + list(v3_candidates or []):
            rows = ((c.get('tuning') or {}).get('rows') or [])
            if rows:
                w(f"{c['id']}: " + '; '.join(f"{r['class']} {_fmt(r['objective'], 4)} {_fmt_constants(r['constants'])}" for r in rows))
    else:
        w("")
        w("(Per-trial CMA rows are summarised per candidate above; the full per-trial record is retained in the dossier file and is not delivered inline.)")
    w("")
    w("--- COMPARISONS: nearest earlier candidate by shared terms; 'simultaneous' counts the changes made at once (>1 prevents causal attribution) ---")
    for cmp in dossier['comparisons']:
        deltas = ' '.join(f"{k} {v:+.3f}" for k, v in cmp['family_deltas'].items())
        w(f"{cmp['candidate']} vs {cmp['reference']}: shared {cmp['shared_term_fraction']}, +{len(cmp['terms_added'])} terms {' '.join(cmp['terms_added'])}, "
          f"-{len(cmp['terms_removed'])} terms {' '.join(cmp['terms_removed'])}, constants changed {','.join(cmp['constants_changed']) or 'none'}; "
          f"simultaneous {cmp['simultaneous_changes']}; obj {('%+.4f' % cmp['objective_delta']) if cmp['objective_delta'] is not None else 'n/a'}; {deltas}; status {cmp['status_pair'][0]} vs {cmp['status_pair'][1]}")
    w("")
    w("--- GATE REJECTION CLASSES (cheap symbolic/1-D gates; a mathematical counterexample, not CFD evidence) ---")
    for kind, item in dossier['gate_rejections'].items():
        rep = item['representative'] or {}
        w(f"{kind}: {item['rejections']} rejections, {item['distinct_names']} structures; e.g. {rep.get('name')}: {rep.get('reason')}")
    w("")
    w("--- HYPOTHESES (proposer rationales, verbatim excerpts; unverified unless the observations above support them) ---")
    for c in dossier['candidates'] + list(v3_candidates or []):
        if c.get('hypothesis'):
            w(f"{c['id']}: {c['hypothesis'][:max_hypothesis_chars]}")
    return '\n'.join(lines) + '\n'


def dossier_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
