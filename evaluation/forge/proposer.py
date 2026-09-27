"""Restricted local CLI proposers and durable, replayable proposal records.

No provider request is made by construction, preflight, or seed loading.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import uuid

from tedp.spec import CandidateSpec, from_json, spec_hash, to_json


class ProposalError(RuntimeError):
    pass


# Provider responses that indicate a temporary account or service limit. These
# wait for capacity instead of consuming the bounded failure attempts.
LIMIT_PATTERN = re.compile(r'usage limit|session limit|weekly limit|hit your [a-z0-9 -]{0,24}limit|limit reached|rate.?limit|'
                           r'overloaded|quota|\b429\b|\b529\b', re.I)
# The CLI's per-request output cap in V3. A cap below the model's thinking length makes the CLI issue a second
# request with a changed thinking configuration, which rewrites the whole evidence prompt to the prompt cache
# (measured in forge-v3-hx1-001 generation 1: two 185 k-token cache writes per call). The configured
# ``max_output_tokens`` remains the response reservation used for budgeting.
V3_TURN_OUTPUT_CAP = 64000
# Evidence tampering or a changed provider/prompt is never retried.
EFFORT_LEVELS = ('low', 'medium', 'high', 'xhigh', 'max')
CODEX_EFFORT_LEVELS = ('low', 'medium', 'high', 'xhigh', 'max')
PROVIDERS = ('claude_cli', 'codex_cli')
CODEX_DISABLED_FEATURES = ('shell_tool', 'unified_exec', 'apps', 'plugins', 'skill_search',
    'multi_agent', 'multi_agent_v2', 'hooks', 'code_mode', 'code_mode_host', 'browser_use',
    'computer_use', 'image_generation', 'view_image', 'sleep_tool')
INTEGRITY_FAILURES = ('proposal replay identity mismatch', 'committed proposal response checksum mismatch',
                      'provider binary changed after the protocol was initialized', 'served model mismatch',
                      'context preflight', 'does not match the frozen dossier_sha256')


@dataclass
class ProposalBatch:
    specs: list[CandidateSpec]
    usage_tokens: int = 0
    metadata: dict = field(default_factory=dict)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def find_claude_binary() -> Path:
    root = Path(__file__).resolve().parents[2]
    manifest = root/'runtime/proposer_manifest.json'
    if manifest.is_file():
        record = json.loads(manifest.read_text())
        pinned = (root/record['path']).resolve()
        if not pinned.is_relative_to(root/'runtime/proposer') or not pinned.is_file():
            raise ValueError('Pinned proposer is missing or outside the isolated runtime')
        if hashlib.sha256(pinned.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Pinned proposer checksum changed')
        return pinned
    found = shutil.which('claude')
    if found:
        return Path(found).resolve()
    candidates = sorted(Path.home().glob('.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude'))
    if not candidates:
        raise FileNotFoundError('Claude CLI is unavailable; configure a real proposer before discovery')
    return candidates[-1].resolve()


SYSTEM_PROMPT = '''You propose candidate algebraic corrections to k-omega SST for FORGE v2.
You have no tools and may use only the supplied development context. Return JSON only.
One equation and coefficient vector must apply across all listed cases; no case identifiers,
coordinates, filenames or mesh/decomposition features may enter the equation.

Implemented model grammar (do not invent new inputs or transport equations):
variables I1 I2 I3 I4 I5 Ret F1 PoE; coefficients c0..c7;
operators + - * / and functions tanh exp sqrt abs min(a,b) max(a,b);
tensors T1..T10 built from dimensionless S/omega and W/omega.
I1=tr(S^2)>=0, I2=tr(W^2)<=0; Ret=k/(nu*omega), PoE=production/dissipation.
Constitutive channel: bDelta=sum g_n*T_n.
Production channel: R=2*k*(sum h_n*T_n):grad(U).
Stress and production channels are independent; their interaction is checked in CFD.
The full ten-tensor basis already exists. Local invariant inputs do not contain flow history.
Use finite, bounded expressions with controlled low-gradient limits. Coefficients and
source caps are fixed by the declared campaign; do not propose alternative cap values.
Numeric literals and AST size are counted in complexity. Do not hide fitted coefficients
in repeated addition or literal arithmetic. Do not assume a low combined score protects Cf,
loads, rotation or secondary flow. Every proposed structure receives the same staged checks.

Growth and limits used by the inexpensive gates:
T1 grows with sqrt(I1); T2 with sqrt(I1*abs(I2)); T3 with I1;
T4 with abs(I2); higher tensors require matching higher-order normalization.
The production gate checks the contracted source 2*bR:S, so a T1 source
needs an I1-scale denominator, not merely sqrt(I1). gMax=10 and
rMaxFactor=5 are safeguards; material reliance on either clamp is rejected.
Tested small-amplitude development probe forms include
stress T2: c0/max(1,c1*sqrt(max(I1*abs(I2),0))) with c0=0.02,c1=4;
source T1: c0*tanh(c1*(I1+I2)/(I1+abs(I2)))/max(1,c1*I1)
with c0=+/-0.02,c1=4. These examples passed inexpensive gates only;
they are not claimed CFD improvements or automatically novel models.
The production-excess switch is optional and need not gate the stress channel.
The failure_summary gives one representative per gate-failure class and the feasible
structures that passed with the same tensors; do not repeat a rejected structure without a
physical repair, and do not read a failure as a reason to retreat into inert forms.

A successful proposal must plausibly move the frontier stated at the top of the development
context, not merely survive the gates. Constraints are hard; the frontier is the objective.
Islands I, II and III are DISCOVERY islands: each owns a distinct mechanism area and is scored
against its own ISLAND FRONTIER, with the campaign-wide best shown only as a reference.
Island IV is the COMBINATION island: it merges mechanisms the discovery islands have proven.
The island objective says which area this island owns.

Risk roles. The first candidate is EXPLOITATION. On a discovery island: extend or repair this
island's own CFD-supported lineage within its area. On the combination island: combine proven
mechanisms from different discovery islands. Every further candidate is EXPLORATION. On a
discovery island: a mechanism new to this island's area, aimed at an unresolved family or
physical deficiency. On the combination island: a new pairing or partition of discovery-island
mechanisms. Both must respect all hard constraints. Novelty alone has no value. Do not make any
candidate near-inert merely to maximize admission probability.

Propose physical structures and starting constants, not claimed CFD outcomes. A seed or a
published formula is not automatically feasible. Signed amplitudes may change sign in tuning;
positive scales must be explicitly typed. Default signed bounds are [-4,4], positive scales
[0.001,100]; the campaign can override them. References, uncertainty and admission rules are
not editable by the proposer. The supplied context contains only development results.

Return an array of exactly the requested number of objects:
{"name":"short_name","role":"exploit|explore","bdelta":[["T2","expression"]],"rsource":[],
 "constants":[0.1],"parameter_types":["signed"],"rationale":"physical mechanism"}.
No Markdown fences or surrounding prose.''' 


def frontier_card(frontier, profile, leaders=None, failures=None, limits=None) -> str:
    """Text rendering of the feasible frontier and the island objective, placed before the archive."""
    if not frontier and not profile and not leaders and not failures and not limits:
        return ''
    lines = []
    if frontier:
        scope = frontier.get('scope')
        lines.append('ISLAND FRONTIER (island %s lineage)' % scope if scope else 'CURRENT FEASIBLE FRONTIER')
        reference = frontier.get('global_reference')
        if reference and reference.get('objective') is not None:
            lines.append('Campaign-wide reference only: %.4f (%s, island %s)'
                         % (reference['objective'], reference.get('name'), reference.get('island')))
        best, form = frontier.get('best_valid_objective'), frontier.get('best_form') or {}
        if best is None:
            lines.append('Best valid objective: none yet (1.0 equals matched SST; lower is better)')
        else:
            lines.append('Best valid objective: %.4f (%s, island %s); 1.0 equals matched SST, lower is better'
                         % (best, form.get('name'), form.get('island')))
        ratios = frontier.get('family_ratios_vs_sst') or {}
        if ratios:
            lines.append('Family ratios vs SST on that form (valid specialist in brackets where better):')
            specialist = frontier.get('specialist_frontier') or {}
            protected = set(frontier.get('protected_families') or [])
            width = max(len(f) for f in ratios)
            for family, ratio in ratios.items():
                special = specialist.get(family) or {}
                extra = ('   [%.3f %s]' % (special['ratio'], special.get('form'))
                         if special.get('ratio') is not None and ratio is not None and special['ratio'] < ratio - 0.01 else '')
                if family in protected:
                    extra += '   protected: hold within allowance, no accuracy credit'
                lines.append('  %-*s  %s%s' % (width, family, 'n/a' if ratio is None else '%.3f' % ratio, extra))
        opportunities = frontier.get('opportunities') or []
        if opportunities:
            lines.append('Highest-value unresolved opportunities:')
            lines.extend('%d. %s' % (index + 1, text) for index, text in enumerate(opportunities))
        if frontier.get('mandate'):
            lines.append(frontier['mandate'])
        lines.append('')
    if failures:
        lines.append('RECENT FAILURES (last two generations; this island first; change/allowance per broken constraint)')
        for item in failures:
            broken = []
            for cid, entry in (item.get('constraint_failures') or {}).items():
                for name, ratio in (entry.get('exceeded_change_over_allowance') or {}).items():
                    broken.append('%s/%s %s' % (cid, name, 'n/a' if ratio is None else '%.2fx' % ratio))
            lines.append('  %s (island %s, objective %s): %s' % ((item.get('spec') or {}).get('name'), item.get('island'),
                         item.get('objective'), '; '.join(broken) or '; '.join(item.get('reasons') or [])[:200]))
        lines.append('')
    if limits:
        lines.append('WHERE TUNING HIT CONSTRAINTS (last generation: lower objectives were reachable only by breaking these)')
        for name, item in limits.items():
            broken = ', '.join('%s x%d (max %.2fx)' % (k, v['trials'], v['max_change_over_allowance'])
                               for k, v in (item.get('constraints_broken_by_better_trials') or {}).items())
            lines.append('  %s (island %s): best feasible %s, best overall %s, %d better-but-infeasible trials%s'
                         % (name, item.get('island'), item.get('best_feasible_objective'), item.get('best_overall_objective'),
                            item.get('better_but_infeasible_trials', 0), (': ' + broken) if broken else ''))
        lines.append('')
    if leaders:
        lines.append('DISCOVERY ISLAND LEADERS (best valid mechanism per discovery island)')
        for island, leader in leaders.items():
            if not leader:
                lines.append('  island %s: no valid form yet' % island)
                continue
            families = ', '.join('%s %.3f' % (k, v) for k, v in (leader.get('family_errors') or {}).items() if v is not None)
            lines.append('  island %s: %s  objective %.4f  [%s]' % (island, leader.get('name'), leader['objective'], families))
        lines.append('')
    if profile:
        lines.append(str(profile))
        lines.append('')
    return '\n'.join(lines) + '\n'


class ClaudeProposer:
    #: Prompt-size calibration: bytes of prompt per input token measured on the campaign-020
    #: records (115572 bytes -> about 78k tokens on this tokenizer); 1.4 is conservative.
    DEFAULT_BYTES_PER_TOKEN = 1.4
    #: Tokens the CLI adds around the user prompt (its own system prompt and tool preamble).
    CLI_OVERHEAD_TOKENS = 16000

    def __init__(self, work_root: Path, model: str = 'opus', *, binary: str | Path | None = None,
                 timeout_s: float = 900., max_output_tokens: int = 4096,
                 retry_deadline_unix: float | None = None, attempts_per_call: int = 3,
                 archived_attempt_limit: int = 6, retry_backoff_s: tuple = (60., 300.),
                 limit_wait_s: float = 1200., limit_max_wait_s: float = 21600.,
                 unknown_usage_charge_cap: int = 300000, effort: str | None = None,
                 provider: str = 'claude_cli', mode: str = 'v2', context_window_tokens: int | None = None,
                 bytes_per_token: float | None = None, allowed_variables=None, repo_root: Path | None = None):
        self.root = Path(work_root).resolve() / 'proposals'
        if provider not in PROVIDERS:
            raise ValueError('proposer provider must be one of ' + ', '.join(PROVIDERS))
        self.provider = provider
        if mode not in ('v2', 'v3'):
            raise ValueError('proposer mode must be v2 or v3')
        self.mode = mode
        # V3 context preflight: the whole evidence is delivered inline or the call is refused, never truncated.
        self.context_window_tokens = int(context_window_tokens) if context_window_tokens else None
        self.bytes_per_token = float(bytes_per_token or self.DEFAULT_BYTES_PER_TOKEN)
        if self.bytes_per_token <= 0:
            raise ValueError('bytes_per_token must be positive')
        self.allowed_variables = tuple(allowed_variables) if allowed_variables else None
        self.repo_root = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
        self._dossier_cache = {}
        # Retry policy is operational; it is not part of the proposal identity.
        self.retry_deadline_unix = retry_deadline_unix
        self.attempts_per_call, self.archived_attempt_limit = attempts_per_call, archived_attempt_limit
        self.retry_backoff_s = tuple(float(value) for value in retry_backoff_s)
        self.limit_wait_s, self.limit_max_wait_s = float(limit_wait_s), float(limit_max_wait_s)
        self.unknown_usage_charge_cap = int(unknown_usage_charge_cap)
        self.model = model
        # Reasoning effort is part of the proposal identity: the same prompt at a
        # different effort is a different proposer. None keeps the CLI default.
        levels = CODEX_EFFORT_LEVELS if provider == 'codex_cli' else EFFORT_LEVELS
        if effort is not None and effort not in levels:
            raise ValueError('proposer effort must be one of ' + ', '.join(levels))
        self.effort = effort
        if binary is None and provider == 'codex_cli':
            raise ValueError('the Codex proposer needs an explicit pinned binary path')
        self.binary = Path(binary).resolve() if binary else find_claude_binary()
        if not self.binary.is_file():
            raise FileNotFoundError(self.binary)
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
        if timeout_s <= 0 or max_output_tokens < 1:
            raise ValueError('positive proposer timeout and output allowance required')
        self.binary_sha256 = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        if mode == 'v3':
            from . import v3 as _v3
            prompt_sha256 = _v3.instruction_sha256()
        else:
            prompt_sha256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
        self.identity = {'provider': provider, 'model': model,
                         'binary_sha256': self.binary_sha256,
                         'prompt_sha256': prompt_sha256,
                         'max_output_tokens': max_output_tokens}
        if mode == 'v3':
            self.identity['mode'] = 'v3'
            self.identity['context_window_tokens'] = self.context_window_tokens
        if effort is not None:
            self.identity['effort'] = effort
        if provider == 'codex_cli':
            # Codex exec has no output-token cap flag. The configured allowance
            # informs reservations; actual total usage is charged after each call.
            self.identity['output_limit_enforced'] = False
            self.identity['disabled_features'] = list(CODEX_DISABLED_FEATURES)
            if mode == 'v3':
                self.identity['repair_delivery'] = 'complete_parent_prompt_and_response'
                self.identity['system_instruction_sha256'] = prompt_sha256

    def command(self, resume: str | None = None) -> list[str]:
        if self.provider == 'codex_cli':
            # Non-interactive Codex: JSONL events on stdout, prompt on stdin ('-'), no session
            # files, the user's config and rules ignored (auth still comes from CODEX_HOME),
            # read-only sandbox inside the isolated working directory.
            effort = ['-c', 'model_reasoning_effort="%s"' % self.effort] if self.effort is not None else []
            disabled = [arg for feature in CODEX_DISABLED_FEATURES for arg in ('--disable', feature)]
            return [str(self.binary), 'exec', '--json', '--ephemeral', '--skip-git-repo-check',
                    '--ignore-user-config', '--ignore-rules', '--color', 'never', '-s', 'read-only',
                    '-m', self.model, *effort, '-c', 'approval_policy="never"',
                    '-c', 'web_search="disabled"', '-c', 'project_doc_max_bytes=0',
                    '--enable', 'skip_host_skill_discovery',
                    '-c', 'model_instructions_file=' + json.dumps(str(self.root/'isolated_cwd/instructions.txt')),
                    *disabled, '-']
        # --safe-mode discards the requested model along with other customizations and
        # silently falls back to the account default; --restricted is what removes tools.
        effort = ['--effort', self.effort] if self.effort is not None else []
        # stream-json keeps every assistant turn: past the per-turn output cap the CLI continues
        # in further turns and its final 'result' carries only the last turn's text.
        # V3 keeps the CLI session of a proposal call so that one bounded repair turn can continue it
        # (the evidence prompt is then served from the prompt cache instead of being sent again).
        persistence = [] if self.mode == 'v3' else ['--no-session-persistence']
        resumed = ['--resume', resume] if resume else []
        return [str(self.binary), '-p', '--output-format', 'stream-json', '--verbose', '--model', self.model, *effort,
                '--tools', '', '--restricted', '--strict-mcp-config',
                '--mcp-config', '{"mcpServers":{}}', '--setting-sources', '', '--settings', '{}',
                '--disable-slash-commands', *persistence, '--no-chrome',
                '--permission-mode', 'dontAsk', *resumed]

    def _dossier(self, reference: dict) -> tuple[dict, str]:
        """The versioned evidence dossier named by the frozen context, integrity-checked."""
        path = Path(reference['dossier_path'])
        if not path.is_absolute():
            path = self.repo_root / path
        expected = reference['dossier_sha256']
        cached = self._dossier_cache.get(str(path))
        if cached and cached[0] == expected:
            return cached[1], cached[2]
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ProposalError(f'evidence dossier {path} does not match the frozen dossier_sha256 (integrity)')
        dossier = json.loads(data)
        self._dossier_cache = {str(path): (expected, dossier, digest)}
        return dossier, digest

    def _prompt(self, n, context, island, generation, seed):
        if context.get('mode') == 'v3' or self.mode == 'v3':
            from . import evidence, v3 as _v3
            if context.get('mode') != 'v3':
                raise ProposalError('a V3 proposer needs a V3 context')
            dossier, _ = self._dossier(context['evidence'])
            text = evidence.render_dossier(dossier, v3_candidates=list(context.get('v3_records') or []),
                                           v3_terms=dict(context.get('v3_terms') or {}),
                                           trial_detail=str(context['evidence'].get('trial_detail', 'summary')))
            return _v3.render_v3_prompt(context, text, n, island, generation, seed)
        # The frontier card and the island objective lead the prompt as text; the archive follows
        # as JSON so that constraints and opportunity carry comparable salience.
        context = dict(context)
        card = frontier_card(context.pop('frontier', None), context.pop('profile', None),
                             context.get('discovery_island_leaders'),  # specs stay in the JSON for combination
                             context.get('recent_failed_candidates'), context.get('tuning_limits_last_generation'))
        return SYSTEM_PROMPT + '\n\n' + card + json.dumps({'requested': n, 'island': island,
            'generation': generation, 'experiment_seed': seed, 'development_context': context},
            sort_keys=True, allow_nan=False)

    def preflight(self, prompt: str) -> dict:
        """Estimated context use of a prompt; refuses (never truncates) when it cannot fit."""
        from .v3 import estimate_tokens
        estimated = estimate_tokens(prompt, self.bytes_per_token)
        report = {'prompt_bytes': len(prompt.encode()), 'estimated_prompt_tokens': estimated,
                  'bytes_per_token': self.bytes_per_token, 'cli_overhead_tokens': self.CLI_OVERHEAD_TOKENS,
                  'response_reserve_tokens': self.max_output_tokens, 'context_window_tokens': self.context_window_tokens,
                  'delivery': 'inline_complete'}
        if self.context_window_tokens is not None:
            needed = estimated + self.CLI_OVERHEAD_TOKENS + self.max_output_tokens
            report['estimated_total_tokens'] = needed
            if needed > self.context_window_tokens:
                report['delivery'] = 'refused'
                raise ProposalError(f'context preflight: the complete evidence prompt needs about {needed} tokens '
                                    f'(prompt {estimated} + CLI overhead {self.CLI_OVERHEAD_TOKENS} + response {self.max_output_tokens}) '
                                    f'but the configured context window is {self.context_window_tokens}; nothing is truncated, '
                                    'reduce trial_detail or hypothesis length in the evidence configuration')
        return report

    def reserve_tokens(self, n, context, island, generation, seed) -> int:
        # The CLI can perform internal thinking/continuations beyond its output
        # allowance. The live pilot consumed 26k aggregate tokens for a 4k
        # allowance. Reserve 64k for that overhead; actual aggregate telemetry
        # is charged and any excess stops subsequent calls. This estimate is
        # not represented as a provider-enforced hard token ceiling.
        prompt = self._prompt(n, context, island, generation, seed)
        if self.mode == 'v3':
            from .v3 import estimate_tokens
            # cache writes of the full prompt plus CLI overhead plus the output allowance and continuations
            return estimate_tokens(prompt, self.bytes_per_token) + self.CLI_OVERHEAD_TOKENS + max(65536, 4 * self.max_output_tokens)
        return len(prompt.encode()) + max(65536,4*self.max_output_tokens)

    @staticmethod
    def envelope(stdout: str):
        """The provider response as one Claude-shaped envelope: result, usage, is_error.

        The Claude CLI prints a single JSON object. The Codex CLI prints JSONL events; the
        final agent message is the result and the turn's usage is mapped onto the Claude
        counters. Cached and cache-write input are subsets of input_tokens, and
        reasoning is a subset of output_tokens; never charge these twice.
        """
        try:
            envelope = json.loads(stdout)
            # Claude CLI prints one object tagged type 'result'; Codex JSONL events never use that type.
            if isinstance(envelope, dict) and envelope.get('type', 'result') == 'result':
                return envelope
        except (ValueError, TypeError):
            pass
        events = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                return None
            if isinstance(event, dict):
                events.append(event)
            else:
                return None
        if not events:
            return None
        if events[-1].get('type') == 'result' and any(e.get('type') == 'assistant' for e in events):
            # Claude stream-json: the final result event carries usage; the reply is the text of
            # every assistant turn in order (the last turn alone is kept as a fallback).
            envelope = dict(events[-1])
            texts = [block.get('text') or '' for e in events if e.get('type') == 'assistant'
                     for block in (e.get('message') or {}).get('content') or []
                     if isinstance(block, dict) and block.get('type') == 'text']
            if texts:
                envelope['last_turn_result'] = envelope.get('result')
                envelope['result'] = ''.join(texts)
            return envelope
        messages = [e['item'].get('text') for e in events if e.get('type') == 'item.completed'
                    and isinstance(e.get('item'), dict) and e['item'].get('type') == 'agent_message']
        errors = [e for e in events if e.get('type') in ('error', 'turn.failed') or 'error' in e and e.get('type') != 'item.completed']
        if any(e.get('type') in ('item.started', 'item.completed') and
               e.get('item', {}).get('type') in ('command_execution', 'file_change', 'mcp_tool_call',
                                                'web_search', 'collab_tool_call') for e in events):
            errors.append({'message': 'Proposer attempted a tool call outside the supplied context'})
        completed = [e for e in events if e.get('type') == 'turn.completed']
        usage = None
        if completed:
            usage = dict.fromkeys(('input_tokens', 'output_tokens', 'cache_creation_input_tokens',
                                  'cache_read_input_tokens'), 0)
            for event in completed:
                raw = event.get('usage')
                fields = ('input_tokens', 'output_tokens', 'cached_input_tokens',
                          'cache_write_input_tokens', 'reasoning_output_tokens')
                if (not isinstance(raw, dict) or not {'input_tokens', 'output_tokens'} <= raw.keys()
                        or any(type(raw.get(k, 0)) is not int or raw.get(k, 0) < 0 for k in fields)):
                    usage = None
                    break
                cached, written = raw.get('cached_input_tokens', 0), raw.get('cache_write_input_tokens', 0)
                if cached + written > raw['input_tokens'] or raw.get('reasoning_output_tokens', 0) > raw['output_tokens']:
                    usage = None
                    break
                usage['input_tokens'] += raw['input_tokens'] - cached - written
                usage['output_tokens'] += raw['output_tokens']
                usage['cache_creation_input_tokens'] += written
                usage['cache_read_input_tokens'] += cached
        result = messages[-1] if messages else None
        if errors:
            detail = errors[-1].get('message') or errors[-1].get('error') or json.dumps(errors[-1])[:500]
            return {'is_error': True, 'result': str(detail), 'usage': usage, 'events': len(events)}
        return {'is_error': not completed, 'result': result, 'usage': usage, 'events': len(events)}

    @staticmethod
    def served_models(stdout: str) -> dict:
        """Model identities the CLI reports for a stream-json call: the init event's model and modelUsage."""
        report = {'init_model': None, 'model_usage': {}}
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get('type') == 'system' and event.get('subtype') == 'init':
                report['init_model'] = event.get('model')
            if event.get('type') == 'result' and isinstance(event.get('modelUsage'), dict):
                report['model_usage'] = {name: {k: v for k, v in (usage or {}).items()
                                                if k in ('inputTokens', 'outputTokens', 'cacheReadInputTokens', 'cacheCreationInputTokens',
                                                         'contextWindow', 'maxOutputTokens', 'thinkingTokens', 'canonicalModel')}
                                         for name, usage in event['modelUsage'].items()}
        return report

    @staticmethod
    def parse(stdout: str, n: int, *, mode: str = 'v2', allowed_variables=None, archive_lookup=None) -> ProposalBatch:
        envelope = ClaudeProposer.envelope(stdout)
        if envelope is None:
            raise ProposalError('provider response is not valid JSON; token usage is unresolved')
        if not isinstance(envelope, dict) or envelope.get('is_error'):
            raise ProposalError('provider response must be a successful JSON envelope with usage')
        usage = envelope.get('usage')
        if not isinstance(usage, dict):
            raise ProposalError('provider token usage is missing; cannot account this call safely')
        fields = ['input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens']
        if not all(isinstance(usage.get(key, 0), int) and usage.get(key, 0) >= 0 for key in fields):
            raise ProposalError('invalid provider usage counters')
        token_count = sum(usage.get(key, 0) for key in fields)
        if token_count <= 0:
            raise ProposalError('provider returned no accountable token usage')
        entries = None
        for content in (envelope.get('result'), envelope.get('last_turn_result')):
            try:
                entries = json.loads(content) if isinstance(content, str) else content
            except (ValueError, TypeError):
                entries = None
            if isinstance(entries, list):
                break
        if not isinstance(entries, list):
            return ProposalBatch([], token_count, {'rejected': [{'reason': 'proposal result is not a JSON array'}],
                'provider_usage': {key: usage.get(key, 0) for key in fields}})
        accepted = []
        parameter_types = {}
        candidate_metadata = {}
        rejected = []
        seen = set()
        for index, entry in enumerate(entries):
            try:
                if not isinstance(entry, dict):
                    raise ValueError('candidate must be an object')
                if mode == 'v3':
                    from . import v3 as _v3
                    spec, metadata = _v3.parse_candidate_entry(entry, allowed_variables)
                    metadata['constants'] = list(spec.constants)
                    if archive_lookup is not None:
                        metadata = _v3.verify_reuse(metadata, archive_lookup)
                    kinds = metadata['parameter_types']
                else:
                    spec = from_json(json.dumps(entry))
                    spec.validate(search_policy=True)
                    kinds = entry.get('parameter_types', ['signed'] * len(spec.constants))
                    if len(kinds) != len(spec.constants) or any(kind not in {'signed', 'positive'} for kind in kinds):
                        raise ValueError('invalid coefficient types')
                    if any(kind == 'positive' and value <= 0 for kind, value in zip(kinds, spec.constants)):
                        raise ValueError('positive scale has a nonpositive starting value')
                    metadata = None
                identity = spec_hash(spec)
                if identity in seen:
                    raise ValueError('duplicate equation and constants')
                seen.add(identity)
                accepted.append(spec)
                parameter_types[identity] = list(kinds)
                if metadata is not None:
                    candidate_metadata[identity] = metadata
            except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
                rejected.append({'index': index, 'reason': str(exc)[:500]})
        metadata_out = {'parameter_types': parameter_types,
            'rejected': rejected, 'requested': n, 'returned': len(entries), 'accepted': min(n, len(accepted)),
            'provider_usage': {key: usage.get(key, 0) for key in fields}}
        if mode == 'v3':
            metadata_out['candidate_metadata'] = {spec_hash(spec): candidate_metadata[spec_hash(spec)] for spec in accepted[:n]}
            metadata_out['served_models'] = ClaudeProposer.served_models(stdout)
        return ProposalBatch(accepted[:n], token_count, metadata_out)

    def _archive_lookup(self):
        """id or spec-hash prefix -> archived dossier record, for verified reuse claims."""
        dossier = next((cached[1] for cached in self._dossier_cache.values()), None)
        if not dossier:
            return None
        by_id = {c['id']: c for c in dossier.get('candidates', [])}
        def lookup(claim):
            claim = str(claim).strip()
            if claim in by_id:
                return by_id[claim]
            return next((c for c in dossier.get('candidates', []) if c['spec_hash'].startswith(claim) and len(claim) >= 8), None)
        return lookup

    def _parse_response(self, stdout: str, n: int) -> ProposalBatch:
        if self.mode != 'v3':
            return self.parse(stdout, n)
        batch = self.parse(stdout, n, mode='v3', allowed_variables=self.allowed_variables, archive_lookup=self._archive_lookup())
        served = batch.metadata.get('served_models') or {}
        usage = served.get('model_usage') or {}
        if usage and self.model not in usage:
            raise ProposalError(f'served model mismatch: requested {self.model}, provider reported {sorted(usage)} (integrity)')
        if served.get('init_model') and served['init_model'] != self.model:
            raise ProposalError(f'served model mismatch: requested {self.model}, CLI initialised {served["init_model"]} (integrity)')
        return batch

    def propose(self, n: int, context: dict, island: str, generation: int,
                seed: int, call_id: str) -> ProposalBatch:
        """Issue one proposal call with bounded retries.

        Each failed attempt is archived and charged: its reported usage when the
        provider returned counters, otherwise the full reservation. The charges
        are added to the eventual successful batch. Integrity failures are never
        retried, and a persistent failure pauses discovery for investigation.
        """
        if n < 1 or re.fullmatch(r'[A-Za-z0-9_.-]{1,180}', call_id) is None:
            raise ValueError('invalid proposal count or call identifier')
        prompt = self._prompt(n, context, island, generation, seed)
        if self.mode == 'v3':
            self.preflight(prompt)
        counted = self._counted_attempts(call_id)
        if counted >= self.archived_attempt_limit:
            raise ProposalError(f'provider call {call_id} already failed {counted} times; '
                                'discovery paused for investigation')
        failures, waited = 0, 0.
        while True:
            try:
                batch = self._attempt(n, prompt, call_id)
                if batch.specs:
                    return self._credit(batch, call_id)
                failure = 'provider returned no valid candidates'
            except ProposalError as exc:
                if any(text in str(exc) for text in INTEGRITY_FAILURES):
                    raise
                failure = str(exc)
            archived = self._archive_attempt(call_id, failure,
                                             self.reserve_tokens(n, context, island, generation, seed))
            if archived['limit']:
                delay = self.limit_wait_s
                if waited + delay > self.limit_max_wait_s:
                    raise ProposalError(f'provider usage or rate limit persisted for {waited/3600:.1f} h: '
                                        f'{archived["archived_reason"]}')
                waited += delay
            else:
                failures += 1
                if (failures >= self.attempts_per_call or
                        self._counted_attempts(call_id) >= self.archived_attempt_limit):
                    raise ProposalError(f'provider call failed after {failures} attempts: '
                                        f'{archived["archived_reason"]}')
                delay = self.retry_backoff_s[min(failures, len(self.retry_backoff_s)) - 1]
            if self._unknown_usage_charges() > self.unknown_usage_charge_cap:
                raise ProposalError('unresolved provider usage charges exceed the declared cap; '
                                    'reconcile before continuing')
            if self.retry_deadline_unix is not None and time.time() + delay >= self.retry_deadline_unix:
                raise ProposalError('a provider retry would pass the search deadline')
            time.sleep(delay)

    def _attempt(self, n: int, prompt: str, call_id: str, *, record_metadata: dict | None = None) -> ProposalBatch:
        fingerprint = _digest({'prompt': prompt, 'provider': self.identity})
        record_path = self.root / f'{call_id}.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get('fingerprint') != fingerprint:
                raise ProposalError('proposal replay identity mismatch')
            if record.get('status') == 'response_committed':
                if hashlib.sha256(record['stdout'].encode()).hexdigest() != record.get('stdout_sha256'):
                    raise ProposalError('committed proposal response checksum mismatch')
                if record.get('returncode') != 0:
                    raise ProposalError('recorded provider call failed; reconcile usage before continuing')
                return self._parse_response(record['stdout'], n)
            if record.get('status') in {'started', 'failed'}:
                raise ProposalError('previous provider call has no committed response; reconcile before retrying to avoid duplicate spend')
        record = {'fingerprint': fingerprint, 'identity': self.identity,
                  'status': 'started', 'prompt': prompt, 'call_id': call_id}
        if record_metadata:
            record.update(record_metadata)
        if self.mode == 'v3':
            record['context_preflight'] = self.preflight(prompt)
            record['evidence_delivery'] = {'mode': 'inline_complete_dossier',
                                           'dossier_sha256': next((c[2] for c in self._dossier_cache.values()), None)}
        if hashlib.sha256(self.binary.read_bytes()).hexdigest() != self.binary_sha256:
            raise ProposalError('provider binary changed after the protocol was initialized')
        _atomic(record_path, record)
        cwd = self.root / 'isolated_cwd'
        cwd.mkdir(parents=True, exist_ok=True)
        if self.provider == 'codex_cli':
            if self.mode == 'v3':
                from .v3 import V3_INSTRUCTION
                instructions = V3_INSTRUCTION
            else:
                instructions = SYSTEM_PROMPT
            (cwd/'instructions.txt').write_text(instructions)
        allowed = ('HOME', 'PATH', 'LANG', 'LC_ALL', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
                   'ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN', 'CLAUDE_CONFIG_DIR',
                   'CODEX_HOME', 'OPENAI_API_KEY')
        env = {key: os.environ[key] for key in allowed if key in os.environ}
        if self.provider == 'claude_cli':
            env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] = str(self._turn_output_cap())
            env['DISABLE_AUTOUPDATER'] = '1'
        try:
            result = subprocess.run(self.command(), input=prompt, cwd=cwd, env=env,
                                    capture_output=True, text=True, timeout=self.timeout_s)
        except (OSError, subprocess.TimeoutExpired) as exc:
            record.update(status='failed', failure_type=type(exc).__name__)
            _atomic(record_path, record)
            raise ProposalError('provider invocation failed; recorded without credential-bearing diagnostics') from exc
        record.update(status='response_committed', stdout=result.stdout,
                      stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                      returncode=result.returncode,
                      stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest())
        _atomic(record_path, record)
        if result.returncode:
            raise ProposalError(f'provider exited {result.returncode}; reconcile usage before continuing')
        return self._parse_response(result.stdout, n)

    def _archived(self, call_id: str | None = None) -> list[dict]:
        failed = self.root / 'failed'
        folders = [failed / call_id] if call_id else sorted(failed.glob('*'))
        return [json.loads(path.read_text()) for folder in folders for path in sorted(folder.glob('attempt-*.json'))]

    def _turn_output_cap(self) -> int:
        return max(int(self.max_output_tokens), V3_TURN_OUTPUT_CAP) if self.mode == 'v3' else int(self.max_output_tokens)

    def _provider_env(self) -> dict:
        allowed = ('HOME', 'PATH', 'LANG', 'LC_ALL', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
                   'ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN', 'CLAUDE_CONFIG_DIR', 'CODEX_HOME', 'OPENAI_API_KEY')
        env = {key: os.environ[key] for key in allowed if key in os.environ}
        env['CLAUDE_CODE_MAX_OUTPUT_TOKENS'] = str(self._turn_output_cap())
        env['DISABLE_AUTOUPDATER'] = '1'
        return env

    @staticmethod
    def session_id(stdout: str) -> str | None:
        """The CLI session of a stream-json call (init or result event)."""
        for line in (stdout or '').splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and isinstance(event.get('session_id'), str) and event['session_id']:
                return event['session_id']
        return None

    def repair(self, n: int, message: str, call_id: str, parent_call_id: str) -> ProposalBatch | None:
        """One bounded repair turn inside the conversation of ``parent_call_id``.

        Never retried and never fatal: without a resumable parent session, after an uncommitted
        earlier attempt, or on a provider error it returns None and the generation proceeds with
        the candidates it has. A committed response is replayed from its record, not called again.
        """
        if self.mode != 'v3' or n < 1:
            return None
        if re.fullmatch(r'[A-Za-z0-9_.-]{1,180}', call_id) is None:
            raise ValueError('invalid repair call identifier')
        if self.provider == 'codex_cli':
            prompt = self._codex_repair_prompt(message, parent_call_id)
            if prompt is None:
                return None
            self.preflight(prompt)
            # One attempt only. _attempt replays committed responses, and refuses an
            # unresolved earlier invocation without buying another request.
            return self._attempt(n, prompt, call_id, record_metadata={
                'kind': 'repair', 'resume_of': parent_call_id,
                'repair_delivery': 'complete_parent_prompt_and_response',
                'reserved_tokens': self.repair_reserve_tokens(message, parent_call_id)})
        parent_path = self.root / f'{parent_call_id}.json'
        if not parent_path.exists():
            return None
        parent = json.loads(parent_path.read_text())
        session = self.session_id(parent.get('stdout') or '') if parent.get('status') == 'response_committed' else None
        if not session:
            return None
        fingerprint = _digest({'prompt': message, 'provider': self.identity, 'resume': parent_call_id, 'session': session})
        record_path = self.root / f'{call_id}.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if (record.get('fingerprint') != fingerprint or record.get('status') != 'response_committed'
                    or record.get('returncode') != 0
                    or hashlib.sha256(record['stdout'].encode()).hexdigest() != record.get('stdout_sha256')):
                return None
            return self._parse_response(record['stdout'], n)
        if hashlib.sha256(self.binary.read_bytes()).hexdigest() != self.binary_sha256:
            raise ProposalError('provider binary changed after the protocol was initialized')
        record = {'fingerprint': fingerprint, 'identity': self.identity, 'status': 'started', 'prompt': message,
                  'call_id': call_id, 'kind': 'repair', 'resume_of': parent_call_id, 'session_id': session}
        _atomic(record_path, record)
        cwd = self.root / 'isolated_cwd'
        cwd.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(self.command(resume=session), input=message, cwd=cwd, env=self._provider_env(),
                                    capture_output=True, text=True, timeout=self.timeout_s)
        except (OSError, subprocess.TimeoutExpired) as exc:
            record.update(status='failed', failure_type=type(exc).__name__)
            _atomic(record_path, record)
            return None
        record.update(status='response_committed', stdout=result.stdout,
                      stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
                      returncode=result.returncode,
                      stderr_sha256=hashlib.sha256(result.stderr.encode()).hexdigest())
        _atomic(record_path, record)
        if result.returncode:
            return None
        return self._parse_response(result.stdout, n)

    def _codex_repair_prompt(self, message: str, parent_call_id: str) -> str | None:
        if re.fullmatch(r'[A-Za-z0-9_.-]{1,180}', parent_call_id) is None:
            raise ValueError('invalid parent call identifier')
        path = self.root / f'{parent_call_id}.json'
        if not path.exists():
            return None
        parent = json.loads(path.read_text())
        if parent.get('status') != 'response_committed' or parent.get('returncode') != 0:
            return None
        if (hashlib.sha256(parent['stdout'].encode()).hexdigest() != parent.get('stdout_sha256')
                or _digest({'prompt': parent['prompt'], 'provider': parent['identity']}) != parent.get('fingerprint')):
            raise ProposalError('committed parent proposal checksum mismatch (integrity)')
        envelope = self.envelope(parent['stdout'])
        if not envelope or envelope.get('is_error') or not isinstance(envelope.get('result'), str):
            return None
        return (parent['prompt'] + '\n\n=== PREVIOUS PROPOSAL RESPONSE ===\n' + envelope['result']
                + '\n\n=== ONE BOUNDED REPAIR TURN ===\n' + message)

    def repair_reserve_tokens(self, message: str, parent_call_id: str) -> int:
        if self.provider != 'codex_cli' or self.mode != 'v3':
            return 0  # Preserve the existing native-session reservation policy.
        prompt = self._codex_repair_prompt(message, parent_call_id)
        if prompt is None:
            return 0
        report = self.preflight(prompt)
        return report['estimated_prompt_tokens'] + self.CLI_OVERHEAD_TOKENS + max(65536, 4 * self.max_output_tokens)

    def repair_usage(self, call_id: str) -> int:
        """Reported token usage of a committed repair record (0 when there is none)."""
        record_path = self.root / f'{call_id}.json'
        if not record_path.exists():
            return 0
        record = json.loads(record_path.read_text())
        envelope = self.envelope(record.get('stdout') or '')
        counters = (envelope or {}).get('usage')
        if counters is None:
            return int(record.get('reserved_tokens', 0))
        return sum(int(counters.get(key) or 0) for key in
                   ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'))

    def _counted_attempts(self, call_id: str) -> int:
        return sum(1 for attempt in self._archived(call_id) if not attempt.get('limit'))

    def _unknown_usage_charges(self) -> int:
        return sum(int(attempt['charged_tokens']) for attempt in self._archived() if not attempt.get('usage_known'))

    def _archive_attempt(self, call_id: str, reason: str, reservation: int) -> dict:
        record_path = self.root / f'{call_id}.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else {'status': 'absent'}
        usage, message = None, reason
        if record.get('status') == 'response_committed':
            envelope = self.envelope(record.get('stdout') or '')
            if isinstance(envelope, dict):
                counters = envelope.get('usage')
                fields = ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')
                if isinstance(counters, dict) and all(isinstance(counters.get(key, 0), int) and counters.get(key, 0) >= 0
                                                      for key in fields):
                    usage = sum(counters.get(key, 0) for key in fields)
                if envelope.get('is_error') and envelope.get('result'):
                    message = f'{reason}: {str(envelope["result"])[:500]}'
        folder = self.root / 'failed' / call_id
        folder.mkdir(parents=True, exist_ok=True)
        index = len(list(folder.glob('attempt-*.json'))) + 1
        archived = {**record, 'archived_reason': message, 'usage_known': usage is not None,
                    'charged_tokens': int(usage if usage is not None else reservation),
                    'limit': bool(LIMIT_PATTERN.search(message)), 'archived_unix': time.time()}
        # Keep the evidence before releasing the call identifier for a new attempt.
        _atomic(folder / f'attempt-{index:03d}.json', archived)
        record_path.unlink(missing_ok=True)
        return archived

    def _credit(self, batch: ProposalBatch, call_id: str) -> ProposalBatch:
        attempts = self._archived(call_id)
        if attempts:
            batch.usage_tokens += sum(int(attempt['charged_tokens']) for attempt in attempts)
            batch.metadata['failed_provider_attempts'] = [
                {'reason': attempt['archived_reason'][:300], 'charged_tokens': attempt['charged_tokens'],
                 'usage_known': attempt['usage_known'], 'limit': attempt['limit']} for attempt in attempts]
        return batch


class ReplayProposer:
    """Explicit deterministic proposal fixture; not presented as an LLM discovery run."""
    def __init__(self, batches: dict[str, list[CandidateSpec]]):
        self.batches = batches
        self.identity = {'provider': 'replay', 'batches_hash': _digest({
            key: [json.loads(to_json(spec)) for spec in specs] for key, specs in sorted(batches.items())})}

    def propose(self, n, context, island, generation, seed, call_id):
        del context, generation, seed
        return ProposalBatch(list(self.batches.get(call_id, self.batches.get(island, [])))[:n], 0,
                             {'source': 'explicit replay fixture'})

    def reserve_tokens(self, *args):
        return 0
