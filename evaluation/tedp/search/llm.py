"""Candidate proposer: headless Claude Code (`claude -p`), per user decision.

Every prompt and raw response is logged verbatim to results/llm_log/ for
reproducibility. The proposer returns parsed CandidateSpecs; malformed
entries are dropped (Tier 0 catches anything that slips through).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import expr
from ..spec import (
    CandidateSpec,
    SpecError,
    Term,
    from_json,
    search_policy_reasons,
    to_json,
)

REPO = Path(__file__).resolve().parents[3]
LLM_LOG = REPO / "results" / "llm_log"

SYSTEM_BLOCK = """You are the proposal engine inside an autonomous turbulence-model discovery loop.
You propose corrections to k-omega SST as JSON. Each candidate has two channels:
  bdelta: list of [tensor, expression] terms — the constitutive correction
          b^Delta = sum g_n(vars) * T^(n)
  rsource: list of [tensor, expression] terms — production correction
          R = 2 k (sum h_m(vars) * T^(m)) : grad(U)

STRICT GRAMMAR (anything else is auto-rejected):
  variables: I1 I2 I3 I4 I5 Ret F1 PoE
    (I1=tr(S^2)>=0, I2=tr(W^2)<=0 with S=strain/omega, W=rotation/omega;
     Ret=k/(nu*omega); F1=SST blending; PoE=production/dissipation)
  free constants: c0..c7 (at most 8 independently tunable coefficients; values
    you give are STARTING POINTS). Every gate-legal structure is tuned by
    CMA-ES on real CFD (two hills calibration cases, 6 evaluations per
    constant) before it is compared with anything, and both the tuned form
    and your original are then confirmed on all 13 cases. So spend capacity
    on structure and give physically sensible starting constants; do not try
    to hand-fit numbers.
    COMPLEXITY CHARGE: the frozen objective adds 0.005 per effective
    parameter = free constants PLUS every distinct non-0/1 numeric literal,
    including epsilons such as 1e-3, 1e-6, 1e-8 and tanh sharpness numbers.
    max(1.0, ...), abs(...), I1+abs(I2) and 1-F1 are free. Prefer those free
    guard idioms; a "9-parameter" model that is really 5 constants plus four
    guards pays for nine. There is no search-admission cap on effective
    parameters, term count, or AST size.
    Do not synthesize an uncharged coefficient by repeated addition
    (I1+I1+I1+I1), repeated identical operands, or literal-only arithmetic
    (3/4 or 1+1). The search layer detects and rejects those forms. Write an
    intended adjustable coefficient once as cN instead.
  operators: + - * /   functions: tanh exp sqrt abs min(a,b) max(a,b)
  tensors: T1..T10 (Pope basis of normalized S,W). The calibration flows are
  2-D: only T1,T2,T3 are independent there; T4+ matter only via 3-D holdout.

HARD GATES a candidate must pass (design for them; in the last run five of
eight gate rejections were the SAME T2 term with a
1/((1e-3+sqrt(I1))*(1e-3+sqrt|I2|)) denominator, which explodes at small
invariants and fails boundedness — never use that recipe):
  - safety clamps are protocol constants, not candidate genes: gMax=10 and
    rMaxFactor=5. A candidate may not materially rely on either clamp.
  - boundedness: aggregate |sum(g*T)|_Frobenius <= 1.65 for bDelta; R is
    bounded on its actual scalar observable 2*bR:Shat. These are tested on
    solver-consistent states with I1 and |I2| up to 900. Tensor norms grow
    as: |T1|~sqrt(I1), |T2|~2*sqrt(I1*|I2|), |T3|~I1, |T4|~|I2|, |T5+|~higher
    powers. EVERY term needs a normalizer matching its tensor's growth.
    Gate-verified idioms (all passed Tier 0 and Tier 1 in this project):
      R  on T1: c0*(1-F1)*tanh(c1*abs(I1+I2)/(I1+abs(I2)))/max(1.0,c2*I1)
      R  on T1: c0*tanh(c1*max(PoE-1,0))*tanh(c2*abs(I1+I2)/(I1+abs(I2)))/max(1.0,c3*I1)
      bDelta T2: c0*(1+F1)/max(1.0,c1*sqrt(max(I1*abs(I2),0.0)))
      bDelta T3: c0*(1-F1)/max(1.0,c1*I1)
      bDelta T1: c0*(1-F1)*tanh(c1*Ret)/max(1.0,c2*sqrt(max(I1,0.0)))
      bDelta T4: c0*tanh(abs(I2))/max(1.0,c1*abs(I2))
    with |c0| <= ~0.5 for bDelta amplitudes.
  - realizability: total b (Boussinesq + b^Delta) inside the Lumley triangle
    on all states with |S/omega|, |W/omega| up to 3 AND on the DNS states —
    the SUM over your bdelta terms of max|g_n*T_n| within that envelope must
    stay below ~0.30 (this was the top rejection reason last generation:
    individually-bounded terms stacked past the triangle)
  - stock-SST recovery: correction TENSORS must vanish as I1,I2 -> 0; a
    constant low-strain T1 coefficient is valid because T1 itself vanishes
  - channel-flow Cf within 2% of SST (a PLAN-MANDATED success criterion,
    not a formality); typical useful magnitudes are SMALL (the physical
    range in the calibration flows is I1 in [0, 0.25])
  - homogeneous shear: positive k/omega and realizable full anisotropy are
    hard requirements throughout. The full k, epsilon, b11, b22, b33 and b12
    histories are scored against bundled AGARD HOM23 DNS as a reported Pareto
    diagnostic (not an uncalibrated hard cutoff); prefer not to worsen it. In
    homogeneous simple
    shear F1=0 and I1+I2=0, so (1-F1) ALONE does not protect this gate. A
    topology factor such as tanh(c*abs(I1+I2)/(eps+I1+abs(I2))) switches off
    exactly in simple shear while remaining active away from that topology.

MEASURED PHYSICS — read before proposing an R (production) term:
  An UNGATED R = c*T1 adds ~c times SST's own production EVERYWHERE.
  At c = 0.41 (the value sparse regression infers from DNS, and close to
  SpaRTA's published 0.4) that is +41% production in every cell: excellent
  in separated regions, but it inflates k in the equilibrium log layer and
  moves channel Cf by +17% -> instant Tier-1 rejection. Measured here:
      R = 0.41*T1            -> Cf ratio 1.173  REJECTED
      R = 0.10*T1            -> Cf ratio 1.051  REJECTED
      R = 0.41*(1-F1)*T1     -> Cf ratio 1.000  PASSES
  F1 -> 1 in equilibrium boundary layers and -> 0 in free-shear/separated
  regions, so (1-F1) (or another indicator that vanishes in equilibrium,
  e.g. a PoE-departure gate like tanh(c*(PoE-1)) or max(PoE-1,0)) buys the
  separated-flow benefit while leaving attached flow untouched. THIS IS THE
  MOST PROMISING OPEN DIRECTION: every high-scoring form found so far
  (including the published SpaRTA models) fails the channel gate precisely
  because its R term is ungated. A gated R that keeps the separated-flow
  gain would be a genuine discovery, but (1-F1) alone worsens the reported
  HOM23 homogeneous-shear metric. To preserve the measured term over the
  calibration range while satisfying the extreme-state boundedness battery,
  use e.g. h = c0*(1-F1)/max(1.0,4.0*I1), c0~0.41.
  - NOVELTY: do not rediscover the library (fitted to 3% rms => rejected):
    QCR (T2/sqrt), Wallin-Johansson (N(PoE)-rational T1+T2), Shih-Zhu-Lumley,
    Craft-Launder-Suga, Gatski-Speziale, Lien-Chen-Leschziner, SpaRTA M1-M3
    and ANY pure polynomial in (I1,I2) times T1..T4 (a linear net catches
    those exactly). Use the nonlinear vocabulary: tanh/exp/sqrt compositions,
    Ret, F1, PoE dependence, ratios — structure the polynomials cannot mimic.

NOVELTY: the search state lists the structural families already evaluated
on every island with their best scores. At least two of your proposals must
be of a family that is NOT on that list (a different tensor pairing, a
different gate variable, or a different channel combination). Families are
capped per generation across the islands; a re-spelled coefficient or a
re-scaled constant is the same family and will be rejected as a duplicate
of effort. Prefer mechanisms that could remove a known weakness of the
current front (listed per case in the state) rather than decorating it.

Respond with ONLY a JSON array (no prose, no code fences) of exactly the
requested number of candidates:
[{"name": "...", "bdelta": [["T2","c0*tanh(c1*I1)*..."]], "rsource": [["T1","..."]],
  "constants": [0.1, 3.0], "rationale": "one line"}]
"""


@dataclass
class Proposal:
    spec: CandidateSpec
    rationale: str


def find_claude_binary() -> str:
    """PATH first; else the newest VS Code extension's bundled binary."""
    import shutil as _shutil

    on_path = _shutil.which("claude")
    if on_path:
        return on_path
    candidates = sorted(
        Path.home().glob(
            ".vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"
        )
    )
    if candidates:
        return str(candidates[-1])
    raise FileNotFoundError("no claude CLI found (PATH or VS Code extension)")


@dataclass
class ProposalCallResult:
    call_id: str
    status: str
    proposals: list[Proposal]
    rejections: list[dict[str, Any]]
    attempts: int
    reason: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _proposal_payload(proposal: Proposal) -> dict[str, Any]:
    return {"spec": json.loads(to_json(proposal.spec)), "rationale": proposal.rationale}


def _proposal_from_payload(payload: dict[str, Any]) -> Proposal:
    return Proposal(
        from_json(json.dumps(payload["spec"])), str(payload.get("rationale", ""))
    )


class ClaudeCodeProposer:
    def __init__(self, model: str = "opus", timeout_s: int = 900):
        self.model = model
        self.timeout_s = timeout_s
        self.binary = str(Path(find_claude_binary()).resolve())
        version = subprocess.run(
            [self.binary, "--version"], capture_output=True, text=True,
            timeout=30, check=False,
        )
        version_text = (version.stdout or version.stderr).strip()
        if version.returncode != 0 or not version_text:
            raise RuntimeError(
                f"cannot identify Claude binary {self.binary}: {version.stderr[:200]}"
            )
        binary_path = Path(self.binary)
        self.binary_manifest = {
            "path": self.binary,
            "sha256": _sha256(binary_path),
            "bytes": binary_path.stat().st_size,
            "version": version_text,
        }

    @staticmethod
    def _prompt(
        n_children: int,
        context_block: str,
        generation: int,
        island: str,
        profile_block: str,
    ) -> str:
        return (
            SYSTEM_BLOCK
            + f"\n\n## Current search state (generation {generation}, island {island})\n"
            + profile_block
            + "\n\n"
            + context_block
            + f"\n\nPropose {n_children} candidates now (JSON array only)."
        )

    def command(self, prompt: str) -> list[str]:
        """Hermetic print-mode invocation: no tools, settings, MCP or session."""
        return [
            self.binary,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
            "--tools",
            "",
            "--safe-mode",
            "--restricted",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--setting-sources",
            "",
            "--settings",
            "{}",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--no-chrome",
            "--permission-mode",
            "dontAsk",
        ]

    def _transaction_path(self, call_id: str) -> Path:
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,180}", call_id) is None:
            raise ValueError(f"unsafe proposal call id {call_id!r}")
        return LLM_LOG / "calls" / f"{call_id}.json"

    def _result_from_record(self, record: dict[str, Any]) -> ProposalCallResult:
        return ProposalCallResult(
            call_id=str(record["call_id"]),
            status=str(record["status"]),
            proposals=[
                _proposal_from_payload(proposal)
                for proposal in record.get("proposals", [])
            ],
            rejections=list(record.get("parse_rejections", [])),
            attempts=len(record.get("attempts", [])),
            reason=str(record.get("reason", "")),
        )

    def propose_transaction(
        self,
        call_id: str,
        n_children: int,
        context_block: str,
        generation: int,
        island: str,
        profile_block: str = "",
        max_attempts: int = 2,
    ) -> ProposalCallResult:
        """Durable bounded proposal call, replayable after a process crash."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        LLM_LOG.mkdir(parents=True, exist_ok=True)
        path = self._transaction_path(call_id)
        prompt = self._prompt(
            n_children, context_block, generation, island, profile_block
        )
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        if path.exists():
            record = json.loads(path.read_text())
            identity = (
                record.get("prompt_sha256"), record.get("model"),
                record.get("requested"), record.get("binary", {}).get("sha256"),
            )
            expected = (
                prompt_sha, self.model, n_children,
                self.binary_manifest["sha256"],
            )
            if identity != expected:
                raise RuntimeError(f"proposal transaction identity mismatch: {call_id}")
            if record.get("status") in {"complete", "paused"}:
                return self._result_from_record(record)
        else:
            record = {
                "schema_version": 1,
                "call_id": call_id,
                "generation": generation,
                "island": island,
                "model": self.model,
                "requested": n_children,
                "prompt_sha256": prompt_sha,
                "profile_sha256": hashlib.sha256(profile_block.encode()).hexdigest(),
                "context_sha256": hashlib.sha256(context_block.encode()).hexdigest(),
                "binary": self.binary_manifest,
                "status": "scheduled",
                "attempts": [],
                "proposals": [],
                "parse_rejections": [],
            }
            prompt_path = LLM_LOG / "calls" / call_id / "prompt.txt"
            _atomic_text(prompt_path, prompt)
            record["prompt_path"] = str(prompt_path.relative_to(REPO))
            _atomic_json(path, record)

        def response_record_path(attempt: dict[str, Any]) -> Path:
            return REPO / str(attempt["response_record_path"])

        def promote_committed_response(attempt: dict[str, Any]) -> dict[str, Any]:
            response_record = response_record_path(attempt)
            payload = json.loads(response_record.read_text())
            if (
                payload.get("call_id") != call_id
                or payload.get("attempt_id") != attempt.get("attempt_id")
                or not isinstance(payload.get("stdout"), str)
                or not isinstance(payload.get("stderr"), str)
            ):
                raise RuntimeError(
                    f"malformed committed proposal response: {response_record}"
                )
            stdout = payload["stdout"]
            stderr = payload["stderr"]
            if hashlib.sha256(stdout.encode()).hexdigest() != payload.get(
                "stdout_sha256"
            ):
                raise RuntimeError("committed Claude stdout hash mismatch")
            if hashlib.sha256(stderr.encode()).hexdigest() != payload.get(
                "stderr_sha256"
            ):
                raise RuntimeError("committed Claude stderr hash mismatch")
            # Plaintext artifacts are derived from the atomic response record.
            # If a crash happened between these writes they can be recreated
            # without issuing a second, nondeterministic model call.
            _atomic_text(REPO / attempt["response_path"], stdout)
            _atomic_text(REPO / attempt["stderr_path"], stderr)
            attempt.update({
                "status": "response_committed",
                "ended_wall_time": payload["ended_wall_time"],
                "returncode": payload.get("returncode"),
                "execution_failure": str(payload.get("failure", "")),
                "stdout_sha256": payload["stdout_sha256"],
                "stderr_sha256": payload["stderr_sha256"],
            })
            record["status"] = "response_committed"
            _atomic_json(path, record)
            return payload

        # A new process may find either a truly interrupted subprocess or a
        # response sidecar committed just before the main transaction record.
        # The sidecar is the commit marker and always wins over rerunning.
        if record.get("attempts"):
            active = record["attempts"][-1]
            if active.get("status") == "running":
                response_record = response_record_path(active)
                if response_record.is_file():
                    promote_committed_response(active)
                else:
                    active.update({
                        "status": "interrupted",
                        "ended_wall_time": time.time(),
                        "reason": "process ended before a response was committed",
                    })
                    record["status"] = "retrying"
                    record["reason"] = active["reason"]
                    _atomic_json(path, record)

        while True:
            if (
                record.get("attempts")
                and record["attempts"][-1].get("status") == "response_committed"
            ):
                attempt = record["attempts"][-1]
                response = json.loads(response_record_path(attempt).read_text())
                stdout = response["stdout"]
                failure = str(response.get("failure", ""))
                # Parsing is intentionally after the response-commit checkpoint.
                # A parser/process crash re-enters here and consumes the same raw
                # output instead of spending another proposal call.
                proposals, rejections = self._parse_with_rejections(
                    stdout, generation, island
                )
                attempt.update({
                    "status": "complete" if not failure and proposals else "failed",
                    "accepted": len(proposals),
                    "rejected": len(rejections),
                    "reason": failure or (
                        "no valid proposals" if not proposals else ""
                    ),
                })
                record["parse_rejections"].extend([
                    {**rejection, "attempt_id": attempt["attempt_id"]}
                    for rejection in rejections
                ])
                if not failure and proposals:
                    record["status"] = "complete"
                    record["proposals"] = [
                        _proposal_payload(proposal)
                        for proposal in proposals[:n_children]
                    ]
                    record["reason"] = ""
                    _atomic_json(path, record)
                    return self._result_from_record(record)
                record["status"] = "retrying"
                record["reason"] = attempt["reason"]
                _atomic_json(path, record)

            if len(record["attempts"]) >= max_attempts:
                break
            attempt_number = len(record["attempts"]) + 1
            attempt_id = uuid.uuid4().hex
            attempt_dir = LLM_LOG / "calls" / call_id
            response_path = attempt_dir / f"attempt_{attempt_number}_{attempt_id}.stdout"
            stderr_path = attempt_dir / f"attempt_{attempt_number}_{attempt_id}.stderr"
            response_record = (
                attempt_dir / f"attempt_{attempt_number}_{attempt_id}.response.json"
            )
            attempt = {
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "status": "running",
                "started_wall_time": time.time(),
                "response_path": str(response_path.relative_to(REPO)),
                "stderr_path": str(stderr_path.relative_to(REPO)),
                "response_record_path": str(response_record.relative_to(REPO)),
            }
            record["attempts"].append(attempt)
            record["status"] = "running"
            _atomic_json(path, record)
            try:
                result = subprocess.run(
                    self.command(prompt),
                    cwd=REPO,
                    env=dict(os.environ),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                stdout = result.stdout
                stderr = result.stderr
                returncode = result.returncode
                failure = "" if returncode == 0 else f"exit code {returncode}"
            except (subprocess.TimeoutExpired, OSError) as exc:
                stdout = getattr(exc, "stdout", "") or ""
                stderr = getattr(exc, "stderr", "") or ""
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                stderr += f"\n{exc}"
                returncode = None
                failure = f"{type(exc).__name__}: {exc}"
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            response_payload = {
                "schema_version": 1,
                "call_id": call_id,
                "attempt_id": attempt_id,
                "attempt_number": attempt_number,
                "ended_wall_time": time.time(),
                "returncode": returncode,
                "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                "failure": failure,
                "stdout": stdout,
                "stderr": stderr,
            }
            _atomic_json(response_record, response_payload)
            promote_committed_response(attempt)
        record["status"] = "paused"
        record["reason"] = (
            f"proposal call exhausted {max_attempts} attempts: "
            f"{record.get('reason', 'unknown failure')}"
        )
        _atomic_json(path, record)
        return self._result_from_record(record)

    def propose(
        self,
        n_children: int,
        context_block: str,
        generation: int,
        island: str,
        profile_block: str = "",
    ) -> list[Proposal]:
        call_id = f"legacy-g{generation:03d}-{island}-{uuid.uuid4().hex}"
        return self.propose_transaction(
            call_id, n_children, context_block, generation, island,
            profile_block, max_attempts=1,
        ).proposals

    def _parse(self, stdout: str, generation: int, island: str) -> list[Proposal]:
        proposals, _ = self._parse_with_rejections(stdout, generation, island)
        return proposals

    def _parse_with_rejections(
        self, stdout: str, generation: int, island: str
    ) -> tuple[list[Proposal], list[dict[str, Any]]]:
        rejections: list[dict[str, Any]] = []
        try:
            envelope = json.loads(stdout)
            if isinstance(envelope, list):
                entries = envelope
            elif isinstance(envelope, dict) and isinstance(
                envelope.get("result"), list
            ):
                entries = envelope["result"]
            elif isinstance(envelope, dict) and isinstance(envelope.get("result"), str):
                text = envelope["result"]
                start = text.find("[")
                if start < 0:
                    return [], [{"scope": "response", "reason": "no JSON array"}]
                entries, _ = json.JSONDecoder().raw_decode(text[start:])
            else:
                return [], [{
                    "scope": "response",
                    "reason": "JSON envelope is neither an array nor a result string",
                }]
        except json.JSONDecodeError:
            text = stdout
            start = text.find("[")
            if start < 0:
                return [], [{"scope": "response", "reason": "no JSON array"}]
            try:
                entries, _ = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError as exc:
                return [], [{
                    "scope": "response", "reason": f"invalid JSON array: {exc}"
                }]
        if not isinstance(entries, list):
            return [], [{"scope": "response", "reason": "proposal payload is not a list"}]
        proposals: list[Proposal] = []
        for i, e in enumerate(entries):
            try:
                if not isinstance(e, dict):
                    raise TypeError("candidate entry is not an object")
                name = e.get("name", "cand")
                if not isinstance(name, str):
                    raise TypeError("name must be a string")

                def terms(channel: str) -> tuple[Term, ...]:
                    raw_terms = e.get(channel, [])
                    if not isinstance(raw_terms, list):
                        raise TypeError(f"{channel} must be a list")
                    parsed: list[Term] = []
                    for index, raw_term in enumerate(raw_terms):
                        if (
                            not isinstance(raw_term, (list, tuple))
                            or len(raw_term) != 2
                            or not all(isinstance(value, str) for value in raw_term)
                        ):
                            raise TypeError(
                                f"{channel}[{index}] must be [tensor, expression] strings"
                            )
                        parsed.append(Term(raw_term[0], raw_term[1]))
                    return tuple(parsed)

                raw_constants = e.get("constants", [])
                if not isinstance(raw_constants, list):
                    raise TypeError("constants must be a list")
                if not all(type(value) in (int, float) for value in raw_constants):
                    raise TypeError("constants must contain only JSON numbers")
                spec = CandidateSpec(
                    name=f"g{generation}_{island}_{i}_{name[:24]}",
                    bdelta=terms("bdelta"),
                    rsource=terms("rsource"),
                    constants=tuple(float(value) for value in raw_constants),
                    gmax=10.0,
                    r_max_factor=5.0,
                )
                spec.validate(search_policy=True)
                policy_reasons = search_policy_reasons(spec)
                if policy_reasons:
                    raise SpecError("; ".join(policy_reasons))
                proposals.append(Proposal(spec, str(e.get("rationale", ""))[:200]))
            except (
                SpecError, expr.GrammarError, ValueError, TypeError, KeyError
            ) as exc:
                rejections.append({
                    "scope": "candidate", "index": i,
                    "reason": f"{type(exc).__name__}: {exc}"[:500],
                    "entry": repr(e)[:1000],
                })
        return proposals, rejections
