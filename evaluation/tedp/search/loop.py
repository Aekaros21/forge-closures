"""Four-island Phase-3 search with CFD-tuned constants (protocol v6).

Design (see PHASE3_PLAN.md):

* four islands are mechanism families, seeded with distinct gate-legal
  mechanisms; island roles are prompt guidance only and never narrow
  admission;
* every gate-legal structure is tuned on CFD (CMA-ES on a two-case
  calibration subset, budget proportional to its constants) before it is
  compared with anything; the untuned original is always confirmed too;
* evaluation is batch-oriented and runs on a local or SLURM backend, with
  every batch checkpointed so an interrupted control process resumes
  without re-running finished cases;
* survival is capacity-stratified (two slots per size class) so a large
  model can only be displaced by a better large model; the plateau counts
  progress in any size class;
* acceptance uses the frozen penalized validation J; the search reports the
  whole raw-error/size front next to that verdict.

Selection uses validation cases only; constants are tuned on calibration
cases only.  The sealed holdout is never touched.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from threadpoolctl import threadpool_info

from .. import candidate as cand
from .. import evaldb, expr, library, runner, scoring, tier0, tier1, tier2
from ..data import mcconkey
from ..spec import (
    CandidateSpec,
    SEARCH_GMAX,
    SEARCH_R_MAX_FACTOR,
    SpecError,
    from_json,
    spec_hash,
    struct_hash,
    to_json,
)
from . import cmaes
from .backend import BudgetExhausted, CaseJob, CaseResult
from .llm import ClaudeCodeProposer, Proposal
from .tuning import CfdTuner, TUNING_CASES, encode_constants, tuning_budget

REPO = Path(__file__).resolve().parents[3]
GENERATIONS_ROOT = REPO / "results" / "phase3_generations"
CHECKPOINT = REPO / "results" / "phase3_population.json"
GATE_LEDGER = REPO / "results" / "phase3_ledger.jsonl"
CANDIDATE_RECORDS = REPO / "results" / "phase3_candidates"
FINALISTS = REPO / "results" / "phase3_finalists.json"
STOCK_CACHE = REPO / "results" / "phase3_stock_cluster.json"
CAMPAIGN = "v1"


def configure_campaign(tag: str) -> None:
    """Point checkpoint, ledger, records and finalists at one campaign.

    Campaign ``v1`` keeps the original file names; any other tag gets its
    own ``results/phase3_<tag>_*`` namespace so campaigns never overwrite
    each other's evidence.
    """
    global CAMPAIGN, GENERATIONS_ROOT, CHECKPOINT, GATE_LEDGER, CANDIDATE_RECORDS, FINALISTS
    if not tag or not tag.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid campaign tag {tag!r}")
    CAMPAIGN = tag
    prefix = "phase3" if tag == "v1" else f"phase3_{tag}"
    GENERATIONS_ROOT = REPO / "results" / f"{prefix}_generations"
    CHECKPOINT = REPO / "results" / f"{prefix}_population.json"
    GATE_LEDGER = REPO / "results" / f"{prefix}_ledger.jsonl"
    CANDIDATE_RECORDS = REPO / "results" / f"{prefix}_candidates"
    FINALISTS = REPO / "results" / f"{prefix}_finalists.json"


ALGORITHM = "synchronous-four-island-cfd-tuned-stratified-v7"
ISLANDS = ("I", "II", "III", "IV")
ISLAND_SIZE = 8              # survivors per island, excluding the stock incumbent
SIZE_CLASSES = (("1-2", 1, 2), ("3-4", 3, 4), ("5-6", 5, 6), ("7+", 7, 10**6))
SLOTS_PER_CLASS = 2
MIGRATE_EVERY = 5
SANITY_CASE = "case_1p0"
SANITY_FACTOR = 1.5
FULL_CASES = tuple(mcconkey.CALIBRATION) + tuple(mcconkey.VALIDATION)
FULL_CASE_EVALS = len(FULL_CASES)
MIN_VAL_IMPROVEMENT = 0.003
DIVERSITY_QUOTA = 2          # children per structural family per island per generation
CROSS_ISLAND_QUOTA = 3       # children per structural family per generation, all islands
REJECTION_ARCHIVE_SIZE = 40
ARCHIVE_CONTEXT_SIZE = 24
CHECKPOINT_SCHEMA = 6
PROTOCOL_MANIFEST_SCHEMA = 6
FINALISTS_SCHEMA = 6
CASE_FAMILIES = {
    "hills": ("case_0p5", "case_0p8", "case_1p0", "case_1p2", "case_1p5"),
    "convdiv": ("convdiv12600", "convdiv20580"),
    "curved_step": ("cbfs13700",),
    "bumps": ("h20", "h26", "h31", "h38", "h42"),
}


def _meaningful_improvement(candidate: float, incumbent: float) -> bool:
    """Protocol threshold with a tiny tolerance for decimal roundoff."""
    return incumbent - candidate >= MIN_VAL_IMPROVEMENT - 1.0e-12


@dataclass(frozen=True)
class IslandProfile:
    key: str
    label: str
    instructions: str

    def prompt_block(self) -> str:
        return (
            f"## Island profile: {self.key} — {self.label}\n"
            f"{self.instructions}\n"
            "This niche is prompt guidance only and never changes admission. "
            "The search adds no cap on effective-parameter count, term count, "
            "or AST size; the unchanged expression grammar provides c0..c7. "
            "Leaving the niche does not cause rejection. Migration examples "
            "are context, not instructions to clone them."
        )


ISLAND_PROFILES: dict[str, IslandProfile] = {
    "I": IslandProfile(
        "I", "production channel, attached flow protected by F1/topology",
        "Refine and extend the production-channel R mechanism whose attached-"
        "flow protection comes from (1-F1) and the simple-shear topology "
        "switch. Push the separated-flow gain further while keeping the "
        "channel Cf gate; local, interpretable mutations of the anchor are "
        "welcome, as are new gates layered on it.",
    ),
    "II": IslandProfile(
        "II", "production channel, F1-free non-equilibrium gates",
        "Explore production-channel R mechanisms that do NOT use F1 at all: "
        "gates built from PoE departure (max(PoE-1,0), tanh(c*(PoE-1))), Ret "
        "ramps and their interactions with the strain/rotation invariants. "
        "The question is whether a state-based indicator can replace SST's "
        "own blending function.",
    ),
    "III": IslandProfile(
        "III", "constitutive channel only (bDelta, no R)",
        "Propose anisotropy corrections in the bDelta channel with an EMPTY "
        "rsource: T2/T3/T4 (and T1 eddy-viscosity relaxation) with F1, Ret or "
        "PoE-dependent amplitudes and normalizers. The question is whether "
        "anisotropy alone can matter once its constants are tuned on CFD.",
    ),
    "IV": IslandProfile(
        "IV", "coupled channels, unrestricted capacity",
        "Explore coupled bDelta + R mechanisms and multi-term nonlinear "
        "interactions with as many independently tunable constants as the "
        "mechanism needs (up to c0..c7). Seek raw validation error below the "
        "measured ceiling; the tuning stage will fit your constants on CFD, "
        "so spend capacity on structure, not on hand-picked numbers.",
    ),
}

PROTOCOL_FILES = (
    "scripts/run_phase3.py",
    "scripts/tedp_cluster_worker.py",
    "src/tedp/search/loop.py",
    "src/tedp/search/llm.py",
    "src/tedp/search/cmaes.py",
    "src/tedp/search/backend.py",
    "src/tedp/search/tuning.py",
    "src/tedp/spec.py",
    "src/tedp/expr.py",
    "src/tedp/candidate.py",
    "src/tedp/tensors.py",
    "src/tedp/scoring.py",
    "src/tedp/evaldb.py",
    "src/tedp/tier0.py",
    "src/tedp/tier1.py",
    "src/tedp/tier2.py",
    "src/tedp/runner.py",
    "src/tedp/foammesh.py",
    "src/tedp/repro.py",
    "src/tedp/library.py",
    "src/tedp/casegen.py",
    "src/tedp/data/mcconkey.py",
    "src/kOmegaSSTBasis/Make/files",
    "src/kOmegaSSTBasis/Make/options",
    "src/kOmegaSSTBasis/basisExpr/exprNode.C",
    "src/kOmegaSSTBasis/basisExpr/exprNode.H",
    "src/kOmegaSSTBasis/basisExpr/exprParser.C",
    "src/kOmegaSSTBasis/basisExpr/exprParser.H",
    "src/kOmegaSSTBasis/basisTensors/integrityBasis.H",
    "src/kOmegaSSTBasis/kOmegaSSTBasis.C",
    "src/kOmegaSSTBasis/kOmegaSSTBasis.H",
    "src/kOmegaSSTBasis/makeKOmegaSSTBasis.C",
    "specs/phase3_islands/manifest.json",
    "data/mcconkey/REF.csv",
    "data/external/agard_HOM23/hom23xU.dat",
    "data/external/agard_HOM23/hom23uU.dat",
    "data/external/agard_HOM23/hom23wU.dat",
    "results/phase2/apriori_table.pkl",
)


@dataclass
class Member:
    spec: CandidateSpec
    val_j: float
    cal_j: float
    per_case: dict[str, float] = field(default_factory=dict)
    novelty_best_rms: float = float("inf")
    val_raw: float = float("nan")
    cal_raw: float = float("nan")
    origin: str = "unknown"

    @property
    def complexity_penalty(self) -> float:
        return scoring.PARSIMONY * self.spec.n_parameters()

    @property
    def n_parameters(self) -> int:
        return self.spec.n_parameters()

    @property
    def size_class(self) -> str:
        return size_class(self.n_parameters)


def size_class(n_parameters: int) -> str:
    for key, low, high in SIZE_CLASSES:
        if low <= n_parameters <= high:
            return key
    return "0"


@dataclass
class SearchCfg:
    max_generations: int = 20
    tier2_eval_cap: int = 20000
    core_hour_cap: float = 11000.0
    plateau_generations: int = 6
    children_per_island: int = 4
    llm_max_attempts: int = 2
    model: str = "opus"
    seed: int = 0
    tuning_enabled: bool = True
    tuning_cases: tuple[str, ...] = TUNING_CASES
    evals_per_constant: int = 6
    max_tuning_evals: int = 48
    inject_apriori_point: bool = True
    campaign: str = "v1"

    def validate(self) -> None:
        if self.children_per_island < 1:
            raise ValueError("children_per_island must be positive")
        if self.llm_max_attempts < 1 or self.llm_max_attempts > 3:
            raise ValueError("llm_max_attempts must be between 1 and 3")
        if self.max_generations < 1 or self.plateau_generations < 1:
            raise ValueError("generation and plateau limits must be positive")
        if self.tier2_eval_cap < FULL_CASE_EVALS + 1:
            raise ValueError("Tier-2 cap must allow at least one complete candidate")
        if self.core_hour_cap <= 0:
            raise ValueError("core-hour cap must be positive")
        if not self.tuning_cases or any(
            case not in mcconkey.CALIBRATION for case in self.tuning_cases
        ):
            raise ValueError("tuning cases must be calibration cases")
        if self.evals_per_constant < 1 or self.max_tuning_evals < 1:
            raise ValueError("tuning budget must be positive")


@dataclass
class SearchState:
    islands: dict[str, list[Member]]
    evaluated_archive: dict[str, list[Member]] = field(
        default_factory=lambda: {island: [] for island in ISLANDS}
    )
    rejection_archive: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {island: [] for island in ISLANDS}
    )
    generation: int = 0
    tier2_evals: int = 0
    tier2_reserved: int = 0
    core_hours_spent: float = 0.0
    best_val_history: list[float] = field(default_factory=list)
    best_raw_history: list[float] = field(default_factory=list)
    best_class_history: list[dict[str, float]] = field(default_factory=list)
    seen_spec_hashes: set[str] = field(default_factory=set)
    pending_generation: dict[str, Any] | None = None
    run_id: str = ""
    protocol_fingerprint: str = ""
    protocol_manifest: dict[str, Any] = field(default_factory=dict)
    status: str = "initialized"
    stop_reason: str = ""
    precheck: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON, checkpoint, ledger, records
# ---------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8") as fh:
            json.dump(_json_safe(payload), fh, indent=1, sort_keys=True, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
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
        if tmp.exists():
            tmp.unlink()


def _member_payload(member: Member) -> dict[str, Any]:
    return {
        "spec": json.loads(to_json(member.spec)),
        "val_j": member.val_j, "cal_j": member.cal_j,
        "val_raw": member.val_raw, "cal_raw": member.cal_raw,
        "per_case": member.per_case, "novelty_best_rms": member.novelty_best_rms,
        "origin": member.origin,
    }


def _member_from_payload(payload: dict[str, Any]) -> Member:
    spec = from_json(json.dumps(payload["spec"]))
    penalty = scoring.PARSIMONY * spec.n_parameters()
    val_j = float(payload["val_j"])
    cal_j = float(payload["cal_j"])
    return Member(
        spec=spec, val_j=val_j, cal_j=cal_j,
        val_raw=float(payload.get("val_raw", val_j - penalty)),
        cal_raw=float(payload.get("cal_raw", cal_j - penalty)),
        per_case={
            k: (float("nan") if v is None else float(v))
            for k, v in payload.get("per_case", {}).items()
        },
        novelty_best_rms=(
            float("inf") if payload.get("novelty_best_rms") is None
            else float(payload["novelty_best_rms"])
        ),
        origin=str(payload.get("origin", "legacy")),
    )


def save_checkpoint(state: SearchState) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "algorithm": ALGORITHM,
        "run_id": state.run_id,
        "protocol_fingerprint": state.protocol_fingerprint,
        "protocol_manifest": state.protocol_manifest,
        "status": state.status,
        "stop_reason": state.stop_reason,
        "precheck": state.precheck,
        "generation": state.generation,
        "tier2_evals": state.tier2_evals,
        "tier2_reserved": state.tier2_reserved,
        "core_hours_spent": state.core_hours_spent,
        "best_val_history": state.best_val_history,
        "best_raw_history": state.best_raw_history,
        "best_class_history": state.best_class_history,
        "seen_spec_hashes": sorted(state.seen_spec_hashes),
        "pending_generation": state.pending_generation,
        "rejection_archive": state.rejection_archive,
        "evaluated_archive": {
            island: [_member_payload(m) for m in state.evaluated_archive[island]]
            for island in ISLANDS
        },
        "islands": {
            island: [_member_payload(m) for m in members]
            for island, members in state.islands.items()
        },
    }
    _atomic_json(CHECKPOINT, payload)


def load_checkpoint(expected_fingerprint: str | None = None) -> SearchState | None:
    if not CHECKPOINT.exists():
        return None
    payload = json.loads(CHECKPOINT.read_text())
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError(
            f"refusing incompatible Phase-3 checkpoint schema "
            f"{payload.get('schema_version')!r}; archive it before a clean run"
        )
    actual = str(payload.get("protocol_fingerprint", ""))
    if expected_fingerprint is not None and actual != expected_fingerprint:
        raise RuntimeError(
            "refusing Phase-3 resume under a different protocol fingerprint: "
            f"checkpoint={actual}, current={expected_fingerprint}"
        )
    return SearchState(
        islands={
            island: [_member_from_payload(m) for m in payload["islands"][island]]
            for island in ISLANDS
        },
        evaluated_archive={
            island: [
                _member_from_payload(m)
                for m in payload.get("evaluated_archive", {}).get(island, [])
            ]
            for island in ISLANDS
        },
        rejection_archive={
            island: list(payload.get("rejection_archive", {}).get(island, []))
            for island in ISLANDS
        },
        generation=int(payload["generation"]),
        tier2_evals=int(payload["tier2_evals"]),
        tier2_reserved=int(payload.get("tier2_reserved", 0)),
        core_hours_spent=float(payload.get("core_hours_spent", 0.0)),
        best_val_history=[float(v) for v in payload["best_val_history"]],
        best_raw_history=[float(v) for v in payload["best_raw_history"]],
        best_class_history=[
            {str(k): float(v) for k, v in row.items()}
            for row in payload.get("best_class_history", [])
        ],
        seen_spec_hashes=set(payload.get("seen_spec_hashes", [])),
        pending_generation=payload.get("pending_generation"),
        run_id=str(payload["run_id"]),
        protocol_fingerprint=actual,
        protocol_manifest=payload.get("protocol_manifest", {}),
        status=str(payload.get("status", "legacy")),
        stop_reason=str(payload.get("stop_reason", "")),
        precheck=dict(payload.get("precheck", {})),
    )


def append_gate_ledger(payload: dict[str, Any]) -> None:
    GATE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(GATE_LEDGER, "a") as fh:
        fh.write(json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _candidate_record_path(state: SearchState, item: dict[str, Any]) -> Path:
    return (
        CANDIDATE_RECORDS / state.protocol_fingerprint / state.run_id
        / f"g{int(item['generation']):03d}_{item['island']}_{item['item_id']}.json"
    )


def _record_item(state: SearchState, item: dict[str, Any], event: str) -> None:
    payload = {
        "schema_version": 2, "run_id": state.run_id,
        "protocol_fingerprint": state.protocol_fingerprint, "event": event,
        "tier2_evals_after_event": state.tier2_evals,
        "core_hours_after_event": state.core_hours_spent,
        "candidate": item,
    }
    _atomic_json(_candidate_record_path(state, item), payload)
    ledger_row = {
        "event_id": hashlib.sha256(
            f"{state.run_id}:{item['item_id']}:{event}:{item.get('status')}:"
            f"{len(item.get('events', []))}".encode()
        ).hexdigest(),
        "event": event, "run_id": state.run_id,
        "protocol_fingerprint": state.protocol_fingerprint,
        "generation": item["generation"], "island": item["island"],
        "item_id": item["item_id"], "spec_hash": item["spec_hash"],
        "origin": item["origin"], "parent_hash": item.get("parent_hash"),
        "n_parameters": item.get("n_parameters"), "status": item.get("status"),
        "tier2_evals": state.tier2_evals, "core_hours": state.core_hours_spent,
    }
    for key in ("apriori_raw", "sanity", "tuning", "reject_reasons"):
        if key in item:
            ledger_row[key] = item[key]
    if "full" in item:
        ledger_row.update({
            key: item["full"][key]
            for key in ("cal_raw", "cal_j", "val_raw", "val_j", "complexity_penalty")
        })
    append_gate_ledger(ledger_row)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _file_provenance(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"sha256": _sha256(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


PACKAGE_NAMES = ("numpy", "scipy", "fluidfoam", "cma", "threadpoolctl")


def _python_manifest() -> dict[str, Any]:
    return {
        "executable": str(Path(sys.executable).resolve()),
        "version": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in PACKAGE_NAMES},
        "effective_threadpools": sorted(
            threadpool_info(),
            key=lambda info: (str(info.get("internal_api")), str(info.get("filepath"))),
        ),
    }


def _normalized_seeds(
    seeds: list[CandidateSpec] | dict[str, list[CandidateSpec]],
) -> dict[str, list[CandidateSpec]]:
    if isinstance(seeds, dict):
        if set(seeds) != set(ISLANDS):
            raise ValueError(f"seed map must have islands {ISLANDS}")
        out = {island: list(seeds[island]) for island in ISLANDS}
    else:
        out = {island: [] for island in ISLANDS}
        for index, spec in enumerate(seeds):
            out[ISLANDS[index % len(ISLANDS)]].append(spec)
    for specs in out.values():
        for spec in specs:
            spec.validate(search_policy=True)
    return out


REMOTE_PROVENANCE_KEYS = (
    "lib_sha256", "simpleFoam_sha256", "worker_sha256", "tier2_sha256",
    "scoring_sha256", "wm_options",
)


def build_protocol_manifest(
    cfg: SearchCfg,
    seeds: list[CandidateSpec] | dict[str, list[CandidateSpec]],
    backend_provenance: dict[str, Any],
    claude_manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    if os.environ.get("TEDP_SHORT") != "1":
        raise RuntimeError("the Phase-3 search requires TEDP_SHORT=1")
    if os.environ.get("TEDP_WARM_START", "0") != "0":
        raise RuntimeError("the Phase-3 search requires TEDP_WARM_START=0")
    seed_map = _normalized_seeds(seeds)
    if claude_manifest is None:
        claude_manifest = ClaudeCodeProposer(model=cfg.model).binary_manifest
    files: dict[str, Any] = {}
    for relative in PROTOCOL_FILES:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = _file_provenance(path)
    case_protocol: list[dict[str, Any]] = []
    for case_id in FULL_CASES:
        template = mcconkey.case_dir(case_id)
        input_files: list[str] = []
        reference_cache = REPO / "data" / "mcconkey" / "cache" / f"REF_{case_id}.npz"
        if not reference_cache.is_file():
            raise FileNotFoundError(reference_cache)
        files[str(reference_cache)] = _file_provenance(reference_cache)
        for subdirectory in ("0", "constant", "system"):
            for path in sorted((template / subdirectory).rglob("*")):
                if not path.is_file():
                    continue
                key = str(path.absolute())
                files[key] = _file_provenance(path)
                input_files.append(key)
        case_protocol.append({
            "case_id": case_id, "short_iters": tier2.SHORT_ITERS.get(case_id),
            "input_files": input_files,
        })
    claude_path = Path(str(claude_manifest.get("path", "")))
    if not claude_path.is_file():
        raise FileNotFoundError(f"Claude proposal binary: {claude_path}")
    observed_claude = _file_provenance(claude_path)
    if (
        observed_claude["sha256"] != claude_manifest.get("sha256")
        or observed_claude["size"] != claude_manifest.get("bytes")
    ):
        raise RuntimeError("Claude binary changed between pre-check and manifest")
    files[str(claude_path)] = observed_claude
    freeze_payload = json.loads(evaldb.FREEZE_PATH.read_text())
    files[str(evaldb.FREEZE_PATH)] = _file_provenance(evaldb.FREEZE_PATH)
    if backend_provenance.get("backend") == "slurm":
        for key in REMOTE_PROVENANCE_KEYS:
            if not backend_provenance.get(key):
                raise RuntimeError(f"remote provenance is missing {key}")
        backend_section = {
            key: value for key, value in backend_provenance.items()
            if key not in {"hostname"}
        }
    else:
        foam_runtime = runner.foam_env()
        backend_section = {
            **backend_provenance,
            "openfoam": {
                key: foam_runtime.get(key)
                for key in ("WM_PROJECT_VERSION", "WM_OPTIONS", "WM_PROJECT_DIR",
                            "FOAM_USER_LIBBIN")
            },
        }
    manifest = {
        "schema_version": PROTOCOL_MANIFEST_SCHEMA,
        "algorithm": ALGORITHM,
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in vars(cfg).items()
        },
        "profiles": {key: vars(profile) for key, profile in ISLAND_PROFILES.items()},
        "profile_policy": "prompt guidance only; no island-specific admission constraints",
        "global_search_envelope": {
            "free_constant_grammar": "c0..c7",
            "max_free_constants": expr.MAX_FREE_CONSTANTS,
            "effective_parameter_admission_cap": None,
            "term_admission_cap": None,
            "ast_node_admission_cap": None,
            "gmax": SEARCH_GMAX,
            "r_max_factor": SEARCH_R_MAX_FACTOR,
        },
        "campaign": cfg.campaign,
        "selection": {
            "island_size": ISLAND_SIZE, "size_classes": [list(c) for c in SIZE_CLASSES],
            "slots_per_class": SLOTS_PER_CLASS, "diversity_quota": DIVERSITY_QUOTA,
            "cross_island_quota": CROSS_ISLAND_QUOTA,
            "plateau_threshold": MIN_VAL_IMPROVEMENT, "migrate_every": MIGRATE_EVERY,
        },
        "tuning": {
            "cases": list(cfg.tuning_cases), "evals_per_constant": cfg.evals_per_constant,
            "max_evals": cfg.max_tuning_evals, "enabled": cfg.tuning_enabled,
            "parametrization": "c = c0*exp(z), |z|<=3, sigma0=0.3",
        },
        "seeds": {
            island: [
                {"spec_hash": spec_hash(spec), "spec": json.loads(to_json(spec))}
                for spec in seed_map[island]
            ]
            for island in ISLANDS
        },
        "files": files,
        "cases": {
            "sanity": SANITY_CASE, "sanity_factor": SANITY_FACTOR,
            "calibration": list(mcconkey.CALIBRATION),
            "validation": list(mcconkey.VALIDATION),
            "ordered_full": list(FULL_CASES), "case_protocol": case_protocol,
            "short_iters_table": [
                [case, int(iterations)] for case, iterations in tier2.SHORT_ITERS.items()
            ],
            "truth_gate_cases": [
                case for case in mcconkey.CALIBRATION if case != "cbfs13700"
            ],
        },
        "scoring_freeze": {"path": str(evaldb.FREEZE_PATH), "payload": freeze_payload},
        "claude": {"binary": claude_manifest},
        "backend": backend_section,
        "runtime": {
            "env": {
                key: os.environ.get(key)
                for key in ("TEDP_SHORT", "TEDP_WARM_START", "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS")
            },
            "sampled_states": {"n": 30_000, "seed": cfg.seed},
            "llm_timeout_s": 900, "llm_max_attempts": cfg.llm_max_attempts,
        },
        "python": _python_manifest(),
    }
    fingerprint_manifest = {
        **manifest,
        "files": {
            path: {"sha256": provenance["sha256"], "size": provenance["size"]}
            for path, provenance in manifest["files"].items()
        },
    }
    canonical = json.dumps(fingerprint_manifest, sort_keys=True, separators=(",", ":"))
    return manifest, hashlib.sha256(canonical.encode()).hexdigest()


def _assert_protocol_unchanged(state: SearchState, backend: Any | None = None) -> None:
    for relative, expected in state.protocol_manifest.get("files", {}).items():
        path = REPO / relative
        if not path.is_file():
            raise RuntimeError(f"protocol input disappeared during search: {relative}")
        stat = path.stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            if _file_provenance(path)["sha256"] != expected["sha256"]:
                raise RuntimeError(f"protocol input changed during search: {relative}")
    expected_env = state.protocol_manifest.get("runtime", {}).get("env", {})
    observed_env = {key: os.environ.get(key) for key in expected_env}
    if observed_env != expected_env:
        raise RuntimeError(
            f"runtime environment changed during search: expected={expected_env}, "
            f"observed={observed_env}"
        )
    if state.protocol_manifest.get("python") != _python_manifest():
        raise RuntimeError("Python packages/runtime changed during search")
    if backend is not None and backend.name == "slurm":
        expected_backend = state.protocol_manifest.get("backend", {})
        observed = backend.provenance()
        for key in REMOTE_PROVENANCE_KEYS:
            if observed.get(key) != expected_backend.get(key):
                raise RuntimeError(
                    f"remote protocol changed during search: {key} "
                    f"{expected_backend.get(key)} -> {observed.get(key)}"
                )


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------
def _case_score(result: CaseResult) -> scoring.CaseScore | None:
    if result.status != "complete" or not result.score:
        return None
    score = result.score
    return scoring.CaseScore(
        result.case_id, float(score["e_u"]), float(score["e_cf"]), float(score["e_uv"])
    )


def _score_objectives(
    scores: dict[str, scoring.CaseScore], diverged: set[str],
    cases: Iterable[str], n_parameters: int,
) -> tuple[float, float]:
    case_set = set(cases)
    selected = {case: score for case, score in scores.items() if case in case_set}
    failed = {case for case in diverged if case in case_set}
    raw = scoring.objective(selected, 0, failed)
    penalized = scoring.objective(selected, n_parameters, failed)
    return raw, penalized


def full_from_results(spec: CandidateSpec, results: dict[str, CaseResult]) -> dict[str, Any]:
    """Raw and penalized objectives from one complete 13-case result set."""
    missing = [case for case in FULL_CASES if case not in results]
    if missing:
        raise RuntimeError(f"full suite is missing cases: {missing}")
    scores: dict[str, scoring.CaseScore] = {}
    diverged: set[str] = set()
    failures: dict[str, str] = {}
    for case in FULL_CASES:
        score = _case_score(results[case])
        if score is None:
            diverged.add(case)
            failures[case] = f"{results[case].failure_class}: {results[case].reason}"
        else:
            scores[case] = score
    n_parameters = spec.n_parameters()
    cal_raw, cal_j = _score_objectives(scores, diverged, mcconkey.CALIBRATION, n_parameters)
    val_raw, val_j = _score_objectives(scores, diverged, mcconkey.VALIDATION, n_parameters)
    per_case = {case: score.composite for case, score in scores.items()}
    for case in diverged:
        per_case[case] = float("nan")
    return {
        "cal_raw": cal_raw, "cal_j": cal_j, "val_raw": val_raw, "val_j": val_j,
        "complexity_penalty": scoring.PARSIMONY * n_parameters,
        "per_case": per_case, "diverged": sorted(diverged), "failures": failures,
        "components": {
            case: {"e_u": s.e_u, "e_cf": s.e_cf, "e_uv": s.e_uv, "composite": s.composite}
            for case, s in scores.items()
        },
    }


def family_key(spec: CandidateSpec) -> str:
    """Structural family: tensor sets per channel plus the variables used."""
    variables: set[str] = set()
    for term in spec.all_terms():
        for name in ("I1", "I2", "I3", "I4", "I5", "Ret", "F1", "PoE"):
            if name in term.expression:
                variables.add(name)
    return json.dumps({
        "bdelta": sorted(t.tensor for t in spec.bdelta),
        "rsource": sorted(t.tensor for t in spec.rsource),
        "variables": sorted(variables),
    }, sort_keys=True, separators=(",", ":"))


def family_residuals(per_case: dict[str, float], stock: dict[str, float]) -> str:
    bits: list[str] = []
    for family, cases in CASE_FAMILIES.items():
        values = [per_case.get(case, float("nan")) for case in cases]
        base = [stock[case] for case in cases]
        if any(not np.isfinite(v) for v in values):
            bits.append(f"{family}: DIVERGED")
            continue
        delta = (float(np.mean(values)) - float(np.mean(base))) / float(np.mean(base)) * 100
        bits.append(f"{family}: {float(np.mean(values)):.4f} ({'+' if delta >= 0 else ''}{delta:.0f}% vs stock)")
    return "; ".join(bits)


def family_map(archives: dict[str, list[Member]]) -> list[dict[str, Any]]:
    """Every structural family evaluated so far, with its count and bests."""
    rows: dict[str, dict[str, Any]] = {}
    for island, members in archives.items():
        for member in members:
            if member.n_parameters == 0:
                continue
            key = family_key(member.spec)
            row = rows.setdefault(key, {"family": key, "count": 0, "best_raw": float("inf"),
                                        "best_j": float("inf"), "islands": set()})
            row["count"] += 1
            row["best_raw"] = min(row["best_raw"], member.val_raw)
            row["best_j"] = min(row["best_j"], member.val_j)
            row["islands"].add(island)
    out = sorted(rows.values(), key=lambda r: (r["best_raw"], r["family"]))
    for row in out:
        row["islands"] = sorted(row["islands"])
    return out


def localize(member: Member, stock: dict[str, float]) -> str:
    bits: list[str] = []
    for case, composite in sorted(member.per_case.items()):
        base = stock.get(case)
        if base is None or not np.isfinite(composite):
            bits.append(f"{case}: DIVERGED")
            continue
        delta = (composite - base) / base * 100.0
        bits.append(f"{case}: {'+' if delta >= 0 else ''}{delta:.0f}%")
    return "; ".join(bits)


def _migration_source(island: str) -> str:
    index = ISLANDS.index(island)
    return ISLANDS[(index - 1) % len(ISLANDS)]


def _should_include_migrant(completed_generations: int) -> bool:
    return completed_generations > 0 and completed_generations % MIGRATE_EVERY == 0


def _spec_text(spec: CandidateSpec) -> str:
    return (
        f"bdelta={[(t.tensor, t.expression) for t in spec.bdelta]} "
        f"rsource={[(t.tensor, t.expression) for t in spec.rsource]} "
        f"constants={[round(c, 6) for c in spec.constants]}"
    )


def context_block(
    snapshot: dict[str, list[Member]], island: str, stock: dict[str, float],
    stock_val_j: float, archive: list[Member], rejections: list[dict[str, Any]],
    design_seeds: list[CandidateSpec] | None = None, include_migrant: bool = False,
    families: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        f"Stock SST: validation J = raw = {stock_val_j:.5f} with 0 parameters. Lower is better.",
        "Two numbers matter. raw = mean validation composite error. valJ = raw +",
        "0.005 per effective parameter (free constants plus every distinct non-0/1",
        "literal) and is the frozen acceptance objective. To beat stock a model with",
        "n parameters needs raw < 0.0818 - 0.005*n (3 -> 0.0668, 4 -> 0.0618,",
        "6 -> 0.0518, 8 -> 0.0418). The search keeps two survivors per size class",
        "(1-2, 3-4, 5-6, 7+) so larger mechanisms compete within their class and",
        "the final report shows the whole raw-versus-size front next to the J verdict.",
        "Every gate-legal structure is tuned by CMA-ES on CFD before it is judged,",
        "so give sensible starting constants and spend capacity on structure.",
        f"Stock per family: {family_residuals(stock, stock)}",
        "",
        f"Island {island} survivors by size class (best raw first within class):",
    ]
    members = sorted(snapshot[island], key=lambda m: (m.size_class, m.val_raw, spec_hash(m.spec)))
    for member in members:
        if member.n_parameters == 0:
            continue
        lines.append(
            f"- class {member.size_class} n={member.n_parameters} raw={member.val_raw:.5f} "
            f"valJ={member.val_j:.5f} calJ={member.cal_j:.5f} origin={member.origin} :: "
            f"{_spec_text(member.spec)}"
        )
        if member.per_case:
            lines.append(f"  families: {family_residuals(member.per_case, stock)}")
            lines.append(f"  cases: {localize(member, stock)}")
    if not any(m.n_parameters > 0 for m in members):
        lines.append("- (no evaluated survivors yet besides stock)")
    seen = {spec_hash(m.spec) for m in members}
    extra = [m for m in sorted(archive, key=lambda m: (m.val_raw, spec_hash(m.spec)))
             if spec_hash(m.spec) not in seen and m.n_parameters > 0]
    if extra:
        lines.extend(["", f"Other evaluated structures on island {island} (best raw first):"])
        for member in extra[:ARCHIVE_CONTEXT_SIZE]:
            lines.append(
                f"- n={member.n_parameters} raw={member.val_raw:.5f} valJ={member.val_j:.5f} "
                f"origin={member.origin} :: {_spec_text(member.spec)}"
            )
    if families:
        lines.extend([
            "", "Structural families already evaluated across ALL islands (tensor sets per",
            "channel + variables used; count, best raw, best J). NOVELTY REQUIREMENT: at",
            "least two of your proposals must belong to a family NOT in this list, and no",
            f"family may receive more than {CROSS_ISLAND_QUOTA} proposals per generation",
            "across the islands. A new gate or a new tensor pairing is a new family; a",
            "re-spelled coefficient is not.",
        ])
        for row in families[:30]:
            lines.append(
                f"- {row['family']} n={row['count']} best_raw={row['best_raw']:.5f} "
                f"best_valJ={row['best_j']:.5f} islands={','.join(row['islands'])}"
            )
    if rejections:
        lines.extend(["", "Recently rejected on this island (do not resubmit these forms):"])
        for row in rejections[-16:]:
            lines.append(
                f"- gen {row['generation']} {row['stage']}: {row['reason']} :: "
                f"bdelta={row['bdelta']} rsource={row['rsource']} constants={row['constants']}"
            )
    if design_seeds:
        lines.extend(["", "Unevaluated, gate-verified starting designs for this island:"])
        for seed in design_seeds:
            lines.append(f"- {seed.name}: parameters={seed.n_parameters()} {_spec_text(seed)}")
    if include_migrant:
        source = _migration_source(island)
        if snapshot[source]:
            lines.extend([
                "", f"Context-only ring migration from island {source} (these models "
                "are not inserted into your population):",
            ])
            shown: set[str] = set()
            ordered = sorted(snapshot[source], key=lambda m: (m.val_raw, spec_hash(m.spec)))
            for member in ordered[:1] + [
                min([m for m in snapshot[source] if m.size_class == key],
                    key=lambda m: (m.val_raw, spec_hash(m.spec)))
                for key, _, _ in SIZE_CLASSES
                if any(m.size_class == key for m in snapshot[source])
            ]:
                identity = spec_hash(member.spec)
                if identity in shown or member.n_parameters == 0:
                    continue
                shown.add(identity)
                lines.append(
                    f"- class {member.size_class} raw={member.val_raw:.5f} "
                    f"valJ={member.val_j:.5f} :: {_spec_text(member.spec)}"
                )
    return "\n".join(lines)


def _snapshot_payload(islands: dict[str, list[Member]]) -> dict[str, Any]:
    return {island: [_member_payload(m) for m in islands[island]] for island in ISLANDS}


def _snapshot_hash(islands: dict[str, list[Member]]) -> str:
    canonical = json.dumps(_json_safe(_snapshot_payload(islands)), sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------
def _new_item(
    spec: CandidateSpec, generation: int, island: str, origin: str,
    rationale: str, ordinal: int, parent_hash: str | None = None,
) -> dict[str, Any]:
    identity = spec_hash(spec)
    item_id = hashlib.sha256(
        f"{generation}:{island}:{origin}:{ordinal}:{identity}".encode()
    ).hexdigest()[:16]
    return {
        "item_id": item_id, "generation": generation, "island": island,
        "origin": origin, "parent_hash": parent_hash, "rationale": rationale,
        "spec": json.loads(to_json(spec)), "spec_hash": identity,
        "struct_hash": struct_hash(spec), "family_key": family_key(spec),
        "n_parameters": spec.n_parameters(), "n_constants": len(spec.constants),
        "status": "proposed",
        "events": [{"event": "proposed", "wall_time": time.time()}],
    }


def _item_spec(item: dict[str, Any]) -> CandidateSpec:
    return from_json(json.dumps(item["spec"]))


def _archive_rejection(state: SearchState, item: dict[str, Any], stage: str, reasons: list[str]) -> None:
    spec = _item_spec(item)
    archive = state.rejection_archive.setdefault(item["island"], [])
    archive.append({
        "generation": int(item["generation"]), "stage": stage,
        "reason": "; ".join(str(r) for r in reasons[:2])[:220],
        "bdelta": [(t.tensor, t.expression) for t in spec.bdelta],
        "rsource": [(t.tensor, t.expression) for t in spec.rsource],
        "constants": [round(c, 6) for c in spec.constants],
        "n_parameters": item.get("n_parameters"),
    })
    del archive[:-REJECTION_ARCHIVE_SIZE]


def _transition(
    state: SearchState, item: dict[str, Any], status: str, event: str, **updates: Any,
) -> None:
    item.update(updates)
    item["status"] = status
    item.setdefault("events", []).append({"event": event, "wall_time": time.time()})
    _record_item(state, item, event)
    save_checkpoint(state)


def _register_item(state: SearchState, pending: dict[str, Any], item: dict[str, Any]) -> bool:
    spec = _item_spec(item)
    if item["spec_hash"] in state.seen_spec_hashes:
        item["status"] = "global_duplicate"
        pending["items"].append(item)
        _record_item(state, item, "global_duplicate")
        save_checkpoint(state)
        return False
    state.seen_spec_hashes.add(item["spec_hash"])
    reasons: list[str] = []
    try:
        spec.validate(search_policy=True)
    except SpecError as exc:
        reasons.append(str(exc))
    if reasons:
        item["status"] = "search_policy_rejected"
        item["reject_reasons"] = reasons
        pending["items"].append(item)
        _archive_rejection(state, item, "policy", reasons)
        _record_item(state, item, "search_policy_rejected")
        save_checkpoint(state)
        return False
    if item["origin"] == "llm_original":
        siblings = [
            other for other in pending["items"]
            if other["island"] == item["island"] and other["origin"] == "llm_original"
            and other.get("family_key") == item["family_key"]
            and other["status"] not in {"global_duplicate", "search_policy_rejected",
                                        "diversity_quota_rejected"}
        ]
        everywhere = [
            other for other in pending["items"]
            if other["origin"] == "llm_original"
            and other.get("family_key") == item["family_key"]
            and other["status"] not in {"global_duplicate", "search_policy_rejected",
                                        "diversity_quota_rejected"}
        ]
        reason = ""
        if len(siblings) >= DIVERSITY_QUOTA:
            reason = (
                f"diversity quota: island {item['island']} already has "
                f"{len(siblings)} proposals of family {item['family_key']} this generation"
            )
        elif len(everywhere) >= CROSS_ISLAND_QUOTA:
            reason = (
                f"diversity quota: {len(everywhere)} proposals of family "
                f"{item['family_key']} already registered across all islands this generation"
            )
        if reason:
            item["status"] = "diversity_quota_rejected"
            item["reject_reasons"] = [reason]
            pending["items"].append(item)
            _archive_rejection(state, item, "diversity", [reason])
            _record_item(state, item, "diversity_quota_rejected")
            save_checkpoint(state)
            return False
    pending["items"].append(item)
    _record_item(state, item, "registered")
    save_checkpoint(state)
    return True


def _tier0_payload(result: Any) -> dict[str, Any]:
    return {
        "passed": result.passed, "reasons": result.reasons,
        "max_observable": result.max_g, "realizability_metric": result.new_violation,
        "recovery_residual": result.recovery_residual,
        "g_clamp_effect": result.clamp_effect,
        "r_clamp_fraction": result.r_clamp_fraction,
        "r_clamp_effect": result.r_clamp_effect,
        "novelty_best_rms": result.novelty_best_rms,
        "equivalent_to": result.equivalent_to,
    }


def _tier1_payload(channel: Any, homogeneous: Any) -> dict[str, Any]:
    return {
        "passed": channel.passed and homogeneous.passed,
        "channel_passed": channel.passed, "channel_reasons": channel.reasons,
        "cf_ratio": channel.cf_ratio,
        "homogeneous_passed": homogeneous.passed,
        "homogeneous_reasons": homogeneous.reasons,
        "homogeneous_error": homogeneous.homogeneous_error,
        "homogeneous_stock_error": homogeneous.homogeneous_stock_error,
        "homogeneous_components": homogeneous.homogeneous_components,
    }


@dataclass
class GateContext:
    sampled_states: Any
    dns_gate_states: list[Any]
    baseline_channel: Any
    ap_states: Any
    b_target: np.ndarray
    r_target: np.ndarray
    ap_weights: np.ndarray | None

    def check(self, spec: CandidateSpec) -> tuple[bool, list[str], dict[str, Any]]:
        try:
            spec.validate(search_policy=True)
        except SpecError as exc:
            return False, [str(exc)], {}
        r0 = tier0.run_tier0(
            spec, self.sampled_states, dns_states=self.dns_gate_states, library=library
        )
        payload: dict[str, Any] = {"tier0": _tier0_payload(r0)}
        if not r0.passed:
            return False, list(r0.reasons), payload
        channel = tier1.channel_gates(spec, self.baseline_channel)
        homogeneous = tier1.homogeneous_gates(spec)
        payload["tier1"] = _tier1_payload(channel, homogeneous)
        if not (channel.passed and homogeneous.passed):
            return False, list(channel.reasons) + list(homogeneous.reasons), payload
        payload["novelty_best_rms"] = r0.novelty_best_rms
        return True, [], payload

    def gate_fn(self, spec: CandidateSpec) -> tuple[bool, list[str]]:
        ok, reasons, _ = self.check(spec)
        return ok, reasons


def _gate_item(state: SearchState, item: dict[str, Any], gates: GateContext) -> None:
    if item["status"] != "proposed":
        return
    spec = _item_spec(item)
    ok, reasons, payload = gates.check(spec)
    item.update({key: value for key, value in payload.items() if key in {"tier0", "tier1"}})
    if not ok:
        stage = "tier1" if "tier1" in payload else "tier0"
        item["reject_reasons"] = reasons
        _archive_rejection(state, item, stage, reasons)
        _transition(state, item, f"{stage}_rejected", f"{stage}_rejected")
        return
    apriori_raw = cmaes.apriori_objective(
        spec, gates.ap_states, gates.b_target, gates.r_target, gates.ap_weights
    )
    _transition(
        state, item, "gates_passed", "gates_passed",
        novelty_best_rms=payload.get("novelty_best_rms"), apriori_raw=apriori_raw,
    )


# ---------------------------------------------------------------------------
# generation preparation and proposals
# ---------------------------------------------------------------------------
def _ensure_generation_one_seeds(
    state: SearchState, pending: dict[str, Any], seed_map: dict[str, list[CandidateSpec]],
) -> None:
    if int(pending["generation"]) != 1:
        return
    registered = {
        (item["island"], item["spec_hash"])
        for item in pending["items"] if item["origin"] == "declared_seed"
    }
    for island in ISLANDS:
        for ordinal, seed in enumerate(seed_map[island]):
            if (island, spec_hash(seed)) in registered:
                continue
            item = _new_item(
                seed, 1, island, "declared_seed",
                "predeclared gate-verified initializer", ordinal,
            )
            _register_item(state, pending, item)


def _prepare_generation(
    state: SearchState, cfg: SearchCfg, seed_map: dict[str, list[CandidateSpec]],
    stock: dict[str, float], stock_val_j: float, proposer: ClaudeCodeProposer,
) -> dict[str, Any]:
    generation = state.generation + 1
    snapshot = {island: list(state.islands[island]) for island in ISLANDS}
    include_migrant = _should_include_migrant(state.generation)
    families = family_map(state.evaluated_archive)
    contexts = {
        island: context_block(
            snapshot, island, stock, stock_val_j,
            archive=state.evaluated_archive[island],
            rejections=state.rejection_archive.get(island, []),
            design_seeds=seed_map[island] if generation == 1 else None,
            include_migrant=include_migrant, families=families,
        )
        for island in ISLANDS
    }
    pending: dict[str, Any] = {
        "generation": generation, "status": "proposing",
        "snapshot": _snapshot_payload(snapshot), "snapshot_hash": _snapshot_hash(snapshot),
        "contexts": contexts,
        "context_sha256": {
            island: hashlib.sha256(contexts[island].encode()).hexdigest()
            for island in ISLANDS
        },
        "context_only_migration": include_migrant,
        "llm_calls": {
            island: {
                "call_id": f"{state.run_id}.g{generation:03d}.{island}",
                "status": "scheduled", "attempts": 0, "reason": "",
                "proposals": [], "parse_rejections": [], "items_registered": False,
            }
            for island in ISLANDS
        },
        "items": [], "created_wall_time": time.time(),
        "evaluation": {"stage": "sanity", "batches": {}, "tuners": {}, "round": 0,
                       "budget_stop": None},
    }
    state.pending_generation = pending
    save_checkpoint(state)
    if generation == 1:
        _ensure_generation_one_seeds(state, pending, seed_map)
        save_checkpoint(state)
    _complete_proposal_transactions(state, pending, cfg, proposer)
    return pending


def _proposal_payload(proposal: Proposal) -> dict[str, Any]:
    return {"spec": json.loads(to_json(proposal.spec)), "rationale": proposal.rationale}


def _proposal_from_payload(payload: dict[str, Any]) -> Proposal:
    return Proposal(from_json(json.dumps(payload["spec"])), str(payload.get("rationale", "")))


def _register_transaction_proposals(state: SearchState, pending: dict[str, Any], island: str) -> None:
    call = pending["llm_calls"][island]
    existing_ids = {item["item_id"] for item in pending["items"]}
    for ordinal, payload in enumerate(call["proposals"]):
        proposal = _proposal_from_payload(payload)
        item = _new_item(
            proposal.spec, int(pending["generation"]), island, "llm_original",
            proposal.rationale, ordinal,
        )
        if item["item_id"] in existing_ids:
            continue
        _register_item(state, pending, item)
        existing_ids.add(item["item_id"])
    call["items_registered"] = True
    save_checkpoint(state)


def _complete_proposal_transactions(
    state: SearchState, pending: dict[str, Any], cfg: SearchCfg,
    proposer: ClaudeCodeProposer,
) -> bool:
    generation = int(pending["generation"])
    for island in ISLANDS:
        call = pending["llm_calls"][island]
        if call["status"] != "complete":
            save_checkpoint(state)
            result = proposer.propose_transaction(
                call["call_id"], cfg.children_per_island, pending["contexts"][island],
                generation, island, ISLAND_PROFILES[island].prompt_block(),
                max_attempts=cfg.llm_max_attempts,
            )
            call.update({
                "status": result.status, "attempts": result.attempts,
                "reason": result.reason,
                "proposals": [_proposal_payload(p) for p in result.proposals],
                "parse_rejections": result.rejections,
            })
            for rejection in result.rejections:
                rejection_identity = json.dumps(
                    _json_safe(rejection), sort_keys=True, separators=(",", ":")
                )
                append_gate_ledger({
                    "event_id": hashlib.sha256(
                        f"{state.run_id}:{call['call_id']}:{rejection_identity}".encode()
                    ).hexdigest(),
                    "event": "llm_parse_rejection", "run_id": state.run_id,
                    "protocol_fingerprint": state.protocol_fingerprint,
                    "generation": generation, "island": island,
                    "call_id": call["call_id"], "rejection": rejection,
                })
            save_checkpoint(state)
        if call["status"] != "complete":
            pending["status"] = "paused_llm"
            state.status = "paused_llm"
            state.stop_reason = f"proposal call {call['call_id']} paused: {call.get('reason', '')}"
            save_checkpoint(state)
            return False
        if not call.get("items_registered"):
            _register_transaction_proposals(state, pending, island)
    pending["status"] = "gating"
    state.status = "running"
    state.stop_reason = ""
    save_checkpoint(state)
    return True


def _resume_or_prepare_generation(
    state: SearchState, cfg: SearchCfg, seed_map: dict[str, list[CandidateSpec]],
    stock: dict[str, float], stock_val_j: float, proposer: ClaudeCodeProposer,
) -> dict[str, Any]:
    pending = state.pending_generation
    if pending is None:
        return _prepare_generation(state, cfg, seed_map, stock, stock_val_j, proposer)
    if int(pending["generation"]) != state.generation + 1:
        raise RuntimeError("checkpoint pending generation is not the next transaction")
    if pending["status"] in {"proposing", "paused_llm"}:
        _ensure_generation_one_seeds(state, pending, seed_map)
        _complete_proposal_transactions(state, pending, cfg, proposer)
    return pending


# ---------------------------------------------------------------------------
# batch evaluation engine
# ---------------------------------------------------------------------------
class CaseBudgetExhausted(BudgetExhausted):
    pass


def _compact_outcome(result: CaseResult) -> dict[str, Any]:
    outcome = dict(result.outcome)
    outcome.pop("run_dir", None)
    if isinstance(outcome.get("log_tail"), str):
        outcome["log_tail"] = outcome["log_tail"][-2000:]
    outcome.update({
        "status": result.status, "score": result.score,
        "failure_class": result.failure_class, "reason": result.reason,
        "case_id": result.case_id, "job_id": result.job_id,
    })
    return outcome


def _batch_id(state: SearchState, pending: dict[str, Any], key: str) -> str:
    return f"{state.run_id[:15]}-g{int(pending['generation']):03d}-{key}"


def _run_batch(
    state: SearchState, pending: dict[str, Any], cfg: SearchCfg, backend: Any,
    key: str, jobs: list[CaseJob],
) -> list[CaseResult]:
    """Run (or replay/resume) one named batch of case jobs for this generation."""
    ev = pending["evaluation"]
    record = ev["batches"].setdefault(key, {"status": "pending", "attempts": []})
    if not jobs:
        record["status"] = "collected"
        record["outcomes"] = {}
        return []
    outcomes: dict[str, dict[str, Any]] = {}

    def merge(attempt: dict[str, Any]) -> None:
        for job_id, outcome in attempt.get("outcomes", {}).items():
            outcomes[job_id] = outcome

    def remaining_jobs() -> list[CaseJob]:
        remaining = []
        for job in jobs:
            outcome = outcomes.get(job.job_id)
            if outcome is None:
                remaining.append(job)
            elif outcome.get("status") != "complete" and CaseResult.from_outcome(
                job, outcome
            ).retryable:
                remaining.append(job)
        return remaining

    for attempt in record["attempts"]:
        if attempt.get("status") == "collected":
            merge(attempt)
    if record["status"] == "collected":
        return [CaseResult.from_outcome(job, outcomes.get(job.job_id)) for job in jobs]

    lib_sha256 = str(state.protocol_manifest.get("backend", {}).get("lib_sha256", ""))

    def settle(attempt: dict[str, Any], results: list[CaseResult], accounting: Any) -> None:
        attempt["status"] = "collected"
        attempt["outcomes"] = {r.job_id: _compact_outcome(r) for r in results}
        attempt["actual_core_hours"] = accounting.actual_core_hours
        attempt["ended_wall_time"] = time.time()
        charged = len(attempt.get("job_ids", []))
        state.tier2_reserved = max(0, state.tier2_reserved - charged)
        state.tier2_evals += charged
        if accounting.actual_core_hours is not None:
            state.core_hours_spent += float(accounting.actual_core_hours)
        merge(attempt)
        save_checkpoint(state)

    last = record["attempts"][-1] if record["attempts"] else None
    if last is not None and last.get("status") == "submitted":
        results, accounting = backend.resume_batch(last["handle"])
        settle(last, results, accounting)

    attempt_index = len(record["attempts"])
    while attempt_index < 2:
        remaining = remaining_jobs()
        if not remaining:
            break
        batch_id = _batch_id(state, pending, key) + (f".retry{attempt_index}" if attempt_index else "")
        if state.tier2_evals + state.tier2_reserved + len(remaining) > cfg.tier2_eval_cap:
            raise CaseBudgetExhausted(
                f"batch {key} needs {len(remaining)} case-evaluations; "
                f"{cfg.tier2_eval_cap - state.tier2_evals - state.tier2_reserved} remain"
            )
        estimate = float(backend.estimate_core_hours(remaining))
        ledger = getattr(backend, "ledger", None)
        committed = 0.0 if ledger is None else ledger.committed
        spent = state.core_hours_spent if ledger is None else ledger.spent
        if spent + committed + estimate > cfg.core_hour_cap:
            raise BudgetExhausted(
                f"batch {key} is estimated at {estimate:.1f} core-hours; "
                f"{cfg.core_hour_cap - spent - committed:.1f} remain under the "
                f"search cap of {cfg.core_hour_cap:.0f}"
            )
        state.tier2_reserved += len(remaining)
        attempt = {
            "status": "submitting", "batch_id": batch_id,
            "job_ids": [job.job_id for job in remaining],
            "started_wall_time": time.time(), "estimate_core_hours": estimate,
        }
        record["attempts"].append(attempt)
        save_checkpoint(state)

        def on_submitted(handle: dict[str, Any], attempt: dict[str, Any] = attempt) -> None:
            attempt["status"] = "submitted"
            attempt["handle"] = handle
            save_checkpoint(state)

        kwargs = {"lib_sha256": lib_sha256} if backend.name == "slurm" else {}
        try:
            results, accounting = backend.run_batch(batch_id, remaining, on_submitted, **kwargs)
        except BudgetExhausted:
            state.tier2_reserved = max(0, state.tier2_reserved - len(remaining))
            attempt["status"] = "refused"
            save_checkpoint(state)
            raise
        settle(attempt, results, accounting)
        attempt_index += 1
    record["status"] = "collected"
    record["outcomes"] = {job.job_id: outcomes.get(job.job_id) for job in jobs}
    save_checkpoint(state)
    return [CaseResult.from_outcome(job, outcomes.get(job.job_id)) for job in jobs]


def _items_by_id(pending: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["item_id"]: item for item in pending["items"]}


def _stage_sanity(
    state: SearchState, pending: dict[str, Any], cfg: SearchCfg, backend: Any,
    stock: dict[str, float],
) -> None:
    items = [item for item in pending["items"] if item["status"] == "gates_passed"]
    jobs = [
        CaseJob(f"{item['item_id']}-sanity-{SANITY_CASE}", SANITY_CASE, _item_spec(item),
                {"item_id": item["item_id"], "stage": "sanity"})
        for item in items
    ]
    results = _run_batch(state, pending, cfg, backend, "sanity", jobs)
    threshold = SANITY_FACTOR * stock[SANITY_CASE]
    for item, result in zip(items, results):
        if item["status"] != "gates_passed":
            continue
        composite = None if result.status != "complete" else float(result.score["composite"])
        item["sanity"] = {
            "case": SANITY_CASE, "composite": composite, "threshold": threshold,
            "stock_composite": stock[SANITY_CASE],
            "failure": None if result.status == "complete" else
            f"{result.failure_class}: {result.reason}",
            "outcome": _compact_outcome(result),
        }
        if composite is None or composite > threshold:
            reason = item["sanity"]["failure"] or f"sanity composite {composite:.4f} > {threshold:.4f}"
            item["reject_reasons"] = [reason]
            _archive_rejection(state, item, "sanity", [reason])
            _transition(state, item, "sanity_rejected", "sanity_rejected")
        else:
            _transition(state, item, "sanity_passed", "sanity_passed")


def _tuner_seed(cfg: SearchCfg, generation: int, island: str, identity: str) -> int:
    return cmaes.deterministic_seed(cfg.seed, generation, island, identity)


def _stage_tuning(
    state: SearchState, pending: dict[str, Any], cfg: SearchCfg, backend: Any,
    gates: GateContext, stock: dict[str, float],
) -> None:
    if not cfg.tuning_enabled:
        return
    ev = pending["evaluation"]
    generation = int(pending["generation"])
    parents = [
        item for item in pending["items"]
        if item["status"] == "sanity_passed" and item["origin"] != "cfd_tuned"
        and _item_spec(item).constants and not item.get("tuning")
    ]
    if not parents:
        return
    tuners: dict[str, CfdTuner] = {}
    for item in parents:
        spec = _item_spec(item)
        stored = ev["tuners"].get(item["item_id"])
        if stored is None:
            budget = min(cfg.max_tuning_evals, cfg.evals_per_constant * len(spec.constants))
            tuner = CfdTuner(
                spec, seed=_tuner_seed(cfg, generation, item["island"], item["spec_hash"]),
                budget=budget,
            )
            if cfg.inject_apriori_point:
                try:
                    apriori_spec, _ = cmaes.tune(
                        spec, gates.ap_states, gates.b_target, gates.r_target,
                        gates.baseline_channel, weights=gates.ap_weights, budget=40,
                        seed=tuner.seed,
                    )
                    z = encode_constants(tuner.c0, np.asarray(apriori_spec.constants, float))
                    if np.all(np.isfinite(z)) and not np.allclose(z, 0.0):
                        tuner.es.inject([z])
                except Exception:
                    pass
            ev["tuners"][item["item_id"]] = tuner.to_state()
        else:
            tuner = CfdTuner.from_state(spec, stored)
        tuners[item["item_id"]] = tuner
    save_checkpoint(state)

    round_index = int(ev.get("round", 0))
    while any(tuner.active for tuner in tuners.values()):
        asks = {
            item_id: tuner.ask(gates.gate_fn)
            for item_id, tuner in tuners.items() if tuner.active
        }
        for item_id in asks:
            ev["tuners"][item_id] = tuners[item_id].to_state()
        save_checkpoint(state)
        jobs: list[CaseJob] = []
        for item_id, points in asks.items():
            for point in points:
                if not point.feasible:
                    continue
                for case in cfg.tuning_cases:
                    jobs.append(CaseJob(
                        f"{item_id}-t{round_index}-{point.index}-{case}", case,
                        tuners[item_id].spec_for(point.constants),
                        {"item_id": item_id, "stage": "tuning", "round": round_index,
                         "index": point.index},
                    ))
        results = _run_batch(state, pending, cfg, backend, f"tune_r{round_index}", jobs)
        meta_by_job = {job.job_id: job.meta for job in jobs}
        by_key: dict[tuple[str, int], dict[str, float | None]] = {}
        for result in results:
            meta = meta_by_job[result.job_id]
            entry = by_key.setdefault((str(meta["item_id"]), int(meta["index"])), {})
            entry[result.case_id] = (
                None if result.status != "complete" else float(result.score["composite"])
            )
        for item_id, points in asks.items():
            tuners[item_id].tell({
                point.index: by_key.get((item_id, point.index), {
                    case: None for case in cfg.tuning_cases
                })
                for point in points if point.feasible
            })
        round_index += 1
        for item_id in asks:
            ev["tuners"][item_id] = tuners[item_id].to_state()
        ev["round"] = round_index
        save_checkpoint(state)

    items = _items_by_id(pending)
    for item_id, tuner in tuners.items():
        item = items[item_id]
        if item.get("tuning"):
            continue
        item["tuning"] = tuner.summary()
        tuned = tuner.tuned_spec()
        if tuned is not None:
            child = _new_item(
                replace(tuned, name=f"{tuned.name[:44]}_tuned"), generation, item["island"],
                "cfd_tuned",
                f"CMA-ES on CFD ({', '.join(cfg.tuning_cases)}, {tuner.state['cfd_evals']} "
                f"evaluations): subset {tuner.state['original_f']:.5f} -> "
                f"{tuner.state['best']['f']:.5f}; parent retained",
                0, parent_hash=item["spec_hash"],
            )
            child["tuning_parent_item_id"] = item_id
            child["tuning_subset"] = tuner.state["best"]
            if _register_item(state, pending, child):
                _gate_item(state, child, gates)
                if child["status"] == "gates_passed":
                    per_case = tuner.state["best"].get("per_case") or {}
                    composite = per_case.get(SANITY_CASE)
                    threshold = SANITY_FACTOR * stock[SANITY_CASE]
                    child["sanity"] = {
                        "case": SANITY_CASE, "composite": composite, "threshold": threshold,
                        "stock_composite": stock[SANITY_CASE],
                        "source": "tuning subset evaluation",
                    }
                    if composite is None or composite > threshold:
                        _transition(state, child, "sanity_rejected", "sanity_rejected")
                    else:
                        _transition(state, child, "sanity_passed", "sanity_passed")
            item["tuned_child_item_id"] = child["item_id"]
        _record_item(state, item, "tuning_complete")
    save_checkpoint(state)


def _stage_confirm(
    state: SearchState, pending: dict[str, Any], cfg: SearchCfg, backend: Any,
) -> None:
    items = [item for item in pending["items"] if item["status"] == "sanity_passed"]
    jobs = [
        CaseJob(f"{item['item_id']}-full-{case}", case, _item_spec(item),
                {"item_id": item["item_id"], "stage": "full"})
        for item in items for case in FULL_CASES
    ]
    results = _run_batch(state, pending, cfg, backend, "confirm", jobs)
    by_item: dict[str, dict[str, CaseResult]] = {}
    for result in results:
        item_id = result.job_id.split("-full-")[0]
        by_item.setdefault(item_id, {})[result.case_id] = result
    for item in items:
        if item["status"] != "sanity_passed":
            continue
        case_results = by_item.get(item["item_id"], {})
        if len(case_results) != FULL_CASE_EVALS:
            raise RuntimeError(f"confirmation batch lost cases for {item['item_id']}")
        full = full_from_results(_item_spec(item), case_results)
        full["outcomes"] = {case: _compact_outcome(r) for case, r in case_results.items()}
        _transition(state, item, "tier2_complete", "tier2_complete", full=full)


def _evaluate_generation(
    state: SearchState, pending: dict[str, Any], cfg: SearchCfg, backend: Any,
    gates: GateContext, stock: dict[str, float],
) -> None:
    ev = pending["evaluation"]
    stages = ("sanity", "tuning", "confirm")
    try:
        start = stages.index(ev.get("stage", "sanity")) if ev.get("stage") in stages else 3
        for stage in stages[start:]:
            ev["stage"] = stage
            save_checkpoint(state)
            if stage == "sanity":
                _stage_sanity(state, pending, cfg, backend, stock)
            elif stage == "tuning":
                _stage_tuning(state, pending, cfg, backend, gates, stock)
            else:
                _stage_confirm(state, pending, cfg, backend)
        ev["stage"] = "done"
    except BudgetExhausted as exc:
        ev["budget_stop"] = f"{type(exc).__name__}: {exc}"
        ev["stage"] = "done"
        for item in pending["items"]:
            if item["status"] in {"gates_passed", "sanity_passed"}:
                _transition(state, item, "budget_exhausted", "budget_exhausted",
                            budget_reason=str(exc)[:300])
    save_checkpoint(state)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def _item_member(item: dict[str, Any]) -> Member:
    full = item["full"]
    return Member(
        spec=_item_spec(item), val_j=float(full["val_j"]), cal_j=float(full["cal_j"]),
        val_raw=float(full["val_raw"]), cal_raw=float(full["cal_raw"]),
        per_case={
            case: (float("nan") if value is None else float(value))
            for case, value in full["per_case"].items()
        },
        novelty_best_rms=float(
            float("inf") if item.get("novelty_best_rms") is None else item["novelty_best_rms"]
        ),
        origin=str(item["origin"]),
    )


def _member_sort_key(member: Member) -> tuple[float, float, str]:
    return member.val_j, member.val_raw, spec_hash(member.spec)


def _dominates(left: Member, right: Member) -> bool:
    if not all(np.isfinite((left.val_j, left.val_raw, right.val_j, right.val_raw))):
        raise RuntimeError("population member has a non-finite validation objective")
    return (
        left.val_j <= right.val_j and left.val_raw <= right.val_raw
        and (left.val_j < right.val_j or left.val_raw < right.val_raw)
    )


def _pareto_fronts(members: Iterable[Member]) -> list[list[Member]]:
    remaining = sorted(list(members), key=_member_sort_key)
    fronts: list[list[Member]] = []
    while remaining:
        front = [
            member for member in remaining
            if not any(other is not member and _dominates(other, member) for other in remaining)
        ]
        if not front:
            raise RuntimeError("failed to construct a validation Pareto front")
        front.sort(key=_member_sort_key)
        selected_ids = {id(member) for member in front}
        remaining = [member for member in remaining if id(member) not in selected_ids]
        fronts.append(front)
    return fronts


def _crowding_order(front: list[Member]) -> list[Member]:
    if len(front) <= 2:
        return sorted(front, key=_member_sort_key)
    distance = {id(member): 0.0 for member in front}
    for attribute in ("val_j", "val_raw"):
        ordered = sorted(
            front, key=lambda member: (float(getattr(member, attribute)), spec_hash(member.spec))
        )
        distance[id(ordered[0])] = float("inf")
        distance[id(ordered[-1])] = float("inf")
        low = float(getattr(ordered[0], attribute))
        high = float(getattr(ordered[-1], attribute))
        if high <= low:
            continue
        for index in range(1, len(ordered) - 1):
            previous = float(getattr(ordered[index - 1], attribute))
            following = float(getattr(ordered[index + 1], attribute))
            distance[id(ordered[index])] += (following - previous) / (high - low)
    return sorted(front, key=lambda member: (-distance[id(member)], *_member_sort_key(member)))


def _pareto_select_members(members: Iterable[Member], capacity: int) -> list[Member]:
    if capacity < 1:
        return []
    selected: list[Member] = []
    for front in _pareto_fronts(members):
        remaining = capacity - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front)
        else:
            selected.extend(_crowding_order(front)[:remaining])
            break
    return sorted(selected, key=_member_sort_key)


def stratified_select(members: Iterable[Member], capacity: int = ISLAND_SIZE) -> list[Member]:
    """Two Pareto survivors per size class, the rest by global Pareto order.

    The stock incumbent (zero parameters) always survives and does not use a
    slot.  A large model can therefore only be displaced by a better large
    model, never by a small one.
    """
    pool = list(members)
    stock = [m for m in pool if m.n_parameters == 0]
    others = [m for m in pool if m.n_parameters > 0]
    selected: list[Member] = []
    chosen: set[int] = set()
    for key, low, high in SIZE_CLASSES:
        candidates = [m for m in others if low <= m.n_parameters <= high]
        for member in _pareto_select_members(candidates, SLOTS_PER_CLASS):
            if id(member) not in chosen:
                chosen.add(id(member))
                selected.append(member)
    leftover = capacity - len(selected)
    if leftover > 0:
        remaining = [m for m in others if id(m) not in chosen]
        selected.extend(_pareto_select_members(remaining, leftover))
    incumbent = sorted(stock, key=_member_sort_key)[:1]
    return sorted(incumbent + selected, key=_member_sort_key)


def _select_generation(state: SearchState, pending: dict[str, Any]) -> None:
    snapshot = {
        island: [_member_from_payload(payload) for payload in pending["snapshot"][island]]
        for island in ISLANDS
    }
    children = {
        island: [
            _item_member(item) for item in pending["items"]
            if item["island"] == island and item.get("status") == "tier2_complete"
        ]
        for island in ISLANDS
    }
    for island in ISLANDS:
        archived = {spec_hash(m.spec): m for m in state.evaluated_archive[island]}
        for member in children[island]:
            archived.setdefault(spec_hash(member.spec), member)
        state.evaluated_archive[island] = sorted(archived.values(), key=_member_sort_key)
        by_structure: dict[str, Member] = {}
        for member in snapshot[island] + children[island]:
            identity = struct_hash(member.spec)
            incumbent = by_structure.get(identity)
            if incumbent is None or (member.val_j, spec_hash(member.spec)) < (
                incumbent.val_j, spec_hash(incumbent.spec)
            ):
                by_structure[identity] = member
        state.islands[island] = stratified_select(by_structure.values())


def class_bests(members: Iterable[Member]) -> dict[str, float]:
    bests: dict[str, float] = {}
    for member in members:
        if member.n_parameters == 0:
            continue
        key = member.size_class
        if key not in bests or member.val_raw < bests[key]:
            bests[key] = float(member.val_raw)
    return bests


def _reconstruct_plateau(
    best_val_history: Iterable[float], best_raw_history: Iterable[float],
    best_class_history: Iterable[dict[str, float]], stock_val_j: float,
    stock_val_raw: float,
) -> tuple[float, float, dict[str, float], int]:
    """Replay progress against stock as generation zero.

    A generation counts as progress when the penalized champion, the raw
    champion, or the best raw error of ANY size class improves by more than
    the protocol threshold.
    """
    val_history = [float(v) for v in best_val_history]
    raw_history = [float(v) for v in best_raw_history]
    class_history = [dict(row) for row in best_class_history]
    if not (len(val_history) == len(raw_history) == len(class_history)):
        raise RuntimeError("progress histories have different lengths")
    best_val, best_raw = float(stock_val_j), float(stock_val_raw)
    best_classes: dict[str, float] = {}
    plateau = 0
    for gen_val, gen_raw, gen_classes in zip(val_history, raw_history, class_history):
        progress = False
        if _meaningful_improvement(gen_val, best_val):
            best_val, progress = gen_val, True
        if _meaningful_improvement(gen_raw, best_raw):
            best_raw, progress = gen_raw, True
        for key, value in gen_classes.items():
            if key not in best_classes or _meaningful_improvement(value, best_classes[key]):
                if key not in best_classes and _meaningful_improvement(value, stock_val_raw):
                    progress = True
                elif key in best_classes:
                    progress = True
                best_classes[key] = min(value, best_classes.get(key, value))
        plateau = 0 if progress else plateau + 1
    return best_val, best_raw, best_classes, plateau


def _generation_counts(pending: dict[str, Any]) -> dict[str, int]:
    items = pending["items"]

    def count(status: str) -> int:
        return sum(item["status"] == status for item in items)

    return {
        "proposed": sum(item["origin"] == "llm_original" for item in items),
        "seeds": sum(item["origin"] == "declared_seed" for item in items),
        "tuned": sum(item["origin"] == "cfd_tuned" for item in items),
        "duplicates": count("global_duplicate"),
        "policy_rejects": count("search_policy_rejected"),
        "diversity_rejects": count("diversity_quota_rejected"),
        "tier0_rejects": count("tier0_rejected"),
        "tier1_rejects": count("tier1_rejected"),
        "sanity_rejects": count("sanity_rejected"),
        "tier2_children": count("tier2_complete"),
        "budget_refused": count("budget_exhausted"),
    }


CSV_HEADER = [
    "run_id", "protocol_fingerprint", "gen", "snapshot_hash", "context_migration",
    "proposed", "seeds", "tuned", "global_duplicates", "policy_rejects",
    "diversity_rejects", "t0_rejects", "t1_rejects", "sanity_rejects",
    "t2_children", "budget_refused",
    "raw_champion_val_raw", "raw_champion_penalty", "raw_champion_val_j",
    "raw_champion_spec_hash", "j_champion_val_raw", "j_champion_penalty",
    "j_champion_val_j", "j_champion_spec_hash",
    "class_best_1_2", "class_best_3_4", "class_best_5_6", "class_best_7p",
    "median_val_j", "tier2_evals_cum", "core_hours_cum", "wall_h", "note",
]


def _generation_csv_path(state: SearchState) -> Path:
    if not state.run_id or not state.protocol_fingerprint:
        raise ValueError("generation telemetry requires a run and fingerprint namespace")
    return GENERATIONS_ROOT / state.protocol_fingerprint / state.run_id / "generations.csv"


def _upsert_generation_row(state: SearchState, row: list[Any]) -> None:
    path = _generation_csv_path(state)
    namespaced_row = [state.run_id, state.protocol_fingerprint, *row]
    if len(namespaced_row) != len(CSV_HEADER):
        raise ValueError(
            f"generation row has {len(namespaced_row)} columns, expected {len(CSV_HEADER)}"
        )
    rows: list[list[str]] = []
    if path.exists():
        with open(path, newline="") as fh:
            reader = list(csv.reader(fh))
        if reader and reader[0] != CSV_HEADER:
            raise RuntimeError("generation CSV has an incompatible schema; archive it before launch")
        rows = reader[1:] if reader else []
    generation = str(row[0])
    rows = [existing for existing in rows if len(existing) > 2 and existing[2] != generation]
    rows.append([str(value) for value in namespaced_row])
    rows.sort(key=lambda values: int(values[2]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "x", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_finalists(state: SearchState) -> None:
    members = [
        member for source in (state.evaluated_archive, state.islands)
        for values in source.values() for member in values
    ]
    unique: list[Member] = []
    seen: set[str] = set()
    for member in sorted(members, key=_member_sort_key):
        identity = spec_hash(member.spec)
        if identity not in seen:
            seen.add(identity)
            unique.append(member)

    def finalist_payload(member: Member) -> dict[str, Any]:
        return {
            **_member_payload(member), "spec_hash": spec_hash(member.spec),
            "struct_hash": struct_hash(member.spec),
            "n_parameters": member.n_parameters, "size_class": member.size_class,
            "complexity_penalty": member.complexity_penalty,
        }

    penalized = sorted(unique, key=_member_sort_key)
    raw = sorted(unique, key=lambda m: (m.val_raw, m.val_j, spec_hash(m.spec)))
    pareto = _pareto_fronts(unique)[0] if unique else []
    by_class = {
        key: [
            finalist_payload(m) for m in sorted(
                [m for m in unique if m.size_class == key],
                key=lambda m: (m.val_raw, m.val_j, spec_hash(m.spec)),
            )[:3]
        ]
        for key, _, _ in SIZE_CLASSES
    }
    payload = {
        "schema_version": FINALISTS_SCHEMA, "run_id": state.run_id,
        "protocol_fingerprint": state.protocol_fingerprint,
        "status": state.status, "stop_reason": state.stop_reason,
        "precheck": state.precheck, "generation": state.generation,
        "tier2_evals": state.tier2_evals, "core_hours_spent": state.core_hours_spent,
        "acceptance_objective": "frozen penalized validation J",
        "validation_pareto_scope": "all Tier-2-evaluated candidates plus stock",
        "finalists": [finalist_payload(m) for m in penalized[:5]],
        "penalized_finalists": [finalist_payload(m) for m in penalized[:5]],
        "raw_finalists": [finalist_payload(m) for m in raw[:5]],
        "validation_pareto_front": [finalist_payload(m) for m in pareto],
        "size_class_fronts": by_class,
    }
    _atomic_json(FINALISTS, payload)


def stock_reference(path: Path = STOCK_CACHE) -> tuple[dict[str, float], float, float, dict[str, Any]]:
    """Per-case stock composites plus its raw validation/calibration means."""
    if not path.is_file():
        raise RuntimeError(f"stock reference cache is missing: {path}")
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 3 or payload.get("status") != "complete":
        raise RuntimeError("refusing legacy or incomplete stock reference cache")
    per_case = {case: float(value) for case, value in payload["per_case"].items()}
    if set(per_case) != set(FULL_CASES) or not all(np.isfinite(list(per_case.values()))):
        raise RuntimeError("stock cache does not contain one finite score per dev case")
    val_raw = float(np.mean([per_case[case] for case in mcconkey.VALIDATION]))
    cal_raw = float(np.mean([per_case[case] for case in mcconkey.CALIBRATION]))
    if abs(val_raw - float(payload["val_j"])) > 1.0e-12:
        raise RuntimeError("stock cache validation objective is internally inconsistent")
    return per_case, val_raw, cal_raw, dict(payload.get("backend_provenance", {}))


def _all_finite(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise RuntimeError(f"a-priori table field {name} is empty or non-finite")
    return array


FEATURE_SEMANTICS = "PoE=SST production/(betaStar*k*omega)"


def _preflight_apriori_table(
    table: dict[str, Any],
) -> tuple[
    cand.States, np.ndarray, np.ndarray, np.ndarray,
    list[cand.States], dict[str, Any],
]:
    """Validate the complete pilot table contract before any proposal/CFD."""
    required = {
        "states", "b", "rhat", "case_id", "weights", "poe_truth",
        "feature_semantics",
    }
    if not required.issubset(table):
        raise RuntimeError(
            f"a-priori table missing required fields: {sorted(required - set(table))}"
        )
    if table["feature_semantics"] != FEATURE_SEMANTICS:
        raise RuntimeError("a-priori feature semantics differ from the pilot protocol")
    states = table["states"]
    if not isinstance(states, cand.States) or len(states) == 0:
        raise RuntimeError("a-priori states are empty or have the wrong type")
    n_rows = len(states)
    expected_arrays = {
        "states.shat": ((n_rows, 3, 3), states.shat),
        "states.what": ((n_rows, 3, 3), states.what),
        "states.T": ((n_rows, 10, 3, 3), states.T),
        "states.inv": ((n_rows, 5), states.inv),
        "states.ret": ((n_rows,), states.ret),
        "states.f1": ((n_rows,), states.f1),
        "states.poe": ((n_rows,), states.poe),
        "b": ((n_rows, 3, 3), table["b"]),
        "rhat": ((n_rows,), table["rhat"]),
        "weights": ((n_rows,), table["weights"]),
        "poe_truth": ((n_rows,), table["poe_truth"]),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (expected_shape, values) in expected_arrays.items():
        array = np.asarray(values)
        if array.shape != expected_shape:
            raise RuntimeError(
                f"a-priori {name} shape {array.shape} != {expected_shape}"
            )
        arrays[name] = _all_finite(name, array)
    case_ids_raw = np.asarray(table["case_id"])
    if case_ids_raw.shape != (n_rows,) or any(
        not isinstance(value, str) or not value
        for value in case_ids_raw.tolist()
    ):
        raise RuntimeError("a-priori case IDs are not aligned nonempty strings")
    case_ids = case_ids_raw.astype(str)
    observed_order = list(dict.fromkeys(case_ids.tolist()))
    expected_order = list(mcconkey.CALIBRATION)
    if observed_order != expected_order:
        raise RuntimeError(
            f"a-priori case order/content {observed_order} != {expected_order}"
        )
    weights = arrays["weights"]
    if np.any(weights < 0.0) or not np.isclose(
        float(np.sum(weights)), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError("a-priori weights must be nonnegative and sum to one")
    dns_gate_states: list[cand.States] = []
    positive_cases: list[str] = []
    zero_cases: list[str] = []
    weight_sums: dict[str, float] = {}
    for case_id in expected_order:
        mask = case_ids == case_id
        case_weights = weights[mask]
        weight_sums[case_id] = float(np.sum(case_weights))
        if np.all(case_weights > 0.0):
            positive_cases.append(case_id)
            dns_gate_states.append(cand.States(
                states.shat[mask], states.what[mask], states.T[mask],
                states.inv[mask], states.ret[mask], states.f1[mask],
                states.poe[mask],
            ))
        elif np.all(case_weights == 0.0):
            zero_cases.append(case_id)
        else:
            raise RuntimeError(f"a-priori {case_id} has mixed zero/positive weights")
    expected_positive = [
        case_id for case_id in expected_order if case_id != "cbfs13700"
    ]
    if positive_cases != expected_positive or zero_cases != ["cbfs13700"]:
        raise RuntimeError("a-priori truth-weight case contract is inconsistent")
    diagnostics = {
        "rows": n_rows,
        "feature_semantics": table["feature_semantics"],
        "case_order": observed_order,
        "positive_weight_cases": positive_cases,
        "zero_weight_cases": zero_cases,
        "weight_sums": weight_sums,
    }
    return (
        states, arrays["b"], arrays["rhat"], weights,
        dns_gate_states, diagnostics,
    )



TERMINAL_STATUSES = {
    "complete_max_generations", "stopped_budget", "stopped_plateau", "stopped_user",
}


def run_search(
    cfg: SearchCfg,
    seeds: list[CandidateSpec] | dict[str, list[CandidateSpec]],
    backend: Any,
    stock_cache: Path = STOCK_CACHE,
    proposer: ClaudeCodeProposer | None = None,
) -> SearchState:
    """Run or resume the four synchronous islands on ``backend``."""
    cfg.validate()
    seed_map = _normalized_seeds(seeds)
    proposer = proposer or ClaudeCodeProposer(model=cfg.model)
    backend_provenance = backend.provenance()
    manifest, fingerprint = build_protocol_manifest(
        cfg, seed_map, backend_provenance, claude_manifest=proposer.binary_manifest
    )
    stock, stock_val_j, stock_cal_j, stock_provenance = stock_reference(stock_cache)
    if backend.name == "slurm":
        for key in ("lib_sha256", "simpleFoam_sha256", "wm_options"):
            if stock_provenance.get(key) != backend_provenance.get(key):
                raise RuntimeError(
                    f"stock reference was produced under a different {key}; rebuild it"
                )

    state = load_checkpoint(expected_fingerprint=fingerprint)
    generation_commit_in_progress = False
    try:
        if state is None:
            stock_spec = CandidateSpec(name="stock_sst")
            stock_member = Member(
                stock_spec, stock_val_j, stock_cal_j, dict(stock),
                val_raw=stock_val_j, cal_raw=stock_cal_j, origin="stock",
            )
            state = SearchState(
                islands={island: [stock_member] for island in ISLANDS},
                evaluated_archive={island: [stock_member] for island in ISLANDS},
                run_id=f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex}",
                protocol_fingerprint=fingerprint, protocol_manifest=manifest,
                seen_spec_hashes={spec_hash(stock_spec)}, status="prechecking",
                precheck={"status": "scheduled", "started_wall_time": time.time()},
            )
            save_checkpoint(state)
        else:
            if state.protocol_manifest != manifest:
                raise RuntimeError("checkpoint manifest differs from the current protocol manifest")
            if state.status in TERMINAL_STATUSES:
                if state.pending_generation is not None and state.status != "stopped_user":
                    raise RuntimeError("terminal checkpoint has unfinished work")
                _write_finalists(state)
                return state
            state.tier2_reserved = 0  # every in-flight batch is resumed from its handle
            print(
                f"resumed four islands after generation {state.generation}; "
                f"case-evals={state.tier2_evals}, core-hours={state.core_hours_spent:.1f}",
                flush=True,
            )

        _assert_protocol_unchanged(state)
        state.precheck = {
            "status": "running", "started_wall_time": time.time(),
            "protocol_fingerprint": fingerprint,
            "claude_binary_sha256": proposer.binary_manifest["sha256"],
            "stock_cache": str(stock_cache), "backend": backend.name,
            "short_protocol": True, "warm_start": False,
        }
        state.status = "prechecking"
        state.stop_reason = ""
        save_checkpoint(state)

        sampled_states = cand.sample_states(n=30_000, seed=cfg.seed)
        with open(REPO / "results" / "phase2" / "apriori_table.pkl", "rb") as fh:
            table = pickle.load(fh)
        ap_states, b_target, r_target, ap_weights, dns_gate_states, table_diagnostics = (
            _preflight_apriori_table(table)
        )
        baseline_channel = tier1.solve_channel1d(None)
        if not baseline_channel.converged or not np.isfinite(baseline_channel.cf):
            raise RuntimeError("stock channel baseline failed during pre-check")
        gates = GateContext(
            sampled_states, dns_gate_states, baseline_channel, ap_states,
            b_target, r_target, ap_weights,
        )
        state.precheck.update({
            "status": "complete", "completed_wall_time": time.time(),
            "apriori_table": table_diagnostics, "dns_gate_groups": len(dns_gate_states),
            "channel_baseline_cf": baseline_channel.cf,
        })
        state.status = "running"
        save_checkpoint(state)

        best_val_ever, best_raw_ever, best_classes_ever, plateau = _reconstruct_plateau(
            state.best_val_history, state.best_raw_history, state.best_class_history,
            stock_val_j, stock_val_j,
        )
        search_start = time.time()
        while state.generation < cfg.max_generations:
            _assert_protocol_unchanged(state, backend)
            pending = _resume_or_prepare_generation(
                state, cfg, seed_map, stock, stock_val_j, proposer
            )
            if pending["status"] == "paused_llm":
                _write_finalists(state)
                return state
            if pending["status"] == "gating":
                for item in pending["items"]:
                    _gate_item(state, item, gates)
                pending["status"] = "evaluating"
                save_checkpoint(state)
            if pending["status"] == "evaluating":
                _evaluate_generation(state, pending, cfg, backend, gates, stock)
                pending["status"] = "selecting"
                save_checkpoint(state)
            if pending["status"] != "selecting":
                raise RuntimeError(f"unknown pending-generation state {pending['status']!r}")

            generation_commit_in_progress = True
            _select_generation(state, pending)
            all_members = [m for values in state.islands.values() for m in values]
            best_member = min(all_members, key=lambda m: (m.val_j, spec_hash(m.spec)))
            raw_member = min(all_members, key=lambda m: (m.val_raw, m.val_j, spec_hash(m.spec)))
            gen_best = best_member.val_j
            gen_best_raw = raw_member.val_raw
            gen_classes = class_bests(all_members)
            median = float(np.median([m.val_j for m in all_members]))
            state.best_val_history.append(gen_best)
            state.best_raw_history.append(gen_best_raw)
            state.best_class_history.append(gen_classes)
            progress = False
            if _meaningful_improvement(gen_best, best_val_ever):
                best_val_ever, progress = gen_best, True
            if _meaningful_improvement(gen_best_raw, best_raw_ever):
                best_raw_ever, progress = gen_best_raw, True
            for key, value in gen_classes.items():
                if key not in best_classes_ever:
                    if _meaningful_improvement(value, stock_val_j):
                        progress = True
                    best_classes_ever[key] = value
                elif _meaningful_improvement(value, best_classes_ever[key]):
                    best_classes_ever[key] = value
                    progress = True
            plateau = 0 if progress else plateau + 1
            counts = _generation_counts(pending)
            budget_reason = pending["evaluation"].get("budget_stop")
            budget_stop = budget_reason is not None or (
                cfg.tier2_eval_cap - state.tier2_evals < FULL_CASE_EVALS + 1
            )
            plateau_stop = plateau >= cfg.plateau_generations
            note = ""
            if budget_stop:
                note = budget_reason or "Tier-2 case-evaluation cap reached"
                state.status = "stopped_budget"
                state.stop_reason = note
            elif plateau_stop:
                note = "plateau"
                state.status = "stopped_plateau"
                state.stop_reason = (
                    f"no >= {MIN_VAL_IMPROVEMENT:.3f} improvement in penalized J, raw "
                    f"error or any size class for {plateau} committed generations"
                )
            else:
                state.status = "running"
                state.stop_reason = ""
            generation = int(pending["generation"])
            state.generation = generation
            state.pending_generation = None
            _upsert_generation_row(state, [
                generation, pending["snapshot_hash"], pending["context_only_migration"],
                counts["proposed"], counts["seeds"], counts["tuned"], counts["duplicates"],
                counts["policy_rejects"], counts["diversity_rejects"],
                counts["tier0_rejects"], counts["tier1_rejects"], counts["sanity_rejects"],
                counts["tier2_children"], counts["budget_refused"],
                f"{gen_best_raw:.8f}", f"{raw_member.complexity_penalty:.8f}",
                f"{raw_member.val_j:.8f}", spec_hash(raw_member.spec),
                f"{best_member.val_raw:.8f}", f"{best_member.complexity_penalty:.8f}",
                f"{gen_best:.8f}", spec_hash(best_member.spec),
                *[
                    "" if key not in gen_classes else f"{gen_classes[key]:.8f}"
                    for key, _, _ in SIZE_CLASSES
                ],
                f"{median:.8f}", state.tier2_evals, f"{state.core_hours_spent:.2f}",
                f"{(time.time() - search_start) / 3600:.3f}", note,
            ])
            save_checkpoint(state)
            generation_commit_in_progress = False
            print(
                f"gen {generation}: proposals={counts['proposed']} tuned={counts['tuned']} "
                f"t0rej={counts['tier0_rejects']} t1rej={counts['tier1_rejects']} "
                f"diversity={counts['diversity_rejects']} sanityrej={counts['sanity_rejects']} "
                f"t2={counts['tier2_children']} budget_refused={counts['budget_refused']} "
                f"J_champion={gen_best:.5f} raw_champion={gen_best_raw:.5f} "
                f"classes={ {k: round(v, 5) for k, v in gen_classes.items()} } "
                f"evals={state.tier2_evals} core_h={state.core_hours_spent:.0f} plateau={plateau}",
                flush=True,
            )
            _assert_protocol_unchanged(state)
            if budget_stop or plateau_stop:
                print(f"stopping four islands: {state.stop_reason}", flush=True)
                break

        if state.status == "running":
            state.status = "complete_max_generations"
            state.stop_reason = f"completed configured {cfg.max_generations} generations"
            save_checkpoint(state)
        _write_finalists(state)
        return state
    except Exception as exc:
        if state is not None and not generation_commit_in_progress:
            state.status = "failed"
            state.stop_reason = f"{type(exc).__name__}: {exc}"[:1000]
            try:
                save_checkpoint(state)
            except Exception:
                pass
        raise
