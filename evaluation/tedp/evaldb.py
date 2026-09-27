"""Append-only evaluation record: results/evaluations.csv.

One row per (candidate, case, tier) evaluation. flock-guarded appends so
parallel workers never interleave. Nothing is ever rewritten; audits and the
leaderboard are derived by reading the whole file.

Also holds the scoring/grammar freeze check: once results/FREEZE_scoring.json
exists, any change to scoring.py or the two grammar files aborts at import.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "results" / "evaluations.csv"
FREEZE_PATH = REPO / "results" / "FREEZE_scoring.json"
AMENDMENTS_PATH = REPO / "results" / "FREEZE_amendments.json"

# files whose hash is pinned by the Phase-1 freeze
FROZEN_FILES = (
    "src/tedp/scoring.py",
    "src/tedp/expr.py",
    "src/kOmegaSSTBasis/basisExpr/exprNode.C",
    "src/kOmegaSSTBasis/basisExpr/exprParser.C",
    "src/kOmegaSSTBasis/basisExpr/exprNode.H",
    "src/kOmegaSSTBasis/basisExpr/exprParser.H",
)

COLUMNS = [
    "ts_utc",
    "spec_hash",
    "struct_hash",
    "case",
    "tier",
    "e_u",
    "e_cf",
    "e_uv",
    "composite",
    "converged",
    "diverged",
    "iterations",
    "res_p",
    "res_Ux",
    "res_k",
    "res_omega",
    "wall_s",
    "nprocs",
    "of_version",
    "lib_sha",
    "scoring_sha",
    "seed",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hashes() -> dict[str, str]:
    return {rel: file_sha256(REPO / rel) for rel in FROZEN_FILES}


def write_freeze(note: str = "") -> dict:
    """Phase-1 exit: record the scoring+grammar hashes. Run exactly once."""
    if FREEZE_PATH.exists():
        raise RuntimeError(f"{FREEZE_PATH} already exists — the freeze is one-shot")
    payload = {
        "note": note,
        "hashes": frozen_hashes(),
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip(),
    }
    FREEZE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


#: Files the freeze can never release, whatever any amendment says. The
#: objective and the parameter-count rule are the verdict itself: amending them
#: would retroactively change every score in the project.
UNAMENDABLE = ("src/tedp/scoring.py",)


def load_amendments() -> list[dict]:
    """Recorded, deliberate changes to frozen files, oldest first.

    The Phase-1 freeze is one-shot and is never rewritten. When a frozen file
    has to change, the change is recorded here with its reason and its new
    hash, so the tripwire keeps firing on everything that was NOT declared.
    An amendment is a signed statement, not an escape hatch: it names the
    files, pins their hashes, and cannot cover anything in UNAMENDABLE.
    """
    if not AMENDMENTS_PATH.exists():
        return []
    return json.loads(AMENDMENTS_PATH.read_text())["amendments"]


def check_freeze() -> None:
    """Abort if any frozen file changed without being declared in an amendment."""
    if not FREEZE_PATH.exists():
        return
    allowed = dict(json.loads(FREEZE_PATH.read_text())["hashes"])
    for amendment in load_amendments():
        for rel, sha in amendment["hashes"].items():
            if rel in UNAMENDABLE:
                raise RuntimeError(
                    f"{AMENDMENTS_PATH.name} tries to amend {rel}, which is "
                    f"permanently frozen"
                )
            if rel not in allowed:
                raise RuntimeError(
                    f"{AMENDMENTS_PATH.name} names {rel}, which is not a frozen file"
                )
            allowed[rel] = sha
    current = frozen_hashes()
    changed = [rel for rel in allowed if allowed[rel] != current.get(rel)]
    if changed:
        raise RuntimeError(
            "FROZEN FILES CHANGED after the Phase-1 freeze: " + ", ".join(changed)
            + " (declare a deliberate change in " + AMENDMENTS_PATH.name + ")"
        )


check_freeze()


def scoring_sha() -> str:
    return file_sha256(REPO / "src/tedp/scoring.py")[:12]


@dataclass
class EvalRow:
    ts_utc: str
    spec_hash: str
    struct_hash: str
    case: str
    tier: int
    e_u: float
    e_cf: float
    e_uv: float
    composite: float
    converged: bool
    diverged: bool
    iterations: int
    res_p: float
    res_Ux: float
    res_k: float
    res_omega: float
    wall_s: float
    nprocs: int
    of_version: str
    lib_sha: str
    scoring_sha: str
    seed: int


assert [f.name for f in fields(EvalRow)] == COLUMNS


def append(row: EvalRow, db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    line_needed_header = not db_path.exists() or db_path.stat().st_size == 0
    with open(db_path, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        writer = csv.writer(fh)
        if line_needed_header:
            writer.writerow(COLUMNS)
        writer.writerow([getattr(row, c) for c in COLUMNS])
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)


def read_all(db_path: Path = DB_PATH) -> list[dict]:
    if not db_path.exists():
        return []
    with open(db_path, newline="") as fh:
        return list(csv.DictReader(fh))
