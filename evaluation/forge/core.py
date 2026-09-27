"""FORGE v2's filesystem, identity and accounting primitives."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def within(root: Path, path: Path, *, allow_root: bool = False) -> Path:
    """Resolve links before granting a write; lexical containment is insufficient."""
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root) or (path == root and not allow_root):
        raise ValueError(f"Output must be inside {root}: {path}")
    return path


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def file_lock(path: Path, *, blocking: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle, operation)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another FORGE process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def source_identity(root: Path = ROOT, *, physical_only: bool = False) -> dict[str, str]:
    """Only executable definitions; generated reports cannot change a protocol."""
    root = Path(root)
    files = []
    for folder in ("src/forge", "src/tedp", "src/kOmegaSSTBasis"):
        files.extend(p for p in (root / folder).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".C", ".H"})
    files.extend(p for p in (root / "requirements.lock", root / "pyproject.toml") if p.is_file())
    if physical_only:
        controller_only = {"search.py", "proposer.py", "seeds.py", "tuning.py", "cli.py", "capability.py", "cfd_capability.py"}
        files = [p for p in files if not (p.parent == root/"src/forge" and p.name in controller_only)]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(files)}


class BudgetExhausted(RuntimeError):
    pass


class Ledger:
    """Process-safe committed and consumed core-hours, including failed runs.

    A killed controller leaves a reservation in place. It cannot disappear on
    resume; the evaluator explicitly closes interrupted attempts conservatively.
    """
    def __init__(self, path: Path, cap: float | None):
        self.path, self.cap = Path(path), None if cap is None else float(cap)
        if self.cap is not None and (not math.isfinite(self.cap) or self.cap <= 0):
            raise ValueError("core-hour cap must be positive and finite")

    def _read(self) -> dict:
        if self.path.exists():
            value = json.loads(self.path.read_text())
            if value["cap_core_hours"] != self.cap:
                raise ValueError("Ledger budget is immutable; create a new campaign")
            return value
        return {"schema_version": 2, "cap_core_hours": self.cap, "jobs": {}}

    def snapshot(self) -> dict:
        with file_lock(self.path.with_suffix(".lock")):
            value = self._read()
        spent = sum(float(j.get("actual_core_hours", 0)) for j in value["jobs"].values())
        committed = sum(float(j["reservation_core_hours"]) for j in value["jobs"].values()
                        if j["status"] == "running")
        return {**value, "spent": spent, "committed": committed,
                "remaining": None if self.cap is None else self.cap - spent - committed}

    def reserve(self, job: str, core_hours: float, identity: dict, *, protected_reserve: float = 0) -> None:
        amount = float(core_hours)
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Reservation must be positive and finite")
        with file_lock(self.path.with_suffix(".lock")):
            value = self._read()
            if job in value["jobs"]:
                raise ValueError(f"Attempt already accounted: {job}")
            charged = sum(float(j.get("actual_core_hours", 0)) +
                          (float(j["reservation_core_hours"]) if j["status"] == "running" else 0)
                          for j in value["jobs"].values())
            if protected_reserve < 0 or not math.isfinite(protected_reserve):
                raise ValueError("Protected reserve must be nonnegative and finite")
            if self.cap is not None and charged + amount + protected_reserve > self.cap + 1e-12:
                raise BudgetExhausted(f"{amount:.3f} core-hours required; {self.cap-charged-protected_reserve:.3f} remain outside the protected reserve")
            value["jobs"][job] = {"status": "running", "reservation_core_hours": amount,
                                  "identity": identity, "started_utc": utc_now()}
            atomic_json(self.path, value)

    def settle(self, job: str, actual: float, status: str, details: dict | None = None) -> None:
        actual = float(actual)
        if not math.isfinite(actual) or actual < 0 or status == "running":
            raise ValueError("Invalid completed accounting")
        with file_lock(self.path.with_suffix(".lock")):
            value = self._read()
            row = value["jobs"][job]
            if row["status"] != "running":
                if row["actual_core_hours"] != actual:
                    raise ValueError("Cannot rewrite settled accounting")
                return
            row.update(status=status, actual_core_hours=actual, finished_utc=utc_now(),
                       details=details or {})
            atomic_json(self.path, value)


def verify_originals(record: Path) -> dict:
    """Read-only content comparison; never reset or clean an original repository."""
    data = json.loads(Path(record).read_text())
    changed, checked = [], 0
    for root, origin in data["origins"].items():
        for rel, expected in origin["tracked_content_sha256"].items():
            p = Path(root) / rel
            checked += 1
            if isinstance(expected, dict):
                actual = {"symlink": str(p.readlink())} if p.is_symlink() else None
            else:
                actual = sha256(p) if p.is_file() else None
            if actual != expected:
                changed.append(str(p))
    return {"unchanged": not changed, "checked_paths": checked, "changed": changed}
