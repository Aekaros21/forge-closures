"""Tier-2 execution backends: a local thread pool or a SLURM cluster.

A batch is a list of :class:`CaseJob` (one candidate on one case).  Both
backends persist per-job results so an interrupted control process can
resume a batch without re-running finished cases, and both report the
compute actually consumed so the search can enforce a core-hour budget.

The SLURM backend keeps the control plane on the workstation: it stages a
manifest per node, submits one single-node job per manifest, polls
``sacct`` and pulls back small result JSON files.  Every remote call is
retried with back-off for a long time, because cluster access on this
project must be refreshed by the user every few hours and a lapse should
pause the search, not fail it.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import tier2
from ..spec import CandidateSpec, from_json, to_json

REPO = Path(__file__).resolve().parents[3]

# Median solver minutes per case at 4 ranks, measured from 278 laptop runs
# (2026-09-06).  Used only for packing and budget estimates.
CASE_COST_MINUTES: dict[str, float] = {
    "case_0p5": 4.6, "case_0p8": 5.8, "case_1p0": 2.4, "case_1p2": 5.8,
    "case_1p5": 4.8, "cbfs13700": 21.3, "convdiv12600": 12.9,
    "convdiv20580": 25.9, "h20": 10.2, "h26": 11.0, "h31": 11.8,
    "h38": 6.5, "h42": 6.0,
}
DEFAULT_COST_MINUTES = 15.0
TERMINAL_SLURM_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
}
RETRYABLE_FAILURES = {
    "infrastructure", "missing", "worker_exception", "setup", "timeout",
}


class BudgetExhausted(RuntimeError):
    """Raised when a batch would exceed the approved core-hour budget."""


@dataclass
class CaseJob:
    job_id: str
    case_id: str
    spec: CandidateSpec | None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def cost_minutes(self) -> float:
        return CASE_COST_MINUTES.get(self.case_id, DEFAULT_COST_MINUTES)

    def payload(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id, "case_id": self.case_id,
            "spec": None if self.spec is None else json.loads(to_json(self.spec)),
            "cost_minutes": self.cost_minutes, "meta": self.meta,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CaseJob":
        spec = payload.get("spec")
        return cls(
            job_id=str(payload["job_id"]), case_id=str(payload["case_id"]),
            spec=None if spec is None else from_json(json.dumps(spec)),
            meta=dict(payload.get("meta", {})),
        )


@dataclass
class CaseResult:
    job_id: str
    case_id: str
    status: str                  # complete | failed
    score: dict[str, float] | None
    failure_class: str | None
    reason: str
    outcome: dict[str, Any]

    @property
    def retryable(self) -> bool:
        return self.status != "complete" and self.failure_class in RETRYABLE_FAILURES

    @classmethod
    def from_outcome(cls, job: CaseJob, outcome: dict[str, Any] | None) -> "CaseResult":
        if outcome is None:
            return cls(
                job.job_id, job.case_id, "failed", None, "missing",
                "no result file returned by the worker", {},
            )
        return cls(
            job.job_id, job.case_id, str(outcome.get("status", "failed")),
            outcome.get("score"), outcome.get("failure_class"),
            str(outcome.get("reason", "")), outcome,
        )


@dataclass
class BatchAccounting:
    batch_id: str
    estimate_core_hours: float
    actual_core_hours: float | None
    node_jobs: list[dict[str, Any]]


# --------------------------------------------------------------------------
# core-hour ledger
# --------------------------------------------------------------------------
class CoreHourLedger:
    """Append-only record of estimated and billed core-hours per batch."""

    def __init__(self, path: Path, cap: float):
        self.path = Path(path)
        self.cap = float(cap)
        if self.path.is_file():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"cap": self.cap, "batches": {}}
        self.data["cap"] = self.cap

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    @property
    def spent(self) -> float:
        return float(sum(
            row["actual"] for row in self.data["batches"].values()
            if row.get("actual") is not None
        ))

    @property
    def committed(self) -> float:
        return float(sum(
            row["estimate"] for row in self.data["batches"].values()
            if row.get("actual") is None
        ))

    def remaining(self) -> float:
        return self.cap - self.spent - self.committed

    def commit(self, batch_id: str, estimate: float, detail: dict[str, Any]) -> None:
        self.data["batches"][batch_id] = {
            "estimate": float(estimate), "actual": None,
            "committed_wall_time": time.time(), **detail,
        }
        self._save()

    def settle(self, batch_id: str, actual: float, detail: dict[str, Any]) -> None:
        row = self.data["batches"].setdefault(batch_id, {"estimate": 0.0})
        row["actual"] = float(actual)
        row["settled_wall_time"] = time.time()
        row.update(detail)
        self._save()


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------
def list_schedule_makespan(costs: list[float], slots: int) -> float:
    """Greedy list-scheduling makespan of ``costs`` on ``slots`` machines."""
    if not costs:
        return 0.0
    loads = [0.0] * max(1, slots)
    for cost in sorted(costs, reverse=True):
        index = min(range(len(loads)), key=lambda i: loads[i])
        loads[index] += cost
    return max(loads)


def plan_nodes(
    jobs: list[CaseJob], slots_per_node: int, target_node_minutes: float,
    max_wall_minutes: float, cost: Callable[[CaseJob], float] | None = None,
) -> list[list[CaseJob]]:
    """Split jobs across single-node workers by longest-processing-time.

    The node count is chosen so each node runs about ``target_node_minutes``
    of packed work; it never exceeds the number of jobs and always grows
    until every node's estimated makespan fits under ``max_wall_minutes``.
    """
    if not jobs:
        return []
    cost = cost or (lambda job: job.cost_minutes)
    total = sum(cost(job) for job in jobs)
    n_nodes = max(1, math.ceil(total / (slots_per_node * target_node_minutes)))
    n_nodes = min(n_nodes, len(jobs))
    while True:
        buckets: list[list[CaseJob]] = [[] for _ in range(n_nodes)]
        loads = [0.0] * n_nodes
        for job in sorted(jobs, key=lambda j: (-cost(j), j.job_id)):
            index = min(range(n_nodes), key=lambda i: (loads[i], i))
            buckets[index].append(job)
            loads[index] += cost(job)
        makespans = [
            list_schedule_makespan([cost(j) for j in bucket], slots_per_node)
            for bucket in buckets
        ]
        if max(makespans) * 1.5 + 10.0 <= max_wall_minutes or n_nodes >= len(jobs):
            return [bucket for bucket in buckets if bucket]
        n_nodes += 1


# --------------------------------------------------------------------------
# local backend
# --------------------------------------------------------------------------
class LocalBackend:
    """Run batches on this machine; results persist per job for resume."""

    name = "local"

    def __init__(
        self, root: Path, nprocs: int = 4, max_workers: int = 5,
        runner: Callable[..., dict[str, Any]] | None = None,
        core_hour_ledger: CoreHourLedger | None = None,
    ):
        self.root = Path(root)
        self.nprocs = int(nprocs)
        self.max_workers = int(max_workers)
        self.runner = runner or tier2.run_case_job
        self.ledger = core_hour_ledger

    def provenance(self) -> dict[str, Any]:
        return {"backend": self.name, "nprocs": self.nprocs,
                "max_workers": self.max_workers}

    def estimate_core_hours(self, jobs: list[CaseJob]) -> float:
        return sum(job.cost_minutes for job in jobs) * self.nprocs / 60.0

    def _results_dir(self, batch_id: str) -> Path:
        return self.root / batch_id / "results"

    def run_batch(
        self, batch_id: str, jobs: list[CaseJob],
        on_submitted: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[CaseResult], BatchAccounting]:
        handle = {"backend": self.name, "batch_id": batch_id,
                  "jobs": [job.payload() for job in jobs], "status": "submitted",
                  "submitted_wall_time": time.time()}
        estimate = self.estimate_core_hours(jobs)
        if self.ledger is not None:
            if self.ledger.remaining() < estimate:
                raise BudgetExhausted(
                    f"batch {batch_id} needs {estimate:.1f} core-hours; "
                    f"{self.ledger.remaining():.1f} remain"
                )
            self.ledger.commit(batch_id, estimate, {"backend": self.name})
        if on_submitted is not None:
            on_submitted(handle)
        return self.resume_batch(handle)

    def resume_batch(
        self, handle: dict[str, Any],
        on_submitted: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[CaseResult], BatchAccounting]:
        del on_submitted
        batch_id = str(handle["batch_id"])
        jobs = [CaseJob.from_payload(p) for p in handle["jobs"]]
        results_dir = self._results_dir(batch_id)
        results_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()

        def run(job: CaseJob) -> tuple[str, dict[str, Any]]:
            path = results_dir / f"{job.job_id}.json"
            if path.is_file():
                return job.job_id, json.loads(path.read_text())
            outcome = self.runner(
                job.spec, job.case_id, self.root / batch_id / "runs" / job.job_id,
                nprocs=self.nprocs, keep_run_dir=False,
            )
            outcome["job_id"] = job.job_id
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_text(json.dumps(outcome, sort_keys=True))
            os.replace(tmp, path)
            return job.job_id, outcome

        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as pool:
            outcomes = dict(pool.map(run, jobs))
        results = [CaseResult.from_outcome(job, outcomes.get(job.job_id)) for job in jobs]
        actual = sum(
            float(r.outcome.get("wall_s", 0.0)) for r in results
        ) * self.nprocs / 3600.0
        if self.ledger is not None:
            self.ledger.settle(batch_id, actual, {"wall_s": time.time() - started})
        return results, BatchAccounting(batch_id, self.estimate_core_hours(jobs), actual, [])


# --------------------------------------------------------------------------
# SLURM backend
# --------------------------------------------------------------------------
@dataclass
class SlurmConfig:
    host: str
    remote_root: str
    account: str
    partition: str
    cores_per_node: int = 96
    # Cores requested per SLURM job.  Sub-node allocations backfill into
    # partially used nodes within seconds on a busy cluster, whereas a full
    # node can wait hours; billing is per core either way.
    cores_per_job: int = 48
    nprocs: int = 4
    # Packed work per job.  Short jobs backfill quickly and keep tuning rounds
    # (whose cases take 2-5 minutes) from waiting on a long tail.
    target_node_minutes: float = 12.0
    max_wall_minutes: float = 180.0
    wall_safety_factor: float = 2.0
    wall_margin_minutes: float = 12.0
    module_line: str = (
        "module purge && module load GCC/13.3.0 OpenMPI/5.0.3 OpenFOAM/v2312"
    )
    poll_interval_s: float = 45.0
    ssh_retry_max_s: float = 48 * 3600.0
    ssh_connect_timeout_s: int = 30
    job_timeout_s: int = 5400
    local_staging: str = "runs/cluster_batches"
    # billing efficiency assumed for the budget estimate (actual is settled)
    billing_efficiency: float = 0.7
    # short (search) protocol or the shipped full-length end times
    short_protocol: bool = True
    # extra environment exported to every job (e.g. a Phase-4 library dir)
    extra_env: dict = field(default_factory=dict)
    # ratio of cluster solver time to the laptop-measured CASE_COST_MINUTES;
    # calibrated from the baseline batch (dedicated cores run the cases
    # several times faster than the oversubscribed workstation)
    cost_scale: float = 1.0

    @property
    def slots_per_node(self) -> int:
        """Concurrent cases per SLURM job (one job = one allocation)."""
        return max(1, self.cores_per_job // self.nprocs)

    @property
    def repo(self) -> str:
        return f"{self.remote_root}/repo"

    @property
    def batches(self) -> str:
        return f"{self.remote_root}/batches"

    @property
    def venv(self) -> str:
        return f"{self.remote_root}/venv"

    @property
    def foam_user_dir(self) -> str:
        return f"{self.remote_root}/foam-user"


class RemoteUnavailable(RuntimeError):
    pass


class SlurmBackend:
    name = "slurm"

    def __init__(
        self, config: SlurmConfig, core_hour_ledger: CoreHourLedger,
        log: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cfg = config
        self.ledger = core_hour_ledger
        self.log = log or (lambda message: print(message, flush=True))
        self.sleep = sleep
        self.staging = REPO / self.cfg.local_staging

    # ---- remote plumbing ---------------------------------------------------
    def _ssh_base(self) -> list[str]:
        return [
            "ssh", "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={self.cfg.ssh_connect_timeout_s}",
            "-o", "ServerAliveInterval=30", self.cfg.host,
        ]

    def _retry(self, what: str, action: Callable[[], Any]) -> Any:
        """Retry a remote action with back-off for up to ``ssh_retry_max_s``."""
        started = time.time()
        delay = 30.0
        attempt = 0
        while True:
            attempt += 1
            try:
                return action()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
                elapsed = time.time() - started
                if elapsed > self.cfg.ssh_retry_max_s:
                    raise RemoteUnavailable(
                        f"{what}: cluster unreachable for {elapsed / 3600:.1f} h"
                    ) from exc
                detail = getattr(exc, "stderr", "") or str(exc)
                if isinstance(detail, bytes):
                    detail = detail.decode(errors="replace")
                self.log(
                    f"[cluster] {what} failed (attempt {attempt}): "
                    f"{str(detail).strip()[:200]}; retrying in {delay:.0f}s "
                    "(cluster access may need a refresh)"
                )
                self.sleep(delay)
                delay = min(delay * 2.0, 600.0)

    def ssh(self, command: str, timeout: int = 300) -> str:
        def action() -> str:
            proc = subprocess.run(
                [*self._ssh_base(), command], capture_output=True, text=True,
                timeout=timeout, check=False,
            )
            if proc.returncode == 255:
                raise subprocess.CalledProcessError(
                    proc.returncode, "ssh", proc.stdout, proc.stderr
                )
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, command, proc.stdout, proc.stderr
                )
            return proc.stdout
        return self._retry(f"ssh {command[:40]}", action)

    def rsync(self, source: str, destination: str, timeout: int = 1800) -> None:
        def action() -> None:
            subprocess.run(
                ["rsync", "-a", "--timeout=120", "-e",
                 " ".join(shlex.quote(part) for part in self._ssh_base()[:-1]),
                 source, destination],
                capture_output=True, text=True, timeout=timeout, check=True,
            )
        self._retry(f"rsync {source[-40:]}", action)

    # ---- provenance ---------------------------------------------------------
    def provenance(self) -> dict[str, Any]:
        """Hashes and versions of everything the remote worker executes."""
        script = f"""
set -e
{self.cfg.module_line} >/dev/null 2>&1
source "$FOAM_BASH" >/dev/null 2>&1 || true
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export WM_PROJECT_USER_DIR={self.cfg.foam_user_dir}
LIB={self.cfg.foam_user_dir}/platforms/$WM_OPTIONS/lib/libkOmegaSSTBasis.so
echo "lib_sha256=$(sha256sum "$LIB" | cut -d' ' -f1)"
echo "lib_bytes=$(stat -c %s "$LIB")"
echo "simpleFoam=$(command -v simpleFoam)"
echo "simpleFoam_sha256=$(sha256sum "$(command -v simpleFoam)" | cut -d' ' -f1)"
echo "mpirun=$(command -v mpirun)"
echo "wm_options=$WM_OPTIONS"
echo "wm_project_dir=$WM_PROJECT_DIR"
echo "worker_sha256=$(sha256sum {self.cfg.repo}/scripts/tedp_cluster_worker.py | cut -d' ' -f1)"
echo "tier2_sha256=$(sha256sum {self.cfg.repo}/src/tedp/tier2.py | cut -d' ' -f1)"
echo "scoring_sha256=$(sha256sum {self.cfg.repo}/src/tedp/scoring.py | cut -d' ' -f1)"
echo "python=$({self.cfg.venv}/bin/python --version 2>&1)"
echo "packages=$({self.cfg.venv}/bin/python -m pip freeze 2>/dev/null | grep -iE '^(numpy|scipy|fluidfoam)==' | tr '\\n' ',')"
echo "modules=$(module list 2>&1 | grep -oE '(GCCcore|GCC|OpenMPI|OpenFOAM|Python|SCOTCH|OpenBLAS)/[A-Za-z0-9_.+-]+' | sort -u | tr '\\n' ',')"
echo "hostname=$(hostname)"
"""
        output = self.ssh(f"bash -l -c {shlex.quote(script)}", timeout=240)
        provenance: dict[str, Any] = {
            "backend": self.name, "host": self.cfg.host,
            "remote_root": self.cfg.remote_root, "account": self.cfg.account,
            "partition": self.cfg.partition, "cores_per_node": self.cfg.cores_per_node,
            "nprocs": self.cfg.nprocs, "module_line": self.cfg.module_line,
        }
        for line in output.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                provenance[key.strip()] = value.strip()
        for required in ("lib_sha256", "simpleFoam", "worker_sha256", "wm_options"):
            if not provenance.get(required):
                raise RuntimeError(f"remote provenance is missing {required}: {output[:400]}")
        if len(provenance["lib_sha256"]) != 64:
            raise RuntimeError("remote model library is missing or unreadable")
        return provenance

    # ---- estimates ----------------------------------------------------------
    def _wall_minutes(self, makespan: float) -> float:
        return min(
            self.cfg.max_wall_minutes,
            self.cfg.wall_safety_factor * makespan + self.cfg.wall_margin_minutes,
        )

    def _node_estimate_core_hours(self, makespan: float) -> float:
        billed_minutes = makespan / self.cfg.billing_efficiency + 3.0
        return self.cfg.cores_per_job * billed_minutes / 60.0

    def _cost(self, job: CaseJob) -> float:
        # meta["cost_factor"] scales the short-protocol estimate for full-length
        # or refined-mesh variants (Phase-4 confirmation)
        return job.cost_minutes * self.cfg.cost_scale * float(job.meta.get("cost_factor", 1.0))

    def _plan(self, jobs: list[CaseJob]) -> list[list[CaseJob]]:
        return plan_nodes(
            jobs, self.cfg.slots_per_node, self.cfg.target_node_minutes,
            self.cfg.max_wall_minutes, cost=self._cost,
        )

    def estimate_core_hours(self, jobs: list[CaseJob]) -> float:
        return sum(
            self._node_estimate_core_hours(list_schedule_makespan(
                [self._cost(job) for job in bucket], self.cfg.slots_per_node
            )) for bucket in self._plan(jobs)
        )

    # ---- submission ---------------------------------------------------------
    def _node_script(self, batch_id: str, index: int, wall_minutes: float) -> str:
        remote_batch = f"{self.cfg.batches}/{batch_id}"
        node_dir = f"{remote_batch}/nodes/{index}"
        minutes = int(math.ceil(wall_minutes))
        hours, minutes = divmod(minutes, 60)
        return f"""#!/usr/bin/env bash
#SBATCH --job-name=tedp_{batch_id[-12:]}_{index}
#SBATCH --account={self.cfg.account}
#SBATCH --partition={self.cfg.partition}
#SBATCH --nodes=1
#SBATCH --ntasks={self.cfg.cores_per_job}
#SBATCH --ntasks-per-node={self.cfg.cores_per_job}
#SBATCH --cpus-per-task=1
#SBATCH --time={hours:02d}:{minutes:02d}:00
#SBATCH --output={node_dir}/slurm.log
set -u
{self.cfg.module_line}
set +u
source "$FOAM_BASH" >/dev/null 2>&1 || true
set -u
# the module chain exports a PYTHONPATH of bundled packages; the worker must
# run on the pinned venv packages only
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export WM_PROJECT_USER_DIR={self.cfg.foam_user_dir}
export FOAM_USER_LIBBIN=$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/lib
export FOAM_USER_APPBIN=$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/bin
export LD_LIBRARY_PATH=$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH
export PATH=$FOAM_USER_APPBIN:$PATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TEDP_SHORT={1 if self.cfg.short_protocol else 0}
export TEDP_WARM_START=0
export TEDP_MPIRUN_ARGS="--bind-to none"
{chr(10).join(f'export {k}={v}' for k, v in self.cfg.extra_env.items())}
source {self.cfg.venv}/bin/activate
mkdir -p {node_dir}/results {remote_batch}/runs/{index}
python {self.cfg.repo}/scripts/tedp_cluster_worker.py \\
  --manifest {node_dir}/manifest.json --results-dir {node_dir}/results \\
  --runs-dir {remote_batch}/runs/{index} --slots {self.cfg.slots_per_node} \\
  --nprocs {self.cfg.nprocs} --timeout-s {self.cfg.job_timeout_s}
rm -rf {remote_batch}/runs/{index}
"""

    def submit(self, batch_id: str, jobs: list[CaseJob], lib_sha256: str) -> dict[str, Any]:
        if not jobs:
            raise ValueError("cannot submit an empty batch")
        buckets = self._plan(jobs)
        nodes: list[dict[str, Any]] = []
        estimate = 0.0
        local_batch = self.staging / batch_id
        for index, bucket in enumerate(buckets):
            makespan = list_schedule_makespan(
                [self._cost(job) for job in bucket], self.cfg.slots_per_node
            )
            wall = self._wall_minutes(makespan)
            estimate += self._node_estimate_core_hours(makespan)
            node_dir = local_batch / "nodes" / str(index)
            node_dir.mkdir(parents=True, exist_ok=True)
            (node_dir / "manifest.json").write_text(json.dumps({
                "batch_id": batch_id, "node_index": index, "short": self.cfg.short_protocol,
                "lib_sha256": lib_sha256,
                "jobs": [job.payload() for job in bucket],
            }, sort_keys=True))
            (node_dir / "submit.sh").write_text(self._node_script(batch_id, index, wall))
            nodes.append({
                "index": index, "job_ids": [job.job_id for job in bucket],
                "estimated_makespan_minutes": makespan, "wall_minutes": wall,
                "slurm_job_id": None, "state": "staged",
            })
        if self.ledger.remaining() < estimate:
            raise BudgetExhausted(
                f"batch {batch_id} needs an estimated {estimate:.1f} core-hours; "
                f"only {self.ledger.remaining():.1f} of the approved "
                f"{self.ledger.cap:.0f} remain"
            )
        self.ledger.commit(batch_id, estimate, {
            "backend": self.name, "nodes": len(nodes), "jobs": len(jobs),
        })
        remote_batch = f"{self.cfg.batches}/{batch_id}"
        self.ssh(f"mkdir -p {shlex.quote(remote_batch)}")
        self.rsync(f"{local_batch}/", f"{self.cfg.host}:{remote_batch}/")
        for node in nodes:
            script = f"{remote_batch}/nodes/{node['index']}/submit.sh"
            output = self.ssh(
                f"cd {shlex.quote(remote_batch)} && sbatch {shlex.quote(script)}"
            )
            job_id = None
            for token in output.split():
                if token.isdigit():
                    job_id = token
            if job_id is None:
                raise RuntimeError(f"sbatch returned no job id: {output.strip()[:200]}")
            node["slurm_job_id"] = job_id
            node["state"] = "PENDING"
        handle = {
            "backend": self.name, "batch_id": batch_id, "status": "submitted",
            "jobs": [job.payload() for job in jobs], "nodes": nodes,
            "estimate_core_hours": estimate, "lib_sha256": lib_sha256,
            "submitted_wall_time": time.time(),
        }
        self.log(
            f"[cluster] submitted batch {batch_id}: {len(jobs)} cases on "
            f"{len(nodes)} node(s), est {estimate:.1f} core-h, "
            f"budget remaining {self.ledger.remaining():.0f}"
        )
        return handle

    # ---- polling ------------------------------------------------------------
    def _sacct(self, slurm_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not slurm_ids:
            return {}
        output = self.ssh(
            "sacct -X -n -P --format=JobID,State,CPUTimeRAW,ElapsedRaw "
            f"-j {','.join(slurm_ids)}", timeout=180,
        )
        rows: dict[str, dict[str, Any]] = {}
        for line in output.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 4 or not parts[0]:
                continue
            state = parts[1].split()[0] if parts[1] else "UNKNOWN"
            rows[parts[0]] = {
                "state": state,
                "cpu_time_raw": int(parts[2]) if parts[2].isdigit() else 0,
                "elapsed_raw": int(parts[3]) if parts[3].isdigit() else 0,
            }
        return rows

    def poll(self, handle: dict[str, Any]) -> bool:
        """Refresh node-job states; True when every node job is terminal."""
        ids = [node["slurm_job_id"] for node in handle["nodes"] if node["slurm_job_id"]]
        rows = self._sacct(ids)
        all_terminal = True
        for node in handle["nodes"]:
            row = rows.get(str(node["slurm_job_id"]))
            if row is None:
                # sacct lag right after submission; a job invisible for a long
                # time after submission is treated as lost.
                age = time.time() - float(handle.get("submitted_wall_time", time.time()))
                node["state"] = "LOST" if age > 1800 else "PENDING"
            else:
                node["state"] = row["state"]
                node["cpu_time_raw"] = row["cpu_time_raw"]
                node["elapsed_raw"] = row["elapsed_raw"]
            if node["state"] not in TERMINAL_SLURM_STATES and node["state"] != "LOST":
                all_terminal = False
        return all_terminal

    def collect(self, handle: dict[str, Any]) -> tuple[list[CaseResult], BatchAccounting]:
        batch_id = str(handle["batch_id"])
        remote_batch = f"{self.cfg.batches}/{batch_id}"
        local_batch = self.staging / batch_id
        local_batch.mkdir(parents=True, exist_ok=True)
        self.rsync(
            f"{self.cfg.host}:{remote_batch}/nodes/", f"{local_batch}/nodes/"
        )
        jobs = [CaseJob.from_payload(p) for p in handle["jobs"]]
        outcomes: dict[str, dict[str, Any]] = {}
        for node in handle["nodes"]:
            results_dir = local_batch / "nodes" / str(node["index"]) / "results"
            for job_id in node["job_ids"]:
                path = results_dir / f"{job_id}.json"
                if path.is_file():
                    try:
                        outcomes[job_id] = json.loads(path.read_text())
                    except ValueError:
                        continue
        results = [CaseResult.from_outcome(job, outcomes.get(job.job_id)) for job in jobs]
        actual = sum(
            float(node.get("cpu_time_raw", 0)) for node in handle["nodes"]
        ) / 3600.0
        states = {str(node["slurm_job_id"]): node["state"] for node in handle["nodes"]}
        self.ledger.settle(batch_id, actual, {"states": states})
        handle["status"] = "collected"
        handle["actual_core_hours"] = actual
        complete = sum(r.status == "complete" for r in results)
        self.log(
            f"[cluster] batch {batch_id}: {complete}/{len(results)} cases complete, "
            f"billed {actual:.1f} core-h (est {handle.get('estimate_core_hours', 0):.1f}); "
            f"spent {self.ledger.spent:.0f} of {self.ledger.cap:.0f}"
        )
        return results, BatchAccounting(
            batch_id, float(handle.get("estimate_core_hours", 0.0)), actual,
            [dict(node) for node in handle["nodes"]],
        )

    def wait(self, handle: dict[str, Any]) -> None:
        while not self.poll(handle):
            self.sleep(self.cfg.poll_interval_s)

    # ---- public API ---------------------------------------------------------
    def run_batch(
        self, batch_id: str, jobs: list[CaseJob],
        on_submitted: Callable[[dict[str, Any]], None] | None = None,
        lib_sha256: str = "",
    ) -> tuple[list[CaseResult], BatchAccounting]:
        handle = self.submit(batch_id, jobs, lib_sha256)
        if on_submitted is not None:
            on_submitted(handle)
        return self.resume_batch(handle)

    def resume_batch(
        self, handle: dict[str, Any],
        on_submitted: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[CaseResult], BatchAccounting]:
        del on_submitted
        if handle.get("status") == "collected":
            raise RuntimeError("batch already collected; results live in the checkpoint")
        self.wait(handle)
        return self.collect(handle)


def run_batch_with_retry(
    backend: Any, batch_id: str, jobs: list[CaseJob],
    on_submitted: Callable[[dict[str, Any]], None] | None = None,
    lib_sha256: str = "", max_retries: int = 1,
    existing_handle: dict[str, Any] | None = None,
) -> tuple[list[CaseResult], list[BatchAccounting]]:
    """Run a batch and resubmit infrastructure failures at most ``max_retries``."""
    accounting: list[BatchAccounting] = []
    if existing_handle is not None and existing_handle.get("status") != "collected":
        results, account = backend.resume_batch(existing_handle)
    else:
        kwargs = {"lib_sha256": lib_sha256} if backend.name == "slurm" else {}
        results, account = backend.run_batch(batch_id, jobs, on_submitted, **kwargs)
    accounting.append(account)
    by_id = {result.job_id: result for result in results}
    for retry in range(1, max_retries + 1):
        retry_jobs = [job for job in jobs if by_id[job.job_id].retryable]
        if not retry_jobs:
            break
        kwargs = {"lib_sha256": lib_sha256} if backend.name == "slurm" else {}
        retry_results, account = backend.run_batch(
            f"{batch_id}.retry{retry}", retry_jobs, on_submitted, **kwargs
        )
        accounting.append(account)
        for result in retry_results:
            by_id[result.job_id] = result
    return [by_id[job.job_id] for job in jobs], accounting
