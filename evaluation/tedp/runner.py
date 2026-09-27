"""Launch and monitor OpenFOAM runs.

foam_env() sources the v2312 bashrc once per process; run_case() executes one
solver (or utility) in a case directory with a log file, wall-clock timeout,
and divergence detection; run_batch() packs jobs onto the machine's cores.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

OPENFOAM_BASHRC = "/usr/lib/openfoam/openfoam2312/etc/bashrc"
TOTAL_CORES = 24

_env_cache: dict[str, str] | None = None


def foam_env() -> dict[str, str]:
    """Environment with OpenFOAM v2312 sourced (cached).

    When the calling process already carries a sourced OpenFOAM environment
    (a cluster job that loaded the OpenFOAM module), that environment is
    inherited verbatim instead of sourcing the workstation bashrc path.
    ``TEDP_OPENFOAM_BASHRC`` overrides the bashrc location otherwise.
    """
    global _env_cache
    if _env_cache is None:
        if os.environ.get("WM_PROJECT_DIR") and os.environ.get("FOAM_APPBIN"):
            _env_cache = dict(os.environ)
            return dict(_env_cache)
        bashrc = os.environ.get("TEDP_OPENFOAM_BASHRC", OPENFOAM_BASHRC)
        out = subprocess.run(
            ["bash", "-c", f"source {bashrc} >/dev/null 2>&1 && env -0"],
            check=True,
            capture_output=True,
        ).stdout
        env: dict[str, str] = {}
        for entry in out.split(b"\0"):
            if b"=" in entry:
                k, _, v = entry.partition(b"=")
                env[k.decode()] = v.decode(errors="replace")
        _env_cache = env
    return dict(_env_cache)


@dataclass
class RunResult:
    case: Path
    command: str
    returncode: int
    converged: bool
    diverged: bool
    iterations: int
    final_residuals: dict[str, float]
    wall_s: float
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.diverged


_RES_RE = re.compile(
    r"Solving for (\w+), Initial residual = ([0-9.eE+-]+)"
)
# tabulatedAccelerationSource also prints ``Time = ... accelerations: ...``.
# Only complete numeric records begin a physical solver iteration; accepting
# that diagnostic used to double counts and could reset bounding streaks.
_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TIME_RE = re.compile(rf"^Time = ({_NUMBER_RE})[ \t\r]*$", re.MULTILINE)
_STEP_RE = re.compile(rf"^(?:Time|Iteration) = ({_NUMBER_RE})[ \t\r]*$", re.MULTILINE)

# divergence signatures in solver logs (note: the startup banner prints
# "trapFpe: Floating point exception trapping enabled" — match the actual
# signal handler backtrace instead)
_FATAL_PATTERNS = (
    "Foam::sigFpe",
    "FOAM FATAL",
    "job aborted",
)

_MAX_CONSECUTIVE_BOUNDED_ITERATIONS = 50


def parse_log(
    log_path: Path, *, require_iterations: bool = True
) -> tuple[int, dict[str, float], bool, str]:
    """Return (iterations, last initial-residual per field, diverged, reason).

    Iterative solver logs must contain at least one ``Time``/``Iteration``
    record.  OpenFOAM utilities such as ``blockMesh`` legitimately do not;
    their callers opt out of that one solver-specific check while retaining
    fatal-error detection.
    """
    if not log_path.exists():
        return 0, {}, True, "no log produced"
    text = log_path.read_text(errors="replace")

    iterations = len(_STEP_RE.findall(text))
    residuals: dict[str, float] = {}
    min_seen: dict[str, float] = {}
    # Preserve each distinct failure category. In particular, a later FPE
    # must remain visible even when an earlier bounding guard already fired.
    # Keep the first residual blow-up per field to avoid an unbounded message.
    failures: dict[str, str] = {}
    if iterations == 0 and require_iterations:
        failures["iterations"] = "no solver iterations"
    bounding_streak = 0
    in_iteration = False
    iteration_was_bounded = False

    def finish_iteration() -> None:
        """Account for bounding once per solver iteration.

        OpenFOAM interleaves bounding messages with residual and continuity
        lines, so consecutive *log-line* matching cannot identify sustained
        bounding.  A clean iteration, rather than an unrelated line within a
        bounded iteration, is what resets the streak.
        """
        nonlocal bounding_streak
        if not in_iteration:
            return
        if iteration_was_bounded:
            bounding_streak += 1
            if bounding_streak > _MAX_CONSECUTIVE_BOUNDED_ITERATIONS:
                failures["bounding"] = "sustained k/omega bounding"
        else:
            bounding_streak = 0

    for line in text.splitlines():
        if _STEP_RE.match(line):
            finish_iteration()
            in_iteration = True
            iteration_was_bounded = False

        m = _RES_RE.search(line)
        if m:
            fieldname, res = m.group(1), float(m.group(2))
            residuals[fieldname] = res
            prev = min_seen.get(fieldname, res)
            min_seen[fieldname] = min(prev, res)
            if res > 1.0e2 * max(min_seen[fieldname], 1.0e-300) and res > 1.0:
                failures.setdefault(
                    f"residual:{fieldname}",
                    f"residual blow-up in {fieldname} ({res:.3g})",
                )
        stripped = line.lstrip()
        if in_iteration and (
            stripped.startswith("bounding k,")
            or stripped.startswith("bounding omega,")
        ):
            iteration_was_bounded = True
        if "nan" in line.lower() and "Solving for" in line:
            failures["nan"] = "NaN residual"

    finish_iteration()

    for pat in _FATAL_PATTERNS:
        if pat in text:
            failures[f"fatal:{pat}"] = pat

    return iterations, residuals, bool(failures), "; ".join(failures.values())


def last_time(log_path: Path) -> float | None:
    """Last solver ``Time =`` value, or ``None`` for a missing/empty log."""
    if not log_path.exists():
        return None
    matches = _TIME_RE.findall(log_path.read_text(errors="replace"))
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except ValueError:
        return None
    return value if value == value and abs(value) != float("inf") else None


def run_case(
    case: Path,
    solver: str = "simpleFoam",
    nprocs: int = 1,
    timeout_s: int = 1800,
    args: tuple[str, ...] = (),
) -> RunResult:
    """Run one solver/utility in `case`, logging to log.<solver>."""
    case = Path(case)
    log_path = case / f"log.{solver}"
    if nprocs > 1:
        cmd = ["mpirun", "--use-hwthread-cpus", "-np", str(nprocs),
               solver, "-parallel", *args]
    else:
        cmd = [solver, *args]

    t0 = time.time()
    with open(log_path, "w") as log:
        try:
            proc = subprocess.run(
                cmd,
                cwd=case,
                env=foam_env(),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
            returncode = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = -9
            timed_out = True

    wall = time.time() - t0
    noniterative_utilities = {
        "blockMesh",
        "checkMesh",
        "decomposePar",
        "reconstructPar",
        "renumberMesh",
    }
    require_iterations = (
        solver not in noniterative_utilities and "-postProcess" not in args
    )
    iterations, residuals, diverged, reason = parse_log(
        log_path, require_iterations=require_iterations
    )
    text = log_path.read_text(errors="replace")
    converged = "SIMPLE solution converged" in text
    if timed_out:
        reason = reason or f"timeout after {timeout_s}s"
    if returncode != 0 and not reason:
        reason = f"exit code {returncode}"

    return RunResult(
        case=case,
        command=" ".join(cmd),
        returncode=returncode,
        converged=converged,
        diverged=diverged or timed_out,
        iterations=iterations,
        final_residuals=residuals,
        wall_s=wall,
        reason=reason,
    )


@dataclass
class Job:
    case: Path
    solver: str = "simpleFoam"
    nprocs: int = 4
    timeout_s: int = 1800
    args: tuple[str, ...] = ()


def run_batch(jobs: list[Job], total_cores: int = TOTAL_CORES) -> list[RunResult]:
    """Run jobs concurrently, never oversubscribing total_cores."""
    results: list[RunResult | None] = [None] * len(jobs)
    max_workers = max(1, total_cores // max(j.nprocs for j in jobs)) if jobs else 1
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                run_case, j.case, j.solver, j.nprocs, j.timeout_s, j.args
            ): i
            for i, j in enumerate(jobs)
        }
        for future, i in futures.items():
            results[i] = future.result()
    return [r for r in results if r is not None]
