"""Phase 0 reproduction: rerun every scored McConkey komegasst case with the
shipped settings and compare the converged fields with the shipped solution.

Exit gate (artifact): our-SST vs dataset-SST within 1% (diagnostic; same
solver family) and SST-vs-DNS composite within 5% of the dataset's own
levels. Output: results/reproduction.csv — the table the human signs at
Checkpoint 1.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fluidfoam import readscalar, readvector

from . import runner
from .data import mcconkey

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
NPROCS = 16


@dataclass
class ReproRow:
    case_id: str
    converged: bool
    iterations: int
    wall_s: float
    rel_u: float      # max-norm relative diff vs shipped fields
    rel_k: float
    rel_omega: float
    rel_nut: float
    reason: str = ""


def clone_case(case_id: str, workdir: Path) -> Path:
    src = mcconkey.case_dir(case_id)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    for sub in ("0", "constant", "system"):
        shutil.copytree(src / sub, workdir / sub, symlinks=True)
    # uniform scotch decomposition for this machine (some shipped dicts use
    # geometric coeffs tied to their original rank count)
    dec = workdir / "system" / "decomposeParDict"
    if dec.exists():
        dec.write_text(
            "FoamFile { version 2.0; format ascii; class dictionary; "
            "object decomposeParDict; }\n"
            f"numberOfSubdomains {NPROCS};\n"
            "method          scotch;\n"
        )
    return workdir


def _read_fields(case: Path | str, t: str):
    case = str(case)
    return {
        "U": readvector(case, t, "U", verbose=False),
        "k": readscalar(case, t, "k", verbose=False),
        "omega": readscalar(case, t, "omega", verbose=False),
        "nut": readscalar(case, t, "nut", verbose=False),
    }


def _rel_max(a: np.ndarray, b: np.ndarray) -> float:
    scale = float(np.max(np.abs(b)))
    if scale == 0.0:
        return float(np.max(np.abs(a - b)))
    return float(np.max(np.abs(a - b))) / scale


def run_repro_case(case_id: str, workdir: Path) -> ReproRow:
    work = clone_case(case_id, workdir)
    t_final = mcconkey.final_time(case_id)

    t0 = time.time()
    env = runner.foam_env()

    def step(cmd: list[str], timeout: int = 7200) -> subprocess.CompletedProcess:
        with open(work / f"log.{cmd[0]}", "a") as log:
            return subprocess.run(
                cmd, cwd=work, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=timeout,
            )

    if (work / "system" / "decomposeParDict").exists():
        if step(["decomposePar", "-force"]).returncode != 0:
            return ReproRow(case_id, False, 0, time.time() - t0,
                            *[float("nan")] * 4, "decomposePar failed")
        r = step(["mpirun", "--use-hwthread-cpus", "-np", str(NPROCS),
                  "simpleFoam", "-parallel"])
        step(["reconstructPar", "-latestTime"])
        for proc in work.glob("processor*"):
            shutil.rmtree(proc)
        rc = r.returncode
    else:
        rc = step(["simpleFoam"]).returncode

    wall = time.time() - t0
    iterations, residuals, diverged, reason = runner.parse_log(
        work / "log.mpirun" if (work / "log.mpirun").exists() else work / "log.simpleFoam"
    )

    times = sorted((int(p.name) for p in work.iterdir() if p.name.isdigit()), reverse=True)
    if rc != 0 or not times or times[0] == 0:
        return ReproRow(case_id, False, iterations, wall,
                        *[float("nan")] * 4, reason or f"exit {rc}")

    ours = _read_fields(work, str(times[0]))
    theirs = _read_fields(mcconkey.case_dir(case_id), t_final)
    rel = {f: _rel_max(ours[f], theirs[f]) for f in ours}
    return ReproRow(
        case_id, not diverged, iterations, wall,
        rel["U"], rel["k"], rel["omega"], rel["nut"], reason,
    )


def run_all(case_ids: list[str] | None = None, out: Path | None = None) -> list[ReproRow]:
    case_ids = case_ids or list(mcconkey.ALL_CASES)
    out = out or RESULTS / "reproduction.csv"
    rows: list[ReproRow] = []
    for cid in case_ids:
        row = run_repro_case(cid, REPO / "runs" / "repro" / cid)
        rows.append(row)
        print(
            f"{cid:16s} conv={row.converged} iters={row.iterations} "
            f"wall={row.wall_s:6.0f}s  relU={row.rel_u:.2e} relk={row.rel_k:.2e} "
            f"relw={row.rel_omega:.2e} {row.reason}",
            flush=True,
        )
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case", "converged", "iterations", "wall_s",
                        "rel_u", "rel_k", "rel_omega", "rel_nut", "reason"])
            for r in rows:
                w.writerow([r.case_id, r.converged, r.iterations, f"{r.wall_s:.0f}",
                            f"{r.rel_u:.3e}", f"{r.rel_k:.3e}",
                            f"{r.rel_omega:.3e}", f"{r.rel_nut:.3e}", r.reason])
    return rows
