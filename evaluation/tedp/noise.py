"""Phase 0 noise floor: what does 'improvement' have to exceed?

Per the plan, measured on periodic hills case_1p0 and the curved step:
  1. mesh refinement x2 (refineMesh in the 2-D plane),
  2. solver tolerance 1e-6 vs 1e-8,
  3. perturbed initial conditions (k, omega scaled by 1.05),
  4. decomposition 16 vs serial-equivalent 8.

Each variant is scored with the E_U metric against REF on the SAME sample
points (cell centres of the baseline mesh via nearest-point lookup for the
refined variant); the noise floor per case is the max |delta E_U| across
variants. Every later gate requires a 2x margin over this number.
Output: results/noise_floor.json
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
from fluidfoam import readmesh, readvector

from . import repro
from .data import mcconkey

REPO = Path(__file__).resolve().parents[2]
CASES = ("case_1p0", "cbfs13700")


def _e_u(case_dir: Path, case_id: str) -> float:
    """E_U vs REF at the reference cell centres (nearest-neighbour lookup
    so refined meshes score on the same points)."""
    ref = mcconkey.load_case_table(case_id, "REF")
    sst = mcconkey.load_case_table(case_id, "komegasst")
    xr, yr = sst["komegasst_C_1"], sst["komegasst_C_2"]
    u_ref = np.stack([ref["REF_U_1"], ref["REF_U_2"], ref["REF_U_3"]], axis=1)

    times = sorted((int(p.name) for p in case_dir.iterdir() if p.name.isdigit()),
                   reverse=True)
    t = str(times[0])
    x, y, _ = readmesh(str(case_dir), verbose=False)
    u = np.asarray(readvector(str(case_dir), t, "U", verbose=False)).T  # (n,3)

    from scipy.spatial import cKDTree

    tree = cKDTree(np.stack([x, y], axis=1))
    _, idx = tree.query(np.stack([xr, yr], axis=1))
    du = u[idx] - u_ref
    scale = float(np.sqrt(np.mean(np.sum(u_ref**2, axis=1))))
    return float(np.sqrt(np.mean(np.sum(du**2, axis=1)))) / scale


def _variant(case_id: str, name: str, mutate) -> float:
    work = REPO / "runs" / "noise" / f"{case_id}_{name}"
    repro.clone_case(case_id, work)
    mutate(work)
    import subprocess  # noqa: E402
    from . import runner  # noqa: E402

    env = runner.foam_env()

    def step(cmd, timeout=10800):
        with open(work / f"log.{cmd[0]}", "a") as log:
            return subprocess.run(cmd, cwd=work, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout)

    assert step(["decomposePar", "-force"]).returncode == 0
    nprocs = repro.NPROCS
    text = (work / "system" / "decomposeParDict").read_text()
    m = re.search(r"numberOfSubdomains\s+(\d+);", text)
    if m:
        nprocs = int(m.group(1))
    assert step(["mpirun", "--use-hwthread-cpus", "-np", str(nprocs),
                 "simpleFoam", "-parallel"]).returncode == 0
    step(["reconstructPar", "-latestTime"])
    for proc in work.glob("processor*"):
        shutil.rmtree(proc)
    return _e_u(work, case_id)


def mutate_none(work: Path) -> None:
    pass


def mutate_refine(work: Path) -> None:
    """x2 refinement restricted to the x-y plane (the z direction is one
    cell thick with empty patches and must not be split)."""
    import subprocess
    from . import runner

    (work / "system" / "topoSetDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }\n"
        "actions\n(\n    {\n        name allCells; type cellSet; action new;\n"
        "        source boxToCell;\n"
        "        box (-1e6 -1e6 -1e6) (1e6 1e6 1e6);\n    }\n);\n"
    )
    (work / "system" / "refineMeshDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object refineMeshDict; }\n"
        "set             allCells;\n"
        "coordinateSystem global;\n"
        "globalCoeffs { tan1 (1 0 0); tan2 (0 1 0); }\n"
        "directions ( tan1 tan2 );\n"
        "useHexTopology  yes;\n"
        "geometricCut    no;\n"
        "writeMesh       no;\n"
    )
    env = runner.foam_env()
    # cid_baseline sibling (runs first) supplies consistently-sized fields
    cid = work.name.rsplit("_", 1)[0]
    baseline = work.parent / f"{cid}_baseline"

    def run_step(cmd):
        with open(work / f"log.{cmd[0]}", "w") as log:
            assert subprocess.run(
                cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT,
            ).returncode == 0, f"{cmd[0]} failed in {work.name}"

    run_step(["topoSet"])
    run_step(["refineMesh", "-dict", "system/refineMeshDict", "-overwrite"])
    # stale fixed-size field lists cannot be read on the refined mesh; wipe
    # and let mapFields recreate the whole 0/ from the baseline clone
    for f in (work / "0").iterdir():
        if f.is_file():
            f.unlink()
    run_step(["mapFields", str(baseline), "-consistent", "-sourceTime", "0"])


def mutate_tight_tol(work: Path) -> None:
    fv = work / "system" / "fvSolution"
    fv.write_text(re.sub(r"tolerance\s+1e-0?6", "tolerance 1e-08", fv.read_text()))


def mutate_perturbed_ic(work: Path) -> None:
    for f in ("k", "omega"):
        p = work / "0" / f
        t = p.read_text()
        m = re.search(r"internalField\s+uniform\s+([0-9.eE+-]+);", t)
        if m:
            v = float(m.group(1)) * 1.05
            t = t.replace(m.group(0), f"internalField   uniform {v:g};")
            p.write_text(t)


def mutate_decomp8(work: Path) -> None:
    dec = work / "system" / "decomposeParDict"
    dec.write_text(re.sub(r"numberOfSubdomains\s+\d+;",
                          "numberOfSubdomains 8;", dec.read_text()))


VARIANTS = {
    "baseline": mutate_none,
    "refined": mutate_refine,
    "tight_tol": mutate_tight_tol,
    "perturbed_ic": mutate_perturbed_ic,
    "decomp8": mutate_decomp8,
}


def run_noise_floor(cases=CASES, out: Path | None = None) -> dict:
    out = out or REPO / "results" / "noise_floor.json"
    results: dict[str, dict] = {}
    for cid in cases:
        results[cid] = {}
        for name, mutate in VARIANTS.items():
            try:
                e_u = _variant(cid, name, mutate)
            except AssertionError as e:
                e_u = float("nan")
                print(f"{cid}/{name}: FAILED ({e})", flush=True)
            results[cid][name] = e_u
            print(f"{cid:12s} {name:14s} E_U = {e_u:.5f}", flush=True)
        base = results[cid]["baseline"]
        deltas = [abs(v - base) for k, v in results[cid].items()
                  if k != "baseline" and np.isfinite(v)]
        results[cid]["noise_floor_E_U"] = max(deltas) if deltas else float("nan")
        print(f"{cid:12s} noise floor dE_U = {results[cid]['noise_floor_E_U']:.5f}",
              flush=True)
    out.write_text(json.dumps(results, indent=2))
    return results
