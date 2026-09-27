"""Tier 2: a-posteriori simpleFoam evaluation on the McConkey families.

A candidate case = the shipped komegasst case with turbulenceProperties
swapped to kOmegaSSTBasis (libs line added) and run with the shipped solver
settings. Scoring is index-aligned with REF.csv (verified machine-exact) and
wall shear comes from wall-adjacent cells via the same discrete formula for
model and reference.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fluidfoam import readscalar, readvector

from . import foammesh, repro, runner, scoring
from .casegen import render_turbulence_properties
from .data import mcconkey
from .spec import CandidateSpec

REPO = Path(__file__).resolve().parents[2]

CALIBRATION_CASES = tuple(mcconkey.CALIBRATION)
VALIDATION_CASES = tuple(mcconkey.VALIDATION)


def _viscosity(case_dir: Path) -> float:
    import re

    text = (case_dir / "constant" / "transportProperties").read_text()
    m = re.search(r"nu\s+(?:\[[^\]]*\]\s*)?([0-9.eE+-]+)\s*;", text)
    if not m:
        raise ValueError(f"no nu in {case_dir}")
    return float(m.group(1))


def _wall_shear(case: Path | str, u: np.ndarray, nu: float) -> np.ndarray:
    """Cf-profile samples: nu * |U_t(owner)| / d over every wall patch,
    concatenated in a deterministic patch order. u is (nCells, 3)."""
    case = Path(case)
    out = []
    for patch in sorted(foammesh.wall_patches(case)):
        wg = foammesh.wall_geometry(case, patch)
        n = wg.face_areas / np.linalg.norm(wg.face_areas, axis=1, keepdims=True)
        from fluidfoam import readmesh

        x, y, z = readmesh(str(case), verbose=False)
        cc = np.stack([x, y, z], 1)[wg.owner_cells]
        d = np.abs(np.einsum("ij,ij->i", cc - wg.face_centres, n))
        uc = u[wg.owner_cells]
        u_t = uc - np.einsum("ij,ij->i", uc, n)[:, None] * n
        out.append(nu * np.linalg.norm(u_t, axis=1) / np.maximum(d, 1e-300))
    return np.concatenate(out)


@dataclass
class CaseData:
    """Static per-case reference bundle (built once, reused per candidate)."""

    case_id: str
    ref: scoring.CaseRef
    nu: float
    template: Path


_case_cache: dict[str, CaseData] = {}


def case_data(case_id: str) -> CaseData:
    if case_id in _case_cache:
        return _case_cache[case_id]
    template = mcconkey.case_dir(case_id)
    ref_tbl = mcconkey.load_case_table(case_id, "REF")
    u_ref = np.stack(
        [ref_tbl["REF_U_1"], ref_tbl["REF_U_2"], ref_tbl["REF_U_3"]], axis=1
    )
    nu = _viscosity(template)
    u_bulk = float(np.sqrt(np.mean(np.sum(u_ref**2, axis=1))))
    cf_ref = _wall_shear(template, u_ref, nu) / (0.5 * u_bulk**2)
    ref = scoring.CaseRef(
        name=case_id,
        u_bulk=u_bulk,
        u_ref=u_ref,
        cf_ref=cf_ref,
        uv_ref=ref_tbl["REF_tau_12"],
    )
    data = CaseData(case_id, ref, nu, template)
    _case_cache[case_id] = data
    return data


def score_run(case_run: Path, case_id: str) -> scoring.CaseScore:
    """Score a finished run directory against REF."""
    data = case_data(case_id)
    times = sorted(
        (int(p.name) for p in case_run.iterdir() if p.name.isdigit()), reverse=True
    )
    t = str(times[0])
    u = np.asarray(readvector(str(case_run), t, "U", verbose=False)).T
    k = readscalar(str(case_run), t, "k", verbose=False)
    nut = readscalar(str(case_run), t, "nut", verbose=False)

    # modelled u'v' = -nut (dU/dy + dV/dx) + nonlinearStress_xy, using the
    # RUN's own velocity gradient (written by `simpleFoam -postProcess
    # -func 'grad(U)'` in evaluate_candidate) and the model's written
    # nonlinear stress (zero/absent for stock SST)
    from fluidfoam import readtensor

    if not (case_run / t / "grad(U)").exists():
        import subprocess

        with open(case_run / "log.postProcess", "a") as log:
            subprocess.run(
                ["simpleFoam", "-postProcess", "-func", "grad(U)", "-latestTime"],
                cwd=case_run, env=runner.foam_env(), stdout=log,
                stderr=subprocess.STDOUT, timeout=600,
            )
    grad_u = np.asarray(readtensor(str(case_run), t, "grad(U)", verbose=False))
    uv = -nut * (grad_u[1] + grad_u[3])  # tensor components xy + yx
    if (case_run / t / "nonlinearStress").exists():
        from fluidfoam import readsymmtensor

        nl = np.asarray(
            readsymmtensor(str(case_run), t, "nonlinearStress", verbose=False)
        )
        uv = uv + nl[1]  # symmTensor xy component

    cf = _wall_shear(case_run, u, data.nu) / (0.5 * data.ref.u_bulk**2)
    sample = scoring.CaseSample(name=case_id, u=u, cf=cf, uv=uv)
    return scoring.case_score(sample, data.ref)


def score_run_resampled(case_run: Path, case_id: str) -> scoring.CaseScore:
    """Score a run whose mesh differs from the reference mesh (e.g. refined):
    fields are sampled at the reference cell centres and wall-face centres
    by nearest neighbour, so every mesh scores on the same points."""
    from scipy.spatial import cKDTree
    from fluidfoam import readmesh, readtensor, readsymmtensor
    import subprocess

    data = case_data(case_id)
    times = sorted((int(p.name) for p in case_run.iterdir() if p.name.isdigit()), reverse=True)
    t = str(times[0])
    if not (case_run / t / "grad(U)").exists():
        with open(case_run / "log.postProcess", "a") as log:
            subprocess.run(
                ["simpleFoam", "-postProcess", "-func", "grad(U)", "-latestTime"],
                cwd=case_run, env=runner.foam_env(), stdout=log,
                stderr=subprocess.STDOUT, timeout=600,
            )
    x, y, _ = readmesh(str(case_run), verbose=False)
    u = np.asarray(readvector(str(case_run), t, "U", verbose=False)).T
    nut = readscalar(str(case_run), t, "nut", verbose=False)
    grad_u = np.asarray(readtensor(str(case_run), t, "grad(U)", verbose=False))
    uv = -nut * (grad_u[1] + grad_u[3])
    if (case_run / t / "nonlinearStress").exists():
        uv = uv + np.asarray(readsymmtensor(str(case_run), t, "nonlinearStress", verbose=False))[1]
    sst = mcconkey.load_case_table(case_id, "komegasst")
    xr, yr = sst["komegasst_C_1"], sst["komegasst_C_2"]
    _, idx = cKDTree(np.stack([x, y], axis=1)).query(np.stack([xr, yr], axis=1))
    # wall shear: refined wall faces sampled at the template's wall-face centres
    cf_run = _wall_shear(case_run, u, data.nu) / (0.5 * data.ref.u_bulk ** 2)
    centres_run = np.concatenate([
        foammesh.wall_geometry(case_run, patch).face_centres[:, :2]
        for patch in sorted(foammesh.wall_patches(case_run))
    ])
    centres_ref = np.concatenate([
        foammesh.wall_geometry(data.template, patch).face_centres[:, :2]
        for patch in sorted(foammesh.wall_patches(data.template))
    ])
    _, widx = cKDTree(centres_run).query(centres_ref)
    sample = scoring.CaseSample(name=case_id, u=u[idx], cf=cf_run[widx], uv=uv[idx])
    return scoring.case_score(sample, data.ref)


def refine_case_x2(work: Path, env: dict, directions: tuple[str, str] = ("(1 0 0)", "(0 1 0)")) -> None:
    """x2 refinement of a prepared case in two directions (the noise-study
    mutation), with the initial fields re-created by mapFields from an
    unrefined clone. The default refines the x-y plane, which is the plane of
    the two-dimensional development cases; the duct is refined in its
    cross-section (y-z) instead, since that is where the secondary flow and
    the corner resolution live."""
    import subprocess

    baseline = work.parent / (work.name + "_unrefined")
    if baseline.exists():
        shutil.rmtree(baseline)
    shutil.copytree(work, baseline, symlinks=True)
    (work / "system" / "topoSetDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object topoSetDict; }\n"
        "actions\n(\n    {\n        name allCells; type cellSet; action new;\n"
        "        source boxToCell;\n        box (-1e6 -1e6 -1e6) (1e6 1e6 1e6);\n    }\n);\n"
    )
    (work / "system" / "refineMeshDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object refineMeshDict; }\n"
        "set             allCells;\ncoordinateSystem global;\n"
        f"globalCoeffs {{ tan1 {directions[0]}; tan2 {directions[1]}; }}\ndirections ( tan1 tan2 );\n"
        "useHexTopology  yes;\ngeometricCut    no;\nwriteMesh       no;\n"
    )
    for cmd in (["topoSet"], ["refineMesh", "-dict", "system/refineMeshDict", "-overwrite"]):
        with open(work / f"log.{cmd[0]}", "w") as log:
            if subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT).returncode != 0:
                raise RuntimeError(f"{cmd[0]} failed while refining {work}")
    for f in (work / "0").iterdir():
        if f.is_file():
            f.unlink()
    with open(work / "log.mapFields", "w") as log:
        if subprocess.run(["mapFields", str(baseline), "-consistent", "-sourceTime", "0"],
                          cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT).returncode != 0:
            raise RuntimeError(f"mapFields failed while refining {work}")
    shutil.rmtree(baseline, ignore_errors=True)


def _freestream_velocity(template: Path, u: np.ndarray) -> float:
    """|U| of the uniform internalField in the template's 0/U (the inlet
    value of the gate cases); the maximum |U| of the solution otherwise."""
    m = re.search(r"internalField\s+uniform\s*\(\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\)",
                  (template / "0" / "U").read_text(errors="replace"))
    if m:
        return float(np.linalg.norm([float(m.group(i)) for i in (1, 2, 3)]))
    return float(np.linalg.norm(u, axis=1).max())


def wall_cf_profile(case_run: Path, case_id: str, bins: int = 200) -> dict:
    """Skin-friction coefficient along the plate, averaged in x bins.

    Gate cases have no field reference on the mesh (the flat plate's REF is
    a boundary-layer profile), so only the viscosity and the freestream
    velocity of the template are used.  The profile is taken on the
    ``bottomWall`` patch (the plate; the top boundary is a slip wall) when
    it exists, otherwise on every wall patch.
    """
    template = mcconkey.case_dir(case_id)
    nu = _viscosity(template)
    times = sorted((int(p.name) for p in case_run.iterdir() if p.name.isdigit()), reverse=True)
    t = str(times[0])
    u = np.asarray(readvector(str(case_run), t, "U", verbose=False)).T
    u_ref = _freestream_velocity(template, u)
    walls = sorted(foammesh.wall_patches(case_run))
    patches = ["bottomWall"] if "bottomWall" in walls else walls
    cf_parts, centre_parts = [], []
    for patch in patches:
        wg = foammesh.wall_geometry(case_run, patch)
        n = wg.face_areas / np.linalg.norm(wg.face_areas, axis=1, keepdims=True)
        from fluidfoam import readmesh

        x, y, z = readmesh(str(case_run), verbose=False)
        cc = np.stack([x, y, z], 1)[wg.owner_cells]
        d = np.abs(np.einsum("ij,ij->i", cc - wg.face_centres, n))
        uc = u[wg.owner_cells]
        u_t = uc - np.einsum("ij,ij->i", uc, n)[:, None] * n
        cf_parts.append(nu * np.linalg.norm(u_t, axis=1) / np.maximum(d, 1e-300) / (0.5 * u_ref ** 2))
        centre_parts.append(wg.face_centres[:, 0])
    cf = np.concatenate(cf_parts)
    centres = np.concatenate(centre_parts)
    edges = np.linspace(centres.min(), centres.max(), bins + 1)
    which = np.clip(np.digitize(centres, edges) - 1, 0, bins - 1)
    sums = np.bincount(which, weights=cf, minlength=bins)
    counts = np.bincount(which, minlength=bins)
    mid = 0.5 * (edges[:-1] + edges[1:])
    ok = counts > 0
    return {"x": mid[ok].tolist(), "cf": (sums[ok] / counts[ok]).tolist(), "u_ref": u_ref,
            "patches": patches, "time": t}


RESIDUAL_CONTROL = (
    "    residualControl\n    {\n"
    "        p               5e-7;\n"
    "        U               1e-8;\n"
    '        "(k|omega)"     1e-8;\n'
    "    }\n"
)


def warm_start_from_repro(case_id: str, work: Path) -> bool:
    """Initialize 0/ from the stock converged solution and add an early-stop
    residualControl. Thresholds sit at/below the WORST natural final level of
    the cold reproduction runs (convdiv20580: p ~1.1e-6), so an early stop is
    always at least as converged as the reproduction the scores were built
    on; the noise study measured tolerance-level effects on E_U at <= 1e-7.
    Cold-start runs remain over-converged to machine level regardless."""
    src = repro.REPO / "runs" / "repro" / case_id
    times = sorted(
        (int(p.name) for p in src.iterdir() if p.name.isdigit()), reverse=True
    ) if src.exists() else []
    if not times:
        return False
    t = str(times[0])
    for f in ("U", "p", "k", "omega", "nut", "phi"):
        if (src / t / f).exists():
            shutil.copy(src / t / f, work / "0" / f)
    fvsol = work / "system" / "fvSolution"
    text = fvsol.read_text()
    if "residualControl" not in text:
        import re as _re

        text = _re.sub(r"(SIMPLE\s*\{)", r"\1\n" + RESIDUAL_CONTROL, text, count=1)
        fvsol.write_text(text)
    return True


# Phase-3 short protocol (validated 2026-08-31): cbfs/bumps match the full
# counts to 1e-5 composite; the hills delta (~1.3e-3) is within the candidate
# limit cycle's own phase noise (warm/cold full-length runs differ by 2.2e-3).
# Applied uniformly to every candidate AND the baselines used in selection;
# finalists get full-length confirmation runs before Phase 4.
SHORT_ITERS = {
    "case_0p5": 8000, "case_0p8": 8000, "case_1p0": 8000,
    "case_1p2": 8000, "case_1p5": 8000,
    "cbfs13700": 8000,
    "h20": 4000, "h26": 4000, "h31": 4000, "h38": 4000, "h42": 4000,
}


def make_candidate_case(
    case_id: str,
    spec: CandidateSpec | None,
    workdir: Path,
    mode: str = "expressions",
    warm_start: bool = False,
    short: bool = False,
) -> Path:
    import os as _os
    import re as _re

    work = repro.clone_case(case_id, workdir)
    if warm_start:
        warm_start_from_repro(case_id, work)
    if (short or _os.environ.get("TEDP_SHORT") == "1") and case_id in SHORT_ITERS:
        n = SHORT_ITERS[case_id]
        cd = work / "system" / "controlDict"
        text = cd.read_text()
        text = _re.sub(r"endTime\s+[\d.]+;", f"endTime         {n};", text)
        text = _re.sub(r"writeInterval\s+[\d.]+;", f"writeInterval   {n};", text)
        cd.write_text(text)
    (work / "constant" / "turbulenceProperties").write_text(
        render_turbulence_properties(spec, mode=mode)
    )
    control = work / "system" / "controlDict"
    text = control.read_text()
    if "libkOmegaSSTBasis" not in text and spec is not None:
        # append as its own entry — anchored replacement broke on cases whose
        # application line is formatted differently (cbfs)
        text += '\nlibs            ("libkOmegaSSTBasis.so");\n'
        control.write_text(text)
    return work


MPIRUN_DEFAULT_ARGS = ("--use-hwthread-cpus",)


def _mpirun_args() -> tuple[str, ...]:
    """Extra mpirun flags; ``TEDP_MPIRUN_ARGS`` overrides the workstation default."""
    import os as _os
    import shlex

    raw = _os.environ.get("TEDP_MPIRUN_ARGS")
    if raw is None:
        return MPIRUN_DEFAULT_ARGS
    return tuple(shlex.split(raw))


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace").splitlines()
    return "\n".join(text[-lines:])


def _prune_run_dir(work: Path, keep: bool) -> None:
    """Delete solver output that has been scored; keep only logs on request."""
    import os as _os

    for proc in work.glob("processor*"):
        shutil.rmtree(proc, ignore_errors=True)
    if keep:
        return
    for child in list(work.iterdir()):
        if child.name.startswith("log."):
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                _os.remove(child)
            except OSError:
                pass


_CLAMP_RE = re.compile(
    r"kOmegaSSTBasis clamps: realizability (\S+) gMax\(bDelta\) (\S+) gMax\(R\) (\S+) rBound (\S+)"
)


def clamp_fractions(log_text: str) -> dict[str, float] | None:
    """Last in-solver clamp report of a run: fraction of cells clipped for
    realizability, fraction of g evaluations clamped at gMax per channel, and
    fraction of cells where |R| hit the rMaxFactor bound.  None for stock or
    a library without the report."""
    last = None
    for last in _CLAMP_RE.finditer(log_text):
        pass
    if last is None:
        return None
    return {
        "realizability": float(last.group(1)), "gmax_bdelta": float(last.group(2)),
        "gmax_r": float(last.group(3)), "r_bound": float(last.group(4)),
    }


def run_case_job(
    spec: CandidateSpec | None,
    case_id: str,
    work_root: Path,
    nprocs: int = 4,
    timeout_s: int = 3600,
    warm_start: bool = False,
    keep_run_dir: bool = True,
    mutation: str | None = None,
    warm_start_time: str | None = None,
    end_time: int | None = None,
) -> dict:
    """Run and score ONE case for one candidate; the unit of cluster work.

    Returns a JSON-serialisable outcome dict with ``status`` ``complete`` or
    ``failed``, the score, a failure class, solver telemetry and the tail of
    the solver log.  Every failure mode of the solver chain is classified so
    the search can distinguish physics divergence from infrastructure loss.
    """
    import os as _os
    import socket
    import subprocess
    import time

    started = time.time()
    outcome: dict = {
        "case_id": case_id, "status": "failed", "score": None,
        "failure_class": None, "reason": "", "iterations": 0,
        "final_time": None, "final_residuals": {}, "wall_s": 0.0,
        "nprocs": int(nprocs), "host": socket.gethostname(),
        "short_iters": SHORT_ITERS.get(case_id)
        if _os.environ.get("TEDP_SHORT") == "1" else None,
        "log_tail": "", "run_dir": "",
    }
    env = runner.foam_env()
    try:
        work = make_candidate_case(
            case_id, spec, Path(work_root) / case_id, warm_start=warm_start
        )
    except Exception as exc:  # template or filesystem problems
        outcome.update(failure_class="setup", reason=f"{type(exc).__name__}: {exc}"[:300])
        outcome["wall_s"] = time.time() - started
        return outcome
    outcome["run_dir"] = str(work)
    outcome["mutation"] = mutation
    if warm_start_time is not None:
        # initialise from a shipped converged time directory of the template
        src = mcconkey.case_dir(case_id) / str(warm_start_time)
        if not src.is_dir():
            outcome.update(failure_class="setup", reason=f"warm start time {warm_start_time} missing")
            outcome["wall_s"] = time.time() - started
            return outcome
        for f in src.iterdir():
            if f.is_file() and f.name in ("U", "p", "k", "omega", "nut", "phi"):
                shutil.copy(f, work / "0" / f.name)
        outcome["warm_start_time"] = str(warm_start_time)
    if end_time is not None:
        cd = work / "system" / "controlDict"
        text = re.sub(r"endTime\s+[\d.]+;", f"endTime         {int(end_time)};", cd.read_text())
        text = re.sub(r"writeInterval\s+[\d.]+;", f"writeInterval   {int(end_time)};", text)
        cd.write_text(text)
        outcome["end_time_override"] = int(end_time)
    if mutation == "refine_x2_yz":
        try:
            refine_case_x2(work, env, directions=("(0 1 0)", "(0 0 1)"))
        except Exception as exc:
            outcome.update(failure_class="setup", reason=f"{type(exc).__name__}: {exc}"[:300])
            outcome["wall_s"] = time.time() - started
            return outcome
    elif mutation == "refine_x2":
        try:
            refine_case_x2(work, env)
        except Exception as exc:
            outcome.update(failure_class="setup", reason=f"refine: {exc}"[:300])
            outcome["wall_s"] = time.time() - started
            return outcome
    elif mutation is not None:
        outcome.update(failure_class="setup", reason=f"unknown mutation {mutation!r}")
        outcome["wall_s"] = time.time() - started
        return outcome

    def step(cmd, timeout=timeout_s):
        with open(work / f"log.{cmd[0]}", "a") as log:
            return subprocess.run(
                cmd, cwd=work, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=timeout,
            )

    def safe_step(cmd, timeout=timeout_s):
        try:
            return step(cmd, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout"
        except OSError as exc:
            return f"oserror: {exc}"

    def fail(failure_class: str, reason: str) -> dict:
        outcome.update(failure_class=failure_class, reason=reason[:300])
        outcome["log_tail"] = _tail(work / "log.mpirun")
        outcome["wall_s"] = time.time() - started
        _prune_run_dir(work, keep=True)
        return outcome

    dec = work / "system" / "decomposeParDict"
    dec.write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; "
        "object decomposeParDict; }\n"
        f"numberOfSubdomains {nprocs};\nmethod          scotch;\n"
    )
    decomposition = safe_step(["decomposePar", "-force"])
    if not hasattr(decomposition, "returncode"):
        return fail("infrastructure", f"decomposePar {decomposition}")
    if decomposition.returncode != 0:
        return fail("decompose", f"decomposePar exit {decomposition.returncode}")

    solve = safe_step(
        ["mpirun", *_mpirun_args(), "-np", str(nprocs), "simpleFoam", "-parallel"]
    )
    solve_log = work / "log.mpirun"
    iterations, residuals, hard_divergence, divergence_reason = runner.parse_log(
        solve_log
    )
    outcome.update(
        iterations=int(iterations),
        final_residuals={key: float(value) for key, value in residuals.items()},
        final_time=runner.last_time(solve_log),
    )
    if solve == "timeout":
        return fail("timeout", f"solver timeout after {timeout_s}s")
    if not hasattr(solve, "returncode"):
        return fail("infrastructure", f"mpirun {solve}")
    if hard_divergence:
        return fail("divergence", divergence_reason or "hard divergence")
    if solve.returncode != 0:
        return fail("solver_exit", f"simpleFoam exit {solve.returncode}")
    control_text = (work / "system" / "controlDict").read_text()
    end_match = re.search(r"\bendTime\s+([0-9.eE+-]+)\s*;", control_text)
    final_time = outcome["final_time"]
    if end_match is None or final_time is None:
        return fail("incomplete", "no endTime/final time recorded")
    expected_end = float(end_match.group(1))
    outcome["expected_end_time"] = expected_end
    log_text = solve_log.read_text(errors="replace")
    outcome["simple_converged"] = "SIMPLE solution converged" in log_text
    outcome["clamp_fractions"] = clamp_fractions(log_text)
    if (
        final_time + 1.0e-9 * max(1.0, abs(expected_end)) < expected_end
        and not outcome["simple_converged"]
    ):
        return fail("incomplete", f"stopped at t={final_time} < {expected_end}")

    reconstruction = safe_step(["reconstructPar", "-latestTime"])
    if not hasattr(reconstruction, "returncode") or reconstruction.returncode != 0:
        return fail("reconstruct", "reconstructPar failed")
    for proc in work.glob("processor*"):
        shutil.rmtree(proc, ignore_errors=True)
    if case_id in mcconkey.HOLDOUT_DUCT:
        # square-duct holdout: the DNS lives on its own cross-section mesh, so
        # the frozen composite is evaluated at the DNS points by duct_scoring,
        # which also returns the secondary-flow diagnostics the success
        # criteria ask for
        from .holdout_cases import duct_scoring

        post = safe_step(["simpleFoam", "-postProcess", "-func", "grad(U)", "-latestTime"])
        if not hasattr(post, "returncode") or post.returncode != 0:
            return fail("score", "grad(U) post-processing failed")
        try:
            reference = duct_scoring.load_reference(mcconkey.duct_reference_file(case_id))
            duct_score, diagnostics = duct_scoring.score_duct(work, reference)
        except Exception as exc:
            return fail("score", f"duct scoring: {type(exc).__name__}: {exc}"[:300])
        outcome.update(
            status="complete", failure_class=None, reason="",
            score={"e_u": duct_score.e_u, "e_cf": duct_score.e_cf,
                   "e_uv": duct_score.e_uv, "composite": duct_score.composite},
            duct_diagnostics=diagnostics, log_tail=_tail(solve_log, 12),
        )
        outcome["wall_s"] = time.time() - started
        if not keep_run_dir:
            shutil.rmtree(work, ignore_errors=True)
            outcome["run_dir"] = ""
        return outcome
    if case_id in mcconkey.GATES:
        # gate cases (flat plate) are judged on wall friction only: no
        # velocity-gradient post-processing, no composite score
        try:
            profile = wall_cf_profile(work, case_id)
        except Exception as exc:
            return fail("score", f"cf profile: {type(exc).__name__}: {exc}")
        outcome.update(
            status="complete", failure_class=None, reason="", gate_only=True,
            score={"e_u": 0.0, "e_cf": 0.0, "e_uv": 0.0, "composite": 0.0},
            cf_profile=profile, log_tail=_tail(solve_log, 12),
        )
        outcome["wall_s"] = time.time() - started
        _prune_run_dir(work, keep=keep_run_dir)
        return outcome
    post = safe_step(
        ["simpleFoam", "-postProcess", "-func", "grad(U)", "-latestTime"]
    )
    if not hasattr(post, "returncode") or post.returncode != 0:
        return fail("postprocess", "grad(U) post-processing failed")
    try:
        result = (score_run_resampled if mutation == "refine_x2" else score_run)(work, case_id)
    except (
        FileNotFoundError, IndexError, OSError, ValueError,
        FloatingPointError, subprocess.TimeoutExpired,
    ) as exc:
        return fail("score", f"{type(exc).__name__}: {exc}")
    values = [result.e_u, result.e_cf, result.e_uv, result.composite]
    if not np.all(np.isfinite(values)):
        return fail("nonfinite", "non-finite score")
    outcome.update(
        status="complete", failure_class=None, reason="",
        score={
            "e_u": float(result.e_u), "e_cf": float(result.e_cf),
            "e_uv": float(result.e_uv), "composite": float(result.composite),
        },
        log_tail=_tail(solve_log, 12),
    )
    if case_id in mcconkey.GATES:
        # gate cases carry their wall-friction profile (binned along x) so
        # the Phase-4 Cf criterion can compare a candidate with stock directly
        try:
            outcome["cf_profile"] = wall_cf_profile(work, case_id)
        except Exception as exc:  # never fail a completed run on diagnostics
            outcome["cf_profile_error"] = f"{type(exc).__name__}: {exc}"[:200]
    outcome["wall_s"] = time.time() - started
    _prune_run_dir(work, keep=keep_run_dir)
    return outcome


def evaluate_candidate(
    spec: CandidateSpec | None,
    case_ids: tuple[str, ...],
    tag: str,
    nprocs: int = 4,
    timeout_s: int = 3600,
    warm_start: bool = False,
) -> tuple[dict[str, scoring.CaseScore], set[str]]:
    """Run + score a candidate on a case set (cases run concurrently at
    `nprocs` ranks each, packed onto the machine). Returns (scores, diverged)."""
    import os as _os
    from concurrent.futures import ThreadPoolExecutor

    warm_start = warm_start or _os.environ.get("TEDP_WARM_START") == "1"
    nprocs = int(_os.environ.get("TEDP_NPROCS", nprocs))
    max_workers = int(_os.environ.get("TEDP_MAX_WORKERS", max(1, 22 // nprocs)))
    root = REPO / "runs" / "tier2" / tag

    def run_one(cid: str):
        outcome = run_case_job(
            spec, cid, root, nprocs=nprocs, timeout_s=timeout_s,
            warm_start=warm_start, keep_run_dir=True,
        )
        if outcome["status"] != "complete":
            return cid, None
        score = outcome["score"]
        return cid, scoring.CaseScore(cid, score["e_u"], score["e_cf"], score["e_uv"])

    scores: dict[str, scoring.CaseScore] = {}
    diverged: set[str] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for cid, result in pool.map(run_one, case_ids):
            if result is None:
                diverged.add(cid)
            else:
                scores[cid] = result
    return scores, diverged
