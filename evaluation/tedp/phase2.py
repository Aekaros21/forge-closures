"""Phase 2: ceiling measurement.

1. build_frozen_case / run_frozen: k-corrective frozen RANS per calibration
   case (DNS U, k, tau frozen; omega iterated) -> inferred bDelta(x), R(x)
   and the regression feature fields.
2. propagation_test: inject the inferred fields via kOmegaSSTBasis mode
   frozenFields into plain simpleFoam; DNS-velocity recovery is the
   representability arbiter for the whole hypothesis class.
3. assemble_table: gather (features, targets) across cases for regression.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fluidfoam import readscalar, readsymmtensor, readtensor, readvector

from . import repro, runner, tier2
from .data import foamwriter, mcconkey

REPO = Path(__file__).resolve().parents[2]
FROZEN_ITERS = 400
FROZEN_OMEGA_RESIDUAL_TOL = 1.0e-4


def build_frozen_case(case_id: str, workdir: Path) -> Path:
    work = repro.clone_case(case_id, workdir)
    src = mcconkey.case_dir(case_id)
    t_final = mcconkey.final_time(case_id)
    ref = mcconkey.load_case_table(case_id, "REF")

    # frozen inputs into 0/, using the shipped final-time fields as BC
    # templates (value lists present, types exact)
    u_ref = np.stack([ref["REF_U_1"], ref["REF_U_2"], ref["REF_U_3"]], axis=1)
    foamwriter.replace_internal(src / t_final / "U", work / "0" / "U", u_ref)
    foamwriter.replace_internal(
        src / t_final / "k", work / "0" / "k", np.asarray(ref["REF_k"])
    )
    # turbulence-model-dependent wall functions cannot evaluate inside the
    # frozen app (no model in the registry); the frozen fields keep their
    # converged wall values as plain fixedValue
    foamwriter.swap_wall_bc(work / "0" / "k", "kLowReWallFunction", "fixedValue")
    foamwriter.swap_wall_bc(work / "0" / "k", "kqRWallFunction", "zeroGradient")
    shutil.copy(src / t_final / "omega", work / "0" / "omega")
    foamwriter.swap_wall_bc(work / "0" / "omega", "omegaWallFunction", "fixedValue")
    tau = np.stack(
        [ref[f"REF_tau_{c}"] for c in ("11", "12", "13", "22", "23", "33")], axis=1
    )
    foamwriter.write_field(work, "0", "tauDNS", tau)

    # solver controls
    control = work / "system" / "controlDict"
    text = control.read_text()
    text = re.sub(r"application\s+\w+;", "application     frozenKOmegaSST;", text)
    text = re.sub(r"startTime\s+[\d.]+;", "startTime       0;", text)
    text = re.sub(r"endTime\s+[\d.]+;", f"endTime         {FROZEN_ITERS};", text)
    text = re.sub(r"writeInterval\s+[\d.]+;", f"writeInterval   {FROZEN_ITERS};", text)
    control.write_text(text)

    fvschemes = work / "system" / "fvSchemes"
    text = fvschemes.read_text()
    if "pcorr" not in text:
        if "fluxRequired" in text:
            # add pcorr inside the existing block
            text = re.sub(
                r"(fluxRequired\s*\{)", r"\1\n    pcorr           ;", text, count=1
            )
        else:
            text += "\nfluxRequired\n{\n    pcorr           ;\n}\n"
        fvschemes.write_text(text)

    fvsol = work / "system" / "fvSolution"
    text = fvsol.read_text()
    if "pcorr" not in text:
        text = text.replace(
            "solvers\n{",
            "solvers\n{\n    pcorr\n    {\n        solver GAMG;\n"
            "        smoother GaussSeidel;\n        tolerance 1e-9;\n"
            "        relTol 0;\n    }\n",
            1,
        )
        m = re.search(r"SIMPLE\s*\{", text)
        if m:
            text = text[: m.end()] + (
                "\n    pcorrRefCell    0;\n    pcorrRefValue   0;"
            ) + text[m.end():]
        fvsol.write_text(text)
    return work


@dataclass
class FrozenResult:
    case_id: str
    work: Path
    ok: bool
    final_time: str
    reason: str = ""


def run_frozen(case_id: str, workdir: Path | None = None) -> FrozenResult:
    work = build_frozen_case(
        case_id, workdir or REPO / "runs" / "phase2" / "frozen" / case_id
    )
    res = runner.run_case(work, solver="frozenKOmegaSST", nprocs=1, timeout_s=3600)
    times = sorted((int(p.name) for p in work.iterdir() if p.name.isdigit()),
                   reverse=True)
    if res.returncode != 0 or res.diverged or not times or times[0] == 0:
        return FrozenResult(case_id, work, False, "0", res.reason or "no output")
    omega_residual = res.final_residuals.get("omega")
    if omega_residual is None or omega_residual > FROZEN_OMEGA_RESIDUAL_TOL:
        return FrozenResult(
            case_id,
            work,
            False,
            str(times[0]),
            "frozen omega did not converge: final initial residual "
            f"{omega_residual!r} > {FROZEN_OMEGA_RESIDUAL_TOL:g}",
        )
    return FrozenResult(case_id, work, True, str(times[0]))


FEATURE_FIELDS = ("I1", "I2", "I3", "I4", "I5", "Ret", "F1out")
TENSOR_COUNT = 10


def read_frozen_fields(fr: FrozenResult) -> dict[str, np.ndarray]:
    """Targets + deployment-parity features from a finished frozen run.

    Older frozen runs wrote the DNS-stress production under ``PoE``.  That is
    useful truth-side diagnostics, but is unavailable to a deployed model.
    For those runs we reconstruct the deployed SST-production coordinate from
    the frozen eddy viscosity and the exact saved velocity gradient when
    present (falling back to the incompressible I1 identity).
    """
    case, t = str(fr.work), fr.final_time
    out: dict[str, np.ndarray] = {}
    out["bDelta"] = np.asarray(readsymmtensor(case, t, "bDelta", verbose=False)).T
    out["RField"] = np.asarray(readscalar(case, t, "RField", verbose=False))
    out["k"] = np.asarray(readscalar(case, "0", "k", verbose=False))
    out["omega"] = np.asarray(readscalar(case, t, "omega", verbose=False))
    out["nutFrozen"] = np.asarray(
        readscalar(case, t, "nutFrozen", verbose=False)
    )
    for f in FEATURE_FIELDS:
        out[f] = np.asarray(readscalar(case, t, f, verbose=False))
    for n in range(1, TENSOR_COUNT + 1):
        out[f"T{n}"] = np.asarray(
            readsymmtensor(case, t, f"T{n}", verbose=False)
        ).T
    poe_model_path = fr.work / t / "PoEModel"
    poe_truth_path = fr.work / t / "PoETruth"
    if poe_model_path.exists():
        poe_model = np.asarray(readscalar(case, t, "PoEModel", verbose=False))
        poe_truth = np.asarray(
            readscalar(case, t, "PoETruth", verbose=False)
        ) if poe_truth_path.exists() else np.full_like(poe_model, np.nan)
    else:
        # Legacy semantic: field PoE was PkDNS/(beta*k*omega).
        poe_truth = np.asarray(readscalar(case, t, "PoE", verbose=False))
        grad_path = fr.work / "0" / "grad(U)"
        if grad_path.exists():
            grad = np.asarray(readtensor(case, "0", "grad(U)", verbose=False)).T
            grad = grad.reshape((-1, 3, 3))
            symm = 0.5 * (grad + np.swapaxes(grad, 1, 2))
            strain_sq = np.einsum("nij,nij->n", symm, symm)
            production = 2.0 * out["nutFrozen"] * strain_sq
            poe_model = production / np.maximum(
                0.09 * out["k"] * out["omega"], 1.0e-300
            )
        else:
            # I1=|dev(symm(gradU))/omega|^2; incompressibility makes this
            # equal to |symm(gradU)/omega|^2 to solver accuracy.
            poe_model = (
                2.0 * out["nutFrozen"] * out["I1"] * out["omega"]
                / np.maximum(0.09 * out["k"], 1.0e-300)
            )
    out["PoETruth"] = poe_truth
    out["PoE"] = np.clip(poe_model, 0.0, 10.0)
    return out


def propagation_test(
    case_id: str,
    fr: FrozenResult,
    channels: tuple[str, ...] = ("bdelta", "rsource"),
    tag: str = "",
    nprocs: int = 8,
) -> tuple[float, float]:
    """Inject inferred fields, run simpleFoam, return (E_U_propagated,
    E_U_stock) — the representability measure for this case. `channels`
    selects which inferred field(s) to inject (the other is zeroed), for
    isolating instabilities."""
    work = tier2.make_candidate_case(
        case_id,
        None,
        REPO / "runs" / "phase2" / "propagation" / (case_id + (f"_{tag}" if tag else "")),
    )
    # frozenFields mode dictionary
    (work / "constant" / "turbulenceProperties").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; "
        "object turbulenceProperties; }\n"
        "simulationType RAS;\nRAS\n{\n"
        "    RASModel kOmegaSSTBasis;\n    turbulence on;\n    printCoeffs off;\n"
        "    kOmegaSSTBasisCoeffs\n    {\n"
        "        mode            frozenFields;\n"
        "        gMax 1e9; rMaxFactor 20.0; bDeltaRelax 0.5; rampIterations 200;\n"
        "        nonlinearProduction on; realizabilityClip on;\n"
        '        a1Expr "";\n'
        "    }\n}\n"
    )
    control = work / "system" / "controlDict"
    text = control.read_text()
    if "libkOmegaSSTBasis" not in text:
        text += '\nlibs            ("libkOmegaSSTBasis.so");\n'
        control.write_text(text)

    # inferred fields into 0/ under the names the model reads, with the
    # physical-range mask applied: the floored-omega cells carry |R| up to
    # ~10x the largest dissipation in the domain and detonate the run
    fields = read_frozen_fields(fr)
    keep = physical_mask(fields)
    bdelta = np.where(keep[:, None], fields["bDelta"], 0.0)
    rfield = np.where(keep, fields["RField"], 0.0)
    if "bdelta" not in channels:
        bdelta = np.zeros_like(bdelta)
    if "rsource" not in channels:
        rfield = np.zeros_like(rfield)
    for src_name, dst_name, values in (
        ("bDelta", "bDeltaField", bdelta),
        ("RField", "RField", rfield),
    ):
        src = fr.work / fr.final_time / src_name
        dst = work / "0" / dst_name
        foamwriter.replace_internal(src, dst, values)
        dst.write_text(
            re.sub(r"object\s+\w+;", f"object      {dst_name};", dst.read_text(), count=1)
        )

    env = runner.foam_env()

    def step(cmd, timeout=14400):
        with open(work / f"log.{cmd[0]}", "a") as log:
            return subprocess.run(cmd, cwd=work, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout)

    dec = work / "system" / "decomposeParDict"
    dec.write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; "
        "object decomposeParDict; }\n"
        f"numberOfSubdomains {nprocs};\nmethod          scotch;\n"
    )
    assert step(["decomposePar", "-force"]).returncode == 0
    r = step(["mpirun", "--use-hwthread-cpus", "-np", str(nprocs),
              "simpleFoam", "-parallel"])
    step(["reconstructPar", "-latestTime"])
    for proc in work.glob("processor*"):
        shutil.rmtree(proc)
    if r.returncode != 0:
        return float("nan"), float("nan")
    # -postProcess reconstructs the turbulence model at the latest time; the
    # frozenFields inputs live in 0/, so copy them forward first
    times = sorted((int(p.name) for p in work.iterdir() if p.name.isdigit()),
                   reverse=True)
    t_last = str(times[0])
    for f in ("bDeltaField", "RField"):
        if not (work / t_last / f).exists():
            shutil.copy(work / "0" / f, work / t_last / f)
    step(["simpleFoam", "-postProcess", "-func", "grad(U)", "-latestTime"])
    score = tier2.score_run(work, case_id)

    # stock E_U for the same case from the repro run
    stock = tier2.score_run(REPO / "runs" / "repro" / case_id, case_id)
    return score.e_u, stock.e_u


I1_MAX = 0.5        # physical range in these flows tops out ~0.25
I2_MAX = 0.5
R_RATIO_MAX = 30.0  # |R| / (beta* k omega); p99.9 of the raw data is ~19
K_FLOOR_REL = 1.0e-6


def physical_mask(fields: dict[str, np.ndarray]) -> np.ndarray:
    """Drop the cells where the frozen omega collapsed to its floor (I1 up
    to 1e29, rhat 1e14 in the raw table) and the floored-k cells: they carry
    no closure information and wreck any least-squares fit."""
    k = fields["k"]
    eps = 0.09 * k * fields["omega"]
    ratio = np.abs(fields["RField"]) / np.maximum(eps, 1e-300)
    m = (
        np.isfinite(fields["I1"]) & np.isfinite(fields["I2"])
        & (fields["I1"] <= I1_MAX) & (np.abs(fields["I2"]) <= I2_MAX)
        & (k > K_FLOOR_REL * float(np.max(k)))
        & (ratio <= R_RATIO_MAX)
        & np.all(np.isfinite(fields["bDelta"]), axis=1)
    )
    return m


def assemble_table(results: list[FrozenResult], mask: bool = True) -> dict[str, np.ndarray]:
    """Concatenate features/targets over cases for the regression, with the
    physical-range mask applied per case."""
    parts: dict[str, list[np.ndarray]] = {}
    for fr in results:
        if not fr.ok:
            continue
        fields = read_frozen_fields(fr)
        keep = physical_mask(fields) if mask else np.ones(len(fields["RField"]), bool)
        for key, val in fields.items():
            parts.setdefault(key, []).append(val[keep])
        parts.setdefault("case_id", []).append(
            np.full(int(keep.sum()), fr.case_id, dtype=object)
        )
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}
