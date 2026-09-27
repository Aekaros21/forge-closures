"""Materialize an OpenFOAM case for one (family, case, candidate) triple.

Templates live in cases/<family>/; per-case mesh and reference data live in
data/ (indexed by tedp.data.fetch). A generated case links the shared
polyMesh read-only and renders constant/turbulenceProperties from the spec.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .spec import CandidateSpec, to_foam_coeffs

REPO = Path(__file__).resolve().parents[2]
CASES = REPO / "cases"
RUNS = REPO / "runs"

TURB_HEADER = (
    "FoamFile { version 2.0; format ascii; class dictionary; "
    "object turbulenceProperties; }\n"
    "simulationType RAS;\n"
)


def render_turbulence_properties(spec: CandidateSpec | None, mode: str = "expressions") -> str:
    """None -> stock kOmegaSST; a spec -> kOmegaSSTBasis with its channels."""
    if spec is None:
        return TURB_HEADER + "RAS\n{\n    RASModel kOmegaSST;\n    turbulence on;\n    printCoeffs off;\n}\n"
    return (
        TURB_HEADER
        + "RAS\n{\n    RASModel kOmegaSSTBasis;\n    turbulence on;\n    printCoeffs off;\n"
        + to_foam_coeffs(spec, mode=mode)
        + "}\n"
    )


def make_case(
    template: Path,
    workdir: Path,
    spec: CandidateSpec | None = None,
    mode: str = "expressions",
    mesh_dir: Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Copy `template` to `workdir`, link the mesh, render the model dict."""
    template = Path(template)
    workdir = Path(workdir)
    if workdir.exists():
        if not overwrite:
            raise FileExistsError(workdir)
        shutil.rmtree(workdir)
    shutil.copytree(template, workdir, symlinks=True)

    if mesh_dir is not None:
        poly = workdir / "constant" / "polyMesh"
        if poly.exists() or poly.is_symlink():
            shutil.rmtree(poly, ignore_errors=True)
            poly.unlink(missing_ok=True)
        poly.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(Path(mesh_dir).resolve(), poly)

    (workdir / "constant" / "turbulenceProperties").write_text(
        render_turbulence_properties(spec, mode=mode)
    )
    return workdir
