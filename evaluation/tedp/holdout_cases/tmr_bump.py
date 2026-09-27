"""TMR 2D bump-in-channel verification case, across the NASA grid family.

Turbulence Modeling Resource case 2DB. A shallow bump on the lower wall of a
channel, used for *verification*: the interest is whether a model converges
under grid refinement and behaves sanely, not whether it matches an experiment.
The five NASA grids (89x41 up to 1409x641) are nested refinements of the same
geometry, so the sequence measures the model's own grid convergence.

Setup follows the TMR case definition: Re = 3e6 per unit grid length, the lower
wall is a symmetry plane ahead of x = 0 and behind x = 1.5 with a viscous wall
between, and the upper boundary is a slip wall. Solved incompressibly with
U_ref = 1, so nu = 1/3e6.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..casegen import render_turbulence_properties
from ..spec import CandidateSpec
from . import hump, structured_mesh, tmr_convergence

GRIDS = {
    "89x41": "bump_4levelsdown_89x41.p2dfmt",
    "177x81": "bump_3levelsdown_177x81.p2dfmt",
    "353x161": "bump_2levelsdown_353x161.p2dfmt.gz",
    "705x321": "bump_1levelsdown_705x321.p2dfmt.gz",
    "1409x641": "bump_0levelsdown_1409x641.p2dfmt.gz",
}
GRID_ROOT = (Path(__file__).resolve().parents[3] / "cases" / "tmr" / "u" / "piyer"
             / "nasa_tmr" / "gitlab" / "turbmodels" / "Bump" / "Grids")
U_REF = 1.0
RE_L = 3.0e6
NU = U_REF / RE_L
WALL_START, WALL_END = 0.0, 1.5      # viscous wall extent on the lower boundary


def make_case(root: Path, grid: str, spec: CandidateSpec | None,
              end_time: int = 6000) -> dict:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    for sub in ("system", "constant", "0"):
        (root / sub).mkdir(parents=True)
    x, y = structured_mesh.read_p2dfmt(GRID_ROOT / GRIDS[grid])
    thickness = 0.05 * (y.max() - y.min())
    counts = structured_mesh.write_polymesh_2d(
        x, y, root, thickness,
        patches={
            "imin": "inlet:patch", "imax": "outlet:patch",
            "jmax": "top:patch", "jmin": "symmetryWall:symmetryPlane",
            "jmin_split": [("bump:wall", lambda xc: WALL_START <= xc <= WALL_END)],
            "z": "frontAndBack",
        })
    # Aref = Sref x span, so the recorded coefficient is the reported one.
    control = tmr_convergence.control_dict(end_time, "bump", 1.5 * thickness)
    if spec is not None:
        control += '\nlibs            ("libkOmegaSSTBasis.so");\n'
    (root / "system" / "controlDict").write_text(control)
    (root / "system" / "fvSchemes").write_text(hump.fv_schemes())
    (root / "system" / "fvSolution").write_text(hump.fv_solution())
    (root / "constant" / "transportProperties").write_text(
        hump._header("dictionary", "transportProperties", "constant")
        + f"\ntransportModel  Newtonian;\nnu              [0 2 -1 0 0 0 0] {NU};\n")
    (root / "constant" / "turbulenceProperties").write_text(
        render_turbulence_properties(spec, mode="expressions"))

    k_in = 1.5 * (0.001 * U_REF) ** 2
    omega_in = k_in ** 0.5 / (0.09 ** 0.25 * 0.01)
    nut_in = 1.0e-3 * NU

    def field(cls, obj, dims, value, wall):
        return hump._header(cls, obj, "0") + f"""
dimensions      {dims};
internalField   uniform {value};

boundaryField
{{
    inlet        {{ type fixedValue; value uniform {value}; }}
    outlet       {{ type zeroGradient; }}
    bump         {{ {wall} }}
    symmetryWall {{ type symmetryPlane; }}
    top          {{ type slip; }}
    frontAndBack {{ type empty; }}
}}
"""
    fields = {
        "U": field("volVectorField", "U", "[0 1 -1 0 0 0 0]", f"({U_REF} 0 0)", "type noSlip;"),
        "k": field("volScalarField", "k", "[0 2 -2 0 0 0 0]", f"{k_in}",
                   f"type kLowReWallFunction; value uniform {k_in};"),
        "omega": field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", f"{omega_in}",
                       f"type omegaWallFunction; value uniform {omega_in};"),
        "nut": field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", f"{nut_in}",
                     "type nutLowReWallFunction; value uniform 0;"),
    }
    fields["p"] = hump._header("volScalarField", "p", "0") + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet        { type zeroGradient; }
    outlet       { type fixedValue; value uniform 0; }
    bump         { type zeroGradient; }
    symmetryWall { type symmetryPlane; }
    top          { type slip; }
    frontAndBack { type empty; }
}
"""
    for name, text in fields.items():
        (root / "0" / name).write_text(text)
    return {"grid": grid, "cells": (x.shape[0] - 1) * (x.shape[1] - 1),
            "patch_faces": counts, "re_l": RE_L, "nu": NU}
