"""TMR 2D NACA 0012 verification case, across the NASA C-grid family.

Turbulence Modeling Resource case 2DN00. Re_c = 6e6, alpha = 0 for the
verification sweep. The grids are C-meshes: the j=0 line runs in along the
lower wake cut, around the aerofoil, and back out along the upper wake cut, so
the cut faces are closed as internal faces joining mirrored cells rather than
left as a boundary (structured_mesh's ``jmin_cut``).

Verification, not validation: the question is whether the model converges under
refinement and leaves an attached, essentially equilibrium aerofoil boundary
layer alone.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np

from ..casegen import render_turbulence_properties
from ..spec import CandidateSpec
from . import hump, structured_mesh, tmr_convergence

GRIDS = {
    "113x33": "n0012_113-33.p2dfmt",
    "225x65": "n0012_225-65.p2dfmt",
    "449x129": "n0012_449-129.p2dfmt.gz",
    "897x257": "n0012_897-257.p2dfmt.gz",
    "1793x513": "n0012_1793-513.p2dfmt.gz",
}
GRID_ROOT = (Path(__file__).resolve().parents[3] / "cases" / "tmr" / "u" / "piyer"
             / "nasa_tmr" / "gitlab" / "turbmodels" / "NACA0012_grids")
U_REF = 1.0
RE_C = 6.0e6
NU = U_REF / RE_C


def wake_cut_faces(x: np.ndarray) -> int:
    """Number of j=0 faces on each wake cut, i.e. cells ahead of the trailing edge.

    The aerofoil occupies 0 <= x <= 1 on the j=0 line; everything outside it at
    either end of the i-range is the cut.
    """
    x0 = x[0]
    nci = x0.size - 1
    centres = 0.5 * (x0[:-1] + x0[1:])
    lower = int(np.argmax(centres <= 1.0 + 1.0e-9))
    upper = nci - 1 - int(np.argmax(centres[::-1] <= 1.0 + 1.0e-9))
    if lower != nci - 1 - upper:
        raise ValueError(f"wake cut is not symmetric: {lower} vs {nci - 1 - upper}")
    return lower


def make_case(root: Path, grid: str, spec: CandidateSpec | None,
              end_time: int = 5000, alpha_deg: float = 0.0) -> dict:
    """Build the case. ``alpha_deg`` rotates the freestream rather than the mesh,
    which keeps one grid usable at every incidence. Forces must then be resolved
    along and normal to that direction, not along x."""
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    for sub in ("system", "constant", "0"):
        (root / sub).mkdir(parents=True)
    a = math.radians(alpha_deg)
    u_inf = (U_REF * math.cos(a), U_REF * math.sin(a), 0.0)
    x, y = structured_mesh.read_p2dfmt(GRID_ROOT / GRIDS[grid])
    n_cut = wake_cut_faces(x)
    span = 0.05
    counts = structured_mesh.write_polymesh_2d(
        x, y, root, span,
        patches={"imin": "outletLower:patch", "imax": "outletUpper:patch",
                 "jmax": "farfield:patch", "jmin": "aerofoil:wall",
                 "jmin_cut": n_cut, "z": "frontAndBack"})
    control = tmr_convergence.control_dict(end_time, "aerofoil", 1.0 * span)
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

    k_in = 1.5 * (0.0005 * U_REF) ** 2
    omega_in = k_in ** 0.5 / (0.09 ** 0.25 * 0.01)
    nut_in = 1.0e-3 * NU

    def field(cls, obj, dims, value, wall):
        return hump._header(cls, obj, "0") + f"""
dimensions      {dims};
internalField   uniform {value};

boundaryField
{{
    farfield     {{ type freestream; freestreamValue uniform {value}; }}
    aerofoil     {{ {wall} }}
    outletLower  {{ type zeroGradient; }}
    outletUpper  {{ type zeroGradient; }}
    frontAndBack {{ type empty; }}
}}
"""
    fields = {
        "U": field("volVectorField", "U", "[0 1 -1 0 0 0 0]",
                   f"({u_inf[0]:.10f} {u_inf[1]:.10f} 0)", "type noSlip;"),
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
    farfield     { type freestreamPressure; freestreamValue uniform 0; }
    aerofoil     { type zeroGradient; }
    outletLower  { type freestreamPressure; freestreamValue uniform 0; }
    outletUpper  { type freestreamPressure; freestreamValue uniform 0; }
    frontAndBack { type empty; }
}
"""
    for name, text in fields.items():
        (root / "0" / name).write_text(text)
    return {"grid": grid, "cells": (x.shape[0] - 1) * (x.shape[1] - 1),
            "wake_cut_faces_each_side": n_cut, "patch_faces": counts, "re_c": RE_C,
            "alpha_deg": alpha_deg, "u_inf": u_inf}
