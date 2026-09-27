"""TMR 2D zero-pressure-gradient flat plate verification case.

Turbulence Modeling Resource case 2DZP, across the five nested NASA grids
(35x25 up to 545x385, each coarser grid being every other point of the next
finer one). Re = 5e6 per unit grid length, plate from x = 0 to x = 2 with a
symmetry plane ahead of the leading edge.

This is the strongest anchor of the three TMR cases we run, because NASA
publishes reference skin friction at x = 0.970084071 from three *incompressible*
codes (FUN3D-incompressible, SC/Tetra and GHOST) as well as the compressible
CFL3D and FUN3D results. Our solver is incompressible, so the incompressible
reference removes the Mach 0.2 caveat that applies to the bump and aerofoil.

Freestream turbulence follows the TMR specification rather than a guessed
turbulence intensity. TMR sets k/a_inf^2 = 9e-9 and omega*mu_inf/(rho a_inf^2)
= 1e-6, which at Mach 0.2 (so a_inf = U_ref/0.2 = 5) gives k = 2.25e-7 and
omega = 1e-6*a_inf^2/nu, i.e. a freestream eddy viscosity ratio of 0.009. That
ratio, not the raw values, is the quantity the reference solutions are sensitive
to. Note this differs from the inflow used in ``tmr_bump``, which was built from
a 0.1% turbulence intensity and lands at a ratio of 20; see LOG.md.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..casegen import render_turbulence_properties
from ..spec import CandidateSpec
from . import hump, structured_mesh, tmr_convergence

GRIDS = {
    "35x25": "flatplate_clust2_4levelsdown_35x25.p2dfmt",
    "69x49": "flatplate_clust2_3levelsdown_69x49.p2dfmt",
    "137x97": "flatplate_clust2_2levelsdown_137x97.p2dfmt",
    "273x193": "flatplate_clust2_1leveldown_273x193.p2dfmt.gz",
    "545x385": "flatplate_clust2.p2dfmt.gz",
}
GRID_ROOT = Path(__file__).resolve().parents[3] / "cases" / "tmr" / "flatplate_grids"

U_REF = 1.0
RE_L = 5.0e6
NU = U_REF / RE_L
MACH = 0.2                      # the TMR reference condition, used only to set
A_INF = U_REF / MACH            # the freestream turbulence scale
PLATE_START = 0.0               # symmetry plane upstream of this, wall after it
CF_STATION = 0.970084071        # where the TMR grid-convergence tables report Cf

# TMR freestream turbulence, converted to our incompressible normalisation.
K_INF = 9.0e-9 * A_INF ** 2
OMEGA_INF = 1.0e-6 * A_INF ** 2 / NU
NUT_INF = K_INF / OMEGA_INF


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
            "jmax": "farfield:patch", "jmin": "symmetryWall:symmetryPlane",
            "jmin_split": [("plate:wall", lambda xc: xc >= PLATE_START)],
            "z": "frontAndBack",
        })
    control = tmr_convergence.control_dict(end_time, "plate", 2.0 * thickness)
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

    def field(cls, obj, dims, value, wall, farfield):
        return hump._header(cls, obj, "0") + f"""
dimensions      {dims};
internalField   uniform {value};

boundaryField
{{
    inlet        {{ type fixedValue; value uniform {value}; }}
    outlet       {{ type zeroGradient; }}
    plate        {{ {wall} }}
    symmetryWall {{ type symmetryPlane; }}
    farfield     {{ {farfield} }}
    frontAndBack {{ type empty; }}
}}
"""
    # The top boundary is a farfield in the TMR definition, not a wall: the
    # boundary layer displaces fluid outward and that flux has to leave the
    # domain. A slip condition would block it.
    fields = {
        "U": field("volVectorField", "U", "[0 1 -1 0 0 0 0]", f"({U_REF} 0 0)",
                   "type noSlip;",
                   f"type pressureInletOutletVelocity; value uniform ({U_REF} 0 0);"),
        "k": field("volScalarField", "k", "[0 2 -2 0 0 0 0]", f"{K_INF}",
                   f"type kLowReWallFunction; value uniform {K_INF};",
                   f"type inletOutlet; inletValue uniform {K_INF}; value uniform {K_INF};"),
        "omega": field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", f"{OMEGA_INF}",
                       f"type omegaWallFunction; value uniform {OMEGA_INF};",
                       f"type inletOutlet; inletValue uniform {OMEGA_INF}; "
                       f"value uniform {OMEGA_INF};"),
        "nut": field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", f"{NUT_INF}",
                     "type nutLowReWallFunction; value uniform 0;",
                     f"type calculated; value uniform {NUT_INF};"),
    }
    fields["p"] = hump._header("volScalarField", "p", "0") + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet        { type zeroGradient; }
    outlet       { type fixedValue; value uniform 0; }
    plate        { type zeroGradient; }
    symmetryWall { type symmetryPlane; }
    farfield     { type fixedValue; value uniform 0; }
    frontAndBack { type empty; }
}
"""
    for name, text in fields.items():
        (root / "0" / name).write_text(text)
    return {"grid": grid, "cells": (x.shape[0] - 1) * (x.shape[1] - 1),
            "patch_faces": counts, "re_l": RE_L, "nu": NU,
            "k_inf": K_INF, "omega_inf": OMEGA_INF, "nut_ratio_inf": NUT_INF / NU}
