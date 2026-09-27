"""NASA wall-mounted hump (Greenblatt et al.), baseline no-flow-control case.

Geometry and grids: NASA Turbulence Modeling Resource "2D NASA Wall-Mounted
Hump Separated Flow Validation Case", no-plenum grids ``hump2newtop_noplenumZ``
(Plot3D 2-D, single block, x and y in chord units; the contoured top wall
accounts for the tunnel blockage).  Boundary map (from the .nmf files):
j-min viscous wall (floor + hump), j-max inviscid wall (tangency),
i-min subsonic inflow at x/c = -6.39, i-max outflow at x/c = 4.

Flow: chord c = 1, U_ref = 1, Re_c = 936,000 (incompressible here; the
experiment is at M = 0.1).  The boundary layer develops naturally from the
inflow so that it is about 0.074 c thick at x/c = -2.14 as in the
experiment.  The sealed experimental data (Cp, Cf, PIV profiles) is never
read by this module.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..casegen import render_turbulence_properties
from ..spec import CandidateSpec
from .structured_mesh import read_p2dfmt, write_polymesh_2d

RE_C = 936_000.0
U_REF = 1.0
CHORD = 1.0
NU = U_REF * CHORD / RE_C
THICKNESS = 0.1
END_TIME = 8000
GRIDS = {
    "coarse": "hump2newtop_noplenumZ103x28.p2dfmt.gz",
    "medium": "hump2newtop_noplenumZ205x55.p2dfmt.gz",
    "fine": "hump2newtop_noplenumZ409x109.p2dfmt.gz",
    "finest": "hump2newtop_noplenumZ1633x433.p2dfmt.gz",
}
PATCHES = {"imin": "inlet:patch", "imax": "outlet:patch", "jmin": "hump:wall",
           "jmax": "top:patch", "z": "frontAndBack"}


def _header(cls: str, obj: str, location: str = "") -> str:
    loc = f'    location    "{location}";\n' if location else ""
    return (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n{loc}    object      {obj};\n}}\n\n"
    )


def control_dict(end_time: int) -> str:
    return _header("dictionary", "controlDict", "system") + f"""
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   {end_time};
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    wallShear
    {{
        type            wallShearStress;
        libs            (fieldFunctionObjects);
        patches         (hump);
        writeControl    writeTime;
    }}
}}
"""


def fv_schemes() -> str:
    return _header("dictionary", "fvSchemes", "system") + """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default                         Gauss linear;
    div(phi,U)                      Gauss linearUpwind grad(U);
    div(phi,k)                      Gauss upwind;
    div(phi,omega)                  Gauss upwind;
    div((nuEff*dev2(T(grad(U)))))   Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
"""


def fv_solution() -> str:
    return _header("dictionary", "fvSolution", "system") + """
solvers
{
    p
    {
        solver          GAMG;
        smoother        DICGaussSeidel;
        tolerance       1e-10;
        relTol          0.05;
    }
    "(U|k|omega)"
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-10;
        relTol          0;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;
}

relaxationFactors
{
    equations
    {
        U               0.7;
        "(k|omega)"     0.7;
        p               0.3;
    }
}
"""


def _field(cls: str, name: str, dims: str, internal: str, bcs: dict[str, str]) -> str:
    body = "".join(f"    {patch}\n    {{\n{text}\n    }}\n" for patch, text in bcs.items())
    return _header(cls, name, "0") + f"""
dimensions      {dims};

internalField   {internal};

boundaryField
{{
{body}}}
"""


def initial_fields() -> dict[str, str]:
    # free-stream turbulence: 0.5% intensity, length scale 0.05 c
    k_inf = 1.5 * (0.005 * U_REF) ** 2
    omega_inf = k_inf ** 0.5 / (0.09 ** 0.25 * 0.05 * CHORD)
    return {
        "U": _field("volVectorField", "U", "[0 1 -1 0 0 0 0]", f"uniform ({U_REF} 0 0)", {
            "inlet": f"        type            fixedValue;\n        value           uniform ({U_REF} 0 0);",
            "outlet": "        type            inletOutlet;\n        inletValue      uniform (0 0 0);\n        value           uniform (1 0 0);",
            "hump": "        type            noSlip;",
            "top": "        type            slip;",
            "frontAndBack": "        type            empty;",
        }),
        "p": _field("volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0", {
            "inlet": "        type            zeroGradient;",
            "outlet": "        type            fixedValue;\n        value           uniform 0;",
            "hump": "        type            zeroGradient;",
            "top": "        type            zeroGradient;",
            "frontAndBack": "        type            empty;",
        }),
        "k": _field("volScalarField", "k", "[0 2 -2 0 0 0 0]", f"uniform {k_inf}", {
            "inlet": f"        type            fixedValue;\n        value           uniform {k_inf};",
            "outlet": f"        type            inletOutlet;\n        inletValue      uniform {k_inf};\n        value           uniform {k_inf};",
            "hump": ("        type            kLowReWallFunction;\n        Ceps2           1.9;\n"
                     "        Ck              -0.416;\n        Bk              8.366;\n"
                     "        C               11;\n        value           $internalField;"),
            "top": "        type            zeroGradient;",
            "frontAndBack": "        type            empty;",
        }),
        "omega": _field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", f"uniform {omega_inf}", {
            "inlet": f"        type            fixedValue;\n        value           uniform {omega_inf};",
            "outlet": f"        type            inletOutlet;\n        inletValue      uniform {omega_inf};\n        value           uniform {omega_inf};",
            "hump": "        type            omegaWallFunction;\n        value           $internalField;",
            "top": "        type            zeroGradient;",
            "frontAndBack": "        type            empty;",
        }),
        "nut": _field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0", {
            "inlet": "        type            calculated;\n        value           uniform 0;",
            "outlet": "        type            calculated;\n        value           uniform 0;",
            "hump": ("        type            nutLowReWallFunction;\n        Cmu             0.09;\n"
                     "        kappa           0.41;\n        E               9.8;\n"
                     "        value           uniform 0;"),
            "top": "        type            calculated;\n        value           uniform 0;",
            "frontAndBack": "        type            empty;",
        }),
    }


def make_case(root: Path, grid_file: Path, spec: CandidateSpec | None, end_time: int = END_TIME) -> dict:
    """Write a complete hump case from a NASA TMR 2-D Plot3D grid."""
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    (root / "system").mkdir(parents=True)
    (root / "constant").mkdir()
    (root / "0").mkdir()
    x, y = read_p2dfmt(Path(grid_file))
    counts = write_polymesh_2d(x, y, root, THICKNESS, PATCHES)
    (root / "system" / "controlDict").write_text(control_dict(end_time))
    (root / "system" / "fvSchemes").write_text(fv_schemes())
    (root / "system" / "fvSolution").write_text(fv_solution())
    (root / "constant" / "transportProperties").write_text(
        _header("dictionary", "transportProperties", "constant")
        + f"transportModel  Newtonian;\nnu              [0 2 -1 0 0 0 0] {NU};\n"
    )
    (root / "constant" / "turbulenceProperties").write_text(
        render_turbulence_properties(spec, mode="expressions")
    )
    if spec is not None:
        control = root / "system" / "controlDict"
        control.write_text(control.read_text() + '\nlibs            ("libkOmegaSSTBasis.so");\n')
    for name, text in initial_fields().items():
        (root / "0" / name).write_text(text)
    return {"grid": str(grid_file), "ni": int(x.shape[1]), "nj": int(x.shape[0]),
            "x_range": (float(x.min()), float(x.max())), "y_range": (float(y.min()), float(y.max())),
            "patch_faces": counts}
