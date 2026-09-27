"""Spanwise-rotating plane channel (AGARD PCH22, Kristoffersen & Andersson 1993).

Definition (from the paper): fully developed pressure-driven channel flow,
Re = U_m h / nu = 2900 with h the channel half-width and U_m the bulk
velocity, rotation number Ro = 2 |Omega| h / U_m about the spanwise axis,
Ro in {0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.50}; the plan scores Ro = 0.10 and
0.50.  Re_tau is about 194 at Ro = 0.

Case: streamwise-periodic channel of half-width h = 1 and bulk velocity
U_m = 1 (so nu = 1/2900), driven by ``meanVelocityForce``; the rotating
frame's Coriolis force enters through ``tabulatedAccelerationSource`` with
a constant angular velocity Omega = (0, 0, Ro/2).  The source also adds the
centrifugal term Omega x (Omega x r); with the streamwise extent centred on
x = 0 and only 0.01 h long, its streamwise component is below 1e-6 of the
driving force and its wall-normal component is absorbed by the pressure.

The wall-normal grid is wall-resolved (256 cells, first cell y+ ~ 0.15).
Grid study with stock SST at Ro 0 (2026-09-07): converged Re_tau 187.3 with
256 cells and 188.3 with 512 (DNS 194); 8,000 iterations are NOT enough on
these meshes (residuals stall near 1e-5 and u_tau is off by up to 10%), so
the protocol runs 40,000 iterations, about one minute per case.  A second
streamwise cell must be avoided: the source's centrifugal term then acts
antisymmetrically on the two cells and shifts the wall shear by ~5%.
Boundary conditions mirror the development hills cases (low-Re wall
functions on k, omega and nut) so a candidate sees the same wall treatment
it was developed with.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..casegen import render_turbulence_properties
from ..spec import CandidateSpec

RE_BULK_HALF = 2900.0          # U_m h / nu
H = 1.0                        # half-width
U_BULK = 1.0
NU = U_BULK * H / RE_BULK_HALF
RO_PLAN = (0.10, 0.50)
RO_ALL = (0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.50)
X_LENGTH = 0.01 * H
Z_LENGTH = 0.1 * H
NX = 1                         # one streamwise cell: fully developed, and no spurious centrifugal x-force
NY = 256
WALL_GRADING = 12.0            # centre-to-wall cell size ratio (per half)
END_TIME = 40000               # the fine wall-resolved mesh needs ~30,000 iterations to converge


def omega_for(ro: float) -> float:
    """Angular velocity about z for a rotation number Ro = 2 Omega h / U_m."""
    return ro * U_BULK / (2.0 * H)


def _header(cls: str, obj: str, location: str = "") -> str:
    loc = f'    location    "{location}";\n' if location else ""
    return (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n{loc}    object      {obj};\n}}\n\n"
    )


def block_mesh_dict(ny: int = NY, wall_grading: float = WALL_GRADING) -> str:
    x0, x1 = -X_LENGTH / 2, X_LENGTH / 2
    return _header("dictionary", "blockMeshDict", "system") + f"""
scale 1;

vertices
(
    ({x0} {-H} 0)
    ({x1} {-H} 0)
    ({x1} {H} 0)
    ({x0} {H} 0)
    ({x0} {-H} {Z_LENGTH})
    ({x1} {-H} {Z_LENGTH})
    ({x1} {H} {Z_LENGTH})
    ({x0} {H} {Z_LENGTH})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({NX} {ny} 1)
    simpleGrading
    (
        1
        (
            (0.5 0.5 {wall_grading})
            (0.5 0.5 {1.0 / wall_grading})
        )
        1
    )
);

edges ();

boundary
(
    bottomWall {{ type wall; faces ((0 1 5 4)); }}
    topWall    {{ type wall; faces ((3 7 6 2)); }}
    inlet
    {{
        type cyclic;
        neighbourPatch outlet;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type cyclic;
        neighbourPatch inlet;
        faces ((1 2 6 5));
    }}
    frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);

mergePatchPairs ();
"""


def fv_options(ro: float) -> str:
    return _header("dictionary", "fvOptions", "constant") + f"""
momentumSource
{{
    type            meanVelocityForce;
    selectionMode   all;
    fields          (U);
    Ubar            ({U_BULK} 0 0);
}}

frameRotation
{{
    type            tabulatedAccelerationSource;
    selectionMode   all;
    timeDataFileName "constant/acceleration-terms.dat";
}}
"""


def acceleration_table(ro: float) -> str:
    w = omega_for(ro)
    return (
        "(\n"
        f"    (0 ((0 0 0) (0 0 {w}) (0 0 0)))\n"
        f"    (1e9 ((0 0 0) (0 0 {w}) (0 0 0)))\n"
        ")\n"
    )


def transport_properties(nu: float = NU) -> str:
    return _header("dictionary", "transportProperties", "constant") + (
        f"transportModel  Newtonian;\nnu              [0 2 -1 0 0 0 0] {nu};\n"
    )


def control_dict(end_time: int = END_TIME) -> str:
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
        patches         (bottomWall topWall);
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
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
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


def _field(cls: str, name: str, dims: str, internal: str, walls: str, extra: str = "") -> str:
    return _header(cls, name, "0") + f"""
dimensions      {dims};

internalField   {internal};

boundaryField
{{
    bottomWall
    {{
{walls}
    }}
    topWall
    {{
{walls}
    }}
    inlet   {{ type cyclic; }}
    outlet  {{ type cyclic; }}
    frontAndBack {{ type empty; }}
{extra}}}
"""


def initial_fields() -> dict[str, str]:
    u_tau = 194.0 * NU / H
    k0 = 1.5 * (0.05 * U_BULK) ** 2
    omega0 = k0 ** 0.5 / (0.09 ** 0.25 * 0.1 * H)
    return {
        "U": _field("volVectorField", "U", "[0 1 -1 0 0 0 0]", f"uniform ({U_BULK} 0 0)",
                    "        type            fixedValue;\n        value           uniform (0 0 0);"),
        "p": _field("volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0",
                    "        type            zeroGradient;"),
        "k": _field("volScalarField", "k", "[0 2 -2 0 0 0 0]", f"uniform {k0}",
                    "        type            kLowReWallFunction;\n        Ceps2           1.9;\n"
                    "        Ck              -0.416;\n        Bk              8.366;\n"
                    "        C               11;\n        value           $internalField;"),
        "omega": _field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", f"uniform {omega0}",
                        "        type            omegaWallFunction;\n        value           $internalField;"),
        "nut": _field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0",
                      "        type            nutLowReWallFunction;\n        Cmu             0.09;\n"
                      "        kappa           0.41;\n        E               9.8;\n"
                      "        value           uniform 0;"),
    }


def make_case(
    root: Path, ro: float, spec: CandidateSpec | None, end_time: int = END_TIME,
    frame_rotation: bool = True, ny: int = NY, wall_grading: float = WALL_GRADING,
    re_bulk_half: float = RE_BULK_HALF,
) -> Path:
    """Write a complete rotating-channel case; ``spec`` None means stock SST.

    ``frame_rotation`` makes a candidate's rotation tensor include the frame
    angular velocity (absolute vorticity), the pre-declared Phase-4 protocol;
    it has no effect on stock SST, which is rotation-blind either way.
    """
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    (root / "system").mkdir(parents=True)
    (root / "constant").mkdir()
    (root / "0").mkdir()
    (root / "system" / "blockMeshDict").write_text(block_mesh_dict(ny, wall_grading))
    (root / "system" / "controlDict").write_text(control_dict(end_time))
    (root / "system" / "fvSchemes").write_text(fv_schemes())
    (root / "system" / "fvSolution").write_text(fv_solution())
    (root / "constant" / "fvOptions").write_text(fv_options(ro))
    (root / "constant" / "acceleration-terms.dat").write_text(acceleration_table(ro))
    (root / "constant" / "transportProperties").write_text(transport_properties(U_BULK * H / re_bulk_half))
    turbulence = render_turbulence_properties(spec, mode="expressions")
    if spec is not None and frame_rotation:
        turbulence = turbulence.replace(
            "        gMax            ",
            f"        frameOmega      (0 0 {omega_for(ro)});\n        gMax            ", 1,
        )
    (root / "constant" / "turbulenceProperties").write_text(turbulence)
    if spec is not None:
        control = root / "system" / "controlDict"
        control.write_text(control.read_text() + '\nlibs            ("libkOmegaSSTBasis.so");\n')
    for name, text in initial_fields().items():
        (root / "0" / name).write_text(text)
    (root / "case.ro").write_text(f"{ro}\n")
    return root
