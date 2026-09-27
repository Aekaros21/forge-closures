"""NASA FAITH hill: a wall-mounted axisymmetric hill in a turbulent boundary layer.

Geometry (NASA TMR, Geometry_files/README and the FISF readme):
    h(r) = 3 cos(pi r / 9) + 3   inches,  r = radial distance from the centre,
so a 6 inch hill on an 18 inch base. Origin at the centre of the model base.

Conditions (FISF readme, model runs 2009-09-02): nominal tunnel dynamic
pressure 6.051 inH2O, i.e. a free-stream velocity of about 163 ft/s = 49.7 m/s,
ambient 23 C so nu = 1.55e-5 m2/s. With H = 0.1524 m that gives Re_H = 4.9e5,
matching the 5.0e5 quoted for this case in the literature.

The mesh is a Cartesian block whose points are then displaced vertically onto
the hill (terrain-following), so the topology comes from blockMesh and stays
valid while the wall follows the analytic surface exactly. Only half the span
is modelled, with a symmetry plane on the centreline.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .. import runner
from ..spec import CandidateSpec
from ..casegen import render_turbulence_properties
from . import hump  # reuse the shared dictionary writers

INCH = 0.0254
H = 6.0 * INCH               # hill height, 0.1524 m
R_BASE = 9.0 * INCH          # hill base radius, 0.2286 m
U_REF = 49.68                # m/s, 163 ft/s
NU = 1.55e-5                 # m2/s at 23 C
RE_H = U_REF * H / NU

# domain, in hill heights
X_MIN, X_MAX = -8.0, 12.0
Y_TOP = 6.0
Z_MAX = 5.0                  # half span; the tunnel is 32 in wide = 5.3 H


def hill_height(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Analytic surface: h = 3 cos(pi r / 9) + 3 inches, zero beyond r = 9 in."""
    r = np.hypot(x, z)
    h = 3.0 * np.cos(np.pi * np.minimum(r, R_BASE) / (9.0 * INCH)) + 3.0
    return np.where(r <= R_BASE, h * INCH, 0.0)


def block_mesh_dict(nx: int, ny: int, nz: int, y_grading: float, x_grading: float) -> str:
    x0, x1 = X_MIN * H, X_MAX * H
    y1 = Y_TOP * H
    z1 = Z_MAX * H
    verts = [(x0, 0, 0), (x1, 0, 0), (x1, y1, 0), (x0, y1, 0),
             (x0, 0, z1), (x1, 0, z1), (x1, y1, z1), (x0, y1, z1)]
    vtxt = "\n".join(f"    ({a} {b} {c})" for a, b, c in verts)
    return hump._header("dictionary", "blockMeshDict", "system") + f"""
scale 1;

vertices
(
{vtxt}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz})
    simpleGrading ({x_grading} {y_grading} 1)
);

edges ();

boundary
(
    inlet     {{ type patch;         faces ((0 4 7 3)); }}
    outlet    {{ type patch;         faces ((1 2 6 5)); }}
    bottom    {{ type wall;          faces ((0 1 5 4)); }}
    top       {{ type patch;         faces ((3 7 6 2)); }}
    symmetry  {{ type symmetryPlane; faces ((0 3 2 1)); }}
    side      {{ type patch;         faces ((4 5 6 7)); }}
);

mergePatchPairs ();
"""


def displace_onto_hill(case: Path) -> tuple[int, float]:
    """Move the block's points onto the hill, keeping the top boundary flat."""
    pfile = case / "constant" / "polyMesh" / "points"
    text = pfile.read_text()
    start = text.index("(", text.index("\n(", text.index("points")) if False else text.index("\n(")) + 1
    end = text.rindex(")")
    body = text[start:end]
    vals = np.fromstring(body.replace("(", " ").replace(")", " "), sep=" ")
    pts = vals.reshape(-1, 3)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    h = hill_height(x, z)
    y_top = Y_TOP * H
    pts[:, 1] = h + (y_top - h) * (y / y_top)
    lines = "\n".join(f"({a:.9g} {b:.9g} {c:.9g})" for a, b, c in pts)
    pfile.write_text(text[:start] + "\n" + lines + "\n" + text[end:])
    return len(pts), float(h.max())


def boundary_fields(spec: CandidateSpec | None) -> dict[str, str]:
    """Inlet: uniform free stream; a turbulent boundary layer develops over the
    8 hill-heights of floor upstream of the hill. Top and side are slip, so the
    tunnel walls impose no boundary layer of their own; the centreline is a
    symmetry plane."""
    nut_in = 1.0e-3 * NU
    k_in = 1.5 * (0.005 * U_REF) ** 2          # 0.5% free-stream turbulence
    omega_in = k_in ** 0.5 / (0.09 ** 0.25 * 0.05 * H)
    walls = ("        type            {t};\n"
             "        value           uniform {v};\n")

    def field(cls, obj, dims, internal, bottom, inlet_val):
        return hump._header(cls, obj, "0") + f"""
dimensions      {dims};
internalField   uniform {internal};

boundaryField
{{
    inlet     {{ type fixedValue; value uniform {inlet_val}; }}
    outlet    {{ type zeroGradient; }}
    bottom    {{ {bottom} }}
    top       {{ type slip; }}
    side      {{ type slip; }}
    symmetry  {{ type symmetryPlane; }}
}}
"""

    return {
        "U": field("volVectorField", "U", "[0 1 -1 0 0 0 0]", f"({U_REF} 0 0)",
                   "type noSlip;", f"({U_REF} 0 0)"),
        "p": hump._header("volScalarField", "p", "0") + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet     { type zeroGradient; }
    outlet    { type fixedValue; value uniform 0; }
    bottom    { type zeroGradient; }
    top       { type slip; }
    side      { type slip; }
    symmetry  { type symmetryPlane; }
}
""",
        "k": field("volScalarField", "k", "[0 2 -2 0 0 0 0]", f"{k_in}",
                   f"type kLowReWallFunction; value uniform {k_in};", f"{k_in}"),
        "omega": field("volScalarField", "omega", "[0 0 -1 0 0 0 0]", f"{omega_in}",
                       f"type omegaWallFunction; value uniform {omega_in};", f"{omega_in}"),
        "nut": field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", f"{nut_in}",
                     "type nutLowReWallFunction; value uniform 0;", f"{nut_in}"),
    }


def write_case(root: Path, spec: CandidateSpec | None, nx: int = 260, ny: int = 90,
               nz: int = 80, end_time: int = 12000, y_grading: float = 4000.0,
               x_grading: float = 1.0) -> dict:
    """Write every case file except the mesh. No OpenFOAM utility is run.

    The mesh is produced afterwards by ``blockMesh`` and then moved onto the hill by
    ``displace_onto_hill``; the V3 evaluator runs both on the compute node.
    """
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    for sub in ("system", "constant", "0"):
        (root / sub).mkdir(parents=True)
    for name, value in (("nx", nx), ("ny", ny), ("nz", nz)):
        if int(value) < 4:
            raise ValueError(f"FAITH mesh needs at least 4 cells in {name}")
    (root / "system" / "blockMeshDict").write_text(block_mesh_dict(nx, ny, nz, y_grading, x_grading))
    control = hump.control_dict(end_time)
    if spec is not None:
        control += '\nlibs            ("libkOmegaSSTBasis.so");\n'
    (root / "system" / "controlDict").write_text(control)
    (root / "system" / "fvSchemes").write_text(hump.fv_schemes())
    (root / "system" / "fvSolution").write_text(hump.fv_solution())
    (root / "constant" / "transportProperties").write_text(
        hump._header("dictionary", "transportProperties", "constant")
        + f"\ntransportModel  Newtonian;\nnu              {NU};\n")
    (root / "constant" / "turbulenceProperties").write_text(
        render_turbulence_properties(spec, mode="expressions"))
    for name, text in boundary_fields(spec).items():
        (root / "0" / name).write_text(text)
    return {"cells": nx * ny * nz, "mesh": {"nx": nx, "ny": ny, "nz": nz, "y_grading": y_grading,
            "x_grading": x_grading}, "re_h": RE_H, "u_ref": U_REF,
            "post_mesh": "displace_onto_hill"}


def make_case(root: Path, spec: CandidateSpec | None, nx: int = 260, ny: int = 90,
              nz: int = 80, end_time: int = 12000, y_grading: float = 4000.0,
              x_grading: float = 1.0) -> dict:
    """Standalone builder: write the case, then mesh it locally (the V3 evaluator does not use this)."""
    root = Path(root)
    info = write_case(root, spec, nx, ny, nz, end_time, y_grading, x_grading)
    env = runner.foam_env()
    with open(root / "log.blockMesh", "w") as log:
        subprocess.run(["blockMesh"], cwd=root, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    n_points, h_max = displace_onto_hill(root)
    with open(root / "log.checkMesh", "w") as log:
        subprocess.run(["checkMesh", "-constant"], cwd=root, env=env, stdout=log,
                       stderr=subprocess.STDOUT)
    info.update(points=n_points, hill_peak_m=h_max)
    return info


# Separation and reattachment on the centreline, from the NASA fringe-imaging skin-friction data
# (tedp-phase4 FAITH study): x/H measured from the hill centre. The FISF magnitude normalisation is
# non-standard, so only the zero crossings are used.
MEASURED_SEPARATION_XH = 0.38
MEASURED_REATTACHMENT_XH = 1.72


def centreline_crossings(x_over_h: np.ndarray, cf: np.ndarray, lo: float = 0.0, hi: float = 6.0) -> list[dict]:
    """Sign changes of the streamwise wall shear along the centreline, in [lo, hi] hill heights."""
    crossings = []
    for i in range(len(x_over_h) - 1):
        a, b = cf[i], cf[i + 1]
        if a * b < 0 and lo < x_over_h[i] < hi:
            x0 = float(x_over_h[i] - a * (x_over_h[i + 1] - x_over_h[i]) / (b - a))
            crossings.append({"x_over_h": x0, "kind": "separation" if a > 0 else "reattachment"})
    return crossings


def bubble_errors(crossings: list[dict]) -> dict:
    """Position errors of the first lee-side separation and the reattachment that follows it.

    A model that predicts no bubble is charged the whole measured bubble length for both points,
    so that a missed separation can never score better than a mislocated one.
    """
    missed = MEASURED_REATTACHMENT_XH - MEASURED_SEPARATION_XH
    seps = [c["x_over_h"] for c in crossings if c["kind"] == "separation"]
    if not seps:
        return {"separation_position_error": missed, "reattachment_position_error": missed,
                "separation_x_over_h": None, "reattachment_x_over_h": None}
    sep = seps[0]
    reatt = [c["x_over_h"] for c in crossings if c["kind"] == "reattachment" and c["x_over_h"] > sep]
    return {"separation_position_error": abs(sep - MEASURED_SEPARATION_XH),
            "reattachment_position_error": abs(reatt[0] - MEASURED_REATTACHMENT_XH) if reatt else missed,
            "separation_x_over_h": sep, "reattachment_x_over_h": reatt[0] if reatt else None}


# ---------------------------------------------------------------------------
# scoring against the NASA PIV centreline data
# ---------------------------------------------------------------------------
PIV_ROOT = Path(__file__).resolve().parents[3] / "cases" / "faith" / "PIV_data"
PIV_SET = "Centerline_FAITH_2Hz_4000samps_scalar"


def read_piv(name: str) -> np.ndarray:
    """One Tecplot-point PIV file as (n, 3): x[mm], y[mm], value.

    x = 0 at the hill centroid and y = 0 at the floor (PIV_data/README); values
    are m/s for velocities and (m/s)^2 for stresses. Zero entries are the mask
    over the model and outside the light sheet, and are dropped.
    """
    rows = []
    for line in (PIV_ROOT / PIV_SET / name).read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            continue
    table = np.asarray(rows, dtype=np.float64)
    return table[table[:, 2] != 0.0]


def score_centreline(case: Path, time: str) -> dict:
    """Compare a run with the PIV centreline plane.

    The run is sampled at the PIV points on the symmetry plane; the error is
    reported in the same form as the frozen composite's velocity term, an RMS
    over the measured points normalised by the free-stream velocity, so it is
    comparable across models even though this case is outside the frozen
    objective.
    """
    from fluidfoam import readmesh, readvector
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    x, y, z = readmesh(str(case), verbose=False)
    u = np.asarray(readvector(str(case), time, "U", verbose=False)).T
    plane = z < z.min() * 1.5 + 1.0e-6          # first cell layer off the symmetry plane
    pts = np.stack([x[plane], y[plane]], axis=1)
    out: dict[str, float] = {}
    for label, fname, comp in (("u", "U_mean_axis00.dat", 0), ("v", "V_mean_axis00.dat", 1)):
        piv = read_piv(fname)
        target = np.stack([piv[:, 0] / 1000.0, piv[:, 1] / 1000.0], axis=1)
        linear = LinearNDInterpolator(pts, u[plane, comp])
        model = linear(target)
        bad = np.isnan(model)
        if bad.any():
            model[bad] = NearestNDInterpolator(pts, u[plane, comp])(target[bad])
        out[f"rmse_{label}"] = float(np.sqrt(np.mean((model - piv[:, 2]) ** 2))) / U_REF
        out[f"n_{label}"] = int(len(piv))
    out["composite_velocity"] = float(np.hypot(out["rmse_u"], out["rmse_v"]))
    return out
