"""Fully developed turbulent flow in rectangular ducts (verification-only, genuinely 3D).

Reference: DNS of Vinuesa et al. (KTH; Phys. Rev. Fluids 3, 054606, 2018; J. Turbul. 15, 677, 2014),
public profiles from https://www.mech.kth.se/~rvinuesa/DuctData/ for aspect ratios AR = 1, 3, 5, 7 at
Re_tau ~ 180 and AR = 1, 3 at Re_tau ~ 360. Statistics are given on the lower-left quadrant
(y in [-1, 0], z in [-AR, 0], lengths in duct half-heights h), velocities scaled with the bulk velocity.

The RANS case reuses the square-duct template of the development set (same numerics, the same
meanVelocityForce drive with U_b = 0.482 and half-height h = 0.5), with a rectangular cross-section
and the viscosity set from the DNS bulk Reynolds number Re_b = U_b h / nu. The reference is written in
the format of ``duct_scoring`` so the development duct scoring applies unchanged.

Wall friction: for AR >= 3 the published friction-velocity rows are exactly half of the value implied
by the DNS wall-velocity gradient (for AR = 1 they agree to 0.4 %), so the wall shear is derived here
from the velocity gradient for every aspect ratio, as it was for the development square ducts.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

U_BULK = 0.482      # m/s, the square-duct template's meanVelocityForce target
H = 0.5             # m, duct half-height (the short half-side)


def _rows(path: Path) -> tuple[list[np.ndarray], dict]:
    rows, meta = [], {}
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("%"):
            match = re.match(r"%\s*([A-Za-z_,]+)\s*=\s*([-0-9.eE+]+)", line)
            if match:
                meta[match.group(1)] = float(match.group(2))
            continue
        rows.append(np.array([float(v) for v in line.split()]))
    return rows, meta


def load_kth(folder: Path, aspect_ratio: int, re_label: int) -> dict:
    """Quadrant statistics of one KTH case: coordinates in h, velocities in U_b, stresses in U_b^2."""
    folder = Path(folder)
    tag = f"{aspect_ratio}_{re_label}"
    z_rows, _ = _rows(folder / f"zcoord_{tag}.prof.txt")
    z = z_rows[1]
    y = z_rows[2] if aspect_ratio == 1 else _rows(folder / f"ycoord_{tag}.prof.txt")[0][0]
    fields, meta = {}, {}
    for name in ("U", "V", "W", "uu", "vv", "ww", "uv", "uw", "vw"):
        rows, info = _rows(folder / f"{name}_{tag}.prof.txt")
        meta.update(info)
        table = np.asarray(rows)
        if table.shape != (len(y), len(z)):
            raise ValueError(f"{name}_{tag}: shape {table.shape} does not match y x z = {len(y)} x {len(z)}")
        fields[name] = table
    if not (np.isclose(y.min(), -1) and np.isclose(y.max(), 0) and np.isclose(z.min(), -aspect_ratio)
            and np.isclose(z.max(), 0)):
        raise ValueError(f"{tag}: unexpected quadrant extent")
    return {"y": y, "z": z, "fields": fields, "meta": meta, "aspect_ratio": aspect_ratio}


def wall_gradient_cf(coordinate: np.ndarray, profile_rows: np.ndarray, re_b: float) -> np.ndarray:
    """cf = 2 / Re_b * dU*/dn* at the wall, second-order one-sided, for each wall-parallel station.

    ``coordinate`` runs away from the wall (distance from the wall in h, starting at 0), and
    ``profile_rows[k]`` is U/U_b at distance coordinate[k] for every station along the wall.
    """
    d1, d2 = coordinate[1] - coordinate[0], coordinate[2] - coordinate[0]
    u1, u2 = profile_rows[1], profile_rows[2]
    # U(0) = 0: fit U = a n + b n^2 through the first two interior points
    slope = (u1 * d2 * d2 - u2 * d1 * d1) / (d1 * d2 * (d2 - d1))
    return 2.0 / re_b * slope


def build_reference(folder: Path, aspect_ratio: int, re_label: int) -> dict:
    """Reference in the ``duct_scoring`` format, in the RANS case's dimensional units."""
    data = load_kth(folder, aspect_ratio, re_label)
    y, z, f, re_b = data["y"], data["z"], data["fields"], data["meta"]["Reb"]
    yy, zz = np.meshgrid(y, z, indexing="ij")
    points = np.stack([yy.ravel() * H, zz.ravel() * H], axis=1)
    u = np.stack([f["U"].ravel(), f["V"].ravel(), f["W"].ravel()], axis=1) * U_BULK
    nu = U_BULK * H / re_b
    # lower wall y = -h: stations along z (the published friction data refer to this wall)
    cf_low = wall_gradient_cf(y - y.min(), f["U"], re_b)
    # side wall z = -AR h: stations along y
    cf_side = wall_gradient_cf(z - z.min(), f["U"].T, re_b)
    wall_points = np.concatenate([
        np.stack([np.zeros_like(z), np.full_like(z, -H), z * H], axis=1),
        np.stack([np.zeros_like(y), y * H, np.full_like(y, -aspect_ratio * H)], axis=1)])
    labels = np.array(["y_min"] * len(z) + ["z_min"] * len(y), dtype="<U5")
    return {
        "points": points, "u": u,
        "tau_xy": f["uv"].ravel() * U_BULK ** 2, "tau_xz": f["uw"].ravel() * U_BULK ** 2,
        "u_bulk": U_BULK, "nu": nu, "wall_points": wall_points, "wall_labels": labels,
        "cf": np.concatenate([cf_low, cf_side]), "x_station": 0.0, "nu_dns_file": nu,
        "re_b": re_b, "re_tau": data["meta"].get("Retau", np.nan), "aspect_ratio": float(aspect_ratio),
    }


def grading_cells(h: float = H, n: int = 48, ratio: float = 0.12) -> tuple[float, float]:
    """Wall and centre cell sizes of the template's geometric half-side grading."""
    q = ratio ** (1.0 / (n - 1))
    first = h * (1 - q) / (1 - q ** n)
    return first * ratio, first            # wall cell, centre-side cell


def block_mesh_dict(aspect_ratio: float, n_wall: int = 48, ratio: float = 0.12, nx: int = 1,
                    length: float = 2.5) -> str:
    """Rectangular cross-section, four quadrant blocks, streamwise cyclic, walls as the default patch.

    The short direction (y, half-height h) keeps the development square duct's 48-cell grading
    towards each wall. The long direction (z, half-width AR h) keeps the same near-wall segment
    over a distance h, followed by a uniform core whose cells match the centre-side cell size.
    """
    ar = float(aspect_ratio)
    if ar < 1:
        raise ValueError("aspect ratio must be at least 1")
    _, centre = grading_cells(H, n_wall, ratio)
    n_core = 0 if ar == 1 else max(1, int(round((ar - 1) * H / centre)))
    nz = n_wall + n_core
    inv = 1.0 / ratio
    w = ar * H
    if n_core:
        wall_frac = 1.0 / ar
        # blocks whose local z axis runs from the wall at -w to the centre (grow away from the wall)
        z_from_wall = (f"(({wall_frac:.9g} {n_wall / nz:.9g} {inv:.9g}) "
                       f"({1 - wall_frac:.9g} {n_core / nz:.9g} 1))")
        # blocks whose local z axis runs from the centre to the wall at +w (shrink toward the wall)
        z_to_wall = (f"(({1 - wall_frac:.9g} {n_core / nz:.9g} 1) "
                     f"({wall_frac:.9g} {n_wall / nz:.9g} {ratio:.9g}))")
    else:
        z_from_wall, z_to_wall = f"{inv:.9g}", f"{ratio:.9g}"
    L, h = length, H
    verts = [(0, 0, -w), (0, 0, 0), (0, h, 0), (0, h, -w), (0, 0, w), (0, h, w),
             (0, -h, -w), (0, -h, 0), (0, -h, w)]
    verts += [(L, a, b) for _, a, b in verts]
    vtxt = "\n".join(f"    ({a:.9g} {b:.9g} {c:.9g})   // {i}" for i, (a, b, c) in enumerate(verts))
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// Rectangular duct, aspect ratio {ar:g}: y in [-{h:g}, {h:g}] m, z in [-{w:g}, {w:g}] m.
// y: {n_wall} cells per half, geometric wall grading {ratio:g} (development square duct);
// z: {n_wall} graded cells over {h:g} m at each wall plus {n_core} uniform core cells per half.
scale 1;

vertices
(
{vtxt}
);

blocks
(
    hex (0 3 2 1  9 12 11 10) ({n_wall} {nz} {nx}) simpleGrading ({ratio:.9g} {z_from_wall} 1)
    hex (1 2 5 4  10 11 14 13) ({n_wall} {nz} {nx}) simpleGrading ({ratio:.9g} {z_to_wall} 1)
    hex (6 0 1 7  15 9 10 16) ({n_wall} {nz} {nx}) simpleGrading ({inv:.9g} {z_from_wall} 1)
    hex (7 1 4 8  16 10 13 17) ({n_wall} {nz} {nx}) simpleGrading ({inv:.9g} {z_to_wall} 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type cyclic;
        neighbourPatch outlet;
        faces
        (
            (0 3 2 1)
            (1 2 5 4)
            (6 0 1 7)
            (7 1 4 8)
        );
    }}
    outlet
    {{
        type cyclic;
        neighbourPatch inlet;
        faces
        (
            (9 12 11 10)
            (10 11 14 13)
            (15 9 10 16)
            (16 10 13 17)
        );
    }}
);

mergePatchPairs
(
);

defaultPatch
{{
    name walls;
    type wall;
}}
"""


def cells(aspect_ratio: float, n_wall: int = 48, ratio: float = 0.12) -> int:
    _, centre = grading_cells(H, n_wall, ratio)
    n_core = 0 if aspect_ratio == 1 else max(1, int(round((aspect_ratio - 1) * H / centre)))
    return 2 * n_wall * 2 * (n_wall + n_core)
