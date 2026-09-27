"""Ahmed body with 25 and 35 degree rear slants (verification only, genuinely 3D).

Geometry: Ahmed, Ramm and Faltin (1984, SAE 840300), length 1044 mm, width 389 mm, height
288 mm, 50 mm ground clearance, rear slant 222 mm long. The front part (first 100 mm) is the
original DLR surface published with ERCOFTAC case 82 (``ahmed-front-geo.dat``: a structured
100 x 200 grid over one quarter of the front outside the flat centre of the nose); it is
triangulated as published, its two outer cut lines are extruded to the body's mid-planes and the
flat centre closes the nose. The body behind it is prismatic with sharp longitudinal edges.
Four cylindrical stilts of 30 mm diameter carry the body; their positions (202.5 mm and
672.5 mm behind the nose, 163.5 mm either side of the centre plane) follow the usual reading of
the original drawing and are not part of the ERCOFTAC files.

Reference: Lienhart and Becker (2003, SAE 2003-01-0656), ERCOFTAC case 82: LDA mean velocity
on the symmetry plane over the slant and in the near wake, on cross planes behind the body, and
surface pressure over the rear. Coordinates as in the data: x = 0 at the rear end of the body,
y = 0 on the symmetry plane, z = 0 on the ground; U = 40 m/s, nu = 1.5e-5 m2/s, Re_H = 7.68e5.

Setup: half model (symmetry plane y = 0), stationary no-slip floor, slip far boundaries for the
3/4-open test section, uniform inflow of 40 m/s with 0.25 % turbulence 1.5 m ahead of the nose,
wall functions on the body, stilts and floor (Spalding's law, valid for any first-cell y+);
snappyHexMesh with prism layers. The resolved-wall treatment of the development cases is not
affordable at this Reynolds number; this is a stated difference, identical for SST and models.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import shutil

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO/'data/assets/references/ahmed'

LENGTH, WIDTH, HEIGHT = 1.044, 0.389, 0.288
CLEARANCE = 0.050
SLANT_LENGTH = 0.222
FRONT_LENGTH = 0.100
U_REF = 40.0
NU = 1.5e-5
RE_H = U_REF*HEIGHT/NU
STILT_DIAMETER = 0.030
STILT_X_FROM_NOSE = (0.2025, 0.6725)
STILT_Y = 0.1635
Z_MID = CLEARANCE + HEIGHT/2
X_NOSE = -LENGTH
TURBULENCE_INTENSITY = 0.0025

# Domain (m) and base background cell; refinement levels halve the cell each level.
DOMAIN = {'x': (-2.56, 3.2), 'y': (0.0, 1.536), 'z': (0.0, 1.536)}


# ---------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------
def front_grid(path: Path = DATA/'ahmed-front-geo.dat') -> np.ndarray:
    """The published front quarter as a (100, 200, 3) grid in metres: x from the nose, y from the
    centre plane, z from the body's mid-height. Ring i runs from the side cut (z = 35 mm) to the
    top cut (y = 80 mm); ring 99 is the end of the front part at x = 100 mm."""
    rows = []
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            rows.append([float(v) for v in line.split()[:3]])
    grid = np.asarray(rows, dtype=float).reshape(100, 200, 3)/1000.0
    if not (np.allclose(grid[:, 0, 2], 0.035) and np.allclose(grid[:, -1, 1], 0.080)):
        raise ValueError('Unexpected front geometry layout')
    # The last ring closes on the prismatic body: put it exactly on the 194.5 x 144 mm section.
    last = grid[-1]
    last[:, 0] = FRONT_LENGTH
    half_w, half_h = WIDTH/2, HEIGHT/2
    on_side = np.abs(last[:, 1]-half_w) <= np.abs(last[:, 2]-half_h)
    last[on_side, 1] = half_w
    last[~on_side, 2] = half_h
    last[:, 1] = np.minimum(last[:, 1], half_w)
    last[:, 2] = np.minimum(last[:, 2], half_h)
    return grid


def _quads(a, b, c, d):
    """Two triangles of the quad a-b-c-d (counter-clockwise seen from outside)."""
    return [(a, b, c), (a, c, d)]


def _strip(p, q):
    """Quads between two polylines of equal length (p[i] -> p[i+1] -> q[i+1] -> q[i])."""
    tris = []
    for i in range(len(p)-1):
        tris += _quads(p[i], p[i+1], q[i+1], q[i])
    return tris


def _front_quarter():
    """Triangles of one front quarter (y >= 0, z >= 0 about mid-height), nose coordinates."""
    g = front_grid()
    tris = []
    for i in range(99):
        tris += _strip(g[i+1], g[i])
    side = g[:, 0]                    # z = 35 mm cut: extrude down to the mid-plane
    side_low = side.copy(); side_low[:, 2] = 0.0
    tris += _strip(side, side_low)
    top = g[:, -1]                    # y = 80 mm cut: extrude to the centre plane
    top_in = top.copy(); top_in[:, 1] = 0.0
    tris += _strip(top_in, top)
    corner = g[0, 0]                  # flat centre of the nose, x = 0
    tris += _quads(np.array([0, 0, 0.0]), np.array([0, corner[1], 0.0]),
                   np.array([0, corner[1], corner[2]]), np.array([0, 0, corner[2]]))
    return tris, g


def _nonzero(tris):
    out = []
    for t in tris:
        a, b, c = (np.asarray(v, dtype=float) for v in t)
        if np.linalg.norm(np.cross(b-a, c-a)) > 1e-14:
            out.append((a, b, c))
    return out


def body_triangles(slant_deg: float, dx: float = 0.02) -> np.ndarray:
    """Closed triangulated surface of the whole body (both sides), outward normals, metres,
    data coordinates (x = 0 rear end, z = 0 ground)."""
    phi = math.radians(slant_deg)
    half_w, half_h = WIDTH/2, HEIGHT/2
    x_slant = -SLANT_LENGTH*math.cos(phi)             # start of the slant on the roof
    z_base_top = HEIGHT/2 - SLANT_LENGTH*math.sin(phi)  # relative to mid-height
    quarter, g = _front_quarter()
    ring = g[-1]                                      # x = 100 mm, from side (z=35) to top (y=80)
    # boundary of the upper-right quarter section at the end of the front part, from the
    # mid-plane on the side up to the top at the centre plane
    side_col = np.array([[FRONT_LENGTH, half_w, 0.0]] + [list(p) for p in ring] + [[FRONT_LENGTH, 0.0, half_h]])
    # prismatic part, x (nose coordinates) from 0.1 m to the rear end at 1.044 m
    xs = np.unique(np.r_[np.arange(FRONT_LENGTH, LENGTH, dx), LENGTH + x_slant, LENGTH])
    xs = xs[xs <= LENGTH + 1e-12]
    side_pts = side_col[side_col[:, 1] >= half_w - 1e-12]        # on the side face: z from 0 up
    top_pts = side_col[side_col[:, 2] >= half_h - 1e-12]         # on the roof: y from half_w to 0
    zs_side = np.unique(np.r_[side_pts[:, 2], half_h])            # 0 .. half_h
    ys_top = np.unique(np.r_[top_pts[:, 1], 0.0, half_w])         # 0 .. half_w
    tris = list(quarter)

    def roof_z(x):   # relative to mid-height, nose coordinates
        return half_h if x <= LENGTH + x_slant + 1e-12 else half_h - (x-(LENGTH+x_slant))*math.tan(phi)

    # side face y = half_w, upper half (z from 0 to the roof line, which drops along the slant)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        f0, f1 = roof_z(x0)/half_h, roof_z(x1)/half_h
        for z0, z1 in zip(zs_side[:-1], zs_side[1:]):
            a = (x0, half_w, z0*f0); b = (x1, half_w, z0*f1); c = (x1, half_w, z1*f1); d = (x0, half_w, z1*f0)
            tris += _quads(a, b, c, d)
    # roof z = half_h up to the slant, then the slant plane down to the base
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys_top[:-1], ys_top[1:]):
            a = (x0, y1, roof_z(x0)); b = (x1, y1, roof_z(x1)); c = (x1, y0, roof_z(x1)); d = (x0, y0, roof_z(x0))
            tris += _quads(a, b, c, d)
    # base x = LENGTH, upper quarter: y from 0 to half_w, z from 0 to z_base_top
    for y0, y1 in zip(ys_top[:-1], ys_top[1:]):
        for z0, z1 in zip(zs_side[:-1], zs_side[1:]):
            f = z_base_top/half_h
            a = (LENGTH, y0, z0*f); b = (LENGTH, y1, z0*f); c = (LENGTH, y1, z1*f); d = (LENGTH, y0, z1*f)
            tris += _quads(a, b, c, d)
    upper = _nonzero(tris)
    # lower quarter: the front is symmetric about mid-height; the lower body has no slant
    lower_tris = list(quarter)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for z0, z1 in zip(zs_side[:-1], zs_side[1:]):
            lower_tris += _quads((x0, half_w, z0), (x1, half_w, z0), (x1, half_w, z1), (x0, half_w, z1))
        for y0, y1 in zip(ys_top[:-1], ys_top[1:]):
            lower_tris += _quads((x0, y1, half_h), (x1, y1, half_h), (x1, y0, half_h), (x0, y0, half_h))
    for y0, y1 in zip(ys_top[:-1], ys_top[1:]):
        for z0, z1 in zip(zs_side[:-1], zs_side[1:]):
            lower_tris += _quads((LENGTH, y0, z0), (LENGTH, y1, z0), (LENGTH, y1, z1), (LENGTH, y0, z1))
    lower = [tuple(np.array([p[0], p[1], -p[2]]) for p in t[::-1]) for t in _nonzero(lower_tris)]
    # The published last ring cuts each section corner diagonally while the prism has a sharp
    # corner: one small forward-facing triangle per corner closes the surface.
    side_last = side_col[side_col[:, 1] >= half_w - 1e-12][-1]
    top_first = side_col[side_col[:, 2] >= half_h - 1e-12][0]
    notch = (side_last, np.array([FRONT_LENGTH, half_w, half_h]), top_first)
    upper = upper + [notch]
    lower = lower + [tuple(np.array([p[0], p[1], -p[2]]) for p in notch)]
    right = upper + lower
    left = [tuple(np.array([p[0], -p[1], p[2]]) for p in t[::-1]) for t in right]
    tris = np.asarray(right + left)
    # nose coordinates -> data coordinates
    tris[..., 0] += X_NOSE
    tris[..., 2] += Z_MID
    return _orient_outward(tris)


def _orient_outward(tris: np.ndarray) -> np.ndarray:
    """Orient every triangle outward. The body is convex in each cross-section, so the outward
    direction is radial from the axis (y = 0, z = mid-height), plus -x over the front part and
    +x on the base; the corner notches at the end of the front face forward."""
    cent = tris.mean(axis=1)
    normal = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0])
    outward = np.c_[np.zeros(len(cent)), cent[:, 1], cent[:, 2]-Z_MID]
    front = cent[:, 0] < X_NOSE + FRONT_LENGTH - 1e-9
    junction = np.isclose(normal[:, 1], 0, atol=1e-15) & np.isclose(normal[:, 2], 0, atol=1e-15) & \
        np.isclose(cent[:, 0], X_NOSE + FRONT_LENGTH, atol=1e-6)
    rear = np.isclose(cent[:, 0], 0.0, atol=1e-9) & np.isclose(normal[:, 1], 0, atol=1e-15) & \
        np.isclose(normal[:, 2], 0, atol=1e-15)
    outward[front, 0] = -1.0
    outward[junction] = [-1.0, 0.0, 0.0]
    outward[rear] = [1.0, 0.0, 0.0]
    flip = np.einsum('ij,ij->i', normal, outward) < 0
    tris[flip] = tris[flip][:, ::-1]
    return tris


def stilt_triangles(n: int = 48) -> np.ndarray:
    """Four closed cylinders from 20 mm below the floor to 10 mm into the body."""
    tris = []
    r = STILT_DIAMETER/2
    z0, z1 = -0.020, CLEARANCE + 0.010
    ang = np.linspace(0, 2*np.pi, n+1)
    for xn in STILT_X_FROM_NOSE:
        for sy in (1, -1):
            cx, cy = X_NOSE + xn, sy*STILT_Y
            ring0 = [np.array([cx + r*math.cos(a), cy + r*math.sin(a), z0]) for a in ang]
            ring1 = [np.array([cx + r*math.cos(a), cy + r*math.sin(a), z1]) for a in ang]
            for i in range(n):
                tris += [(ring0[i], ring0[i+1], ring1[i+1]), (ring0[i], ring1[i+1], ring1[i])]
                tris += [(np.array([cx, cy, z0]), ring0[i+1], ring0[i]), (np.array([cx, cy, z1]), ring1[i], ring1[i+1])]
    return np.asarray(tris)


def write_stl(path: Path, tris: np.ndarray, name: str) -> None:
    lines = [f'solid {name}']
    for a, b, c in tris:
        nrm = np.cross(b-a, c-a)
        nrm = nrm/np.linalg.norm(nrm)
        lines += [f' facet normal {nrm[0]:.9e} {nrm[1]:.9e} {nrm[2]:.9e}', '  outer loop']
        lines += [f'   vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}' for v in (a, b, c)]
        lines += ['  endloop', ' endfacet']
    lines.append(f'endsolid {name}')
    Path(path).write_text('\n'.join(lines)+'\n')


def closed_surface_report(tris: np.ndarray, tol: float = 1e-9) -> dict:
    """Edge-manifold check, enclosed volume and area of a triangulated surface."""
    keys = np.round(tris.reshape(-1, 3)/tol).astype(np.int64)
    _, index = np.unique(keys, axis=0, return_inverse=True)
    index = index.reshape(-1, 3)
    edges = {}
    for t in index:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges[(a, b)] = edges.get((a, b), 0) + 1
    unmatched = sum(1 for (a, b), n in edges.items() if edges.get((b, a), 0) != n)
    volume = float(np.einsum('ij,ij->i', tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum()/6)
    area = float(0.5*np.linalg.norm(np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0]), axis=1).sum())
    return {'triangles': int(len(tris)), 'unmatched_directed_edges': int(unmatched), 'volume': volume, 'area': area}


# ---------------------------------------------------------------------------------------------
# mesh: background block, snappyHexMesh (castellation, snapping, prism layers)
# ---------------------------------------------------------------------------------------------
# Mesh levels. ``cell`` is the background cell (m); refinement level n halves it n times.
# Body surface at level 5 (2 mm at the base level), stilts at level 5, the slant and near wake
# at level 4, the body region and wake at level 3, a halo at level 2.
LEVELS = {
    'coarse': {'cell': 0.128, 'surface': (4, 5), 'layers': 4},
    'base': {'cell': 0.064, 'surface': (4, 5), 'layers': 5},
    'fine': {'cell': 0.0508, 'surface': (4, 5), 'layers': 5},
}
BOXES = (   # (name, level, (xmin, ymin, zmin), (xmax, ymax, zmax)); names must differ from surfaces
    ('box_halo', 2, (-1.25, 0.0, 0.0), (0.90, 0.40, 0.52)),
    ('box_near', 3, (-1.10, 0.0, 0.0), (0.12, 0.26, 0.39)),
    ('box_wake', 3, (-0.25, 0.0, 0.0), (0.65, 0.30, 0.42)),
    ('box_slant', 4, (-0.25, 0.0, 0.03), (0.20, 0.23, 0.36)),
)


def _header(cls: str, obj: str, location: str = '') -> str:
    loc = f'    location    "{location}";\n' if location else ''
    return ('FoamFile\n{\n    version     2.0;\n    format      ascii;\n'
            f'    class       {cls};\n{loc}    object      {obj};\n}}\n')


def block_mesh_dict(cell: float) -> str:
    (x0, x1), (y0, y1), (z0, z1) = DOMAIN['x'], DOMAIN['y'], DOMAIN['z']
    n = [max(4, int(round((b-a)/cell))) for a, b in ((x0, x1), (y0, y1), (z0, z1))]
    return _header('dictionary', 'blockMeshDict') + f"""
scale 1;
vertices
(
    ({x0} {y0} {z0}) ({x1} {y0} {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0})
    ({x0} {y0} {z1}) ({x1} {y0} {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({n[0]} {n[1]} {n[2]}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet    {{ type patch; faces ((0 4 7 3)); }}
    outlet   {{ type patch; faces ((1 2 6 5)); }}
    floor    {{ type wall; faces ((0 3 2 1)); }}
    top      {{ type patch; faces ((4 5 6 7)); }}
    side     {{ type patch; faces ((3 7 6 2)); }}
    symmetry {{ type symmetryPlane; faces ((0 1 5 4)); }}
);
mergePatchPairs ();
"""


def surface_feature_extract_dict() -> str:
    return _header('dictionary', 'surfaceFeatureExtractDict') + """
body.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    writeObj            no;
}
stilts.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs { includedAngle 150; }
    writeObj            no;
}
"""


def snappy_hex_mesh_dict(level: str) -> str:
    spec = LEVELS[level]
    lo, hi = spec['surface']
    boxes = '\n'.join(f'    {name} {{ type searchableBox; min ({a[0]} {a[1]} {a[2]}); max ({b[0]} {b[1]} {b[2]}); }}'
                      for name, _, a, b in BOXES)
    regions = '\n'.join(f'        {name} {{ mode inside; levels ((1e15 {lvl})); }}' for name, lvl, _, _ in BOXES)
    return _header('dictionary', 'snappyHexMeshDict') + f"""
castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    body.stl   {{ type triSurfaceMesh; name body; }}
    stilts.stl {{ type triSurfaceMesh; name stilts; }}
{boxes}
}}

castellatedMeshControls
{{
    maxLocalCells       20000000;
    maxGlobalCells      60000000;
    minRefinementCells  0;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;
    features
    (
        {{ file "body.eMesh";   level {hi}; }}
        {{ file "stilts.eMesh"; level {hi}; }}
    );
    refinementSurfaces
    {{
        body   {{ level ({lo} {hi}); patchInfo {{ type wall; }} }}
        stilts {{ level ({hi} {hi}); patchInfo {{ type wall; }} }}
    }}
    resolveFeatureAngle 30;
    refinementRegions
    {{
{regions}
    }}
    locationInMesh (-2.0 0.7 0.7);
    allowFreeStandingZoneFaces false;
}}

snapControls
{{
    nSmoothPatch            3;
    tolerance               2.0;
    nSolveIter              100;
    nRelaxIter              5;
    nFeatureSnapIter        15;
    implicitFeatureSnap     false;
    explicitFeatureSnap     true;
    multiRegionFeatureSnap  false;
}}

addLayersControls
{{
    relativeSizes       true;
    layers
    {{
        body   {{ nSurfaceLayers {spec['layers']}; }}
        stilts {{ nSurfaceLayers 3; }}
        floor  {{ nSurfaceLayers 3; }}
    }}
    expansionRatio          1.2;
    finalLayerThickness     0.4;
    minThickness            0.05;
    nGrow                   0;
    featureAngle            130;
    slipFeatureAngle        30;
    nRelaxIter              5;
    nSmoothSurfaceNormals   1;
    nSmoothNormals          3;
    nSmoothThickness        10;
    maxFaceThicknessRatio   0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle      90;
    nBufferCellsNoExtrude   0;
    nLayerIter              50;
}}

meshQualityControls
{{
    maxNonOrtho             65;
    maxBoundarySkewness     20;
    maxInternalSkewness     4;
    maxConcave              80;
    minVol                  1e-13;
    minTetQuality           1e-15;
    minArea                 -1;
    minTwist                0.02;
    minDeterminant          0.001;
    minFaceWeight           0.05;
    minVolRatio             0.01;
    minTriangleTwist        -1;
    nSmoothScale            4;
    errorReduction          0.75;
    relaxed {{ maxNonOrtho 75; }}
}}

mergeTolerance 1e-6;
"""


def mesh_inputs(root: Path, slant_deg: float, level: str) -> None:
    """Write the geometry and meshing dictionaries of one Ahmed mesh into a case folder."""
    root = Path(root)
    (root/'constant/triSurface').mkdir(parents=True, exist_ok=True)
    (root/'system').mkdir(parents=True, exist_ok=True)
    write_stl(root/'constant/triSurface/body.stl', body_triangles(slant_deg), 'body')
    write_stl(root/'constant/triSurface/stilts.stl', stilt_triangles(), 'stilts')
    (root/'system/blockMeshDict').write_text(block_mesh_dict(LEVELS[level]['cell']))
    (root/'system/surfaceFeatureExtractDict').write_text(surface_feature_extract_dict())
    (root/'system/snappyHexMeshDict').write_text(snappy_hex_mesh_dict(level))


# ---------------------------------------------------------------------------------------------
# solver case (the mesh is a frozen asset built by scripts/verify_build_ahmed_mesh.py)
# ---------------------------------------------------------------------------------------------
PATCHES = ('inlet', 'outlet', 'floor', 'top', 'side', 'symmetry', 'body', 'stilts')
WALLS = ('floor', 'body', 'stilts')


def inflow_turbulence() -> tuple[float, float]:
    """k and omega of the 0.25 % free stream with an eddy-viscosity ratio of one."""
    k = 1.5*(TURBULENCE_INTENSITY*U_REF)**2
    return k, k/NU


def fv_schemes() -> str:
    """First-order momentum convection, first-order turbulence, limited non-orthogonal corrections.

    Steady SST with second-order momentum does not converge on this body: on the coarse frozen
    mesh the residuals stall near 1e-3 (U) and 3e-3 (p) with linearUpwind, with limitedLinearV 1
    and with heavier under-relaxation (the separated wake is unsteady), so no run could meet the
    frozen 1e-6 residual rule. With first-order momentum the same run converges below 1e-6 within
    about 600 iterations. The numerical diffusion is identical for SST and every model, but it
    partly masks closure effects in the wake: a stated limitation of this case."""
    return _header('dictionary', 'fvSchemes', 'system') + """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default                         Gauss linear;
    div(phi,U)                      bounded Gauss upwind;
    div(phi,k)                      bounded Gauss upwind;
    div(phi,omega)                  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U)))))   Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited corrected 0.5; }
wallDist        { method meshWave; }
"""


def fv_solution() -> str:
    return _header('dictionary', 'fvSolution', 'system') + """
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


def boundary_fields() -> dict[str, str]:
    k_in, omega_in = inflow_turbulence()

    def field(cls, obj, dims, internal, walls, inlet, outlet):
        body = {'inlet': inlet, 'outlet': outlet, 'top': 'type slip;', 'side': 'type slip;',
                'symmetry': 'type symmetryPlane;'}
        body.update({w: walls for w in WALLS})
        entries = '\n'.join(f'    {name:<9}{{ {body[name]} }}' for name in PATCHES)
        return _header(cls, obj, '0') + f"""
dimensions      {dims};
internalField   uniform {internal};

boundaryField
{{
{entries}
}}
"""
    zero_grad = 'type zeroGradient;'
    return {
        'U': field('volVectorField', 'U', '[0 1 -1 0 0 0 0]', f'({U_REF} 0 0)', 'type noSlip;',
                   f'type fixedValue; value uniform ({U_REF} 0 0);',
                   'type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0);'),
        'p': field('volScalarField', 'p', '[0 2 -2 0 0 0 0]', '0', zero_grad, zero_grad,
                   'type fixedValue; value uniform 0;'),
        'k': field('volScalarField', 'k', '[0 2 -2 0 0 0 0]', f'{k_in}',
                   f'type kqRWallFunction; value uniform {k_in};', f'type fixedValue; value uniform {k_in};',
                   f'type inletOutlet; inletValue uniform {k_in}; value uniform {k_in};'),
        'omega': field('volScalarField', 'omega', '[0 0 -1 0 0 0 0]', f'{omega_in}',
                       f'type omegaWallFunction; value uniform {omega_in};',
                       f'type fixedValue; value uniform {omega_in};',
                       f'type inletOutlet; inletValue uniform {omega_in}; value uniform {omega_in};'),
        'nut': field('volScalarField', 'nut', '[0 2 -1 0 0 0 0]', '0',
                     'type nutUSpaldingWallFunction; value uniform 0;', 'type calculated; value uniform 0;',
                     'type calculated; value uniform 0;'),
    }


def control_dict(end_time: int) -> str:
    from . import hump
    return hump.control_dict(end_time)


def write_case(root: Path, spec, slant_deg: float, mesh_dir: Path, end_time: int) -> dict:
    """Write every case file and copy the frozen mesh. No OpenFOAM utility is run."""
    from ..casegen import render_turbulence_properties
    root = Path(root)
    for sub in ('system', 'constant', '0'):
        (root/sub).mkdir(parents=True, exist_ok=True)
    source = Path(mesh_dir)
    if not (source/'points').is_file():
        raise FileNotFoundError(f'Frozen Ahmed mesh missing: {source}')
    shutil.copytree(source, root/'constant/polyMesh')
    boundary = (root/'constant/polyMesh/boundary').read_text()
    missing = [p for p in PATCHES if not re.search(rf'\b{p}\b\s*\{{', boundary)]
    if missing:
        raise ValueError(f'Frozen Ahmed mesh lacks patches {missing}')
    control = control_dict(end_time)
    if spec is not None:
        control += '\nlibs            ("libkOmegaSSTBasis.so");\n'
    (root/'system/controlDict').write_text(control)
    (root/'system/fvSchemes').write_text(fv_schemes())
    (root/'system/fvSolution').write_text(fv_solution())
    (root/'constant/transportProperties').write_text(
        _header('dictionary', 'transportProperties', 'constant') + f'\ntransportModel  Newtonian;\nnu              {NU};\n')
    (root/'constant/turbulenceProperties').write_text(render_turbulence_properties(spec, mode='expressions'))
    for name, text in boundary_fields().items():
        (root/'0'/name).write_text(text)
    cells = _mesh_cells(root/'constant/polyMesh')
    return {'cells': cells, 'slant_deg': slant_deg, 're_h': RE_H, 'u_ref': U_REF, 'mesh': str(source)}


def _mesh_cells(poly: Path) -> int | None:
    match = re.search(r'nCells:\s*(\d+)', (poly/'owner').read_text(errors='replace')[:4000])
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------------------------
# scoring against Lienhart and Becker (ERCOFTAC case 82)
# ---------------------------------------------------------------------------------------------
SLANT_PLANES = ('yp000-xz', 'yp100-xz', 'yp180-xz')       # over the slant, both angles
WAKE_PLANES = ('xp000-yz', 'xp080-yz', 'xp200-yz', 'xp500-yz')
FRONTAL_HALF_AREA = WIDTH*HEIGHT/2


def read_table(path: Path) -> np.ndarray:
    rows = []
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            rows.append([float(v) for v in line.split()])
        except ValueError:
            continue
    width = min(len(r) for r in rows)
    return np.asarray([r[:width] for r in rows], dtype=float)


def reference(slant_deg: int) -> dict:
    """LDA points (m) and mean velocities (m/s) on the half model (y >= 0), and rear-surface
    pressure coefficients. Locations measured on both sides of the centre plane (and repeated
    points) are averaged after mirroring, spanwise velocity reversed: the model is symmetric by
    construction, and half of the measured left-right difference is kept as ``asymmetry``."""
    def planes(names):
        tables = [read_table(DATA/f'ahmed-{slant_deg}-{n}.dat') for n in names]
        t = np.vstack([tab[:, :6] for tab in tables])
        points, velocity = _mirror(t[:, :3]/1000.0, t[:, 3:6])
        key = np.round(points*1e4).astype(np.int64)      # 0.1 mm
        unique, inverse = np.unique(key, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        count = np.bincount(inverse)
        mean = np.stack([np.bincount(inverse, velocity[:, c])/count for c in range(3)], axis=1)
        spread = np.sqrt(np.bincount(inverse, np.sum((velocity-mean[inverse])**2, axis=1))/count)
        return unique/1e4, mean, spread
    slant = planes(SLANT_PLANES)
    wake = planes(WAKE_PLANES)
    press = read_table(DATA/f'ahmed-{slant_deg}-press.dat')
    return {'slant': slant[:2], 'wake': wake[:2], 'asymmetry': {'slant': slant[2], 'wake': wake[2]},
            'pressure': (press[:, :3]/1000.0, press[:, 3])}


def _mirror(xyz: np.ndarray, u: np.ndarray | None = None):
    """Half model: data points at y < 0 are compared at |y| with the spanwise velocity reversed."""
    sign = np.where(xyz[:, 1] < 0, -1.0, 1.0)
    points = xyz*np.c_[np.ones(len(xyz)), sign, np.ones(len(xyz))]
    if u is None:
        return points
    return points, u*np.c_[np.ones(len(u)), sign, np.ones(len(u))]


def sample(tree, values: np.ndarray, targets: np.ndarray, k: int = 8) -> np.ndarray:
    """Inverse-distance-squared interpolation of cell values at the target points."""
    dist, index = tree.query(targets, k=k)
    weight = 1.0/np.maximum(dist, 1e-9)**2
    return np.einsum('ij,ij...->i...', weight, values[index])/weight.sum(axis=1).reshape((-1,)+(1,)*(values.ndim-1))


def score(centres: np.ndarray, u: np.ndarray, p: np.ndarray, wall_centres: np.ndarray, wall_p: np.ndarray,
          wall_force: np.ndarray, slant_deg: int) -> tuple[dict, dict]:
    """Errors and observables from cell centres, cell velocity and pressure, and the body patch
    (face centres, face pressure, total force of the half body in N/(kg/m3))."""
    from scipy.spatial import cKDTree
    ref = reference(slant_deg)
    region = ((centres[:, 0] > -0.40) & (centres[:, 0] < 0.65) & (centres[:, 1] < 0.50) & (centres[:, 2] < 0.55))
    tree = cKDTree(centres[region])
    uu = u[region]
    out, obs = {}, {}
    for key, name in (('slant', 'slant_velocity_rmse'), ('wake', 'wake_velocity_rmse')):
        points, measured = ref[key]
        model = sample(tree, uu, points)
        out[name] = float(np.sqrt(np.mean(np.sum((model-measured)**2, axis=1))))/U_REF
        obs[f'{key}_points'] = int(len(points))
        obs[f'{key}_u_mean'] = float(np.mean(model[:, 0]))/U_REF
        obs[f'{key}_reversed_fraction'] = float(np.mean(model[:, 0] < 0))
        obs[f'{key}_reversed_fraction_measured'] = float(np.mean(measured[:, 0] < 0))
        obs[f'{key}_measured_asymmetry_rms'] = float(np.sqrt(np.mean(ref['asymmetry'][key]**2)))/U_REF
    upstream = (centres[:, 0] < DOMAIN['x'][0] + 0.25)
    p_ref = float(np.mean(p[upstream]))
    points, cp_measured = ref['pressure']
    points = _mirror(points)
    nearest = cKDTree(wall_centres).query(points)[1]
    cp = (wall_p[nearest]-p_ref)/(0.5*U_REF**2)
    out['rear_pressure_rmse'] = float(np.sqrt(np.mean((cp-cp_measured)**2)))
    obs.update(rear_cp_mean=float(np.mean(cp)), rear_cp_mean_measured=float(np.mean(cp_measured)),
               pressure_points=int(len(points)), reference_pressure=p_ref,
               drag_coefficient=float(wall_force[0]/(0.5*U_REF**2*FRONTAL_HALF_AREA)),
               lift_coefficient=float(wall_force[2]/(0.5*U_REF**2*FRONTAL_HALF_AREA)))
    return out, obs
