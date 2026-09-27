"""Wing-body junction (Devenport and Simpson 1990; Fleming et al. 1993): verification only, 3D.

Reference: ERCOFTAC classic case 008, "Wing/body junction with separation": a cylindrical wing
of 3:2 elliptic nose and NACA 0020 tail (thickness T = 71.7 mm, chord 305 mm) standing on a flat
wall in a turbulent boundary layer (Re_theta ~ 6300 at x/T = -2.15, U_ref = 26.75 m/s). The
approaching layer separates in front of the wing and rolls up into the horseshoe vortex, whose
legs are measured downstream. Scored against the laser-anemometer profiles in the symmetry plane
ahead of the wing (Devenport and Simpson), the cross-plane profiles behind it (Fleming's hot-wire
and Devenport's laser data), and the wall and wing pressures. Data sets of the other studies in
the case (different inflow boundary layers) are not used.

Coordinates as in the data, in metres: x along the chord from the leading edge, y normal to the
flat wall, z across the span (the data's z < 0 side is compared on the half model's z > 0 side).

Setup (the case's recommendations): half domain with the symmetry plane z = 0 through the chord;
inlet at x = -18.24 T with the measured boundary layer; outlet 12 T behind the trailing edge;
a slip boundary at y = 3 T; a half-elliptic far field (1 m across the span, farther than the
tunnel's 6.35 T half width) with the measured inflow on its upstream half and the pressure outlet on
its downstream half; resolved walls (first cell
~1e-5 m, y+ < 1) with the development cases' low-Re wall treatment. The NACA 0020 tail is closed
at the trailing edge (the standard 0.1036 quartic coefficient) so the O-grid ends on the symmetry
plane. The mesh is structured (a two-layer O-grid in the x-z plane extruded in y) and is written
directly as an OpenFOAM polyMesh by ``write_mesh``; no meshing utility is involved.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO/'data/assets/references/wingbody'

T = 0.0717                      # maximum thickness
NOSE_A = 1.5*T/2                # 3:2 ellipse: semi-axis along the chord
NACA_CHORD = T/0.20             # NACA 0020 chord whose thickness is T
CHORD = NOSE_A + 0.7*NACA_CHORD
U_REF = 26.99                   # inflow table (x/T = -18.24)
NU = 1.660802e-5
X_IN, X_OUT = -18.24*T, CHORD + 12*T
Z_MAX = 0.455                   # tunnel half width, 6.35 T
Y_TOP = 3*T

LEVELS = {   # points along the wing half, inner and outer wall-normal cells, cells in y
    'coarse': {'wing': 71, 'inner': 30, 'outer': 30, 'height': 44},
    'base': {'wing': 141, 'inner': 50, 'outer': 50, 'height': 70},
}
FIRST_CELL = 1.0e-5
INNER_LAYER = 0.35*T


# ---------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------
def half_thickness(x: np.ndarray) -> np.ndarray:
    """z of the wing surface (z >= 0) at chordwise x from the leading edge."""
    x = np.asarray(x, dtype=float)
    b = T/2
    nose = b*np.sqrt(np.clip(1-((NOSE_A-x)/NOSE_A)**2, 0, None))
    xi = 0.3 + (x-NOSE_A)/NACA_CHORD
    tail = NACA_CHORD*(0.2969*np.sqrt(np.clip(xi, 0, None)) - 0.1260*xi - 0.3516*xi**2
                       + 0.2843*xi**3 - 0.1036*xi**4)
    return np.where(x <= NOSE_A, nose, np.clip(tail, 0, None))


TE_RADIUS = 1.0e-3              # the closed trailing edge is rounded (O-grid without a singular corner)


def _rounded_tail():
    """Tangent point (x, z) and centre x of the trailing-edge rounding arc of radius TE_RADIUS.

    The inward normal from a tail point meets the chord line at distance z*sqrt(1+z'^2), which
    falls monotonically to zero at the closed trailing edge; bisection finds the point where it
    equals TE_RADIUS, so the arc is tangent to the section and continuous with it. (A grid search
    over only the last 0.05 T never reached 1 mm on this thin tail: the arc started 0.11 mm off the
    surface, a kink that folded the base-level wall normals and inverted cells behind it.)"""
    def radius(x):
        h = 1e-9
        z = float(half_thickness(np.array([x]))[0])
        slope = float((half_thickness(np.array([x+h]))[0]-half_thickness(np.array([x-h]))[0])/(2*h))
        return z*math.sqrt(1+slope*slope), z, slope
    lo, hi = CHORD-0.5*T, CHORD-1e-9
    if not radius(lo)[0] > TE_RADIUS > radius(hi)[0]:
        raise ValueError('trailing-edge rounding radius is not bracketed on the tail')
    for _ in range(200):
        mid = 0.5*(lo+hi)
        lo, hi = (mid, hi) if radius(mid)[0] > TE_RADIUS else (lo, mid)
    x = 0.5*(lo+hi)
    r, z, slope = radius(x)
    if abs(r-TE_RADIUS) > 1e-9:
        raise ValueError(f'trailing-edge arc not tangent: radius {r} m')
    return x, z, x + z*slope


def wing_curve(n: int) -> np.ndarray:
    """(n, 2) points (x, z) along the upper half of the section from the leading edge to the
    trailing edge, clustered at both ends; the nose is sampled by angle for its curvature and the
    closed trailing edge ends in a small tangent arc centred on the chord line."""
    s = 0.5*(1-np.cos(np.linspace(0, math.pi, n)))              # 0..1, clustered at the ends
    x_t, z_t, x_c = _rounded_tail()
    xx = np.r_[NOSE_A*(1-np.cos(np.linspace(0, math.pi/2, 4000))), np.linspace(NOSE_A, x_t, 8001)[1:]]
    zz = half_thickness(xx)
    zz[0] = 0.0
    start = math.atan2(z_t, x_t-x_c)
    angle = np.linspace(start, 0.0, 400)[1:]
    xx = np.r_[xx, x_c + TE_RADIUS*np.cos(angle)]
    zz = np.r_[zz, TE_RADIUS*np.sin(angle)]
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(xx), np.diff(zz)))]
    target = s*arc[-1]
    curve = np.c_[np.interp(target, arc, xx), np.interp(target, arc, zz)]
    curve[0], curve[-1] = (0.0, 0.0), (x_c + TE_RADIUS, 0.0)
    return curve


def _normals(curve: np.ndarray) -> np.ndarray:
    """Outward unit normals of the curve: -x at the leading edge, +x at the rounded trailing edge."""
    t = np.gradient(curve, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    n = np.c_[-t[:, 1], t[:, 0]]                    # rotate the tangent (x forward, z up): outward is +z
    n[0] = (-1.0, 0.0)
    n[-1] = (1.0, 0.0)
    return n/np.linalg.norm(n, axis=1)[:, None]


def _geometric(first: float, total: float, cells: int) -> np.ndarray:
    """Cumulative distances (cells+1 values from 0 to total) of a geometric distribution."""
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):
        r = 0.5*(lo+hi)
        s = first*(r**cells-1)/(r-1)
        lo, hi = (r, hi) if s < total else (lo, r)
    d = first*np.r_[0, np.cumsum(r**np.arange(cells))]
    return d*total/d[-1]


FAR_WIDTH = 1.0                 # half-ellipse far field: semi-axis across the span (m)


def _outer_boundary(n: int, fractions: np.ndarray) -> np.ndarray:
    """Points on the half-elliptic far field from (X_IN, 0) over (centre, FAR_WIDTH) to (X_OUT, 0)
    at the given arc-length fractions: O-grid lines meet it nearly at right angles."""
    centre, a = 0.5*(X_IN+X_OUT), 0.5*(X_OUT-X_IN)
    angle = np.linspace(math.pi, 0.0, 4001)
    path = np.c_[centre + a*np.cos(angle), FAR_WIDTH*np.sin(angle)]
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    target = fractions*arc[-1]
    return np.c_[np.interp(target, arc, path[:, 0]), np.interp(target, arc, path[:, 1])]


def grid_2d(level: str = 'base') -> np.ndarray:
    """(n_wing, n_normal+1, 2) structured points in the x-z plane: an inner layer along the wall
    normals, then straight lines to the far field."""
    spec = LEVELS[level]
    curve = wing_curve(spec['wing'])
    normals = _normals(curve)
    inner = _geometric(FIRST_CELL, INNER_LAYER, spec['inner'])
    layer = curve[:, None, :] + inner[None, :, None]*normals[:, None, :]
    edge = layer[:, -1, :]
    # far-field points evenly spaced along the ellipse: the lines leaving the clustered leading and
    # trailing edges fan out, so the wake cells behind the trailing edge widen downstream
    outer_pts = _outer_boundary(len(edge), np.linspace(0.0, 1.0, len(edge)))
    last = inner[-1]-inner[-2]
    rows = []
    for i in range(len(edge)):
        dist = np.linalg.norm(outer_pts[i]-edge[i])
        d = _geometric(min(1.3*last, dist/spec['outer']), dist, spec['outer'])
        direction = (outer_pts[i]-edge[i])/dist
        rows.append(edge[i][None, :] + d[1:, None]*direction[None, :])
    outer = np.stack(rows)
    return np.concatenate([layer, outer], axis=1)


def heights(level: str = 'base') -> np.ndarray:
    return _geometric(FIRST_CELL, Y_TOP, LEVELS[level]['height'])


# ---------------------------------------------------------------------------------------------
# structured polyMesh writer
# ---------------------------------------------------------------------------------------------
def _write_list(path: Path, cls: str, obj: str, rows: list[str], note: str = '') -> None:
    note_line = f'    note        "{note}";\n' if note else ''
    head = ('FoamFile\n{\n    version     2.0;\n    format      ascii;\n'
            f'    class       {cls};\n{note_line}    location    "constant/polyMesh";\n    object      {obj};\n}}\n\n')
    path.write_text(head + f'{len(rows)}\n(\n' + '\n'.join(rows) + '\n)\n')


def write_mesh(case: Path, level: str = 'base') -> dict:
    """Write constant/polyMesh of the structured junction mesh; returns sizes and quality checks."""
    g2 = grid_2d(level)
    ys = heights(level)
    ni, nj, nk = g2.shape[0], g2.shape[1], len(ys)
    pts = np.empty((ni, nj, nk, 3))
    pts[..., 0] = g2[:, :, None, 0]
    pts[..., 1] = ys[None, None, :]
    pts[..., 2] = g2[:, :, None, 1]
    vid = np.arange(ni*nj*nk).reshape(ni, nj, nk, order='F')
    ci, cj, ck = ni-1, nj-1, nk-1
    cid = np.arange(ci*cj*ck).reshape(ci, cj, ck, order='F')
    # handedness of (i, j, k): quads below are written for a right-handed index system
    e_i = pts[1, 0, 0]-pts[0, 0, 0]
    e_j = pts[0, 1, 0]-pts[0, 0, 0]
    e_k = pts[0, 0, 1]-pts[0, 0, 0]
    flip = np.dot(e_i, np.cross(e_j, e_k)) < 0

    def quad(a, b, c, d):
        q = np.stack([a, b, c, d], axis=-1)
        return q[..., ::-1] if flip else q

    def i_face(i, j, k):   # face at vertex plane i, normal +i
        return quad(vid[i, j, k], vid[i, j+1, k], vid[i, j+1, k+1], vid[i, j, k+1])

    def j_face(i, j, k):   # normal +j
        return quad(vid[i, j, k], vid[i, j, k+1], vid[i+1, j, k+1], vid[i+1, j, k])

    def k_face(i, j, k):   # normal +k
        return quad(vid[i, j, k], vid[i+1, j, k], vid[i+1, j+1, k], vid[i, j+1, k])

    faces, owner, neighbour = [], [], []
    I, J, K = np.meshgrid(np.arange(ci), np.arange(cj), np.arange(ck), indexing='ij')
    # internal faces, upper-triangular order: owner ascending, then neighbour ascending
    blocks = []
    m = I < ci-1
    blocks.append((cid[I[m], J[m], K[m]], cid[I[m]+1, J[m], K[m]], i_face(I[m]+1, J[m], K[m])))
    m = J < cj-1
    blocks.append((cid[I[m], J[m], K[m]], cid[I[m], J[m]+1, K[m]], j_face(I[m], J[m]+1, K[m])))
    m = K < ck-1
    blocks.append((cid[I[m], J[m], K[m]], cid[I[m], J[m], K[m]+1], k_face(I[m], J[m], K[m]+1)))
    own = np.concatenate([b[0] for b in blocks])
    nei = np.concatenate([b[1] for b in blocks])
    fac = np.concatenate([b[2] for b in blocks])
    order = np.lexsort((nei, own))
    own, nei, fac = own[order], nei[order], fac[order]
    # boundary faces, outward normals
    patches = {}

    def add(name, kind, owners, quads):
        rows = patches.setdefault(name, [kind, [], []])
        rows[1].append(owners)
        rows[2].append(quads)
    jj, kk = np.meshgrid(np.arange(cj), np.arange(ck), indexing='ij')
    add('symmetry', 'symmetryPlane', cid[0, jj, kk].ravel(), i_face(0, jj, kk)[..., ::-1].reshape(-1, 4))
    add('symmetry', 'symmetryPlane', cid[ci-1, jj, kk].ravel(), i_face(ci, jj, kk).reshape(-1, 4))
    ii, kk = np.meshgrid(np.arange(ci), np.arange(ck), indexing='ij')
    add('wing', 'wall', cid[ii, 0, kk].ravel(), j_face(ii, 0, kk)[..., ::-1].reshape(-1, 4))
    far_q = j_face(ii, cj, kk).reshape(-1, 4)
    far_o = cid[ii, cj-1, kk].ravel()
    corners = pts.reshape(-1, 3, order='F')[far_q]
    normal = np.cross(corners[:, 1]-corners[:, 0], corners[:, 3]-corners[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    inlet = normal[:, 0] < 0                  # upstream half of the far field: measured inflow
    outlet = ~inlet                           # downstream half: pressure outlet
    side = np.zeros_like(inlet)               # no side boundary with the elliptic far field
    for name, kind, mask in (('inlet', 'patch', inlet), ('outlet', 'patch', outlet)):
        add(name, kind, far_o[mask], far_q[mask])
    ii, jj = np.meshgrid(np.arange(ci), np.arange(cj), indexing='ij')
    add('floor', 'wall', cid[ii, jj, 0].ravel(), k_face(ii, jj, 0)[..., ::-1].reshape(-1, 4))
    add('top', 'patch', cid[ii, jj, ck-1].ravel(), k_face(ii, jj, ck).reshape(-1, 4))
    names = ('inlet', 'outlet', 'top', 'symmetry', 'floor', 'wing')
    b_owner = [np.concatenate(patches[n][1]) for n in names]
    b_faces = [np.concatenate(patches[n][2]) for n in names]
    all_faces = np.concatenate([fac] + b_faces)
    all_owner = np.concatenate([own] + b_owner)
    poly = Path(case)/'constant/polyMesh'
    poly.mkdir(parents=True, exist_ok=True)
    flat = pts.reshape(-1, 3, order='F')
    n_cells = ci*cj*ck
    note = f'nPoints:{len(flat)}  nCells:{n_cells}  nFaces:{len(all_faces)}  nInternalFaces:{len(fac)}'
    _write_list(poly/'points', 'vectorField', 'points', [f'({p[0]:.12g} {p[1]:.12g} {p[2]:.12g})' for p in flat])
    _write_list(poly/'faces', 'faceList', 'faces', [f'4({q[0]} {q[1]} {q[2]} {q[3]})' for q in all_faces])
    _write_list(poly/'owner', 'labelList', 'owner', [str(o) for o in all_owner], note)
    _write_list(poly/'neighbour', 'labelList', 'neighbour', [str(n) for n in nei], note)
    start, entries = len(fac), []
    for n, f in zip(names, b_faces):
        kind = patches[n][0]
        in_group = '\n        inGroups        1(wall);' if kind == 'wall' else ''
        entries.append(f'    {n}\n    {{\n        type            {kind};{in_group}\n'
                       f'        nFaces          {len(f)};\n        startFace       {start};\n    }}')
        start += len(f)
    (poly/'boundary').write_text(
        'FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       polyBoundaryMesh;\n'
        '    location    "constant/polyMesh";\n    object      boundary;\n}\n\n'
        f'{len(names)}\n(\n' + '\n'.join(entries) + '\n)\n')
    inlet_centres = flat[b_faces[0]].mean(axis=1)
    return {'cells': int(n_cells), 'points': int(len(flat)), 'faces': int(len(all_faces)),
            'patch_faces': {n: int(len(f)) for n, f in zip(names, b_faces)}, 'left_handed_index': bool(flip),
            'inlet_y': inlet_centres[:, 1]}


# ---------------------------------------------------------------------------------------------
# case: measured inflow boundary layer, resolved walls
# ---------------------------------------------------------------------------------------------
def _header(cls: str, obj: str, location: str = '') -> str:
    loc = f'    location    "{location}";\n' if location else ''
    return ('FoamFile\n{\n    version     2.0;\n    format      ascii;\n'
            f'    class       {cls};\n{loc}    object      {obj};\n}}\n')


MISSING = 9.9e9                 # the data files mark missing values with 9.9999E+9


def _table(path: Path) -> list:
    rows = []
    for line in Path(path).read_text(errors='replace').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            row = [float(v) for v in line.split()]
        except ValueError:
            continue
        if all(abs(v) < MISSING for v in row):
            rows.append(row)
    return rows


FREESTREAM_INTENSITY = 0.002


def inflow_profiles(y: np.ndarray) -> dict:
    """U, k and omega at the inlet heights y (m) from the measured layer at x/T = -18.24: k from the
    recommended v'2 = 2 w'2 - u'2 (k = 1.5 w'2), omega from a mixing length min(0.41 y, 0.09 delta)
    and the measured velocity gradient; free stream above the layer, a linear sublayer below it."""
    data = np.asarray([r for r in _table(DATA/'wbj-etab1.dat') if len(r) == 5])
    yd, u = data[:, 0]*T, data[:, 1]*U_REF
    k = np.maximum(1.5*data[:, 3]*U_REF**2, 1e-10)
    delta = float(np.interp(0.99, u/U_REF, yd)) if (u/U_REF).max() > 0.99 else float(yd[-1])
    dudy = np.gradient(u, yd)
    mixing = np.minimum(0.41*yd, 0.09*delta)
    nut = np.maximum(mixing**2*np.abs(dudy), NU*1e-3)
    omega = k/nut
    k_inf = 1.5*(FREESTREAM_INTENSITY*U_REF)**2
    omega_inf = k_inf/NU
    y = np.asarray(y, dtype=float)
    uu = np.interp(y, yd, u, right=U_REF)
    kk = np.interp(y, yd, k, right=k_inf)
    ww = np.exp(np.interp(y, yd, np.log(omega), right=np.log(omega_inf)))
    below = y < yd[0]
    uu[below] = u[0]*y[below]/yd[0]
    kk[below] = k[0]*(y[below]/yd[0])**2
    ww[below] = np.maximum(6*NU/(0.075*y[below]**2), omega[0])
    above = y > delta
    kk[above] = np.maximum(kk[above], k_inf)
    return {'U': uu, 'k': np.maximum(kk, 1e-12), 'omega': ww, 'delta': delta}


def _nonuniform(values, vector=False):
    if vector:
        body = '\n'.join(f'({v:.10g} 0 0)' for v in values)
        return f'nonuniform List<vector> {len(values)}\n(\n{body}\n)'
    body = '\n'.join(f'{v:.10g}' for v in values)
    return f'nonuniform List<scalar> {len(values)}\n(\n{body}\n)'


PATCHES = ('inlet', 'outlet', 'top', 'symmetry', 'floor', 'wing')


def boundary_fields(inlet_y: np.ndarray) -> dict[str, str]:
    prof = inflow_profiles(inlet_y)
    k_inf = 1.5*(FREESTREAM_INTENSITY*U_REF)**2
    omega_inf = k_inf/NU

    def field(cls, obj, dims, internal, inlet, outlet, walls):
        entries = {'inlet': inlet, 'outlet': outlet, 'top': 'type slip;',
                   'symmetry': 'type symmetryPlane;', 'floor': walls, 'wing': walls}
        body = '\n'.join(f'    {n}\n    {{\n        {entries[n]}\n    }}' for n in PATCHES)
        return _header(cls, obj, '0') + f"""
dimensions      {dims};
internalField   uniform {internal};

boundaryField
{{
{body}
}}
"""
    return {
        'U': field('volVectorField', 'U', '[0 1 -1 0 0 0 0]', f'({U_REF} 0 0)',
                   'type fixedValue; value ' + _nonuniform(prof['U'], vector=True) + ';',
                   'type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0);', 'type noSlip;'),
        'p': field('volScalarField', 'p', '[0 2 -2 0 0 0 0]', '0', 'type zeroGradient;',
                   'type fixedValue; value uniform 0;', 'type zeroGradient;'),
        'k': field('volScalarField', 'k', '[0 2 -2 0 0 0 0]', f'{k_inf}',
                   'type fixedValue; value ' + _nonuniform(prof['k']) + ';',
                   f'type inletOutlet; inletValue uniform {k_inf}; value uniform {k_inf};',
                   f'type kLowReWallFunction; value uniform {k_inf};'),
        'omega': field('volScalarField', 'omega', '[0 0 -1 0 0 0 0]', f'{omega_inf}',
                       'type fixedValue; value ' + _nonuniform(prof['omega']) + ';',
                       f'type inletOutlet; inletValue uniform {omega_inf}; value uniform {omega_inf};',
                       f'type omegaWallFunction; value uniform {omega_inf};'),
        'nut': field('volScalarField', 'nut', '[0 2 -1 0 0 0 0]', '0', 'type calculated; value uniform 0;',
                     'type calculated; value uniform 0;', 'type nutLowReWallFunction; value uniform 0;'),
    }


def write_case(root: Path, spec, level: str = 'base', end_time: int = 8000) -> dict:
    """Write the mesh and every case file (pure Python, deterministic); no OpenFOAM utility."""
    from ..casegen import render_turbulence_properties
    from . import hump
    root = Path(root)
    for sub in ('system', 'constant', '0'):
        (root/sub).mkdir(parents=True, exist_ok=True)
    mesh = write_mesh(root, level)
    control = hump.control_dict(end_time)
    if spec is not None:
        control += '\nlibs            ("libkOmegaSSTBasis.so");\n'
    (root/'system/controlDict').write_text(control)
    (root/'system/fvSchemes').write_text(hump.fv_schemes().replace(
        'laplacianSchemes { default Gauss linear corrected; }', 'laplacianSchemes { default Gauss linear limited corrected 0.5; }').replace(
        'snGradSchemes   { default corrected; }', 'snGradSchemes   { default limited corrected 0.5; }'))
    (root/'system/fvSolution').write_text(hump.fv_solution())
    (root/'constant/transportProperties').write_text(
        _header('dictionary', 'transportProperties', 'constant') + f'\ntransportModel  Newtonian;\nnu              {NU};\n')
    (root/'constant/turbulenceProperties').write_text(render_turbulence_properties(spec, mode='expressions'))
    for name, text in boundary_fields(mesh.pop('inlet_y')).items():
        (root/'0'/name).write_text(text)
    return {**mesh, 'level': level, 're_T': U_REF*T/NU, 'u_ref': U_REF, 'chord_m': float(wing_curve(5)[-1, 0])}


# ---------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------
SYMMETRY_FILES = tuple(f'wbj-el{i:02d}tc.dat' for i in range(1, 12))
PLANE_FILES = ('wbj-fp05tc.dat', 'wbj-fp06tc.dat', 'wbj-fp07tc.dat', 'wbj-fp09tc.dat', 'wbj-fp10tc.dat',
               'wbj-fp11tc.dat', 'wbj-lp05tc.dat', 'wbj-lp08tc.dat', 'wbj-lp10tc.dat')


def _symmetry_plane() -> tuple[np.ndarray, np.ndarray]:
    """Points (m) and (U, V)/Uref of the laser profiles in the symmetry plane ahead of the wing."""
    import re
    points, values = [], []
    for name in SYMMETRY_FILES:
        text = (DATA/name).read_text(errors='replace')
        x = float(re.search(r'Location of traverse;\s*X/T\s*=\s*([-.\dEe+]+)', text).group(1))*T
        for row in _table(DATA/name):
            if len(row) >= 11:
                points.append([x, row[0]*T, 0.0])
                values.append([row[1], row[6]])
    return np.asarray(points), np.asarray(values)


def _cross_planes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points (m, mirrored to z >= 0), velocity/Uref (U, V, W; NaN where not measured) and a mask."""
    import re
    points, values = [], []
    for name in PLANE_FILES:
        text = (DATA/name).read_text(errors='replace')
        x = float(re.search(r'x/C=\s*([-.\dEe+]+)', text).group(1))*CHORD
        laser = name.startswith('wbj-lp')
        for row in _table(DATA/name):
            if len(row) < 7 or (laser and len(row) < 11):
                continue
            y, z, u, w = row[0]*T, row[1]*T, row[2], row[4]
            v = row[8] if len(row) >= 11 else np.nan
            points.append([x, y, abs(z)])
            values.append([u, v, -w if z < 0 else w])
    return np.asarray(points), np.asarray(values)


def _pressures() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points (m, mirrored), measured Cp, and whether each tap is on the wing (else the floor)."""
    pts, cps, wing = [], [], []
    for name, on_wing in (('wbj-cp-body-27.dat', False), ('wbj-cp-wing-27.dat', True)):
        for row in _table(DATA/name):
            if len(row) == 4:
                pts.append([row[0]*T, row[1]*T, abs(row[2])*T])
                cps.append(row[3])
                wing.append(on_wing)
    return np.asarray(pts), np.asarray(cps), np.asarray(wing)


def sample(tree, values, targets, k=8):
    dist, index = tree.query(targets, k=k)
    weight = 1.0/np.maximum(dist, 1e-12)**2
    return np.einsum('ij,ij...->i...', weight, values[index])/weight.sum(axis=1).reshape((-1,)+(1,)*(values.ndim-1))


def score(centres, u, p, floor, wing) -> tuple[dict, dict]:
    """Errors and observables. ``floor`` and ``wing`` are (face centres, kinematic pressure) pairs."""
    from scipy.spatial import cKDTree
    region = (centres[:, 0] > -1.2*T) & (centres[:, 0] < CHORD + 5*T) & (centres[:, 2] < 2.5*T)
    tree = cKDTree(centres[region])
    uu = u[region]/U_REF
    sym_pts, sym_ref = _symmetry_plane()
    sym = sample(tree, uu, sym_pts)[:, :2]
    errors = {'symmetry_plane_velocity_rmse': float(np.sqrt(np.mean(np.sum((sym-sym_ref)**2, axis=1))))}
    cross_pts, cross_ref = _cross_planes()
    model = sample(tree, uu, cross_pts)
    diff = np.where(np.isnan(cross_ref), 0.0, model-cross_ref)
    errors['crossflow_velocity_rmse'] = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    # undisturbed free-stream static pressure: the upstream 15 % of the domain, above half height
    # (13 T and more ahead of the wing, outside the boundary layer)
    free = (centres[:, 0] < X_IN + 0.15*(X_OUT-X_IN)) & (centres[:, 1] > 0.5*Y_TOP)
    if not free.any():
        raise ValueError('No free-stream cells for the reference pressure')
    p_ref = float(np.mean(p[free]))
    taps, cp_ref, on_wing = _pressures()
    cp = np.empty(len(taps))
    for mask, (fc, fp) in ((~on_wing, floor), (on_wing, wing)):
        if mask.any():
            cp[mask] = (fp[cKDTree(fc).query(taps[mask])[1]]-p_ref)/(0.5*U_REF**2)
    errors['wall_pressure_rmse'] = float(np.sqrt(np.mean((cp-cp_ref)**2)))
    obs = {'symmetry_points': int(len(sym_pts)), 'crossflow_points': int(len(cross_pts)), 'pressure_taps': int(len(taps)),
           'symmetry_u_min': float(sym[:, 0].min()), 'symmetry_u_min_measured': float(sym_ref[:, 0].min()),
           'symmetry_reversed_fraction': float(np.mean(sym[:, 0] < 0)),
           'symmetry_reversed_fraction_measured': float(np.mean(sym_ref[:, 0] < 0)),
           'crossflow_u_mean': float(np.mean(model[:, 0])), 'floor_cp_mean': float(np.mean(cp[~on_wing])),
           'reference_pressure': p_ref}
    return errors, obs
