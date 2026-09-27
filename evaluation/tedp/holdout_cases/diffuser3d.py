"""Stanford three-dimensional asymmetric diffuser, Diffuser 1 (verification only, genuinely 3D).

Reference: Cherry, Elkins and Eaton (2008, Int. J. Heat Fluid Flow 29, 803), ERCOFTAC
Knowledge Base UFR 4-16: magnetic-resonance velocimetry of the three-component mean velocity
on a 0.9 mm grid through the whole diffuser, and the wall pressure along the centre line of the
flat bottom wall. Water, bulk velocity 1 m/s, Re = 10,000 on the inlet duct height.

Geometry (metres; origin where the two flat walls meet at the start of the expansion, x
downstream, y across the height, z across the width): inlet duct h = 0.010 by B = 0.0333,
fully developed; over L = 0.150 the top wall rises to 0.040 (11.3 degrees) and one side wall
moves out to 0.040 (2.56 degrees), the bottom wall y = 0 and the side wall z = 0 stay flat;
every bend is rounded with a 0.060 radius; then a straight 0.040 x 0.040 outlet duct. The
experiment's outlet contraction is replaced by a further straight 0.100 so the outlet is far
from the separated region.

Setup: resolved walls (first cell y+ ~ 1, the development cases' low-Re wall treatment); a
short inlet duct whose inlet is mapped from 0.100 downstream with the bulk velocity held at
1 m/s, so the fully developed duct flow, including any secondary flow a closure produces,
forms without the experiment's 63-height development channel. The mesh is a graded unit block
from blockMesh, moved onto the walls on the compute node (``map_onto_walls``), so every
cross-section is an exact rectangle.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import shutil

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = REPO/'data/assets/references/diffuser3d'

H_IN, B_IN = 0.010, 0.0333
LENGTH = 0.150
H_OUT, B_OUT = 0.040, 0.040
RADIUS = 0.060
U_BULK = 1.0
NU = 1.0e-6
X_IN, X_OUT = -0.200, 0.375
INLET_SAMPLE_OFFSET = 0.100
TOP_ANGLE = math.atan((H_OUT-H_IN)/LENGTH)
SIDE_ANGLE = math.atan((B_OUT-B_IN)/LENGTH)

# Mesh levels: (inlet-duct, diffuser, outlet cells along x; cells over the height; cells over the width)
LEVELS = {'coarse': (24, 100, 36, 36, 52), 'base': (36, 150, 54, 56, 80), 'fine': (48, 200, 72, 72, 104)}
# Block boundaries along x: the diffuser segment covers both rounded bends with margin.
X_SEGMENTS = (X_IN, -0.020, 0.170, X_OUT)


def wall(x: np.ndarray, start: float, end: float, angle: float) -> np.ndarray:
    """Height (or width) of the duct: start, a rounded bend at x = 0, a straight ramp, a rounded
    bend at x = LENGTH, then end. Both bends are circular arcs of RADIUS tangent to the lines."""
    x = np.asarray(x, dtype=float)
    t = RADIUS*math.tan(angle/2)                  # tangent length of each bend
    out = np.where(x <= 0, start, np.where(x >= LENGTH, end, start + x*math.tan(angle)))
    first = (x > -t) & (x < t*math.cos(angle))
    out = np.where(first, start + RADIUS - np.sqrt(np.maximum(RADIUS**2 - (x + t)**2, 0.0)), out)
    second = (x > LENGTH - t*math.cos(angle)) & (x < LENGTH + t)
    out = np.where(second, end - RADIUS + np.sqrt(np.maximum(RADIUS**2 - (x - LENGTH - t)**2, 0.0)), out)
    return out


def top(x):
    return wall(x, H_IN, H_OUT, TOP_ANGLE)


def side(x):
    return wall(x, B_IN, B_OUT, SIDE_ANGLE)


def _two_sided(cells: int, first_fraction: float) -> tuple[float, str]:
    """Geometric expansion ratio of each half of a two-sided graded direction so the wall cell is
    ``first_fraction`` of the full extent; blockMesh multi-grading entry."""
    half = cells//2
    lo, hi = 1.0, 1.6
    for _ in range(200):
        r = 0.5*(lo+hi)
        first = 0.5*(r-1)/(r**half-1)
        lo, hi = (r, hi) if first > first_fraction else (lo, r)
    ratio = r**(half-1)
    return ratio, f'((0.5 0.5 {ratio:.6g}) (0.5 0.5 {1/ratio:.6g}))'


def block_mesh_dict(level: str = 'base') -> str:
    """A unit block in y and z (mapped onto the walls afterwards), three graded segments in x."""
    n_in, n_diff, n_out, ny, nz = LEVELS[level]
    # first cells ~1.6e-5 m (height) and ~2e-5 m (width) in the inlet duct: y+ about 1
    _, gy = _two_sided(ny, 1.6e-5/H_IN)
    _, gz = _two_sided(nz, 2.0e-5/B_IN)
    lengths = np.diff(X_SEGMENTS)
    total = lengths.sum()
    nx = n_in + n_diff + n_out
    # inlet duct uniform, diffuser uniform, outlet duct expanding 4x towards the outlet
    gx = (f'(({lengths[0]/total:.6g} {n_in/nx:.6g} 1) ({lengths[1]/total:.6g} {n_diff/nx:.6g} 1) '
          f'({lengths[2]/total:.6g} {n_out/nx:.6g} 4))')
    x0, x1 = X_SEGMENTS[0], X_SEGMENTS[-1]
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
// Unit block in y and z: map_onto_walls() scales y by the local height and z by the local width.
scale 1;
vertices
(
    ({x0} 0 0) ({x1} 0 0) ({x1} 1 0) ({x0} 1 0)
    ({x0} 0 1) ({x1} 0 1) ({x1} 1 1) ({x0} 1 1)
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading ({gx} {gy} {gz}) );
edges ();
boundary
(
    inlet
    {{
        type            mappedPatch;
        sampleMode      nearestCell;
        sampleRegion    region0;
        samplePatch     none;
        offsetMode      uniform;
        offset          ({INLET_SAMPLE_OFFSET} 0 0);
        faces           ((0 4 7 3));
    }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    walls  {{ type wall; faces ((0 3 2 1) (4 5 6 7) (0 1 5 4) (3 7 6 2)); }}
);
mergePatchPairs ();
"""


def cells(level: str = 'base') -> int:
    n_in, n_diff, n_out, ny, nz = LEVELS[level]
    return (n_in+n_diff+n_out)*ny*nz


def map_onto_walls(case: Path) -> dict:
    """Scale the unit-block points onto the duct walls; returns the point count and extremes."""
    path = Path(case)/'constant/polyMesh/points'
    text = path.read_text()
    match = re.search(r'\n(\d+)\s*\n\(\n', text)
    if not match:
        raise ValueError('Unexpected points file layout')
    n = int(match.group(1))
    start = match.end()
    end = text.index('\n)', start)
    pts = np.array([[float(v) for v in line.strip('()').split()] for line in text[start:end].splitlines()])
    if len(pts) != n:
        raise ValueError('Point count mismatch')
    unit = all(abs(pts[:, c].min()) < 1e-12 and abs(pts[:, c].max()-1) < 1e-12 for c in (1, 2))
    if not unit:
        raise ValueError('Points already mapped or not a unit block')
    pts[:, 1] *= top(pts[:, 0])
    pts[:, 2] *= side(pts[:, 0])
    body = '\n'.join(f'({p[0]:.12g} {p[1]:.12g} {p[2]:.12g})' for p in pts)
    path.write_text(text[:start] + body + text[end:])
    return {'mapped_points': n, 'max_height': float(pts[:, 1].max()), 'max_width': float(pts[:, 2].max())}


# ---------------------------------------------------------------------------------------------
# case files
# ---------------------------------------------------------------------------------------------
def _header(cls: str, obj: str, location: str = '') -> str:
    loc = f'    location    "{location}";\n' if location else ''
    return ('FoamFile\n{\n    version     2.0;\n    format      ascii;\n'
            f'    class       {cls};\n{loc}    object      {obj};\n}}\n')


def inflow_turbulence() -> tuple[float, float]:
    """Initial and inlet-start k and omega: 5 % intensity, mixing length 0.07 h."""
    k = 1.5*(0.05*U_BULK)**2
    return k, math.sqrt(k)/(0.09**0.25*0.07*H_IN)


def boundary_fields() -> dict[str, str]:
    k, omega = inflow_turbulence()

    def field(cls, obj, dims, internal, inlet, outlet, walls):
        return _header(cls, obj, '0') + f"""
dimensions      {dims};
internalField   uniform {internal};

boundaryField
{{
    inlet  {{ {inlet} }}
    outlet {{ {outlet} }}
    walls  {{ {walls} }}
}}
"""
    mapped = 'type mapped; interpolationScheme cell; setAverage false; average {avg}; value uniform {val};'
    return {
        'U': field('volVectorField', 'U', '[0 1 -1 0 0 0 0]', f'({U_BULK} 0 0)',
                   f'type mapped; interpolationScheme cell; setAverage true; average ({U_BULK} 0 0); '
                   f'value uniform ({U_BULK} 0 0);',
                   'type inletOutlet; inletValue uniform (0 0 0); value uniform (0 0 0);', 'type noSlip;'),
        'p': field('volScalarField', 'p', '[0 2 -2 0 0 0 0]', '0', 'type zeroGradient;',
                   'type fixedValue; value uniform 0;', 'type zeroGradient;'),
        'k': field('volScalarField', 'k', '[0 2 -2 0 0 0 0]', f'{k}', mapped.format(avg=k, val=k),
                   f'type inletOutlet; inletValue uniform {k}; value uniform {k};',
                   f'type kLowReWallFunction; value uniform {k};'),
        'omega': field('volScalarField', 'omega', '[0 0 -1 0 0 0 0]', f'{omega}', mapped.format(avg=omega, val=omega),
                       f'type inletOutlet; inletValue uniform {omega}; value uniform {omega};',
                       f'type omegaWallFunction; value uniform {omega};'),
        'nut': field('volScalarField', 'nut', '[0 2 -1 0 0 0 0]', '0', 'type calculated; value uniform 0;',
                     'type calculated; value uniform 0;', 'type nutLowReWallFunction; value uniform 0;'),
    }


def write_case(root: Path, spec, level: str = 'base', end_time: int = 8000) -> dict:
    """Write every case file except the mesh. No OpenFOAM utility is run: blockMesh and then
    ``map_onto_walls`` run on the compute node."""
    from ..casegen import render_turbulence_properties
    from . import hump
    root = Path(root)
    for sub in ('system', 'constant', '0'):
        (root/sub).mkdir(parents=True, exist_ok=True)
    (root/'system/blockMeshDict').write_text(block_mesh_dict(level))
    control = hump.control_dict(end_time)
    if spec is not None:
        control += '\nlibs            ("libkOmegaSSTBasis.so");\n'
    (root/'system/controlDict').write_text(control)
    (root/'system/fvSchemes').write_text(hump.fv_schemes())
    (root/'system/fvSolution').write_text(hump.fv_solution())
    (root/'constant/transportProperties').write_text(
        _header('dictionary', 'transportProperties', 'constant') + f'\ntransportModel  Newtonian;\nnu              {NU};\n')
    (root/'constant/turbulenceProperties').write_text(render_turbulence_properties(spec, mode='expressions'))
    for name, text in boundary_fields().items():
        (root/'0'/name).write_text(text)
    return {'cells': cells(level), 'level': level, 're_h': U_BULK*H_IN/NU, 'u_bulk': U_BULK,
            'post_mesh': 'map_onto_walls'}


# ---------------------------------------------------------------------------------------------
# scoring against Cherry, Elkins and Eaton (2008)
# ---------------------------------------------------------------------------------------------
WALL_MARGIN = 0.0015      # MRV points closer than this to a wall are not scored (partial volume)
X_SCORED = (0.0, 0.200)


def mrv_bulk_velocity(m: dict) -> float:
    """Bulk velocity of the MRV data: its volume flow through the inlet duct (voxels inside the
    walls, x from -20 to -3 mm) over the duct area. About 0.87 m/s against the nominal 1 m/s: the
    experiment matched Re = 10,000, so velocities are compared as U/U_b with each side's own bulk."""
    x, y, z, vx = m['x'], m['y'], m['z'], m['vx']
    dy = float(np.abs(np.diff(y[:, 0, 0])).mean())
    dz = float(np.abs(np.diff(z[0, 0, :])).mean())
    flows = []
    for j in np.flatnonzero((x[0, :, 0] > -0.020) & (x[0, :, 0] < -0.003)):
        Y, Z, V = y[:, j, :], z[:, j, :], vx[:, j, :]
        inside = (Y > 0) & (Y < H_IN) & (Z > 0) & (Z < B_IN)
        flows.append(V[inside].sum()*dy*dz)
    return float(np.mean(flows))/(H_IN*B_IN)


def reference(stride: int = 2) -> dict:
    """MRV points inside the diffuser (wall margin 1.5 mm, signal present), their mean velocity
    scaled to the model's bulk velocity (see mrv_bulk_velocity), and the bottom-wall pressure
    coefficients (x in metres)."""
    from scipy.io import loadmat
    m = loadmat(DATA/'Diffuser 1 data.mat')
    ub = mrv_bulk_velocity(m)
    x, y, z, mg = m['x'], m['y'], m['z'], m['mg']
    v = np.stack([m['vx'], m['vy'], m['vz']], axis=-1)*(U_BULK/ub)
    sl = (slice(None, None, stride),)*3
    x, y, z, mg, v = x[sl].ravel(), y[sl].ravel(), z[sl].ravel(), mg[sl].ravel(), v[sl].reshape(-1, 3)
    inside = ((x >= X_SCORED[0]) & (x <= X_SCORED[1]) & (y > WALL_MARGIN) & (y < top(x)-WALL_MARGIN)
              & (z > WALL_MARGIN) & (z < side(x)-WALL_MARGIN))
    signal = mg > 0.05*np.percentile(mg[inside], 99)
    keep = inside & signal
    cp = []
    import zipfile
    from xml.etree import ElementTree as ET
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with zipfile.ZipFile(DATA/'UFR4-16_Cp_Re%3D10000.xlsx') as book:
        root = ET.fromstring(book.read('xl/worksheets/sheet1.xml'))
    for row in root.iter(ns+'row'):
        cells_ = row.findall(ns+'c')
        if any(c.get('t') == 's' for c in cells_):
            continue                                  # header row (shared strings)
        values = [c.find(ns+'v') for c in cells_]
        try:
            cp.append([float(values[0].text), float(values[1].text)])
        except (TypeError, ValueError, IndexError, AttributeError):
            continue
    cp = np.asarray(cp)
    return {'points': np.c_[x[keep], y[keep], z[keep]], 'velocity': v[keep],
            'cp_x': cp[:, 0]*LENGTH, 'cp': cp[:, 1], 'mrv_bulk_velocity': ub}


def sample(tree, values, targets, k=8):
    dist, index = tree.query(targets, k=k)
    weight = 1.0/np.maximum(dist, 1e-12)**2
    return np.einsum('ij,ij...->i...', weight, values[index])/weight.sum(axis=1).reshape((-1,)+(1,)*(values.ndim-1))


def score(centres: np.ndarray, u: np.ndarray, wall_centres: np.ndarray, wall_p: np.ndarray) -> tuple[dict, dict]:
    """Errors and observables from cell centres and velocity and the wall patch (face centres and
    kinematic pressure)."""
    from scipy.spatial import cKDTree
    ref = reference()
    region = (centres[:, 0] > X_SCORED[0]-0.01) & (centres[:, 0] < X_SCORED[1]+0.01)
    tree = cKDTree(centres[region])
    model = sample(tree, u[region], ref['points'])
    measured = ref['velocity']
    errors = {'velocity_rmse': float(np.sqrt(np.mean(np.sum((model-measured)**2, axis=1))))/U_BULK}
    f_model, f_measured = float(np.mean(model[:, 0] < 0)), float(np.mean(measured[:, 0] < 0))
    errors['backflow_fraction_error'] = abs(f_model-f_measured)
    # bottom-wall centre line: faces on y = 0 nearest to (x, 0, width/2)
    bottom = np.abs(wall_centres[:, 1]) < 1e-9
    targets = np.c_[ref['cp_x'], np.zeros(len(ref['cp_x'])), 0.5*side(ref['cp_x'])]
    nearest = cKDTree(wall_centres[bottom]).query(targets)[1]
    p = wall_p[bottom][nearest]
    cp = (p-p[0])/(0.5*U_BULK**2) + ref['cp'][0]
    errors['wall_pressure_rmse'] = float(np.sqrt(np.mean((cp-ref['cp'])**2)))
    streamwise = {}
    for xh in (2, 5, 8, 12, 15):
        at = np.abs(ref['points'][:, 0]-xh*H_IN) < 0.6e-3*2
        if at.any():
            streamwise[f'x{xh}h'] = float(np.sqrt(np.mean((model[at, 0]-measured[at, 0])**2)))/U_BULK
    obs = {'backflow_fraction': f_model, 'backflow_fraction_measured': f_measured,
           'mrv_bulk_velocity': ref['mrv_bulk_velocity'],
           'pressure_recovery': float(cp[-1]), 'pressure_recovery_measured': float(ref['cp'][-1]),
           'scored_points': int(len(measured)), 'station_streamwise_rmse': streamwise,
           'diffuser_u_mean': float(np.mean(model[:, 0]))/U_BULK}
    return errors, obs
