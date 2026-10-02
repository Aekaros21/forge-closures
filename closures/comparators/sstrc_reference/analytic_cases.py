"""Manufactured velocity fields with closed-form S, Omega, r*, r~ and f_r1 of SST-RC.

Each family below gives, at chosen points, the inputs of the pointwise correction (velocity
gradient A_ij = du_i/dx_j of the velocity relative to the frame, the steady Lagrangian
derivative DS_ij/Dt = u_k dS_ij/dx_k in that frame WITHOUT the frame terms, the frame angular
velocity Omega^rot, and the SST omega) and the closed-form values derived by hand from
[SM09] Eqs. (1), (4)-(11) (Smirnov and Menter 2009, J. Turbomach. 131, 041010, pp. 041010-1/2;
see reference.py for the full citation list). The derivations are in the docstrings.

Families (all steady):

1. ``couette``   plane shear u = G (n.x) t in an inertial frame (plane Couette flow; any
                 orientation t, n): S = W = |G|, r* = 1, r~ = 0, f_r1 = 1 exactly ([SS97]
                 p. 301: "the constraint f_r1(1, 0) = 1 is satisfied, for thin shear flows
                 without curvature").
2. ``rotshear``  the same plane shear seen in a frame rotating at Omega^rot (the 1D rotating
                 channel locally; Omega_z = Ro/2 with h = U_m = 1). With e1 = t, e2 = n,
                 e3 = t x n and O_a = Omega.e_a:
                     S = |G|,  W^2 = (G - 2 O_3)^2 + 4 O_1^2 + 4 O_2^2,
                     numerator of Eq. (6) N = (G^2/2) [O_1^2 + O_2^2 - 2 G O_3 + 4 O_3^2],
                     r* = S/W,  r~ = N/(W D^3),  D = max(|G|, 0.3 omega).
                 For Omega = O_3 e3 and D = S: r~ = -O_3 sign(G - 2 O_3)/|G| (destabilising,
                 r~ < 0, where G O_3 > 0 and |G| > 2|O_3|: the pressure side of the channel).
3. ``solidbody`` u = Omega_0 x x (seen from a frame rotating at Omega_f parallel to Omega_0:
                 u_rel = (Omega_0 - Omega_f) x x): S = 0, W = 2|Omega_0| in every frame,
                 r* = 0, r~ = 0 (D = 0.3 omega > 0), f_rotation = -c_r1 = -1, f_r1 = 0.
4. ``azimuthal`` u = f(r) e_theta about the z axis (curved channel / Taylor-Couette), absolute
                 profile f_abs, seen from a frame rotating at Omega_f e_z (f = f_abs - Omega_f r):
                     A = f' e_theta e_r^T - (f/r) e_r e_theta^T,
                     S_ij = (f' - f/r)/2 (e_theta e_r^T + e_r e_theta^T)_ij,
                     DS/Dt (relative) = (f/r) dS/dtheta = (f/r)(f' - f/r)(e_theta e_theta^T - e_r e_r^T),
                     S = |f_abs' - f_abs/r|,  W = |f_abs' + f_abs/r| (absolute vorticity),
                     r~ = sign(f_abs' + f_abs/r) (f_abs/r) S^2/D^3.
                 The result depends on the absolute flow only (frame invariance), and with D = S
                 reduces to r~ = e/S with e = (U_theta/r) sign(d[r U_theta]/dr) of [SS97] p. 299.
                 Profiles: Taylor-Couette f = a r + b/r (exact laminar solution) and a curved
                 channel with R1/R2 = 0.8 ([SM09] Fig. 3, p. 041010-3) and a parabolic profile.
5. ``general3d`` a divergence-free quadratic 3D field (no closed form; the reference value is
                 the expectation). Exercises every tensor component for the C++ comparison.

``build_states()`` returns the states; ``write_states``/``write_expected`` write the files
read by the C++ unit executable closures/comparators/sstrc_unit and by compare_cpp.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import reference as ref

HERE = Path(__file__).resolve().parent
STATES_FILE = HERE / 'manufactured_states.txt'
EXPECTED_FILE = HERE / 'manufactured_expected.txt'

#: rotating channels of this work: Omega = (0, 0, Ro/2) with h = U_m = 1 (tedp rotchan.omega_for)
CHANNEL_RO = (0.10, 0.50)


@dataclass
class State:
    id: str
    family: str
    A: np.ndarray          # du_i/dx_j (relative velocity), row i, column j
    DSDt: np.ndarray       # u_k dS_ij/dx_k (relative frame, no frame terms)
    Om: np.ndarray         # frame angular velocity
    omega: float           # SST omega
    u: np.ndarray | None = None      # relative velocity (for information / FD checks)
    x: np.ndarray | None = None      # position
    closed: dict | None = None       # closed-form S, W, D, rStar, rTilde, fRotation, fr1 (None: no closed form)
    note: str = ''
    meta: dict = field(default_factory=dict)


def closed_from(S, W, D, N):
    r_star = S / W
    r_tilde = N / (W * D ** 3)
    f_rot = float(ref.f_rotation(r_star, r_tilde))
    return dict(S=S, W=W, D=D, numerator=N, rStar=r_star, rTilde=r_tilde, fRotation=f_rot,
                fr1=min(max(f_rot, ref.FR1_MIN), ref.FR1_MAX))


def _omega_for_branch(S, branch, W=None):
    """SST omega that puts D on the requested branch of Eq. (11)."""
    if branch == 'DS':        # 0.09 omega^2 = 0.25 S^2 < S^2  -> D = S
        return 0.5 * S / 0.3
    if branch == 'Dw':        # 0.09 omega^2 = 9 S^2 > S^2     -> D = 3 S = 0.3 omega
        return 10.0 * S
    raise ValueError(branch)


def _frame(t, n):
    t = np.asarray(t, float); n = np.asarray(n, float)
    t = t / np.linalg.norm(t)
    n = n - (n @ t) * t
    n = n / np.linalg.norm(n)
    return t, n, np.cross(t, n)


# ---------------------------------------------------------------------------------------------
# 1-2. plane shear, inertial and rotating
# ---------------------------------------------------------------------------------------------
def plane_shear(sid, G, Om, t=(1, 0, 0), n=(0, 1, 0), branch='DS', x=(0.1, -0.3, 0.2), family='rotshear'):
    t, n, b = _frame(t, n)
    Om = np.asarray(Om, float)
    x = np.asarray(x, float)
    A = G * np.outer(t, n)                      # u = G (n.x) t  ->  du_i/dx_j = G t_i n_j
    u = G * (n @ x) * t
    DSDt = np.zeros((3, 3))                     # S uniform
    O1, O2, O3 = Om @ t, Om @ n, Om @ b
    S = abs(G)
    W = np.sqrt((G - 2 * O3) ** 2 + 4 * O1 ** 2 + 4 * O2 ** 2)
    N = 0.5 * G ** 2 * (O1 ** 2 + O2 ** 2 - 2 * G * O3 + 4 * O3 ** 2)
    omega = _omega_for_branch(S, branch)
    D = max(S, 0.3 * omega)
    return State(sid, family, A, DSDt, Om, omega, u=u, x=x, closed=closed_from(S, W, D, N),
                 meta=dict(G=G, t=t.tolist(), n=n.tolist(), branch=branch))


# ---------------------------------------------------------------------------------------------
# 3. solid-body rotation
# ---------------------------------------------------------------------------------------------
def solid_body(sid, Om0, Omf, x=(0.4, -0.7, 0.25), omega=3.0):
    Om0 = np.asarray(Om0, float); Omf = np.asarray(Omf, float); x = np.asarray(x, float)
    rel = Om0 - Omf
    # u = rel x x -> du_i/dx_j = eps_ikj rel_k
    A = np.einsum('ikj,k->ij', ref.EPS, rel)
    u = np.cross(rel, x)
    DSDt = np.zeros((3, 3))                     # S = 0 everywhere
    S, W, D, N = 0.0, 2 * np.linalg.norm(Om0), 0.3 * omega, 0.0
    closed = dict(S=S, W=W, D=D, numerator=N, rStar=0.0, rTilde=0.0, fRotation=-ref.CR1, fr1=0.0)
    return State(sid, 'solidbody', A, DSDt, Omf, omega, u=u, x=x, closed=closed,
                 meta=dict(Om0=Om0.tolist()))


# ---------------------------------------------------------------------------------------------
# 4. azimuthal flows (curved channel, Taylor-Couette), possibly seen from a rotating frame
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    name: str
    f: object      # absolute u_theta(r)
    fp: object     # d u_theta/dr


def taylor_couette(a, b):
    return Profile(f'tc_a{a}_b{b}', lambda r: a * r + b / r, lambda r: a - b / r ** 2)


def curved_channel(R1, R2, U0):
    """Parabolic profile in the gap R1 < r < R2 (curved channel, [SM09] Fig. 3 geometry R1/R2)."""
    h = R2 - R1
    return Profile(f'cc_R1{R1}_R2{R2}', lambda r: 4 * U0 * (r - R1) * (R2 - r) / h ** 2,
                   lambda r: 4 * U0 * (R1 + R2 - 2 * r) / h ** 2)


def azimuthal(sid, prof: Profile, r, theta, Omf_z=0.0, z=0.3, branch='DS'):
    f_abs, fp_abs = prof.f(r), prof.fp(r)
    f, fp = f_abs - Omf_z * r, fp_abs - Omf_z            # relative profile
    er = np.array([np.cos(theta), np.sin(theta), 0.0])
    et = np.array([-np.sin(theta), np.cos(theta), 0.0])
    x = r * er + np.array([0, 0, z])
    u = f * et
    A = fp * np.outer(et, er) - (f / r) * np.outer(er, et)
    DSDt = (f / r) * (fp - f / r) * (np.outer(et, et) - np.outer(er, er))
    S = abs(fp_abs - f_abs / r)
    vort = fp_abs + f_abs / r
    W = abs(vort)
    omega = _omega_for_branch(S, branch)
    D = max(S, 0.3 * omega)
    N = vort * S ** 2 * (f_abs / r)                     # = W * sign(vort) (f_abs/r) S^2
    return State(sid, 'azimuthal', A, DSDt, np.array([0.0, 0.0, Omf_z]), omega, u=u, x=x,
                 closed=closed_from(S, W, D, N),
                 meta=dict(profile=prof.name, r=r, theta=theta, branch=branch))


# ---------------------------------------------------------------------------------------------
# 5. divergence-free quadratic 3D field (no closed form)
# ---------------------------------------------------------------------------------------------
def quadratic_field(seed=20261002):
    """u_i = a_i + B_ij x_j + C_ijk x_j x_k / 2, with tr B = 0 and C_iik = 0 (div u = 0)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=3)
    B = rng.normal(size=(3, 3)); B -= np.trace(B) / 3 * np.eye(3)
    C = rng.normal(size=(3, 3, 3)); C = 0.5 * (C + np.swapaxes(C, 1, 2))
    for k in range(3):
        tr = C[0, 0, k] + C[1, 1, k] + C[2, 2, k]
        C[0, 0, k] -= tr
        if k != 0:
            C[0, k, 0] = C[0, 0, k]
    return a, B, C


def quadratic_state(sid, x, Om, omega, field_=None):
    a, B, C = field_ if field_ is not None else quadratic_field()
    x = np.asarray(x, float)
    u = a + B @ x + 0.5 * np.einsum('ijk,j,k->i', C, x, x)
    A = B + np.einsum('ijk,k->ij', C, x)
    H = C                                              # d2u_i/dx_j dx_k
    DSDt = ref.lagrangian_strain_derivative(u, H)
    return State(sid, 'general3d', A, DSDt, np.asarray(Om, float), float(omega), u=u, x=x, closed=None,
                 meta=dict(field_seed=20261002))


# ---------------------------------------------------------------------------------------------
def build_states():
    st = []
    # 1. plane Couette shear, inertial frame: f_r1 = 1 exactly
    for i, (G, t, n, br) in enumerate([(1.0, (1, 0, 0), (0, 1, 0), 'DS'), (-3.7, (1, 0, 0), (0, 1, 0), 'DS'),
                                       (250.0, (1, 0, 0), (0, 1, 0), 'Dw'),
                                       (2.3, (1, 2, -1), (0.5, -1, 3), 'DS'), (-0.8, (0, 0, 1), (1, 1, 0), 'Dw')]):
        st.append(plane_shear(f'couette_{i}', G, (0, 0, 0), t, n, br, family='couette'))
    # 2. plane shear in the rotating channels' frames: Omega = (0, 0, Ro/2)
    for ro in CHANNEL_RO:
        oz = ro / 2
        tag = f'ro{int(round(ro * 100)):02d}'
        for j, (G, br) in enumerate([(50.0, 'DS'), (-50.0, 'DS'), (2.0, 'DS'), (-2.0, 'DS'),
                                     (2 * oz * 1.05, 'DS'), (2 * oz * 0.95, 'DS'), (0.5 * oz, 'DS'),
                                     (-0.5 * oz, 'DS'), (2.0, 'Dw'), (-2.0, 'Dw'), (2 * oz * 1.05, 'Dw')]):
            st.append(plane_shear(f'rotshear_{tag}_{j}', G, (0, 0, oz), branch=br))
    # general rotation vector and orientation (tests every eps component)
    st.append(plane_shear('rotshear_gen_0', 1.7, (0.3, -0.2, 0.45), (1, 0, 0), (0, 1, 0), 'DS'))
    st.append(plane_shear('rotshear_gen_1', -0.9, (0.3, -0.2, 0.45), (1, 2, -1), (0.5, -1, 3), 'DS'))
    st.append(plane_shear('rotshear_gen_2', 4.0, (-1.1, 0.7, 0.2), (0, 1, 1), (1, 0, 0), 'Dw'))
    st.append(plane_shear('rotshear_gen_3', 0.6, (0.0, 0.0, -0.25), (0, 1, 0), (0, 0, 1), 'DS'))
    # 3. solid-body rotation, inertial and co-rotating frames
    st.append(solid_body('solidbody_0', (0, 0, 1.0), (0, 0, 0)))
    st.append(solid_body('solidbody_1', (0, 0, 1.0), (0, 0, 0.4)))
    st.append(solid_body('solidbody_2', (0.3, -0.5, 0.8), (0, 0, 0)))
    st.append(solid_body('solidbody_3', (0.3, -0.5, 0.8), (0.15, -0.25, 0.4)))
    # 4. azimuthal flows: Taylor-Couette and curved channel, inertial and rotating frames
    tc = taylor_couette(0.6, 0.35)
    tc_neg = taylor_couette(-0.4, 1.2)
    cc = curved_channel(0.8, 1.0, 1.0)
    k = 0
    for prof, radii in ((tc, (0.7, 1.0, 1.6)), (tc_neg, (0.8, 1.3)), (cc, (0.82, 0.86, 0.93, 0.97))):
        for r in radii:
            for theta, Omf, br in ((0.0, 0.0, 'DS'), (2.1, 0.0, 'DS'), (-0.7, 0.35, 'DS'), (1.3, -0.8, 'Dw')):
                st.append(azimuthal(f'azimuthal_{k}', prof, r, theta, Omf, branch=br))
                k += 1
    # 5. general 3D field
    fld = quadratic_field()
    rng = np.random.default_rng(7)
    for i in range(6):
        x = rng.uniform(-1, 1, 3)
        Om = (0, 0, 0) if i < 2 else rng.normal(scale=0.3, size=3)
        st.append(quadratic_state(f'general3d_{i}', x, Om, omega=float(rng.uniform(0.2, 8.0)), field_=fld))
    return st


HEADER = ('# SST-RC manufactured states (Smirnov & Menter 2009 Eqs. (1),(4)-(11)); generated by '
          'src/comparators/sstrc_reference/analytic_cases.py\n'
          '# columns: id A11 A12 A13 A21 A22 A23 A31 A32 A33 D11 D12 D13 D22 D23 D33 Om1 Om2 Om3 omega\n'
          '# A_ij = du_i/dx_j of the velocity relative to the frame (row i, column j); D_ij = DS_ij/Dt '
          'relative frame, steady, WITHOUT frame terms; Om = frame angular velocity Omega^rot; '
          'omega = SST omega\n')


def _fmt(v):
    return format(float(v) + 0.0, '.17g')  # no negative zeros


def write_states(path=STATES_FILE, states=None):
    states = build_states() if states is None else states
    lines = [HEADER.rstrip('\n')]
    for s in states:
        D = s.DSDt
        vals = list(s.A.reshape(-1)) + [D[0, 0], D[0, 1], D[0, 2], D[1, 1], D[1, 2], D[2, 2]] + list(s.Om) + [s.omega]
        lines.append(' '.join([s.id] + [_fmt(v) for v in vals]))
    Path(path).write_text('\n'.join(lines) + '\n')
    return path


def read_states(path=STATES_FILE):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        tok = line.split()
        if len(tok) != 20:
            raise ValueError(f'{path}: expected 20 columns, got {len(tok)}: {line[:60]}')
        v = np.array([float(t) for t in tok[1:]])
        A = v[:9].reshape(3, 3)
        d = v[9:15]
        D = np.array([[d[0], d[1], d[2]], [d[1], d[3], d[4]], [d[2], d[4], d[5]]])
        out[tok[0]] = dict(A=A, DSDt=D, Om=v[15:18], omega=v[18])
    return out


EXPECTED_COLUMNS = ('S', 'W', 'rStar', 'rTilde', 'fRotation', 'fr1')


def write_expected(path=EXPECTED_FILE, states=None):
    """Reference values (and closed-form values where they exist) for every state."""
    states = build_states() if states is None else states
    lines = ['# SST-RC expected values: reference.py (ref_*) and hand-derived closed forms (cf_*; nan '
             'where none); generated by analytic_cases.py',
             '# id family ' + ' '.join(f'ref_{c}' for c in EXPECTED_COLUMNS) + ' '
             + ' '.join(f'cf_{c}' for c in EXPECTED_COLUMNS)]
    for s in states:
        r = ref.evaluate(s.A, s.DSDt, s.Om, s.omega).as_dict()
        cf = s.closed or {}
        lines.append(' '.join([s.id, s.family] + [_fmt(r[c]) for c in EXPECTED_COLUMNS]
                              + [_fmt(cf.get(c, np.nan)) for c in EXPECTED_COLUMNS]))
    Path(path).write_text('\n'.join(lines) + '\n')
    return path


if __name__ == '__main__':
    print(write_states())
    print(write_expected())
