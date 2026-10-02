"""pytest: the NumPy reference of SST-RC (reference.py) reproduces the closed forms of
analytic_cases.py and independent formulas of Spalart and Shur (1997).

Run:  python3 -m pytest -q closures/comparators/sstrc_reference
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analytic_cases as ac  # noqa: E402
import reference as ref  # noqa: E402

RTOL = 1e-12
COLS = ('S', 'W', 'D', 'numerator', 'rStar', 'rTilde', 'fRotation', 'fr1')


def _close(a, b, rtol=RTOL, atol=1e-14):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=rtol, atol=atol)


STATES = ac.build_states()
CLOSED = [s for s in STATES if s.closed is not None]


# --------------------------------------------------------------------------- published constants
def test_published_constants():
    # Smirnov & Menter 2009, p. 041010-2, end of Sec. 2; Eq. (4); Eq. (11)
    assert (ref.CR1, ref.CR2, ref.CR3) == (1.0, 2.0, 1.0)
    assert (ref.FR1_MIN, ref.FR1_MAX) == (0.0, 1.25)
    assert ref.D_OMEGA_COEFF == 0.09


def test_levi_civita():
    for (i, j, k), s in {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}.items():
        assert ref.EPS[i, j, k] == s
    assert np.count_nonzero(ref.EPS) == 6


def test_f_rotation_neutral_point():
    # Spalart & Shur 1997 p. 301: f_r1(1, 0) = 1 for thin shear flows without curvature
    assert ref.f_rotation(1.0, 0.0) == pytest.approx(1.0, abs=0)
    # r* = 0 -> -c_r1; r* -> inf -> 2(1 + c_r1)[...] - c_r1
    assert ref.f_rotation(0.0, 0.3) == -1.0
    assert ref.f_rotation(np.inf, 0.0) == pytest.approx(3.0)


def test_limiter():
    f = ref.f_r1(np.array([-3.0, 0.0, 0.7, 1.25, 1.3, 9.0]))
    assert f.tolist() == [0.0, 0.0, 0.7, 1.25, 1.25, 1.25]


# --------------------------------------------------------------------------- closed forms
@pytest.mark.parametrize('state', CLOSED, ids=[s.id for s in CLOSED])
def test_closed_form(state):
    r = ref.evaluate(state.A, state.DSDt, state.Om, state.omega).as_dict()
    for c in COLS:
        assert _close(r[c], state.closed[c]), (state.id, c, float(r[c]), state.closed[c])


def test_every_family_present_and_both_D_branches():
    fams = {s.family for s in STATES}
    assert fams == {'couette', 'rotshear', 'solidbody', 'azimuthal', 'general3d'}
    branches = {s.meta.get('branch') for s in STATES}
    assert {'DS', 'Dw'} <= branches
    # both limiter bounds and the unlimited range are exercised
    fr = np.array([float(ref.evaluate(s.A, s.DSDt, s.Om, s.omega).fRotation) for s in STATES])
    assert (fr > 1.25).any() and (fr < 0).any() and ((fr > 0.05) & (fr < 1.2) & (np.abs(fr - 1) > 1e-3)).any()


def test_couette_is_neutral():
    for s in STATES:
        if s.family == 'couette':
            r = ref.evaluate(s.A, s.DSDt, s.Om, s.omega)
            assert float(r.rStar) == pytest.approx(1.0, rel=1e-14)
            assert abs(float(r.rTilde)) < 1e-15
            assert float(r.fr1) == pytest.approx(1.0, rel=1e-14)


def test_solid_body_is_fully_stabilised_in_every_frame():
    for s in STATES:
        if s.family == 'solidbody':
            r = ref.evaluate(s.A, s.DSDt, s.Om, s.omega)
            assert float(r.S) < 1e-15 and float(r.fr1) == 0.0
            assert float(r.W) == pytest.approx(2 * np.linalg.norm(s.meta['Om0']), rel=1e-14)


# --------------------------------------------------------------------------- sign conventions
@pytest.mark.parametrize('ro', ac.CHANNEL_RO)
def test_channel_sign_convention(ro):
    """Flow +x, Omega = +z (Ro/2): the -y wall (dU/dy > 0) is the pressure side, destabilised
    (r~ < 0, f_r1 > 1); the +y wall (dU/dy < 0) is stabilised (r~ > 0, f_r1 < 1). The Coriolis
    acceleration -2 Omega x u of tabulatedAccelerationSource points to -y."""
    oz = ro / 2
    for G, sign in ((40.0, +1), (-40.0, -1)):
        s = ac.plane_shear('x', G, (0, 0, oz))
        r = ref.evaluate(s.A, s.DSDt, s.Om, s.omega)
        assert np.sign(float(r.rTilde)) == -sign
        assert np.sign(float(r.fr1) - 1.0) == sign
        assert float(r.rTilde) == pytest.approx(-oz / G, rel=1e-13)   # r~ = -O_z/G for |G| > 2|O_z|
    assert np.sign(np.cross([0, 0, oz], [1, 0, 0])[1] * -2) == -1


def test_frame_rotation_tensor_convention():
    # Om_F,ij = eps_mji Omega_m: Omega = (0,0,Oz) -> Om_F,12 = -Oz, Om_F,21 = +Oz
    T = ref.frame_rotation_tensor([0, 0, 0.7])
    assert T[0, 1] == -0.7 and T[1, 0] == 0.7 and np.count_nonzero(T) == 2
    # absolute rotation of shear G in frame Oz: Omega_12 = G/2 - Oz
    W = ref.rotation_rate_absolute(np.array([[0, 2.0, 0], [0, 0, 0], [0, 0, 0]]), [0, 0, 0.3])
    assert W[0, 1] == pytest.approx(1.0 - 0.3)


def test_frame_term_is_corotational_commutator():
    rng = np.random.default_rng(1)
    for _ in range(20):
        S = rng.normal(size=(3, 3)); S = S + S.T
        Om = rng.normal(size=3)
        F = ref.frame_rotation_tensor(Om)
        assert _close(ref.frame_term(S, Om), F @ S - S @ F)


# --------------------------------------------------------------------------- frame invariance
def test_azimuthal_frame_invariance():
    """The same absolute azimuthal flow computed in frames rotating at different rates gives the
    same S, Omega, r*, r~ and f_r1: checks the sign of both frame terms of Eq. (6) and Eq. (8)."""
    rng = np.random.default_rng(3)
    for prof in (ac.taylor_couette(0.6, 0.35), ac.curved_channel(0.8, 1.0, 1.0), ac.taylor_couette(-0.4, 1.2)):
        for r in (0.83, 0.9, 0.97):
            base = ac.azimuthal('b', prof, r, 0.4, 0.0)
            rb = ref.evaluate(base.A, base.DSDt, base.Om, base.omega).as_dict()
            for Omf in rng.normal(scale=1.0, size=4):
                s = ac.azimuthal('s', prof, r, float(rng.uniform(-3, 3)), float(Omf))
                rs = ref.evaluate(s.A, s.DSDt, s.Om, base.omega).as_dict()
                for c in COLS:
                    assert _close(rs[c], rb[c], rtol=1e-11), (c, Omf)
            # dropping the frame term of Eq. (6) breaks the invariance (so the test has teeth)
            s = ac.azimuthal('s', prof, r, 0.4, 0.5)
            S_t = ref.strain_rate(s.A)
            W_t = ref.rotation_rate_absolute(s.A, s.Om)
            N_wrong = 2 * np.einsum('ik,jk,ij->', W_t, S_t, s.DSDt)
            assert not np.isclose(N_wrong, float(rb['numerator']), rtol=1e-6)


def test_rotation_of_coordinates_objectivity():
    """Proper rotation Q of the coordinates (A -> Q A Q^T, D -> Q D Q^T, Omega -> Q Omega) leaves
    every scalar unchanged."""
    rng = np.random.default_rng(5)
    for s in STATES:
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        a = ref.evaluate(s.A, s.DSDt, s.Om, s.omega).as_dict()
        b = ref.evaluate(q @ s.A @ q.T, q @ s.DSDt @ q.T, q @ s.Om, s.omega).as_dict()
        for c in COLS:
            assert _close(a[c], b[c], rtol=1e-10, atol=1e-13), (s.id, c)


# --------------------------------------------------------------------------- Spalart & Shur 1997
def test_numerator_equals_spalart_shur_eq4():
    """[SM09] Eq. (6) numerator = 2 S_mn S_mn e with e of [SS97] Eq. (4), p. 300 (explicit loops)."""
    rng = np.random.default_rng(11)
    for _ in range(40):
        A = rng.normal(size=(3, 3))
        D = rng.normal(size=(3, 3)); D = D + D.T
        Om = rng.normal(size=3)
        S = ref.strain_rate(A)
        e = ref.spalart_shur_e3d(A, D, Om)
        assert ref.r_tilde_numerator(A, D, Om) == pytest.approx(2 * np.sum(S * S) * e, rel=1e-11)


def test_2d_identity_with_spalart_shur_eq2():
    """2D incompressible: r~ = e S^2/D^3 with e = (Dalpha/Dt) sign(omega_z + 2 Omega) and
    Dalpha/Dt from [SS97] Eq. (2), p. 299 (the rate of turn of the strain principal axes)."""
    rng = np.random.default_rng(13)
    for _ in range(60):
        a, b, c = rng.normal(size=3)
        A = np.array([[a, b, 0], [c, -a, 0], [0, 0, 0]])              # div u = 0
        p, q = rng.normal(size=2)
        D = np.array([[p, q, 0], [q, -p, 0], [0, 0, 0]])              # trace-free DS/Dt
        Oz = rng.normal()
        omega = abs(rng.normal()) * 3
        r = ref.evaluate(A, D, [0, 0, Oz], omega)
        S11, S12 = a, 0.5 * (b + c)
        dadt = ref.spalart_shur_dalpha_dt_2d(S11, S12, p, q, Oz)
        vort = c - b                                                    # omega_z = dv/dx - du/dy
        e = dadt * np.sign(vort + 2 * Oz)
        assert float(r.W) == pytest.approx(abs(vort + 2 * Oz), rel=1e-13)
        assert float(r.rTilde) == pytest.approx(e * float(r.S) ** 2 / float(r.D) ** 3, rel=1e-10)


def test_azimuthal_matches_spalart_shur_curvature_measure():
    """[SS97] p. 299: in an azimuthal flow e = (U_theta/r) sign(d[r U_theta]/dr); with D = S the
    [SM09] measure is r~ = e/S."""
    for prof in (ac.taylor_couette(0.6, 0.35), ac.curved_channel(0.8, 1.0, 1.0)):
        for r in (0.81, 0.88, 0.95):
            s = ac.azimuthal('s', prof, r, 0.0, 0.0, branch='DS')
            res = ref.evaluate(s.A, s.DSDt, s.Om, s.omega)
            drU = prof.f(r) + r * prof.fp(r)
            e = prof.f(r) / r * np.sign(drU)
            assert float(res.rTilde) == pytest.approx(e / float(res.S), rel=1e-12)


def test_shur2000_variant_normalisation():
    """[S00] p. 785: D^2 = (S^2 + Omega^2)/2 and r~ = numerator/D^4 (SARC)."""
    s = ac.plane_shear('x', 3.0, (0, 0, 0.4))
    r = ref.evaluate(s.A, s.DSDt, s.Om, s.omega, variant='shur2000', cr2=12.0)
    G, Oz = 3.0, 0.4
    N = -G ** 2 * Oz * (G - 2 * Oz)
    D4 = (0.5 * (G ** 2 + (G - 2 * Oz) ** 2)) ** 2
    assert float(r.rTilde) == pytest.approx(N / D4, rel=1e-13)


# --------------------------------------------------------------------------- DS/Dt
def test_lagrangian_derivative_matches_finite_differences():
    """u_k dS_ij/dx_k from the analytic Hessian equals a central difference of S along u."""
    a, B, C = ac.quadratic_field()
    rng = np.random.default_rng(17)

    def S_at(x):
        return ref.strain_rate(B + np.einsum('ijk,k->ij', C, x))

    for _ in range(10):
        x = rng.uniform(-1, 1, 3)
        u = a + B @ x + 0.5 * np.einsum('ijk,j,k->i', C, x, x)
        h = 1e-5
        fd = (S_at(x + h * u) - S_at(x - h * u)) / (2 * h)
        assert _close(ref.lagrangian_strain_derivative(u, C), fd, rtol=1e-7, atol=1e-9)
    # divergence-free field
    assert abs(np.trace(B)) < 1e-14
    assert np.allclose(np.einsum('iik->k', C), 0, atol=1e-14)


def test_azimuthal_lagrangian_derivative_finite_difference():
    prof = ac.curved_channel(0.8, 1.0, 1.0)
    r, th, Omf = 0.9, 0.7, 0.3
    s = ac.azimuthal('s', prof, r, th, Omf)

    def S_at(x):
        rr = np.hypot(x[0], x[1]); tt = np.arctan2(x[1], x[0])
        return ref.strain_rate(ac.azimuthal('t', prof, rr, tt, Omf).A)

    h = 1e-6
    fd = (S_at(s.x + h * s.u) - S_at(s.x - h * s.u)) / (2 * h)
    assert _close(s.DSDt, fd, rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------- files
def test_states_file_is_current(tmp_path):
    p = ac.write_states(tmp_path / 's.txt')
    assert p.read_text() == ac.STATES_FILE.read_text()
    e = ac.write_expected(tmp_path / 'e.txt')
    assert e.read_text() == ac.EXPECTED_FILE.read_text()


def test_states_file_round_trip():
    back = ac.read_states()
    assert len(back) == len(STATES)
    for s in STATES:
        b = back[s.id]
        r1 = ref.evaluate(s.A, s.DSDt, s.Om, s.omega).as_dict()
        r2 = ref.evaluate(b['A'], b['DSDt'], b['Om'], b['omega']).as_dict()
        for c in COLS:
            assert _close(r1[c], r2[c], rtol=1e-15, atol=0)


def test_compare_cpp_tool(tmp_path):
    """compare_cpp.py reports zero for the reference itself and the injected error otherwise."""
    import compare_cpp
    out = tmp_path / 'cpp_fr1.txt'
    compare_cpp.write_reference_as_cpp(out)
    rep = compare_cpp.compare(out)
    assert rep['passed'] and rep['max_abs_diff']['fr1'] == 0.0
    lines = out.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('rotshear_ro50_3 '))
    tok = lines[i].split(); tok[6] = repr(float(tok[6]) + 1e-6); lines[i] = ' '.join(tok)
    out.write_text('\n'.join(lines) + '\n')
    rep = compare_cpp.compare(out)
    assert not rep['passed'] and rep['max_abs_diff']['fr1'] == pytest.approx(1e-6, rel=1e-6)
    assert rep['worst']['fr1']['id'] == 'rotshear_ro50_3'
    res = subprocess.run([sys.executable, str(HERE / 'compare_cpp.py'), str(out)], capture_output=True, text=True)
    assert res.returncode == 1 and 'FAIL' in res.stdout


def test_published_targets_consistent():
    t = json.loads((HERE / 'published_targets.json').read_text())
    for key in ('ro10', 'ro50'):
        c = t['rotating_channel'][key]
        y0 = c['velocity_maximum_y_over_H']['value']
        ratio = c['wall_shear_ratio_pressure_over_suction']['value']
        assert ratio == pytest.approx(y0 / (1 - y0), rel=1e-12)
        assert c['frame_omega'] == [0.0, 0.0, c['Ro'] / 2]
        lo, hi = c['wall_shear_ratio_pressure_over_suction']['accept']
        assert lo < ratio < hi and lo > 1.0
    assert t['flat_plate']['f_r1_expected'] == 1.0


# --------------------------------------------------------------------------- preflight checker
def _channel_result(tmp_path, name, case, y, u, ratio):
    p = tmp_path / f'{name}.json'
    p.write_text(json.dumps(dict(case_id=case, observables=dict(y=list(y), u=list(u), wall_friction_ratio=ratio))))
    return p


@pytest.mark.parametrize('key', ['ro10', 'ro50'])
def test_check_targets_channel(tmp_path, key):
    import check_targets
    t = json.loads((HERE / 'published_targets.json').read_text())['rotating_channel'][key]
    y = 2 * np.asarray(t['published_profile']['y_over_H']) - 1
    u = np.asarray(t['published_profile']['u_over_Um'])
    ratio = t['wall_shear_ratio_pressure_over_suction']['value']
    good = _channel_result(tmp_path, 'good', t['case_id'], y, u, ratio)
    assert check_targets.check(good)['passed']
    # sign error: mirrored profile and inverted ratio
    bad = _channel_result(tmp_path, 'mirror', t['case_id'], -y[::-1], u[::-1], 1 / ratio)
    rep = check_targets.check(bad)
    assert not rep['passed'] and not rep['checks']['velocity_maximum_y_over_h_case']['passed']
    # no correction: symmetric profile
    sym = 0.5 * (u + u[::-1])
    rep = check_targets.check(_channel_result(tmp_path, 'sym', t['case_id'], y, sym, 1.0))
    assert not rep['passed']


def test_check_targets_plate(tmp_path):
    import check_targets
    x = np.linspace(0.002, 1.98, 112)
    cf = 0.0025 + 0.003 * np.exp(-x / 0.05)

    def res(name, c, cd):
        p = tmp_path / f'{name}.json'
        p.write_text(json.dumps(dict(case_id='tmr_plate_137x97', observables=dict(
            wall_x=list(x), wall_cf=list(c), drag_coefficient=cd, wall_cf_rms=float(np.sqrt(np.mean(c ** 2)))))))
        return p
    sst = res('sst', cf, 0.0028)
    le = np.where(x < 0.1, 5e-3, 5e-4)               # leading edge may move by 0.5 %
    assert check_targets.check(res('ok', cf * (1 + le), 0.0028 * (1 + 5e-4)), sst)['passed']
    assert not check_targets.check(res('bad', cf * (1 + 2e-3), 0.0028), sst)['passed']
