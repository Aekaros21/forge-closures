"""Digitize Fig. 1 of Smirnov and Menter (2009) and write the SST-RC rotating-channel targets.

Fig. 1 (J. Turbomach. 131, 041010, p. 041010-3, "Developed channel flow at Re = 5800;
comparison with DNS of Kristoffersen and Andersson [6]") is vector graphics in the publisher's
PDF: each SST and SST-CC profile is a polyline whose vertices are the plotted solution points.
The page is converted to SVG with poppler's ``pdftocairo -svg`` and the polylines are read
exactly (no raster tracing). Panel (a): U/U_m against y/H, the profile for the k-th rotation
number (Ro = 0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.50) shifted up by 0.5 k; panel (b): u'v'/u_tau^2
against y/H, shifted up by k. The offsets are confirmed by every polyline starting and ending
exactly on its offset (wall value zero) and by the bulk integral of U/U_m being 1 to 0.04 %.

Outputs (this directory): ``sm09_fig1_digitized.json`` (all 14 profiles of each panel) and
``published_targets.json`` (targets and acceptance bands for the comparator preflight).

Usage: python digitize_sm09_fig1.py path/to/smirnov2009.pdf   (needs pdftocairo; the PDF is not distributed here)
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]   # repository root
PDF_DEFAULT = HERE / 'smirnov2009.pdf'   # the publisher's PDF of the paper, not distributed here
ROS = (0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.50)
NS = '{http://www.w3.org/2000/svg}'

# Axis calibration in page points (SVG of page 3, 612 x 792 pt), read from the tick marks and
# axis lines of Fig. 1 and verified against them below.
LEFT = dict(x0=76.20312509131399, x1=168.023437143676, y_at_0=160.23046910393793, y_at_top=40.679687247665925,
            top=4.5, bottom=0.0, offset_step=0.5)
RIGHT = dict(x0=188.934, x1=280.75, y_at_0=160.23046910393793, y_at_top=40.679687247665925, top=8.0, bottom=-2.0, offset_step=1.0)


def _transform(t):
    M = np.eye(3)
    for name, args in re.findall(r'(\w+)\(([^)]*)\)', t or ''):
        a = [float(v) for v in re.split(r'[ ,]+', args.strip()) if v]
        if name == 'matrix':
            T = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif name == 'translate':
            T = np.array([[1, 0, a[0]], [0, 1, a[1] if len(a) > 1 else 0], [0, 0, 1]])
        elif name == 'scale':
            T = np.diag([a[0], a[1] if len(a) > 1 else a[0], 1])
        else:
            raise ValueError(name)
        M = M @ T
    return M


def _subpaths(d):
    toks = re.findall(r'[MLCZmlcz]|-?[0-9.]+(?:e-?\d+)?', d)
    subs, cur, cmd, i = [], [], None, 0
    while i < len(toks):
        t = toks[i]
        if t in 'MLCZ':
            cmd = t; i += 1
            if t == 'M' and cur:
                subs.append(cur); cur = []
            if t == 'Z' and cur:
                cur.append(cur[0])
            continue
        if cmd in 'ML':
            cur.append((float(toks[i]), float(toks[i + 1]))); i += 2
        elif cmd == 'C':
            cur.append((float(toks[i + 4]), float(toks[i + 5]))); i += 6
        else:
            raise ValueError(f'unsupported path command {cmd}')
    if cur:
        subs.append(cur)
    return subs


def stroked_paths(svg_file):
    out = []

    def walk(el, M):
        for ch in el:
            tag = ch.tag.replace(NS, '')
            if tag == 'defs':
                continue
            T = M @ _transform(ch.get('transform'))
            if tag == 'path' and ch.get('stroke') and ch.get('stroke') != 'none':
                subs = []
                for s in _subpaths(ch.get('d', '')):
                    a = np.c_[np.array(s), np.ones(len(s))] @ T.T
                    subs.append(a[:, :2])
                out.append(dict(stroke_width=ch.get('stroke-width'), dash=ch.get('stroke-dasharray'), subs=subs))
            elif tag != 'path':
                walk(ch, T)
    walk(ET.parse(svg_file).getroot(), np.eye(3))
    return out


def _to_data(sub, cal, k):
    x = (sub[:, 0] - cal['x0']) / (cal['x1'] - cal['x0'])
    scale = (cal['y_at_0'] - cal['y_at_top']) / (cal['top'] - cal['bottom'])
    v = cal['bottom'] + (cal['y_at_0'] - sub[:, 1]) / scale - k * cal['offset_step']
    o = np.argsort(x, kind='stable')
    x, v = x[o], v[o]
    ux, inv = np.unique(np.round(x, 7), return_inverse=True)      # merge coincident vertices
    return ux, np.bincount(inv, weights=v) / np.bincount(inv)


def extract(pdf):
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / 'p3.svg'
        subprocess.run(['pdftocairo', '-svg', '-f', '3', '-l', '3', str(pdf), str(svg)], check=True)
        paths = stroked_paths(svg)
    curves = {}
    for panel, cal, xr in (('a', LEFT, (70, 175)), ('b', RIGHT, (183, 290))):
        cands = [p for p in paths if len(p['subs']) == 7 and all(len(s) > 50 for s in p['subs'])
                 and all(xr[0] < s[:, 0].min() and s[:, 0].max() < xr[1] for s in p['subs'])]
        solid = [p for p in cands if p['dash'] is None]
        dashed = [p for p in cands if p['dash'] is not None]
        if len(solid) != 1 or len(dashed) != 1:
            raise RuntimeError(f'panel {panel}: expected one solid (SST-CC) and one dashed (SST) set, '
                               f'found {len(solid)}/{len(dashed)}')
        for model, p in (('SST-CC', solid[0]), ('SST', dashed[0])):
            subs = sorted(p['subs'], key=lambda s: -s[0, 1])          # bottom (k = 0) first
            for k, s in enumerate(subs):
                # every polyline starts and ends on the axis ends at its offset (wall value 0)
                assert abs(s[0, 0] - cal['x0']) < 0.01 and abs(s[-1, 0] - cal['x1']) < 0.01
                scale = (cal['y_at_0'] - cal['y_at_top']) / (cal['top'] - cal['bottom'])
                ends = cal['bottom'] + (cal['y_at_0'] - s[[0, -1], 1]) / scale - k * cal['offset_step']
                assert np.all(np.abs(ends) < 2e-3), (panel, model, k, ends)   # 0.05 pt
                x, v = _to_data(s, cal, k)
                curves[(panel, model, ROS[k])] = (x, v)
    return curves


def velocity_max(x, u):
    """Vertex maximum refined by the parabola through it and its two neighbours."""
    k = int(np.argmax(u))
    q = np.polyfit(x[k - 1:k + 2], u[k - 1:k + 2], 2)
    return -q[1] / (2 * q[0]), float(np.polyval(q, -q[1] / (2 * q[0])))


def stress_zero(x, t):
    m = (x > 0.2) & (x < 0.9)
    xs, ts = x[m], t[m]
    j = np.where((ts[:-1] > 0) & (ts[1:] <= 0))[0]
    if len(j) != 1:
        raise RuntimeError('expected one zero crossing of the shear stress')
    j = j[0]
    return xs[j] - ts[j] * (xs[j + 1] - xs[j]) / (ts[j + 1] - ts[j])


def metrics(curves):
    out = {}
    for model in ('SST-CC', 'SST'):
        for ro in ROS:
            x, u = curves[('a', model, ro)]
            xb, tb = curves[('b', model, ro)]
            ym, um = velocity_max(x, u)
            y0 = stress_zero(xb, tb)
            u25, u75 = np.interp([0.25, 0.75], x, u)
            out[(model, ro)] = dict(
                bulk=float(np.trapezoid(u, x)), y_umax=float(ym), u_max=float(um), y_stress_zero=float(y0),
                u_025=float(u25), u_075=float(u75), du=float(u75 - u25))
    return out


def main(pdf=PDF_DEFAULT):
    pdf = Path(pdf)
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    curves = extract(pdf)
    met = metrics(curves)
    # --- consistency checks of the reading
    for (model, ro), m in met.items():
        assert abs(m['bulk'] - 1) < 1e-3, (model, ro, m['bulk'])
        assert abs(m['y_umax'] - m['y_stress_zero']) < 5e-3, (model, ro)
        if model == 'SST':
            assert abs(m['y_stress_zero'] - 0.5) < 1e-3 and abs(m['du']) < 1e-3
    plate_file = HERE / 'plate_apriori.json'
    plate = json.loads(plate_file.read_text()) if plate_file.exists() else None
    reading_unc = max(abs(m['y_umax'] - m['y_stress_zero']) for m in met.values())
    sst_sym = max(abs(m['y_stress_zero'] - 0.5) for (mod, _), m in met.items() if mod == 'SST')

    source = dict(citation='smirnov2009', reference='P. E. Smirnov and F. R. Menter, J. Turbomach. 131(4), 041010 (2009)',
                  figure='Fig. 1, p. 041010-3', pdf='archive/paper_pof/literature/pdfs/smirnov2009.pdf', pdf_sha256=sha,
                  method='vector polylines read from pdftocairo -svg of PDF page 3 (no raster tracing)',
                  conditions='fully developed plane channel, spanwise rotation, Re = U_m H/nu = 5800 (H channel width), '
                             'Ro = 2 Omega h/U_m = Omega H/U_m (Kristoffersen and Andersson 1993); SST-CC solved with ANSYS CFX 11, '
                             'y+ < 1, grid-independent (Sec. 4.1, pp. 041010-2/3)')
    dig = dict(source=source, axes=dict(a='U/U_m vs y/H, curve k shifted by +0.5 k', b="u'v'/u_tau^2 vs y/H, curve k shifted by +k"),
               rotation_numbers=list(ROS), profiles={})
    for (panel, model, ro), (x, v) in sorted(curves.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        dig['profiles'].setdefault(f'{panel}:{model}', {})[f'{ro:.2f}'] = dict(
            y_over_H=[round(float(a), 6) for a in x], value=[round(float(b), 6) for b in v])
    dig['metrics'] = {f'{model}:{ro:.2f}': m for (model, ro), m in met.items()}
    (HERE / 'sm09_fig1_digitized.json').write_text(json.dumps(dig, indent=1) + '\n')

    # ------------------------------------------------------------------ targets
    REL_RATIO = 0.08     # see 'tolerance_rationale'
    DU_ABS, DU_REL = 0.008, 0.08
    UMAX_REL = 0.01
    RMS_MAX = 0.012
    rot = {}
    for ro in (0.10, 0.50):
        m = met[('SST-CC', ro)]
        y0 = m['y_stress_zero']
        ratio = y0 / (1 - y0)
        tol_du = DU_ABS + DU_REL * m['du']
        x, u = curves[('a', 'SST-CC', ro)]
        keep = (x >= 0.05) & (x <= 0.95)
        rot[f'ro{int(round(ro * 100)):02d}'] = dict(
            case_id=f'rotchan_ro{int(round(ro * 100)):02d}', Ro=ro, frame_omega=[0.0, 0.0, ro / 2],
            wall_shear_ratio_pressure_over_suction=dict(
                value=ratio, accept=[ratio * (1 - REL_RATIO), ratio * (1 + REL_RATIO)],
                observable='wall_friction_ratio = wall_cf_bottom/wall_cf_top (bottomWall, y = -h, is the pressure side)',
                derivation='tau_p/tau_s = y0/(H - y0): the total shear stress of a fully developed channel is linear '
                           'and vanishes where dU/dy = 0, which for an eddy-viscosity model is where the modelled '
                           "Reynolds stress of Fig. 1(b) crosses zero (y0, measured from the pressure wall)"),
            velocity_maximum_y_over_H=dict(
                value=y0, accept=[y0 - REL_RATIO * ratio * (1 - y0) ** 2, y0 + REL_RATIO * ratio * (1 - y0) ** 2],
                cross_check_velocity_maximum=m['y_umax'],
                case_coordinate=dict(value=2 * y0 - 1, note='y/h in the case (walls at -1 and +1), = 2 y/H - 1')),
            velocity_asymmetry_dU=dict(
                value=m['du'], accept=[m['du'] - tol_du, m['du'] + tol_du],
                definition='[U(0.75 H) - U(0.25 H)]/U_m, y from the pressure wall (case y/h = +0.5 and -0.5)'),
            u_max_over_Um=dict(value=m['u_max'], accept=[m['u_max'] * (1 - UMAX_REL), m['u_max'] * (1 + UMAX_REL)],
                               role='secondary'),
            profile_rms_max=dict(
                value=RMS_MAX, definition='RMS over 0.05 <= y/H <= 0.95 of U_case(y)/U_m - U_SST-CC(y)/U_m, '
                                          'published profile linearly interpolated', role='secondary'),
            published_profile=dict(y_over_H=[round(float(a), 6) for a in x[keep]],
                                   u_over_Um=[round(float(b), 6) for b in u[keep]]),
        )
    targets = dict(
        schema='sstrc_published_targets/1',
        model='SST-RC (SST-CC) of Smirnov and Menter (2009), c_r1 = 1, c_r2 = 2, c_r3 = 1, f_r1 in [0, 1.25]',
        source=source,
        orientation='Fig. 1 y/H = 0 is the pressure (destabilised, high-friction) wall. In this work the flow is +x, '
                    'Omega = +z, the Coriolis acceleration -2 Omega x U points to -y, so y/H = 0 maps to bottomWall '
                    '(y = -h) and y_case/h = 2 y/H - 1.',
        case_match='tedp.holdout_cases.rotchan: Re = U_m h/nu = 2900 (= 5800 on the width), Ro = 2 Omega h/U_m, '
                   'Omega = (0, 0, Ro/2) with h = U_m = 1; frameOmega of the comparator = the same vector',
        primary=['wall_shear_ratio_pressure_over_suction', 'velocity_maximum_y_over_H', 'velocity_asymmetry_dU'],
        check_tool='python closures/comparators/sstrc_reference/check_targets.py <model result.json or case dir> '
                   '<SST result.json or case dir> --out <report.json>  (exit 0 = primary criteria met)',
        rotating_channel=rot,
        other_published_rotation_numbers={f'{ro:.2f}': dict(
            wall_shear_ratio=met[('SST-CC', ro)]['y_stress_zero'] / (1 - met[('SST-CC', ro)]['y_stress_zero']),
            velocity_maximum_y_over_H=met[('SST-CC', ro)]['y_stress_zero'], velocity_asymmetry_dU=met[('SST-CC', ro)]['du'],
            u_max_over_Um=met[('SST-CC', ro)]['u_max']) for ro in ROS},
        sst_control=dict(
            note='the published SST (no correction) profiles are identical for all Ro (symmetric): the preflight SST '
                 'baseline must give wall_friction_ratio = 1 within 1e-3 at every Ro',
            published_max_asymmetry_y_over_H=sst_sym,
            this_work_sst_wall_friction_ratio=dict(rotchan_ro10=0.9999997654622291, rotchan_ro50=0.999970683207263,
                                                   source='results/forge-v2-hx1-011/baseline_qualification.json')),
        tolerance_rationale=dict(
            reading=f'two independent readings of y0 (velocity maximum of Fig. 1a by a three-point parabola, zero of '
                    f'the Reynolds stress of Fig. 1b) agree within {reading_unc:.4f} H over all seven Ro; the bulk '
                    'velocity of every digitized profile is 1 within 0.04 %; the published SST profiles are symmetric '
                    f'within {sst_sym:.4f} H. Reading uncertainty of the wall-shear ratio: 1.2 % (Ro 0.10), 1.4 % (Ro 0.50).',
            code_to_code='SST at Ro = 0 of this work (OpenFOAM v2312, 256 cells) against the published SST profile '
                         '(ANSYS CFX 11): U_max 1.1600 vs 1.1638 (-0.33 %), RMS profile difference 0.0039 U_m over '
                         '0.05-0.95 H, near-wall gradient about 4 % higher; the two walls can be affected differently '
                         'once rotation makes them asymmetric, so 5 % is allowed on the ratio.',
            band=f'ratio band +-{REL_RATIO:.0%}: about 1.5 x the quadrature sum of reading and code-to-code allowances; '
                 'it still rejects the published responses at the neighbouring rotation numbers (Ro 0.05 -> 1.26 and '
                 'Ro 0.15 -> 1.64 around Ro 0.10; Ro 0.20 -> 1.80 below Ro 0.50), i.e. a frame rate wrong by a factor '
                 '1.5-2, the uncorrected SST (1.0) and an inverted sign (< 1). velocity_maximum band = the same band '
                 'mapped through ratio = y0/(1 - y0). dU band = 0.008 U_m + 8 %: rejects Ro 0.05 (0.030) and Ro 0.15 '
                 '(0.077) around Ro 0.10. profile_rms_max 0.012 = 3 x the SST code-to-code RMS.'),
        flat_plate=dict(
            case_id='tmr_plate_137x97', f_r1_expected=1.0,
            criterion='drag_coefficient and wall_cf at every station with x >= 0.1 of SST-RC within 0.1 % '
                      '(relative, scale floor 1e-5) of the SST baseline of the same case and protocol; wall_cf_rms '
                      '(within 0.1 %) and the leading-edge stations x < 0.1 (within 1 %) are reported, not pass/fail, '
                      'because the leading-edge strain legitimately gives f_r1 < 1 there',
            check='python closures/comparators/sstrc_reference/check_targets.py MODEL_result.json SST_result.json',
            relative_tolerance=1e-3, scale_floor=1e-5, x_min=0.1, leading_edge_relative_tolerance=1e-2,
            basis='thin shear flow without curvature: r* = 1, r~ = 0 -> f_rotation = f_r1 = 1 ([SM09] Eq. (1); '
                  '[SS97] p. 301 "f_r1(1, 0) = 1"; p. 300 "near a flat surface |r~| << 1"); 0.1 % is the relative '
                  'tolerance the harness applies to wall_cf and drag_coefficient (required_observables)',
            a_priori=plate,
            a_priori_summary=('f_r1 evaluated by reference.py on the converged SST solution of the same TMR plate '
                              '(12138 x 385 cells, data/mcconkey): production-weighted mean f_r1 - 1 = '
                              f"{plate['regions']['plate x > 0']['production_weighted_mean_fr1_minus_1']:+.1e} over the plate, "
                              f"{plate['stations'][6]['production_weighted_mean_fr1_minus_1']:+.1e} at x = 0.97, but "
                              f"{plate['stations'][0]['production_weighted_mean_fr1_minus_1']:+.1e} at x = 0.002 (leading-edge "
                              'strain, r* < 1): a production change of about 0.02 % downstream, hence the 0.1 % band '
                              'with x >= 0.1') if plate else None),
        not_used=dict(shur2000='Shur et al. (2000) Figs. 3, 5, 6 (pp. 786-787) are the SA-based SARC model (c_r2 = 12, '
                               'no limiter, D^2 = (S^2 + Omega^2)/2); not a target for SST-RC.'),
    )
    (HERE / 'published_targets.json').write_text(json.dumps(targets, indent=1) + '\n')
    for ro in (0.10, 0.50):
        t = rot[f'ro{int(round(ro * 100)):02d}']
        print(f"Ro {ro:.2f}: tau_p/tau_s = {t['wall_shear_ratio_pressure_over_suction']['value']:.4f} "
              f"accept {np.round(t['wall_shear_ratio_pressure_over_suction']['accept'], 4).tolist()}; "
              f"y0/H = {t['velocity_maximum_y_over_H']['value']:.4f}; dU = {t['velocity_asymmetry_dU']['value']:.4f}; "
              f"Umax = {t['u_max_over_Um']['value']:.4f}")
    print(f'reading uncertainty {reading_unc:.4f} H, SST symmetry {sst_sym:.4f} H')


if __name__ == '__main__':
    main(*sys.argv[1:])
