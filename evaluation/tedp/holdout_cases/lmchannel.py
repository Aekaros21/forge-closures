"""Plane channel at Re_tau = 5186 (Lee and Moser 2015 DNS): verification only.

The development set's plane channel (Kristoffersen and Andersson, Re_tau ~ 194) run 27 times
faster: the same streamwise-periodic one-cell channel of the rotating-channel adapter at Ro = 0,
with the viscosity of the DNS bulk Reynolds number and a 512-cell wall-normal mesh graded to a
first cell at y+ ~ 0.25. The reference is converted to the rotating-channel reference format
(y/h on [-1, 1], U/U_m, -u'v'/U_m^2, wall Cf) so the development scoring applies unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MEAN = REPO/'data/external/LM_Channel_5200_mean_prof.dat'
FLUCTUATIONS = REPO/'data/external/LM_Channel_5200_vel_fluc_prof.dat'
RE_TAU = 5186.0
NY = 512
WALL_GRADING = 540.0      # first cell y+ ~ 0.25 with 256 cells per half


def _rows(path: Path) -> np.ndarray:
    rows = [line.split() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('%')]
    return np.asarray(rows, dtype=float)


def half_channel() -> dict:
    """y/delta, U+ and u'v'+ of the half channel (wall to centre line)."""
    mean, fluc = _rows(MEAN), _rows(FLUCTUATIONS)
    if not np.allclose(mean[:, 0], fluc[:, 0]):
        raise ValueError('Lee-Moser mean and fluctuation files use different grids')
    return {'y': mean[:, 0], 'u_plus': mean[:, 2], 'uv_plus': fluc[:, 5]}


def bulk_plus() -> float:
    half = half_channel()
    return float(np.trapezoid(half['u_plus'], half['y']))


def re_bulk_half() -> float:
    """U_m h / nu of the DNS (h the half height): U_m+ Re_tau."""
    return bulk_plus()*RE_TAU


def reference() -> dict:
    """The rotating-channel reference dict: y on [-1, 1], U/U_m, -u'v'/U_m^2 and both walls' Cf."""
    half = half_channel()
    ub = bulk_plus()
    y = np.r_[half['y']-1.0, (1.0-half['y'])[::-1][1:]]
    u = np.r_[half['u_plus'], half['u_plus'][::-1][1:]]/ub
    muv = np.r_[-half['uv_plus'], half['uv_plus'][::-1][1:]]/ub**2
    cf = 2.0/ub**2
    return {'y': y, 'u': u, 'muv': muv, 'cf': (cf, cf), 're_tau': RE_TAU, 'bulk_plus': ub}
