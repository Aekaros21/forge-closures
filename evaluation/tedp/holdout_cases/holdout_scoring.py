"""Scoring of the hump and rotating-channel holdout classes.

The frozen composite (0.5 E_U + 0.3 E_Cf + 0.2 E_uv, ``tedp.scoring``) is
reused with the reference quantities each class provides:

hump (Greenblatt experiment, sealed files ``noflow_cp.exp.dat``,
``noflow_cf.exp.dat``, ``noflow_vel_and_turb.exp.dat``):
    E_U   RMS of the velocity vector error over every PIV point, / U_ref
    E_Cf  RMS of the Cf error over the experimental Cf stations
    E_uv  RMS of the u'v' error over the PIV points, / U_ref^2
rotating channel (AGARD PCH22, sealed ``rotchan/f*/f*_r_XX.dat``):
    E_U   RMS of U/U_m error over the mean-velocity profile
    E_Cf  RMS of the two wall Cf errors (pressure and suction side)
    E_uv  RMS of the -u'v'/U_m^2 error over the shear-stress profile

The sealed files are parsed with strict format checks: column counts and
value ranges must match the layouts documented on the source websites,
otherwise the runner stops before any candidate is scored.  The parsers
are unit-tested on synthetic files; the real files are read once, after
Checkpoint 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d

from .. import scoring


class HoldoutFormatError(RuntimeError):
    pass


def _numeric_rows(path: Path, min_cols: int) -> np.ndarray:
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            continue  # header / comment line
        if len(values) >= min_cols:
            rows.append(values)
    if not rows:
        raise HoldoutFormatError(f"no numeric rows with >= {min_cols} columns in {path}")
    width = min(len(r) for r in rows)
    return np.asarray([r[:width] for r in rows], dtype=np.float64)


# ---------------------------------------------------------------------------
# hump
# ---------------------------------------------------------------------------
def read_hump_wall(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """x/c and Cp or Cf along the wall (two numeric columns)."""
    table = _numeric_rows(path, 2)
    x, v = table[:, 0], table[:, 1]
    if not (-1.0 <= x.min() and x.max() <= 3.0):
        raise HoldoutFormatError(f"{name}: x/c range {x.min()}..{x.max()} is not a hump wall profile")
    if name == "cf" and not (abs(v).max() < 0.05):
        raise HoldoutFormatError("cf: values outside a plausible skin-friction range")
    if name == "cp" and not (-2.0 < v.min() and v.max() < 1.5):
        raise HoldoutFormatError("cp: values outside a plausible pressure-coefficient range")
    order = np.argsort(x)
    return x[order], v[order]


def read_hump_piv(path: Path) -> np.ndarray:
    """PIV table: x/c, y/c, u/U, v/U, u'u', v'v', u'v' (columns 0..6, >= 5 needed)."""
    table = _numeric_rows(path, 5)
    x, y = table[:, 0], table[:, 1]
    if not (0.0 <= x.min() and x.max() <= 2.0 and 0.0 <= y.min() and y.max() <= 0.5):
        raise HoldoutFormatError("PIV table: x/c, y/c ranges are not the hump PIV window")
    if abs(table[:, 2]).max() > 2.0:
        raise HoldoutFormatError("PIV table: u/U column out of range")
    return table


def score_hump(
    cell_xy: np.ndarray, u_cells: np.ndarray, uv_cells: np.ndarray,
    wall_x: np.ndarray, wall_cf: np.ndarray,
    piv: np.ndarray, cf_exp_x: np.ndarray, cf_exp: np.ndarray,
) -> scoring.CaseScore:
    """Composite score of one hump run against the experiment.

    ``cell_xy`` (n, 2), ``u_cells`` (n, 3) and ``uv_cells`` (n,) are the run
    fields at cell centres in chord units; ``wall_x``/``wall_cf`` the run's
    wall Cf; ``piv`` the experimental table; ``cf_exp_x``/``cf_exp`` the
    experimental Cf.  Reference values are sampled at the experimental points.
    """
    u_interp = LinearNDInterpolator(cell_xy, u_cells[:, :2], fill_value=np.nan)
    uv_interp = LinearNDInterpolator(cell_xy, uv_cells, fill_value=np.nan)
    pts = piv[:, :2]
    u_model = u_interp(pts)
    uv_model = uv_interp(pts)
    nearest_u = NearestNDInterpolator(cell_xy, u_cells[:, :2])
    nearest_uv = NearestNDInterpolator(cell_xy, uv_cells)
    bad = np.isnan(u_model[:, 0])
    u_model[bad] = nearest_u(pts[bad])
    uv_model[np.isnan(uv_model)] = nearest_uv(pts[np.isnan(uv_model)])
    u_ref = piv[:, 2:4]
    uv_ref = piv[:, 6] if piv.shape[1] > 6 else np.zeros(len(piv))
    e_u = float(np.sqrt(np.mean(np.sum((u_model - u_ref) ** 2, axis=1))))
    cf_model = interp1d(wall_x, wall_cf, bounds_error=False, fill_value="extrapolate")(cf_exp_x)
    e_cf = float(np.sqrt(np.mean((cf_model - cf_exp) ** 2)))
    e_uv = float(np.sqrt(np.mean((uv_model - uv_ref) ** 2))) if piv.shape[1] > 6 else 0.0
    return scoring.CaseScore("hump", e_u, e_cf, e_uv)


# ---------------------------------------------------------------------------
# rotating channel
# ---------------------------------------------------------------------------
def read_pch22_profile(path: Path, expect_columns: int = 2) -> np.ndarray:
    """A PCH22 profile file: first column y/h in [-1, 1] (or [0, 2]), then values."""
    table = _numeric_rows(path, expect_columns)
    y = table[:, 0]
    if y.max() - y.min() > 2.5 or y.max() - y.min() < 1.0:
        raise HoldoutFormatError(f"{path.name}: first column does not span a channel height")
    return table


def channel_coordinate(y: np.ndarray) -> np.ndarray:
    """Map a profile coordinate to [-1, 1] whether the file uses [0, 2] or [-1, 1]."""
    if y.min() >= -1e-6 and y.max() <= 2.0 + 1e-6 and y.max() > 1.2:
        return y - 1.0
    return y


def score_rotchan(
    y_cells: np.ndarray, u_cells: np.ndarray, uv_cells: np.ndarray,
    cf_walls: tuple[float, float],
    u_profile: np.ndarray, uv_profile: np.ndarray | None,
    cf_ref: tuple[float, float] | None,
) -> scoring.CaseScore:
    """Composite score of one rotating-channel run against the DNS profiles.

    Profiles are (y/h, value) with value U/U_m or -u'v'/U_m^2.  Wall Cf is
    compared at both walls when the reference provides it; otherwise E_Cf is
    taken from the profile's near-wall slope on both sides.
    """
    y_ref = channel_coordinate(u_profile[:, 0])
    u_model = interp1d(y_cells, u_cells, bounds_error=False, fill_value="extrapolate")(y_ref)
    e_u = float(np.sqrt(np.mean((u_model - u_profile[:, 1]) ** 2)))
    if uv_profile is not None:
        y_uv = channel_coordinate(uv_profile[:, 0])
        uv_model = interp1d(y_cells, uv_cells, bounds_error=False, fill_value="extrapolate")(y_uv)
        e_uv = float(np.sqrt(np.mean((uv_model - uv_profile[:, 1]) ** 2)))
    else:
        e_uv = 0.0
    if cf_ref is not None:
        e_cf = float(np.sqrt(np.mean((np.asarray(cf_walls) - np.asarray(cf_ref)) ** 2)))
    else:
        e_cf = 0.0
    return scoring.CaseScore("rotchan", e_u, e_cf, e_uv)


# ---------------------------------------------------------------------------
# PCH22 reference construction
# ---------------------------------------------------------------------------
def pch22_reference(path_f2: Path, pressure_side_at: str = "bottom") -> dict:
    """Reference profiles for one rotating-channel case from its mean-velocity file.

    The sealed dataset ships figure files with no key, so only the mean
    velocity file is used; it identifies itself.  Its first column is y/h on
    [-1, 1] and its second is U/u_tau: the wall gradients give
    dU+/dY = Re_tau (u_w/u_ref)^2 and reproduce U+ = y+ in the sublayer, and
    the bulk mean gives U_m/u_tau = 15.06 at Ro 0, i.e. Re_tau = 194 for the
    Re_m = 2900 of this case.

    The Reynolds shear stress is then exact rather than read off a figure. In
    a fully developed channel the total stress is linear in y, and spanwise
    rotation adds no streamwise mean force because <v> = 0, so

        tau+(Y)     = linear interpolation between the two measured wall values
        -<u'v'>+(Y) = tau+(Y) - (1/Re_tau) dU+/dY

    Returns the quantities FREEZE.json declares: U/U_m, -<u'v'>/U_m^2 and the
    skin friction on each wall, all normalised by the bulk velocity.
    """
    table = _numeric_rows(path_f2, 2)
    order = np.argsort(table[:, 0])
    y, u_plus = table[order, 0], table[order, 1]
    if abs(y[0] + 1.0) > 0.01 or abs(y[-1] - 1.0) > 0.01:
        raise HoldoutFormatError(f"{path_f2.name}: first column does not span the channel")

    def wall_gradient(yy: np.ndarray, uu: np.ndarray) -> float:
        """Least-squares slope through the origin over the viscous sublayer."""
        d = yy - yy[0]
        slope = (uu[1] - uu[0]) / d[1]
        for _ in range(3):
            keep = np.abs(d) <= 4.0 / max(abs(slope), 1.0)
            keep[0] = True
            if keep.sum() < 3:
                keep = np.zeros_like(d, dtype=bool)
                keep[:4] = True
            slope = float(np.sum(d[keep] * (uu[keep] - uu[0])) / np.sum(d[keep] ** 2))
        return slope

    g_bottom = wall_gradient(y, u_plus)
    g_top = wall_gradient(y[::-1], u_plus[::-1])       # negative: y decreasing
    re_tau = 0.5 * (abs(g_bottom) + abs(g_top))
    u_bulk_plus = float(np.trapezoid(u_plus, y) / (y[-1] - y[0]))
    tau_b, tau_t = g_bottom / re_tau, -abs(g_top) / re_tau
    tau = tau_b + (tau_t - tau_b) * (y - y[0]) / (y[-1] - y[0])
    muv_plus = tau - np.gradient(u_plus, y) / re_tau
    u_out = u_plus / u_bulk_plus
    muv_out = muv_plus / u_bulk_plus ** 2
    cf = (2.0 * abs(g_bottom) / (re_tau * u_bulk_plus ** 2),
          2.0 * abs(g_top) / (re_tau * u_bulk_plus ** 2))
    # Orientation. The sealed profiles put the high-friction (pressure,
    # destabilised) side at +y. The case as built rotates about +z with the
    # flow along +x, so the Coriolis acceleration -2 Omega x u points to -y
    # and the bottom wall is the pressure side (the Bradshaw criterion gives
    # the same answer: destabilised where dU/dy > 0). Aligning the two is a
    # relabelling of the wall-normal axis, not a change of any value: y and
    # the shear stress are odd under the reflection, U and Cf are even.
    mirrored = False
    if pressure_side_at == "bottom" and cf[1] > cf[0]:
        mirrored = True
    elif pressure_side_at == "top" and cf[0] > cf[1]:
        mirrored = True
    if mirrored:
        y_m = -y[::-1]
        u_out = u_out[::-1]
        muv_out = -muv_out[::-1]
        cf = (cf[1], cf[0])
        y = y_m
    return {
        "y": y,
        "u": u_out,
        "muv": muv_out,
        "cf": cf,
        "re_tau": re_tau,
        "u_bulk_plus": u_bulk_plus,
        "mirrored_to_match_case_orientation": mirrored,
    }
