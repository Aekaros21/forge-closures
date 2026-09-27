"""Scoring of the square-duct holdout class.

The sealed dataset ships, per Reynolds number, a RANS case (96x96x75 cells,
cyclic in x) and a DNS case on a 400x400 cross-section at one streamwise
station carrying the mean velocity and the full Reynolds-stress tensor.  The
two meshes differ, so the frozen composite is applied at the DNS points, as
FREEZE.json declares:

    E_U   error of the full velocity vector at the DNS points, / U_bulk
    E_Cf  RMS wall-shear error over all four walls, at the DNS wall faces
    E_uv  L2 error of the two wall-normal shear components (xy, xz), / U_bulk^2

with U_bulk defined exactly as in the development cases, the RMS velocity
magnitude over the reference points.

The candidate solution is averaged over the streamwise direction first (the
flow is fully developed and the case is cyclic, so this removes only
round-off) and then interpolated onto the DNS points.  The wall faces are
added to the interpolation with U = 0 and zero Reynolds stress, which is the
no-slip condition, so the near-wall DNS points that lie inside the first RANS
cell are interpolated rather than extrapolated.

Beside the composite, the secondary-flow check of the plan's success criteria
is reported: its sign as the spatial correlation of the in-plane velocity with
the DNS, and its strength as the ratio of RMS in-plane velocity.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from .. import foammesh, scoring

WALLS = ("y_min", "y_max", "z_min", "z_max")


def read_internal_field(path: Path, n_components: int) -> np.ndarray:
    """internalField of an OpenFOAM field file as (n, n_components).

    Written because the sealed DNS files end without a trailing newline,
    which the installed fluidfoam cannot parse.
    """
    text = Path(path).read_text(errors="replace")
    head = text.index("internalField")
    body_end = text.index("boundaryField", head)
    chunk = text[head:body_end]
    uniform = re.search(r"internalField\s+uniform\s+\(?([^;)]+)\)?\s*;", chunk)
    if uniform and "nonuniform" not in chunk[: uniform.start() + 40]:
        values = np.fromstring(uniform.group(1).replace("(", " ").replace(")", " "), sep=" ")
        return values.reshape(1, -1)
    match = re.search(r"nonuniform\s+List<\w+>\s*\n?\s*(\d+)\s*\n?\s*\(", chunk)
    if match is None:
        raise ValueError(f"{path}: cannot find a nonuniform internalField")
    count = int(match.group(1))
    data = chunk[match.end():]
    data = data[: data.rindex(")")]
    values = np.fromstring(data.replace("(", " ").replace(")", " "), sep=" ")
    if values.size != count * n_components:
        raise ValueError(f"{path}: read {values.size} values, expected {count * n_components}")
    return values.reshape(count, n_components)


def wall_faces(case: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wall-face centres, owner cells and outward unit normals of every wall patch."""
    centres, owners, normals = [], [], []
    for patch in sorted(foammesh.wall_patches(case)):
        wg = foammesh.wall_geometry(case, patch)
        centres.append(wg.face_centres)
        owners.append(wg.owner_cells)
        normals.append(wg.face_areas / np.linalg.norm(wg.face_areas, axis=1, keepdims=True))
    return np.concatenate(centres), np.concatenate(owners), np.concatenate(normals)


def wall_labels(centres: np.ndarray) -> np.ndarray:
    """Which of the four duct walls each face belongs to, and the free coordinate."""
    y, z = centres[:, 1], centres[:, 2]
    label = np.empty(len(centres), dtype="<U5")
    dy = np.minimum(np.abs(y - y.min()), np.abs(y - y.max()))
    dz = np.minimum(np.abs(z - z.min()), np.abs(z - z.max()))
    on_y = dy <= dz
    label[on_y & (y < 0)] = "y_min"
    label[on_y & (y >= 0)] = "y_max"
    label[~on_y & (z < 0)] = "z_min"
    label[~on_y & (z >= 0)] = "z_max"
    return label


def wall_shear(case: Path, u_cells: np.ndarray, nu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """nu |U_t(owner)| / d at every wall face: centres, labels, shear."""
    centres, owners, normals = wall_faces(case)
    cell_centres = np.stack(foammesh_cell_centres(case), axis=1)[owners]
    distance = np.abs(np.einsum("ij,ij->i", cell_centres - centres, normals))
    u_owner = u_cells[owners]
    u_tangential = u_owner - np.einsum("ij,ij->i", u_owner, normals)[:, None] * normals
    shear = nu * np.linalg.norm(u_tangential, axis=1) / np.maximum(distance, 1e-300)
    return centres, wall_labels(centres), shear


def foammesh_cell_centres(case: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from fluidfoam import readmesh

    return readmesh(str(case), verbose=False)


def viscosity(case: Path) -> float:
    text = (Path(case) / "constant" / "transportProperties").read_text()
    match = re.search(r"nu\s+(?:\[[^\]]*\]\s*)?([0-9.eE+-]+)\s*;", text)
    if match is None:
        raise ValueError(f"no nu in {case}")
    return float(match.group(1))


def build_reference(dns_case: Path, rans_case: Path) -> dict:
    """Reference arrays for one duct case, read once from the sealed DNS case.

    The viscosity is taken from the RANS case, not the DNS one: the sealed DNS
    directories all carry the same value, the template's, while the three
    Reynolds numbers are actually set by the viscosity in the RANS cases
    (2.191e-4, 1.205e-4, 6.886e-5 for a bulk velocity of 0.482 and a half
    width of 0.5, i.e. Re_b = 1100, 2000, 3500 exactly). Velocities and
    stresses are in the same units in both, so the fields compare directly.
    """
    x, y, z = foammesh_cell_centres(dns_case)
    u = read_internal_field(dns_case / "0" / "U", 3)
    tau = read_internal_field(dns_case / "0" / "tau", 6)      # xx xy xz yy yz zz
    if len(u) != len(y) or len(tau) != len(y):
        raise ValueError(f"{dns_case}: field/mesh size mismatch")
    nu = viscosity(rans_case)
    u_bulk = float(np.sqrt(np.mean(np.sum(u ** 2, axis=1))))
    centres, labels, shear = wall_shear(dns_case, u, nu)
    return {
        "points": np.stack([y, z], axis=1),
        "u": u,
        "tau_xy": tau[:, 1],
        "tau_xz": tau[:, 2],
        "u_bulk": u_bulk,
        "nu": nu,
        "wall_points": centres,
        "wall_labels": labels,
        "cf": shear / (0.5 * u_bulk ** 2),
        "x_station": float(np.mean(x)),
        "nu_dns_file": viscosity(dns_case),
    }


def _column_average(y: np.ndarray, z: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average a cell field over the streamwise direction, per (y, z) column."""
    key = np.round(np.stack([y, z], axis=1), 9)
    _, index, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    counts = np.bincount(inverse)
    values = np.atleast_2d(values.T).T
    out = np.stack([np.bincount(inverse, weights=values[:, c]) / counts
                    for c in range(values.shape[1])], axis=1)
    return key[index], out


def _interpolate(points: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    linear = LinearNDInterpolator(points, values)
    out = linear(targets)
    missing = np.isnan(out[:, 0]) if out.ndim > 1 else np.isnan(out)
    if np.any(missing):
        nearest = NearestNDInterpolator(points, values)
        out[missing] = nearest(targets[missing])
    return out


def score_duct(run_case: Path, reference: dict) -> tuple[scoring.CaseScore, dict]:
    """Frozen composite plus the secondary-flow diagnostics for one duct run."""
    from fluidfoam import readscalar, readtensor

    times = sorted((int(p.name) for p in Path(run_case).iterdir() if p.name.isdigit()), reverse=True)
    time = str(times[0])
    u = read_internal_field(run_case / time / "U", 3)
    nut = np.asarray(readscalar(str(run_case), time, "nut", verbose=False))
    grad_u = np.asarray(readtensor(str(run_case), time, "grad(U)", verbose=False))
    tau_xy = -nut * (grad_u[1] + grad_u[3])
    tau_xz = -nut * (grad_u[2] + grad_u[6])
    if (Path(run_case) / time / "nonlinearStress").exists():
        nl = read_internal_field(run_case / time / "nonlinearStress", 6)
        tau_xy = tau_xy + nl[:, 1]
        tau_xz = tau_xz + nl[:, 2]
    x, y, z = foammesh_cell_centres(run_case)

    field = np.column_stack([u, tau_xy, tau_xz])
    columns, averaged = _column_average(y, z, field)
    # no-slip wall faces carry U = 0 and zero Reynolds stress
    wall_centres, _, _ = wall_faces(Path(run_case))
    wall_columns, _ = _column_average(wall_centres[:, 1], wall_centres[:, 2],
                                      np.zeros((len(wall_centres), field.shape[1])))
    points = np.vstack([columns, wall_columns])
    values = np.vstack([averaged, np.zeros((len(wall_columns), field.shape[1]))])
    sampled = _interpolate(points, values, reference["points"])

    u_bulk = reference["u_bulk"]
    du = sampled[:, :3] - reference["u"]
    e_u = float(np.sqrt(np.mean(np.sum(du ** 2, axis=1)))) / u_bulk
    d_uv = np.stack([sampled[:, 3] - reference["tau_xy"], sampled[:, 4] - reference["tau_xz"]], axis=1)
    e_uv = float(np.sqrt(np.mean(np.sum(d_uv ** 2, axis=1)))) / u_bulk ** 2

    nu = reference["nu"]
    centres, labels, shear = wall_shear(Path(run_case), u, nu)
    cf_model = shear / (0.5 * u_bulk ** 2)
    cf_sampled = np.empty(len(reference["cf"]))
    for wall in WALLS:
        target = reference["wall_labels"] == wall
        source = labels == wall
        if not np.any(target):
            continue
        free = 2 if wall.startswith("y") else 1
        s_src, cf_src = _column_average(centres[source][:, free], np.zeros(source.sum()),
                                        cf_model[source])
        order = np.argsort(s_src[:, 0])
        cf_sampled[target] = np.interp(reference["wall_points"][target][:, free],
                                       s_src[order, 0], cf_src[order, 0])
    e_cf = float(np.sqrt(np.mean((cf_sampled - reference["cf"]) ** 2)))

    inplane_model = sampled[:, 1:3]
    inplane_ref = reference["u"][:, 1:3]
    num = float(np.sum(inplane_model * inplane_ref))
    den = float(np.sqrt(np.sum(inplane_model ** 2) * np.sum(inplane_ref ** 2)))
    rms_model = float(np.sqrt(np.mean(np.sum(inplane_model ** 2, axis=1))))
    rms_ref = float(np.sqrt(np.mean(np.sum(inplane_ref ** 2, axis=1))))
    diagnostics = {
        "secondary_correlation": (num / den) if den > 0 else 0.0,
        "secondary_rms_model_over_bulk": rms_model / u_bulk,
        "secondary_rms_dns_over_bulk": rms_ref / u_bulk,
        "secondary_magnitude_ratio": (rms_model / rms_ref) if rms_ref > 0 else float("inf"),
        "time": time,
    }
    return scoring.CaseScore(Path(run_case).name, e_u, e_cf, e_uv), diagnostics


def save_reference(reference: dict, path: Path) -> None:
    """Cache a reference so cluster nodes need only this file, not the DNS case."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in reference.items()})


def load_reference(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    for scalar in ("u_bulk", "nu", "x_station"):
        out[scalar] = float(out[scalar])
    out["wall_labels"] = out["wall_labels"].astype("<U5")
    return out


def correlation_scale(references: dict[str, dict]) -> dict:
    """What the secondary-flow correlation means, measured on the DNS itself.

    The metric is the cosine similarity of the in-plane velocity over the DNS
    points, so the DNS scores 1 against itself. The useful calibration is the
    DNS against DNS at a different Reynolds number, because the secondary-flow
    pattern genuinely changes with Reynolds number: a model cannot be expected
    to beat that.
    """
    names = sorted(references)
    out: dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i:]:
            ia = references[a]["u"][:, 1:3]
            ib = references[b]["u"][:, 1:3]
            out[f"{a}|{b}"] = float(np.sum(ia * ib) / np.sqrt(np.sum(ia * ia) * np.sum(ib * ib)))
    return out
