"""Minimal ASCII polyMesh reader: wall-patch geometry for Cf scoring.

Parses constant/polyMesh/{boundary,owner,faces,points} — enough to give, for
every face of a named patch: the face centre, the outward area vector, and
the adjacent (owner) cell index. Pure text parsing, no OpenFOAM needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _strip_header(text: str) -> str:
    """Drop the FoamFile header block."""
    m = re.search(r"FoamFile\s*\{.*?\}", text, re.DOTALL)
    return text[m.end():] if m else text


def _read_label_list(path: Path) -> np.ndarray:
    body = _strip_header(path.read_text())
    m = re.search(r"^\s*(\d+)\s*\(\s*", body, re.MULTILINE)
    if m is None:
        raise ValueError(f"no list header in {path}")
    n = int(m.group(1))
    start = m.end()
    end = body.index(")", start)
    return np.fromstring(body[start:end], dtype=np.int64, sep=" ")[:n]


def read_points(mesh_dir: Path) -> np.ndarray:
    body = _strip_header((mesh_dir / "points").read_text())
    m = re.search(r"^\s*(\d+)\s*\(\s*", body, re.MULTILINE)
    n = int(m.group(1))
    chunk = body[m.end(): body.rindex(")")]
    vals = np.fromstring(chunk.replace("(", " ").replace(")", " "), sep=" ")
    return vals.reshape(-1, 3)[:n]


def read_faces(mesh_dir: Path) -> list[np.ndarray]:
    body = _strip_header((mesh_dir / "faces").read_text())
    m = re.search(r"^\s*(\d+)\s*\(", body, re.MULTILINE)
    n = int(m.group(1))
    out: list[np.ndarray] = []
    pos = m.end()
    face_re = re.compile(r"(\d+)\(([^)]*)\)")
    for fm in face_re.finditer(body, pos):
        nv = int(fm.group(1))
        verts = np.fromstring(fm.group(2), dtype=np.int64, sep=" ")
        if len(verts) != nv:
            raise ValueError("face vertex count mismatch")
        out.append(verts)
        if len(out) == n:
            break
    if len(out) != n:
        raise ValueError(f"expected {n} faces, parsed {len(out)}")
    return out


@dataclass
class Patch:
    name: str
    kind: str
    start_face: int
    n_faces: int


def read_boundary(mesh_dir: Path) -> dict[str, Patch]:
    body = _strip_header((mesh_dir / "boundary").read_text())
    patches: dict[str, Patch] = {}
    for m in re.finditer(
        r"(\w+)\s*\{([^}]*)\}", body
    ):
        name, inner = m.group(1), m.group(2)
        kind = re.search(r"type\s+(\w+);", inner)
        nf = re.search(r"nFaces\s+(\d+);", inner)
        sf = re.search(r"startFace\s+(\d+);", inner)
        if kind and nf and sf:
            patches[name] = Patch(name, kind.group(1), int(sf.group(1)), int(nf.group(1)))
    return patches


@dataclass
class WallGeometry:
    face_centres: np.ndarray   # (nF, 3)
    face_areas: np.ndarray     # (nF, 3) outward area vectors
    owner_cells: np.ndarray    # (nF,) adjacent cell indices


def wall_geometry(case: Path | str, patch_name: str) -> WallGeometry:
    mesh_dir = Path(case) / "constant" / "polyMesh"
    patches = read_boundary(mesh_dir)
    if patch_name not in patches:
        raise KeyError(f"{patch_name!r} not in {sorted(patches)}")
    p = patches[patch_name]
    points = read_points(mesh_dir)
    faces = read_faces(mesh_dir)
    owner = _read_label_list(mesh_dir / "owner")

    centres = np.empty((p.n_faces, 3))
    areas = np.empty((p.n_faces, 3))
    for i in range(p.n_faces):
        verts = points[faces[p.start_face + i]]
        c = verts.mean(axis=0)
        area = np.zeros(3)
        for j in range(len(verts)):
            a, b = verts[j], verts[(j + 1) % len(verts)]
            area += np.cross(a - c, b - c)
        areas[i] = 0.5 * area
        centres[i] = c
    return WallGeometry(
        face_centres=centres,
        face_areas=areas,
        owner_cells=owner[p.start_face: p.start_face + p.n_faces],
    )


def wall_patches(case: Path | str) -> list[str]:
    patches = read_boundary(Path(case) / "constant" / "polyMesh")
    return [p.name for p in patches.values() if p.kind == "wall"]
