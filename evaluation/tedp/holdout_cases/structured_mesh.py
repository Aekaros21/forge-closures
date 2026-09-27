"""Write a 2-D structured (i, j) grid as an OpenFOAM polyMesh with one cell
in z and named patches.

Used for the NASA hump grids (Plot3D 2-D formatted, single block).  The
grid is assumed right-handed: i increases roughly with x and j with y, so
the face orderings below give outward normals; ``checkMesh`` verifies it.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np


def read_p2dfmt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a single-block Plot3D 2-D formatted grid; returns x, y of shape (NJ, NI)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        tokens = fh.read().split()
    nblocks = int(tokens[0])
    if nblocks != 1:
        raise ValueError(f"expected a single block, found {nblocks}")
    ni, nj = int(tokens[1]), int(tokens[2])
    values = np.asarray(tokens[3:3 + 2 * ni * nj], dtype=np.float64)
    if values.size != 2 * ni * nj:
        raise ValueError("Plot3D file is truncated")
    x = values[: ni * nj].reshape(nj, ni)
    y = values[ni * nj:].reshape(nj, ni)
    return x, y


def _write_list(fh, header: dict, cls: str, obj: str, count: int, body_lines) -> None:
    fh.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n")
    fh.write(f"    class       {cls};\n    location    \"constant/polyMesh\";\n    object      {obj};\n}}\n\n")
    fh.write(f"{count}\n(\n")
    for line in body_lines:
        fh.write(line)
    fh.write(")\n")


def write_polymesh_2d(
    x: np.ndarray, y: np.ndarray, case: Path, thickness: float,
    patches: dict[str, str] | None = None,
) -> dict[str, int]:
    """Write constant/polyMesh for a (NJ, NI) point grid extruded by ``thickness``.

    ``patches`` maps the six logical sides to patch names/types:
    keys ``imin, imax, jmin, jmax`` (values ``name:type``) and ``z`` for the
    empty front/back patch name.  ``jmin_split`` optionally divides the jmin
    boundary into several patches by a predicate on the face-centre x, which
    the TMR bump-in-channel needs.  ``jmin_cut`` turns the first and last N
    faces of the jmin line into internal faces joining mirrored cells, which is
    how a C-grid's wake cut is closed.  Returns the face count per patch.
    """
    patches = patches or {}
    imin = patches.get("imin", "inlet:patch")
    imax = patches.get("imax", "outlet:patch")
    jmin = patches.get("jmin", "bottomWall:wall")
    jmax = patches.get("jmax", "topWall:wall")
    zname = patches.get("z", "frontAndBack")
    nj, ni = x.shape
    nci, ncj = ni - 1, nj - 1

    def pid(i: int, j: int, k: int) -> int:
        return (k * nj + j) * ni + i

    def cid(i: int, j: int) -> int:
        return j * nci + i

    points = np.empty((2 * ni * nj, 3), dtype=np.float64)
    for k, z in enumerate((0.0, thickness)):
        base = k * ni * nj
        points[base:base + ni * nj, 0] = x.reshape(-1)
        points[base:base + ni * nj, 1] = y.reshape(-1)
        points[base:base + ni * nj, 2] = z

    # Internal faces are collected first and then sorted into upper-triangular
    # order (owner ascending, then neighbour), which lets a branch cut add
    # faces that connect cells far apart in the index space.
    internal: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for j in range(ncj):
        for i in range(nci):
            c = cid(i, j)
            if i + 1 < nci:
                internal.append((c, cid(i + 1, j),
                                 (pid(i + 1, j, 0), pid(i + 1, j + 1, 0), pid(i + 1, j + 1, 1), pid(i + 1, j, 1))))
            if j + 1 < ncj:
                internal.append((c, cid(i, j + 1),
                                 (pid(i, j + 1, 0), pid(i, j + 1, 1), pid(i + 1, j + 1, 1), pid(i + 1, j + 1, 0))))

    # A C-grid's wake cut is not a boundary: the j=0 faces of cell i and of its
    # mirror cell (nci-1-i) are geometrically the same surface, so they become
    # one internal face joining the two cells. ``jmin_cut`` gives the number of
    # face pairs at each end of the jmin line.
    n_cut = int(patches.get("jmin_cut", 0) or 0)
    cut_indices: set[int] = set()
    for i in range(n_cut):
        lo, hi = i, nci - 1 - i
        cut_indices.update((lo, hi))
        a, b = cid(lo, 0), cid(hi, 0)
        quad = (pid(lo, 0, 0), pid(lo + 1, 0, 0), pid(lo + 1, 0, 1), pid(lo, 0, 1))
        internal.append((min(a, b), max(a, b), quad if a < b else
                         (quad[0], quad[3], quad[2], quad[1])))

    internal.sort(key=lambda row: (row[0], row[1]))
    faces: list[tuple[int, int, int, int]] = [row[2] for row in internal]
    owner: list[int] = [row[0] for row in internal]
    neighbour: list[int] = [row[1] for row in internal]
    n_internal = len(faces)

    boundary: list[tuple[str, str, int, int]] = []

    def add_patch(spec: str, quads: list[tuple[int, int, int, int]], owners: list[int]) -> None:
        name, ptype = spec.split(":")
        start = len(faces)
        faces.extend(quads)
        owner.extend(owners)
        boundary.append((name, ptype, start, len(quads)))

    add_patch(imin, [(pid(0, j, 0), pid(0, j, 1), pid(0, j + 1, 1), pid(0, j + 1, 0)) for j in range(ncj)],
              [cid(0, j) for j in range(ncj)])
    add_patch(imax, [(pid(nci, j, 0), pid(nci, j + 1, 0), pid(nci, j + 1, 1), pid(nci, j, 1)) for j in range(ncj)],
              [cid(nci - 1, j) for j in range(ncj)])
    # jmin may be split into several patches along i, which the TMR
    # bump-in-channel needs: symmetry ahead of the plate, a viscous wall over
    # the bump, symmetry again behind it. ``jmin_split`` is a list of
    # (name:type, predicate on the face-centre x) applied in order; the first
    # matching entry wins and anything unmatched falls back to ``jmin``.
    split = patches.get("jmin_split")
    if split:
        xc = 0.5 * (x[0, :-1] + x[0, 1:])
        assigned: dict[str, list[int]] = {spec: [] for spec, _ in split}
        rest: list[int] = []
        for i in range(nci):
            for spec, predicate in split:
                if predicate(xc[i]):
                    assigned[spec].append(i)
                    break
            else:
                rest.append(i)
        for spec, _ in split:
            idx = assigned[spec]
            if idx:
                add_patch(spec, [(pid(i, 0, 0), pid(i + 1, 0, 0), pid(i + 1, 0, 1), pid(i, 0, 1)) for i in idx],
                          [cid(i, 0) for i in idx])
        if rest:
            add_patch(jmin, [(pid(i, 0, 0), pid(i + 1, 0, 0), pid(i + 1, 0, 1), pid(i, 0, 1)) for i in rest],
                      [cid(i, 0) for i in rest])
    else:
        keep = [i for i in range(nci) if i not in cut_indices]
        add_patch(jmin, [(pid(i, 0, 0), pid(i + 1, 0, 0), pid(i + 1, 0, 1), pid(i, 0, 1)) for i in keep],
                  [cid(i, 0) for i in keep])
    add_patch(jmax, [(pid(i, ncj, 0), pid(i, ncj, 1), pid(i + 1, ncj, 1), pid(i + 1, ncj, 0)) for i in range(nci)],
              [cid(i, ncj - 1) for i in range(nci)])
    front = [(pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0), pid(i + 1, j, 0)) for j in range(ncj) for i in range(nci)]
    back = [(pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j + 1, 1), pid(i, j + 1, 1)) for j in range(ncj) for i in range(nci)]
    cells_order = [cid(i, j) for j in range(ncj) for i in range(nci)]
    add_patch(f"{zname}:empty", front + back, cells_order + cells_order)

    mesh_dir = Path(case) / "constant" / "polyMesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    with open(mesh_dir / "points", "w") as fh:
        _write_list(fh, {}, "vectorField", "points", len(points),
                    (f"({p[0]:.12g} {p[1]:.12g} {p[2]:.12g})\n" for p in points))
    with open(mesh_dir / "faces", "w") as fh:
        _write_list(fh, {}, "faceList", "faces", len(faces),
                    (f"4({f[0]} {f[1]} {f[2]} {f[3]})\n" for f in faces))
    with open(mesh_dir / "owner", "w") as fh:
        fh.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       labelList;\n")
        fh.write(f"    note        \"nPoints:{len(points)}  nCells:{nci * ncj}  nFaces:{len(faces)}  nInternalFaces:{n_internal}\";\n")
        fh.write("    location    \"constant/polyMesh\";\n    object      owner;\n}\n\n")
        fh.write(f"{len(owner)}\n(\n" + "\n".join(str(o) for o in owner) + "\n)\n")
    with open(mesh_dir / "neighbour", "w") as fh:
        fh.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       labelList;\n")
        fh.write("    location    \"constant/polyMesh\";\n    object      neighbour;\n}\n\n")
        fh.write(f"{len(neighbour)}\n(\n" + "\n".join(str(n) for n in neighbour) + "\n)\n")
    with open(mesh_dir / "boundary", "w") as fh:
        fh.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       polyBoundaryMesh;\n")
        fh.write("    location    \"constant/polyMesh\";\n    object      boundary;\n}\n\n")
        fh.write(f"{len(boundary)}\n(\n")
        for name, ptype, start, count in boundary:
            fh.write(f"    {name}\n    {{\n        type            {ptype};\n")
            if ptype == "wall":
                fh.write("        inGroups        1(wall);\n")
            fh.write(f"        nFaces          {count};\n        startFace       {start};\n    }}\n")
        fh.write(")\n")
    return {name: count for name, _, _, count in boundary}
