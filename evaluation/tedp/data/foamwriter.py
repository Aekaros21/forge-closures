"""Write OpenFOAM ASCII fields for the frozen-RANS cases.

Two modes:
  * replace_internal: keep a shipped field file's header + boundaryField
    (preserving the exact BC types/values) and swap the internalField —
    used for U, k, omega where a template field exists.
  * write_field: create a field from scratch with boundary types derived
    from the mesh's patch kinds (wall/patch -> zeroGradient, cyclic ->
    cyclic, empty -> empty, symmetry -> symmetryPlane) — used for tauDNS.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .. import foammesh

_DIMS = {
    "U": "[0 1 -1 0 0 0 0]",
    "k": "[0 2 -2 0 0 0 0]",
    "omega": "[0 0 -1 0 0 0 0]",
    "tauDNS": "[0 2 -2 0 0 0 0]",
}

_KIND_TO_BC = {
    "wall": "zeroGradient",
    "patch": "zeroGradient",
    "mappedPatch": "zeroGradient",
    "cyclic": "cyclic",
    "cyclicAMI": "cyclicAMI",
    "empty": "empty",
    "symmetry": "symmetry",
    "symmetryPlane": "symmetryPlane",
    "processor": "processor",
}


def _format_values(values: np.ndarray) -> tuple[str, str]:
    """Return (foam List type, body lines)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        body = "\n".join(f"{v:.12g}" for v in values)
        return "scalar", body
    if values.shape[1] == 3:
        body = "\n".join(f"({v[0]:.12g} {v[1]:.12g} {v[2]:.12g})" for v in values)
        return "vector", body
    if values.shape[1] == 6:
        body = "\n".join(
            f"({v[0]:.12g} {v[1]:.12g} {v[2]:.12g} {v[3]:.12g} {v[4]:.12g} {v[5]:.12g})"
            for v in values
        )
        return "symmTensor", body
    raise ValueError(f"unsupported value shape {values.shape}")


def _internal_block(values: np.ndarray) -> str:
    ftype, body = _format_values(values)
    return (
        f"internalField   nonuniform List<{ftype}> \n{len(values)}\n(\n{body}\n)\n;"
    )


def replace_internal(template_field: Path, out_field: Path, values: np.ndarray) -> None:
    """Copy a field file, replacing its internalField with `values`."""
    text = template_field.read_text()
    # match 'internalField ... ;' whether uniform or nonuniform (list body
    # includes parens; use a robust two-part match)
    m = re.search(
        r"internalField\s+(uniform[^;]*;|nonuniform\s+List<\w+>\s*\n?\s*\d+\s*\(.*?\)\s*;)",
        text,
        re.DOTALL,
    )
    if not m:
        raise ValueError(f"no internalField in {template_field}")
    text = text[: m.start()] + _internal_block(values) + text[m.end():]
    out_field.parent.mkdir(parents=True, exist_ok=True)
    out_field.write_text(text)


def swap_wall_bc(field_path: Path, old: str, new: str) -> None:
    """In-place BC type swap (e.g. omegaWallFunction -> fixedValue)."""
    text = field_path.read_text()
    field_path.write_text(
        text.replace(f"type            {old};", f"type            {new};")
    )


def write_field(
    case: Path,
    time_name: str,
    name: str,
    values: np.ndarray,
    dims: str | None = None,
) -> Path:
    """Create a field with mesh-derived boundary types."""
    values = np.asarray(values, dtype=np.float64)
    ftype, _ = _format_values(values)
    foam_class = {
        "scalar": "volScalarField",
        "vector": "volVectorField",
        "symmTensor": "volSymmTensorField",
    }[ftype]

    patches = foammesh.read_boundary(Path(case) / "constant" / "polyMesh")
    bc_lines = []
    for p in patches.values():
        bc = _KIND_TO_BC.get(p.kind, "zeroGradient")
        bc_lines.append(f"    {p.name}\n    {{\n        type            {bc};\n    }}")
    boundary = "boundaryField\n{\n" + "\n".join(bc_lines) + "\n}\n"

    out = Path(case) / time_name / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        f"    class {foam_class};\n    object {name};\n}}\n"
        f"dimensions      {dims or _DIMS.get(name, '[0 0 0 0 0 0 0]')};\n"
        + _internal_block(values)
        + "\n"
        + boundary
    )
    return out
