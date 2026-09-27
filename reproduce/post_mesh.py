#!/usr/bin/env python3
"""Geometry step that follows blockMesh in the FAITH hill and Stanford diffuser cases (called by their Allrun):
the Cartesian block is moved onto the analytic hill surface, or mapped onto the diffuser walls, by the paper's
evaluation code (evaluation/forge/cases.py, post_mesh)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'evaluation'))
from forge.cases import post_mesh   # noqa: E402

work = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
cid = json.loads((work/'reproduce.json').read_text())['case_id']
case = next(c for c in json.loads((ROOT/'reproduce/cases.json').read_text())['cases'] if c['id'] == cid)
print(post_mesh(case, work))
