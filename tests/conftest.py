"""Puts the evaluation packages and the QCR2000 script on the import path of the tests.

Run from the repository root:  python3 -m pytest -q tests
(the two reference suites, closures/comparators/sstrc_reference and earsm_reference, run separately: both
import a module named reference).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT/'evaluation', ROOT/'closures/comparators/qcr2000'):
    sys.path.insert(0, str(path))
