# Changelog

## 1.1.0 (2 October 2026)

Additions for the revised paper; the corrections, the library of the corrections and the reproduction of the paper's
cases are unchanged.

- `closures/comparators/`: the four closures compared with the corrections in Sec. V. QCR2000 (Spalart, 2000) as
  the dictionary `models/qcr2000` of the existing library; SST-RC (Smirnov and Menter, 2009), the EARSM of Menter,
  Garbaruk and Egorov (2012) and AutoTurb (Zhang et al., 2025) as OpenFOAM v2312 turbulence models with their own
  libraries (`closures/comparators/Allwmake`), unchanged from the sources that ran in the paper
  (`comparator_models.json` records their SHA-256). With them: the independent reference implementations of SST-RC
  and EARSM, the unit programs, and the verification and preflight scripts of each closure.
- `closures/forgeTermFields/`: the utility that evaluates each term of a correction in a converged solution
  (Sec. IV, Fig. 15), its algebra test, `termfields_dict.py` (term dictionaries and the check of the split) and
  `term_fields.py` (the script that computed Fig. 15, as it ran).
- `evaluation/forge_fixed` and `evaluation/forge_deploy`: the evaluation support for compiled closures used by the
  comparison, unchanged.
- `reproduce/use_comparator.py`: turns a fresh SST case into a case of SST-RC, EARSM or AutoTurb.
- `tests/`: unit tests of QCR2000, of the support for compiled closures and of the case conversion, and the unit
  checks of EARSM and AutoTurb. The tests of the verification driver and of the cluster controller scripts, which
  are not part of this repository, are not included. In the copied tests and reference scripts, paths of the
  paper's working folders were changed to those of this repository.
- README: sections on the compared closures, the term fields and the tests; table numbers updated to the revised
  paper.

## 1.0.0 (27 September 2026)

The corrections of the paper, the OpenFOAM v2312 library `libforgeClosures.so`, the evaluation code, and the
scripts that build, run and score the paper's cases (https://doi.org/10.5281/zenodo.22996127).
