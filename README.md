# FORGE closures

[![reproduce](https://github.com/Aekaros21/forge-closures/actions/workflows/reproduce.yml/badge.svg)](https://github.com/Aekaros21/forge-closures/actions/workflows/reproduce.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22996126.svg)](https://doi.org/10.5281/zenodo.22996126)

Explicit corrections to the k–ω SST turbulence model, the OpenFOAM library that runs them, and the cases and
scripts to test them. This repository accompanies

> C. Charalampous, "Toward non-regressing turbulence closures from large language models: explicit corrections
> to SST tested on unseen flows," submitted to *Physics of Fluids* (2026).

The paper seeks corrections to SST that lower its error on flows it predicts poorly without raising the error on
flows it already predicts well. A genetic algorithm searched for them: large language models proposed new
equations, whose coefficients were fitted and whose errors were measured in OpenFOAM simulations. This repository
holds the resulting corrections and what is needed to rerun the paper's cases with them. It also holds the
implementations of the four closures that the paper compares with the corrections (Sec. V) and the utility that
evaluates each term of a correction in a converged solution (Sec. IV). The records of the searches, of the holdout
verification, of the term ablation and of the comparison, including the prompts and responses of the language
models, are archived separately on Zenodo: https://doi.org/10.5281/zenodo.22996003.

## Contents

| Folder | Content |
|---|---|
| [`closures/`](closures) | `kOmegaSSTBasis`, the OpenFOAM v2312 turbulence model that adds a correction to SST (C++ source of `libforgeClosures.so`) |
| [`closures/comparators/`](closures/comparators) | The closures compared in Sec. V: QCR2000, SST-RC, EARSM and AutoTurb, with their reference implementations and verification ([closures/comparators/README.md](closures/comparators/README.md)) |
| [`closures/forgeTermFields/`](closures/forgeTermFields) | The utility that evaluates each term of a correction in a converged solution (Sec. IV, Fig. 15) |
| [`models/`](models) | The corrections of the paper, and QCR2000, as OpenFOAM dictionaries ([models/README.md](models/README.md)) |
| [`reproduce/`](reproduce) | Case definitions, the paper's case errors, and scripts that build, run and score the cases |
| [`evaluation/`](evaluation) | The evaluation code of the paper, unchanged: case generation, scoring, the search itself, and the support for compiled closures |
| [`tests/`](tests) | Unit tests of the compared closures, the evaluation support for them and the case conversion (no simulation) |
| [`data/`](data) | Grids and reference data, their checksums, and the script that downloads the rest ([data/README.md](data/README.md)) |
| [`cases/faith/`](cases/faith) | Reference data of the FAITH hill |

## Requirements

- OpenFOAM v2312 (ESI-OpenCFD, https://www.openfoam.com), with which every simulation of the paper was run.
- Python 3.10 or later with numpy, scipy and fluidfoam (`requirements.txt`), and pytest for the unit tests.
- For the square ducts, the Kaggle command-line client; for the Ahmed bodies, the diffuser and the wing–body
  junction, data from ERCOFTAC ([data/README.md](data/README.md)).

## Quick start

```sh
source /usr/lib/openfoam/openfoam2312/etc/bashrc   # the etc/bashrc of your OpenFOAM v2312 installation
./closures/Allwmake                                 # builds libforgeClosures.so into $FOAM_USER_LIBBIN
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 data/fetch_data.py pch22                    # rotating-channel DNS, 25 kB

python3 reproduce/make_case.py rotchan_ro10 sst
python3 reproduce/make_case.py rotchan_ro10 flow_state
runs/rotchan_ro10_sst/Allrun
runs/rotchan_ro10_flow_state/Allrun
python3 reproduce/score_case.py runs/rotchan_ro10_flow_state runs/rotchan_ro10_sst
```

Each run takes about a minute on one core. The last command prints the errors of both runs against the DNS,
their ratios, and the case error, the weighted mean of the ratios, next to the paper's value:

```
case error relative to SST: 0.3399
paper:                      0.3399   (difference -0.0000)
```

To run and score every case whose inputs are present, with SST and the corrections, on the cores of one
machine:

```sh
python3 reproduce/run_cases.py            # options: --cases, --closures, --cores, --tolerance
```

## Cases

`reproduce/cases.json` defines 19 of the paper's cases with the settings of the paper, and
`reproduce/make_case.py --list` lists them with the closures. `make_case.py` writes a case to
`runs/<case>_<closure>` with an `Allrun` script that runs the command sequence of the paper's evaluator: mesh
generation, `simpleFoam` (in parallel, between `decomposePar` and `reconstructPar`, where a case uses several
ranks) and the post-processing the scorer needs. `score_case.py` then compares a closure run with the SST run of
the same case.

| Case | Flow | Role in the paper | Cells | Ranks | Iterations | Data |
|---|---|---|---|---|---|---|
| `rotchan_ro00`, `rotchan_ro10`, `rotchan_ro50` | Rotating channel, Ro = 0, 0.10, 0.50 | Development; Ro = 0.10 fitted | 256 | 1 | 40,000 | `pch22` |
| `rotchan_ro01`, `rotchan_ro05`, `rotchan_ro15`, `rotchan_ro20` | Rotating channel, Ro = 0.01, 0.05, 0.15, 0.20 | Holdout | 256 | 1 | 40,000 | `pch22` |
| `squareDuct_Re_2000` | Square duct, Re_b = 2000 | Development | 9,216 | 4 | 5,000 | `ducts` |
| `squareDuct_Re_3200` | Square duct, Re_b = 3200 | Holdout | 9,216 | 4 | 5,000 | `ducts` |
| `nasa_hump_fine` | NASA wall-mounted hump | Development | 44,064 | 4 | 8,000 | included |
| `tmr_plate_137x97` | Flat plate | Development, protection | 13,056 | 4 | 8,000 | included |
| `tmr_bump_177x81` | Bump in channel | Development, protection | 14,080 | 4 | 8,000 | included |
| `naca0012_a010_225x65` | NACA 0012 at 10° | Development, protection | 14,336 | 4 | 40,000 | included |
| `naca0012_a015_225x65` | NACA 0012 at 15° | Holdout | 14,336 | 4 | 40,000 | included |
| `faith_hill` | FAITH hill | Holdout | 1.87 M | 48 | 6,000 | included |
| `diffuser3d` | Stanford three-dimensional diffuser | Holdout | 1.08 M | 32 | 6,000 | ERCOFTAC |
| `ahmed_25`, `ahmed_35` | Ahmed body, 25° and 35° slants | Holdout | 1.2 M | 48 | 6,000 | ERCOFTAC |
| `wingbody` | Wing–body junction | Holdout | 0.98 M | 32 | 6,000 | ERCOFTAC |

The first 14 cases run on a workstation. On a 12-core laptop, with several cases sharing the cores, a solver run
took 1 to 2 min for a rotating channel, 1 to 4 min for a square duct, 2 to 5 min for the plate or bump, 6 to 8 min
for the hump and 7 to 10 min for an airfoil; together, the 49 runs of these cases take roughly an hour on such a
machine. The five three-dimensional cases took about 1 to 1.5 h each on 32 or 48 cores of a cluster. The
Ahmed-body meshes are built by `reproduce/build_ahmed_mesh.py`, which runs snappyHexMesh around the body surface
from ERCOFTAC; snappyHexMesh in parallel is not bit-reproducible, so a rebuilt mesh can differ slightly from the
paper's. The other cases generate their meshes themselves; checkMesh flags the wall-resolved cells of the
wing–body mesh for their aspect ratio (up to about 8,000), which is expected there.

A parallel case can be run on fewer ranks than in the paper with `make_case.py --np N`, for example on a
workstation. The domain decomposition then differs from the paper's, so the errors can differ slightly;
`score_case.py` notes it.

## Reproduced values

The table lists the case errors reproduced with `reproduce/run_cases.py` on a laptop (OpenFOAM v2312 on
Pop!_OS 24.04, which is based on Ubuntu 24.04) next to the paper's values, which were computed on a cluster
(`reproduce/expected.json`). The development values are those behind the family averages of Tables III and V of
the paper, the holdout values those of Table VI, and the rotation-limited values those of Sec. III B 3 and Table VII.

| Case | Invariant | Flow-state | Rotation-limited |
|---|---|---|---|
| `rotchan_ro00` | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| `rotchan_ro10` | 0.9750 / 0.9750 | 0.3399 / 0.3399 | 0.4744 / 0.4744 |
| `rotchan_ro50` | 0.9422 / 0.9422 | 0.5987 / 0.5987 | 0.6093 / 0.6093 |
| `rotchan_ro01` | 0.9998 / 0.9998 | 1.6322 / 1.6325 | 0.5804 / 0.5804 |
| `rotchan_ro05` | 0.9906 / 0.9906 | 0.2949 / 0.2949 | 0.5104 / 0.5104 |
| `rotchan_ro15` | 0.9644 / 0.9644 | 0.3752 / 0.3752 | 0.4750 / 0.4750 |
| `rotchan_ro20` | 0.9595 / 0.9595 | 0.4539 / 0.4539 | 0.5158 / 0.5158 |
| `squareDuct_Re_2000` | 0.6123 / 0.6123 | 0.6109 / 0.6109 |  |
| `squareDuct_Re_3200` | 0.5860 / 0.5860 | 0.5834 / 0.5834 |  |
| `nasa_hump_fine` | 0.8960 / 0.8960 | 0.7584 / 0.7584 |  |
| `tmr_plate_137x97` | 0.9983 / 0.9983 | 1.0004 / 1.0004 |  |
| `tmr_bump_177x81` | 0.9960 / 0.9960 | 1.0028 / 1.0027 |  |
| `naca0012_a010_225x65` | 1.0299 / — | 1.0107 / — |  |
| `naca0012_a015_225x65` | 1.1303 / 1.1300 | 1.1560 / 1.1560 |  |

Each entry is the reproduced case error / the paper's value; 33 values are compared, and the largest difference is
0.0004. The rotation-limited correction is listed for the rotating channels only, since elsewhere it equals the
flow-state correction. The paper gives no case error for the airfoil at 10°, a preservation check: there the lift
and drag and their errors had to stay within 2% of SST, with floors of 0.001 in lift and 10⁻⁵ in drag (Sec. II A of
the paper), which they do. The invariant correction, for example, raises the lift error from 0.0144 to 0.0153,
within its allowance of 0.0013.

Differences of up to 4×10⁻⁴ come from floating-point differences between machines. They are largest on the
airfoil at 15° and on the rotating channel at Ro = 0.01, where the error ratios are large. The paper's evaluator
also required each run to be converged by criteria that compare several saved states (Appendix B);
`score_case.py` scores only the final state.

## How the case error is computed

The errors of a run against the reference data are computed by the paper's scorer (`evaluation/forge/cases.py`).
The case error is the weighted mean of the ratios of the closure's errors to those of SST, with the metric weights
of the paper (Sec. II A of the paper and the `contract` in `reproduce/cases.json`); a value below 1 is an
improvement on SST. Two conventions of the paper's tables are applied in `score_case.py`:

- The airfoil cases have no accuracy metric in the search, where they only guard against regressions; they are
  scored by their drag and lift errors with equal weights.
- The FAITH separation and reattachment errors are taken against the measured positions x/H = 0.38 and 1.78;
  the reattachment is where the reversed near-wall flow of the centerline PIV plane ends. The scorer frozen
  before the holdout used 1.72, the zero crossing of the measured skin friction; `reproduce/cases.json` records
  the change.

## Compared closures

Section V of the paper compares the corrections with QCR2000 (Spalart, 2000), SST-RC (Smirnov and Menter, 2009), the
EARSM of Wallin and Johansson in the form of Menter, Garbaruk and Egorov (2012), and AutoTurb (Zhang et al., 2025),
each with its published equations and coefficients; the fifth closure of Sec. V, the correction of Rincón et al.,
ran with its authors' library, which is not included here. QCR2000 is exactly a term of `kOmegaSSTBasis` and runs as
the dictionary `models/qcr2000`. The other three are OpenFOAM v2312 turbulence models in their own libraries, built
by `closures/comparators/Allwmake`; their sources are those that ran in the paper, unchanged, and
`closures/comparators/comparator_models.json` records their coefficients and source digests. Each implementation was
checked without simulation against an independent evaluation of the published equations: QCR2000 to a relative
difference of 5×10⁻¹¹ on 20,000 velocity gradients, the rotation function of SST-RC to 4×10⁻¹⁶ at 77 manufactured
states, the EARSM stress to 10⁻¹² on 235 states and the AutoTurb source to 10⁻¹² of the magnitude of its terms on
27,474 states. Before the comparison, each closure was also run on the cluster against published or expected
behaviour on the square duct, the flat plate, the rotating channels or the periodic hill; those records are in the
archive. [closures/comparators/README.md](closures/comparators/README.md) gives the equations as implemented, the
departures from the publications that leave the converged models unchanged, and the tests.

```sh
./closures/comparators/Allwmake                                         # libkOmegaSSTRC, libkOmegaEARSM, libkOmegaSSTAutoTurb
python3 reproduce/make_case.py squareDuct_Re_2000 qcr2000              # QCR2000, as a correction
python3 reproduce/make_case.py rotchan_ro10 sst --name rotchan_ro10_sstrc
python3 reproduce/use_comparator.py runs/rotchan_ro10_sstrc sstrc     # SST-RC, EARSM or AutoTurb
runs/rotchan_ro10_sstrc/Allrun
python3 reproduce/score_case.py runs/rotchan_ro10_sstrc runs/rotchan_ro10_sst
```

`use_comparator.py` makes the case of a compiled closure as the paper's comparison did: the SST case of the same
flow with its turbulence dictionary and library list changed, and nothing else.

## Term fields

`closures/forgeTermFields` is an OpenFOAM v2312 utility that evaluates, in a converged solution of a correction, the
stress and production of each of its three terms: the normal-stress term, the excess-production term and the
rotation term (Sec. IV of the paper, Fig. 15). It reads the velocity, pressure, k, ω and ν_t of the solution,
rebuilds the inputs of the correction with the sources of `libforgeClosures.so`, which it includes without
rebuilding or loading the library, and splits the correction with the same limits as the library. The terms are
listed in `system/termFieldsDict`, written by `termfields_dict.py`; the utility stops if the full correction listed
there differs from that in the case's `constant/turbulenceProperties`, and it compares the stress it rebuilds with
the `nonlinearStress` field the solver wrote. `termfields_dict.py check` verifies the split without a case: on
20,000 random states the terms add up to the full correction to within 10⁻¹⁵ of its magnitude, and the compiled
algebra of the utility (`testTermAlgebra`) agrees with a NumPy twin to within 10⁻¹⁵. `term_fields.py` is the script
that computed the term fields of Fig. 15 from the paper's stored solutions, with a second, independent evaluation in
NumPy compared cell by cell; it is included as it ran, and its paths refer to the paper's working folders.

```sh
./closures/forgeTermFields/Allwmake                                      # forgeTermFields and testTermAlgebra
python3 closures/forgeTermFields/termfields_dict.py dict --model rotation_limited \
    --out runs/nasa_hump_fine_rotation_limited/system/termFieldsDict      # after the run of that case
(cd runs/nasa_hump_fine_rotation_limited && forgeTermFields -latestTime)
python3 closures/forgeTermFields/termfields_dict.py check --binary "$FOAM_USER_APPBIN/testTermAlgebra" --work /tmp/tf
```

## Evaluation code

`evaluation/forge` and `evaluation/tedp` are the Python packages of the paper, copied unchanged. Besides case
generation and scoring, they contain the genetic algorithm (`evaluation/tedp/search/`) and the interface to the
language models that proposed the equations (`evaluation/forge/proposer.py`). Running the search itself needs access
to those models and a cluster scheduler and is not covered by the scripts here; its records are in the archive on
Zenodo (https://doi.org/10.5281/zenodo.22996003). `evaluation/forge_fixed` and `evaluation/forge_deploy`, also
unchanged, let the evaluator run a compiled closure (a RAS model, its library and its coefficients) through the same
cases, cache, convergence admission and scoring as the corrections, and built and checked those libraries on the
cluster. Their paths `src/comparators/<folder>` are `closures/comparators/<folder>` here.

## Tests

The unit tests run without OpenFOAM simulations:

```sh
python3 -m pytest -q tests                                   # QCR2000, evaluation/forge_fixed and forge_deploy, use_comparator.py
python3 -m pytest -q closures/comparators/sstrc_reference    # SST-RC reference and the output of its C++ kernel
python3 -m pytest -q closures/comparators/earsm_reference    # EARSM reference
```

With OpenFOAM v2312, `./closures/comparators/Allwmake -test` builds the unit programs of the compiled closures,
and `tests/comparators/earsm/check_earsm_unit.py`, `tests/comparators/autoturb/check_autoturb_source.py` and
`closures/forgeTermFields/termfields_dict.py check` compare them with the independent implementations.

## Citation

Please cite the paper when using the closures, the scripts or the case set (see also `CITATION.cff`). The code is
archived on Zenodo at https://doi.org/10.5281/zenodo.22996126 (all versions; version 1.0 is
https://doi.org/10.5281/zenodo.22996127). Version 1.1 adds the compared closures, the term-field utility and the
evaluation support for compiled closures ([CHANGELOG.md](CHANGELOG.md)). When using a compared closure, please cite
its publication as well (`closures/comparators/README.md`).

```bibtex
@article{charalampous2026forge,
  author  = {Charalampous, Charalampos},
  title   = {Toward non-regressing turbulence closures from large language models: explicit corrections to {SST} tested on unseen flows},
  journal = {Physics of Fluids},
  year    = {2026},
  note    = {Submitted}
}
```

## License

The code (`closures/`, `evaluation/`, `reproduce/`, `tests/`, `data/fetch_data.py`) is licensed under the GNU
General Public License, version 3 or later ([LICENSE](LICENSE)), as is OpenFOAM, against which the libraries are
built. The closure dictionaries, case definitions, expected values, checksums, and the states and targets of the
reference implementations in `closures/comparators` are licensed under CC BY 4.0 ([LICENSE-data](LICENSE-data));
the targets read from Fig. 1 of Smirnov and Menter (2009) are values of that publication. The NASA grids and reference data are in the public domain in the United States;
the other reference data remain under the terms of their distributors ([data/README.md](data/README.md)).
