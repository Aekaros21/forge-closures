# FORGE closures

[![reproduce](https://github.com/Aekaros21/forge-closures/actions/workflows/reproduce.yml/badge.svg)](https://github.com/Aekaros21/forge-closures/actions/workflows/reproduce.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22996126.svg)](https://doi.org/10.5281/zenodo.22996126)

Explicit corrections to the k–ω SST turbulence model, the OpenFOAM library that runs them, and the cases and
scripts to test them. This repository accompanies

> C. Charalampous, "Toward non-regressing turbulence closures from large language models: explicit corrections
> to SST tested on unseen flows," submitted to *Physics of Fluids* (2026).

The paper seeks corrections to SST that lower its error on flows it predicts poorly without raising the error on
flows it already predicts well. A genetic algorithm searched for them: large language models proposed new
equations, whose coefficients were fitted and whose errors were measured in OpenFOAM simulations. This repository
holds the resulting corrections and what is needed to rerun the paper's cases with them. The records of the
searches and of the holdout verification, including the prompts and responses of the language models, are
archived separately on Zenodo: https://doi.org/10.5281/zenodo.22996003.

## Contents

| Folder | Content |
|---|---|
| [`closures/`](closures) | `kOmegaSSTBasis`, the OpenFOAM v2312 turbulence model that adds a correction to SST (C++ source of `libforgeClosures.so`) |
| [`models/`](models) | The corrections of the paper as OpenFOAM dictionaries ([models/README.md](models/README.md)) |
| [`reproduce/`](reproduce) | Case definitions, the paper's case errors, and scripts that build, run and score the cases |
| [`evaluation/`](evaluation) | The evaluation code of the paper, unchanged: case generation, scoring and the search itself |
| [`data/`](data) | Grids and reference data, their checksums, and the script that downloads the rest ([data/README.md](data/README.md)) |
| [`cases/faith/`](cases/faith) | Reference data of the FAITH hill |

## Requirements

- OpenFOAM v2312 (ESI-OpenCFD, https://www.openfoam.com), with which every simulation of the paper was run.
- Python 3.10 or later with numpy, scipy and fluidfoam (`requirements.txt`).
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
(`reproduce/expected.json`). The development values are those behind the family averages of Tables II and IV of
the paper, the holdout values those of Table V, and the rotation-limited values those of Sec. III B 3 and Table VI.

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

## Evaluation code

`evaluation/forge` and `evaluation/tedp` are the Python packages of the paper, copied unchanged. Besides case
generation and scoring, they contain the genetic algorithm (`evaluation/tedp/search/`) and the interface to the
language models that proposed the equations (`evaluation/forge/proposer.py`). Running the search itself needs
access to those models and a cluster scheduler and is not covered by the scripts here; its records are in the
archive on Zenodo (https://doi.org/10.5281/zenodo.22996003).

## Citation

Please cite the paper when using the closures, the scripts or the case set (see also `CITATION.cff`). The code
is archived on Zenodo at https://doi.org/10.5281/zenodo.22996126 (all versions; version 1.0, as submitted with
the paper, is https://doi.org/10.5281/zenodo.22996127).

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

The code (`closures/`, `evaluation/`, `reproduce/`, `data/fetch_data.py`) is licensed under the GNU General Public
License, version 3 or later ([LICENSE](LICENSE)), as is OpenFOAM, against which the library is built. The closure
dictionaries, case definitions, expected values and checksums are licensed under CC BY 4.0
([LICENSE-data](LICENSE-data)). The NASA grids and reference data are in the public domain in the United States;
the other reference data remain under the terms of their distributors ([data/README.md](data/README.md)).
