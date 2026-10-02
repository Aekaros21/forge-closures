# Compared closures

Section V of the paper compares the corrections with four established or published closures under the same
rule: QCR2000, SST-RC, EARSM and AutoTurb. Each runs with its published equations and coefficients through the
same cases, numerics, convergence admission and scoring as the corrections; Appendix A 6 of the paper gives
their equations. The fifth closure of Sec. V, the correction of Rincón et al., was run with its authors'
library (https://github.com/AUfluids/KOSSTPDA), which is not redistributed here.

| Closure | Published source | Implementation | Label in the records |
|---|---|---|---|
| QCR2000 | Spalart, Int. J. Heat Fluid Flow **21**, 252 (2000), https://doi.org/10.1016/S0142-727X(00)00007-2, with C_cr1 as given by Prudenzano, Gand and Deck, AIAA J. **63**, 1819 (2025), https://doi.org/10.2514/1.J064610, Eqs. (5)–(6) | Expression for `kOmegaSSTBasis`: [`models/qcr2000`](../../models/qcr2000) | `QCR` |
| SST-RC | Smirnov and Menter, J. Turbomach. **131**, 041010 (2009), https://doi.org/10.1115/1.3070573, Eqs. (1)–(11), with the rotation function of Spalart and Shur, Aerosp. Sci. Technol. **1**, 297 (1997), https://doi.org/10.1016/S1270-9638(97)90051-1 | RAS model `kOmegaSSTRC`, library `libkOmegaSSTRC.so`: [`sstrc/`](sstrc) | `SSTRC` |
| EARSM | Menter, Garbaruk and Egorov, Prog. Flight Phys. **3**, 89 (2012), https://doi.org/10.1051/eucass/201203089, the BSL form of the model of Wallin and Johansson, J. Fluid Mech. **403**, 89 (2000), https://doi.org/10.1017/S0022112099007004 | RAS model `kOmegaEARSM`, library `libkOmegaEARSM.so`: [`earsm/`](earsm) | `EARSM` |
| AutoTurb | Zhang, Zheng, Liu, Zhang and Wang, Phys. Fluids **37**, 015211 (2025), https://doi.org/10.1063/5.0247759 (arXiv:2410.10657), the selected model of Eq. (11) | RAS model `kOmegaSSTAutoTurb`, library `libkOmegaSSTAutoTurb.so`: [`autoturb/`](autoturb) | `AutoTurb` |

`comparator_models.json` is the definition of the four closures frozen for the comparison campaign
(`forge-v3-verify-008-comparators` in the records archive, https://doi.org/10.5281/zenodo.22996003): the
expression of QCR2000, and for the other three the RAS model, library, coefficients and the SHA-256 of their
source folder. The folders `sstrc/`, `earsm/` and `autoturb/` are the sources that ran, unchanged; the paths in
the records, `src/comparators/<folder>`, are this folder. The source digest is checked with

```sh
PYTHONPATH=evaluation python3 -m forge_fixed hash closures/comparators/sstrc   # prints the source_sha256
```

## QCR2000

The quadratic constitutive relation replaces the SST Reynolds stress R by R − C_cr1 (O R + (O R)ᵀ), with O the
rotation tensor normalized by the magnitude of the velocity gradient and C_cr1 = 0.3. Applied to the SST stress,
the added stress is exactly the T2 term of `kOmegaSSTBasis` with the coefficient
g2 = C_cr1 β* Π / (I1 √(I1 − I2)), where Π is the production-to-dissipation ratio without its upper bound. The
closure therefore runs in `libforgeClosures.so` as the expression in `models/qcr2000.json`, with the realizability
check, ramp and under-relaxation of the corrections, and with the coefficient limit removed (gMax = 10³⁰), since
the published model has none. Above the bound of Π at 10, Π is recovered from the variable `phiDkPk` of the library.
The frame rotation enters the rotation tensor, so QCR2000 is evaluated in the inertial frame in the rotating
channels.

Verification (`qcr2000/qcr2000_comparator.py`): on 20,000 random velocity gradients, including rotating frames,
the floor of ω and Π beyond its bound, the expression reproduces an independent NumPy evaluation of QCR2000 to a
relative difference of 4.6×10⁻¹¹; the solver's own C++ expression parser gives the same coefficient to the last
bit; C_cr1 = 0 gives SST exactly; in simple shear the normal stresses are ordered as in Spalart's calibration.
`qcr2000_apriori_duct.py` evaluates the closure on a converged SST solution of the square duct, and
`qcr2000_preflight_check.py` checks the secondary flow of a QCR2000 run of the duct against the DNS (sign along
the four corner bisectors and the eight-vortex pattern).

```sh
python3 reproduce/make_case.py squareDuct_Re_2000 qcr2000     # then Allrun and score_case.py, as for a correction
PYTHONPATH=evaluation:closures/comparators/qcr2000 python3 closures/comparators/qcr2000/qcr2000_comparator.py --out <dir>
```

## SST-RC

The rotation function f_r1 of Spalart and Shur multiplies both production terms of SST, after the SST limiters,
with c_r1 = 1, c_r2 = 2, c_r3 = 1 and f_r1 limited to [0, 1.25]. The eddy viscosity, dissipation, diffusion and
stress are those of SST (`kOmegaSSTBase` of OpenFOAM v2312). The strain-rate derivative DS/Dt is the steady
material derivative, evaluated from the face fluxes as Smirnov and Menter describe, minus the discrete continuity
error. f_r1 is evaluated from the current velocity and ω before the k and ω equations and is written as the field
`fr1`. The frame rotation is the `frameOmega` entry of the coefficient dictionary; at the first iteration the
model checks it against the rotating-frame source of the case and stops on a mismatch. Two entries are not in the
publication and leave the converged model unchanged: `rampIterations 200` raises the departure of f_r1 from one
linearly over the first 200 iterations, as for the corrections (0 gives the published start), and `Cscale 1`
returns f_r1 bit for bit.

Verification: the pointwise kernel (`sstrc/sstrcKernel.H`, the function the model calls per cell), run by the unit
executable `sstrc_unit` on 77 manufactured states (plane shear with and without frame rotation, solid-body
rotation, Taylor–Couette and curved-channel flows in inertial and rotating frames, general three-dimensional
fields), agrees with the independent NumPy reference in `sstrc_reference/` to 4.4×10⁻¹⁶ in f_r1, and with
hand-derived closed forms; plane shear gives f_r1 = 1 and solid-body rotation f_r1 = 0. `sstrc_dsdt` checks the
DS/Dt operator on manufactured fields on in-memory grids (second order, observed order 1.99).
`sstrc_reference/published_targets.json` holds the rotating-channel targets read from the vector paths of Fig. 1
of Smirnov and Menter (`digitize_sm09_fig1.py`), and `check_targets.py` compares a run with them.

## EARSM

The explicit algebraic Reynolds-stress model of Wallin and Johansson in the BSL form of Menter, Garbaruk and Egorov:
the anisotropy a = β1 T1 + β3 T3 + β4 T4 + β6 T6 on the regrouped basis of the publication, N the largest real root
of the cubic in its closed form, A1 = 1.245, C1 = 1.8, the time scale with C_τ = 6, and the BSL k–ω equations with
k/ω in the diffusion terms and the production of the full stress. The linear part of the stress replaces the SST
eddy viscosity and its limiter; the rest enters the momentum equation as the explicit stress `nonlinearStress`,
which is set to zero on walls. Readings of the publication: β4 = −1/Q as in the 2009 printing of the same authors
(the only value that satisfies the implicit algebraic relation), and β9 = 0 as published (`includeT9 false`). The
publication has no system-rotation term, so `frameOmega` is read and not used. `rampIterations 200` passes the
stress and eddy viscosity linearly from those of SST to those of the model over the first 200 iterations.

Verification (`tests/comparators/earsm/check_earsm_unit.py`, which builds the library and the test programs in a
scratch folder): the unit program `testEARSM` agrees with the independent NumPy reference in `earsm_reference/` to
9.2×10⁻¹³ on 235 shared states (`earsm_reference/compare_cpp.py`); the three-dimensional implicit relation is
satisfied to 1.2×10⁻¹⁵; the closed-form root equals the largest real root to 1.9×10⁻¹⁴ on 1,824 invariant pairs,
including the branch boundary; the divergence of the explicit stress converges at an observed order of 1.88 on
in-memory box meshes; `testEARSMLoad` checks that the library registers the model. `earsm_reference/basis_mapping.md` maps the
basis of the publication onto that of `kOmegaSSTBasis`.

## AutoTurb

The selected model of AutoTurb keeps the SST stress and adds the source
R = 2k ∂U_i/∂x_j [(sin λ1 + 0.5) T1_ij + T2_ij] to the production of k, and γR/ν_t to that of ω, with
T1 = S/ω, T2 = (SΩ − ΩS)/ω² and λ1 = tr(S/ω)². R is added after the SST production limiter and is neither
limited nor relaxed. The frame rotation cannot enter (T2 does no work for any rotation), so `frameOmega` is read
and not used. `rampIterations 200` raises R linearly over the first 200 iterations.

Verification (`tests/comparators/autoturb/check_autoturb_source.py`): the one-point source of the model
(`autoturb/autoTurbProduction.H`), run by the unit program `testAutoTurbSource` on 27,474 manufactured states,
agrees with an independent NumPy evaluation of Eq. (11) to 4.2×10⁻¹³ relative on the well-conditioned states and to
2×10⁻¹³ of the magnitude of the source term on all; zero coefficients give R = 0 bit for bit; `testAutoTurbLoad`
checks that the library registers the model.

## Building and running

```sh
source /usr/lib/openfoam/openfoam2312/etc/bashrc     # the etc/bashrc of your OpenFOAM v2312 installation
./closures/Allwmake                                  # libforgeClosures.so, which also runs QCR2000
./closures/comparators/Allwmake                      # the three libraries; with -test also the unit programs

python3 reproduce/make_case.py rotchan_ro10 sst --name rotchan_ro10_sstrc
python3 reproduce/use_comparator.py runs/rotchan_ro10_sstrc sstrc
runs/rotchan_ro10_sstrc/Allrun
python3 reproduce/score_case.py runs/rotchan_ro10_sstrc runs/rotchan_ro10_sst
```

`use_comparator.py` turns a fresh SST case into a case of SST-RC, EARSM or AutoTurb by changing two files, as the
campaign did (`evaluation/forge_fixed`): `constant/turbulenceProperties` selects the RAS model with the coefficients
of `comparator_models.json` and the frame rotation of the case, and `system/controlDict` loads the closure's
library after `libforgeClosures.so`. To use a closure in another case, load its library in `system/controlDict`
and select it in `constant/turbulenceProperties`, for example

```
RAS
{
    RASModel        kOmegaSSTRC;
    turbulence      on;
    printCoeffs     off;
    kOmegaSSTRCCoeffs
    {
        Cscale          1;
        cr1             1;
        cr2             2;
        cr3             1;
        fr1Max          1.25;
        rampIterations  200;
        rotationCurvatureCorrection true;
        frameOmega      (0 0 0);
    }
}
```

The paper's values of the compared closures are in Table IX and in the records of campaign
`forge-v3-verify-008-comparators`. Some of them come from runs extended to two or four times the iterations of the
case, which the convergence admission of Appendix B required; `make_case.py` sets up the paper's iteration count.

## Unit tests

```sh
python3 -m pytest -q tests                                   # QCR2000, the harness and use_comparator.py
python3 -m pytest -q closures/comparators/sstrc_reference    # the SST-RC reference and the C++ kernel output
python3 -m pytest -q closures/comparators/earsm_reference    # the EARSM reference (and the C++ output, if built)
python3 tests/comparators/earsm/check_earsm_unit.py          # needs OpenFOAM v2312
python3 tests/comparators/autoturb/check_autoturb_source.py --out autoturb_unit_check.json   # needs OpenFOAM v2312
```

`check_earsm_unit.py` writes its report and the C++ output read by the EARSM reference tests to
`build/qualification/earsm/`. The records of the verification as run for the paper, including the cluster
preflights of each closure on the square duct, the flat plate, the rotating channels and the periodic hill, are in
the records archive (`comparator_verification/`).
