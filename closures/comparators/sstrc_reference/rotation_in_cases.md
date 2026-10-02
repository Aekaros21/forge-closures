# How the rotating-channel cases impose system rotation, and what SST-RC must read

Independent reference, 2 Oct 2026. Sources: this repository's code (paths below) and OpenFOAM v2312
(`/usr/lib/openfoam/openfoam2312`). Searched with `grep -rIl -i "tabulatedAcceleration|SRFModel|MRFProperties|coriolis|frameOmega"`
over `src/`, `scripts/`, `config/`, `cases/`, `data/` and `tests/`, and for `Omega`, `rotation` and `fvOptions` in
`evaluation/forge/cases.py`.

## 1. Which cases rotate

Only the cases with `"adapter": "rotation"` in `config/verification_cases.json`. All of them are built by
`evaluation/tedp/holdout_cases/rotchan.py` through `evaluation/forge/cases.py:prepare_case` (branch `adapter == 'rotation'`,
`rotchan.make_case(work, ro, spec, frame_rotation=True, ny=..., wall_grading=..., re_bulk_half=...)`).

| case | role | Ro | frame angular velocity (rad/s) |
|---|---|---|---|
| rotchan_ro00 | calibration | 0 | (0 0 0) |
| rotchan_ro10, rotchan_ro10_ny128, rotchan_ro10_ny512 | calibration / numerical study | 0.10 | (0 0 0.05) |
| rotchan_ro50, rotchan_ro50_ny128, rotchan_ro50_ny512 | calibration / numerical study | 0.50 | (0 0 0.25) |
| rotchan_ro01, ro05, ro15, ro20 | verification (held-out conditions) | 0.01, 0.05, 0.15, 0.20 | (0 0 0.005), (0 0 0.025), (0 0 0.075), (0 0 0.1) |
| channel_lm5200 | verification | 0 | (0 0 0) (same adapter, inertial) |

All eleven rotating-channel ids are in the comparator campaign `config/verify_campaign_comparators.json`. No case
uses SRF, MRF, a moving mesh or a Coriolis source other than this one; every other case is inertial and the
harness writes `frameOmega (0 0 0)` there.

## 2. How the rotation enters the momentum equation

`rotchan.py` writes (one streamwise cell, flow along +x driven by `meanVelocityForce`, `Ubar (1 0 0)`, half-width
h = 1, U_m = 1, Re = U_m h/nu = 2900):

```
constant/fvOptions
    frameRotation
    {
        type            tabulatedAccelerationSource;
        selectionMode   all;
        timeDataFileName "constant/acceleration-terms.dat";
    }
constant/acceleration-terms.dat       (time ((linear acceleration) (angular velocity) (angular acceleration)))
    (
        (0   ((0 0 0) (0 0 W) (0 0 0)))
        (1e9 ((0 0 0) (0 0 W) (0 0 0)))
    )
with W = rotchan.omega_for(Ro) = Ro U_m/(2 h) = Ro/2      (Ro = 2 |Omega| h/U_m, Kristoffersen and Andersson 1993)
```

OpenFOAM v2312, `src/fvOptions/sources/derived/tabulatedAccelerationSource/tabulatedAccelerationSourceTemplates.C`,
`addSup`: `Omega = acceleration.y()` (the second vector of the row) and

```
eqn -= rho*(2*Omega ^ U) + rho*(Omega ^ (Omega ^ C)) + rho*(dOmegaDT ^ C)    (and -= rho*a)
```

The fvOption is the right-hand side of the momentum equation of the velocity RELATIVE to the frame, so the solver
solves Du/Dt = ... - 2 Omega x u - Omega x (Omega x r): `Omega` is the angular velocity of the frame of the
calculation with respect to inertial space. The centrifugal part is absorbed by the pressure (rotchan.py docstring).

Orientation: u = U e_x, Omega = W e_z, so the Coriolis acceleration -2 Omega x u = -2 W U (e_z x e_x) = -2 W U e_y
points to -y. The -y wall (`bottomWall`) is the pressure side (dU/dy > 0, mean vorticity anti-parallel to Omega,
destabilised, higher friction); `topWall` is the suction side. The scored observable
`wall_friction_ratio = wall_cf_bottom/wall_cf_top` (`evaluation/forge/cases.py`, rotation scoring) is therefore
pressure/suction and must exceed 1 for a rotation-sensitive model.

## 3. What the turbulence models receive

* Basis library (the paper's corrections): `rotchan.make_case` inserts `frameOmega (0 0 W);` into
  `kOmegaSSTBasisCoeffs`; `kOmegaSSTBasis::checkFrameConfiguration` (closures/kOmegaSSTBasis/kOmegaSSTBasis.C) reads the
  same table and stops with a fatal error if a rotating-frame source exists without `frameOmega`, if they differ
  by more than 1e-9 relative, or if `frameOmega` is non-zero without a source.
* Comparators (`evaluation/forge_fixed/__init__.py`): `prepare_fixed_case` builds the stock SST case
  (`base_prepare(case, None, ...)`, identical fvOptions and table) and replaces `turbulenceProperties` with
  `render_turbulence_properties(model, frame_omega(case))`, where `frame_omega(case) = (0, 0, rotchan.omega_for(ro))`
  for the rotation adapter and `(0, 0, 0)` otherwise. The model therefore finds, always present,

```
RAS { RASModel kOmegaSSTRC; ... kOmegaSSTRCCoeffs { cr1 ...; cr2 ...; cr3 ...; frameOmega (0 0 W); } }
```

## 4. Exactly which vector SST-RC must use

**`Omega^rot = frameOmega`, read as a required entry `this->coeffDict_.get<vector>("frameOmega")`, in rad/s, with no
factor 2 and no sign change.** It is the same vector, component by component, as the second vector of each row of
`constant/acceleration-terms.dat` and as the `Omega` of the solver's Coriolis term above, which is the meaning of
Smirnov and Menter (2009), p. 041010-2: "all the variables and their derivatives are defined with respect to the
reference frame of the calculation, which is rotating with a rate Omega^rot". Values: Ro 0.10 -> (0 0 0.05),
Ro 0.50 -> (0 0 0.25). Recommended: repeat the basis library's consistency check against the table (a mismatch
or a missing entry should be fatal, never a silent zero).

It enters twice, both from Eqs. (6) and (8) of Smirnov and Menter (2009), p. 041010-2:

1. absolute rotation tensor, Eq. (8): Omega_ij = (du_i/dx_j - du_j/dx_i)/2 + eps_mji Omega^rot_m
   (= W_ij - eps_ijm Omega^rot_m); its magnitude sqrt(2 Omega_ij Omega_ij) is the Omega of r* = S/Omega (Eq. 5)
   and of the denominator Omega D^3 (Eq. 6), and the tensor itself multiplies S in the numerator;
2. frame term of the strain derivative, Eq. (6): DS_ij/Dt + (eps_imn S_jn + eps_jmn S_in) Omega^rot_m.

OpenFOAM notation (`gradU = fvc::grad(U)`, `(gradU)_ij = du_j/dx_i`, the transpose of the paper's):

```
S      = symm(gradU)                                   // S_ij, Eq. (7)
OmAbs  = -skew(gradU) + (*frameOmega)                  // Eq. (8); OpenFOAM (*v)_ij = -eps_ijk v_k (TensorI.H)
T      = twoSymm((*frameOmega) & S)                    // (eps_imn S_jn + eps_jmn S_in) Omega_m = OmF S - S OmF
num    = 2*((OmAbs & S) && (DSDt + T))                 // 2 Omega_ik S_jk [ ... ], Eq. (6)
W      = sqrt(2*magSqr(OmAbs)),  S = sqrt(2*magSqr(S)),  D = sqrt(max(S^2, 0.09 omega^2))
rTilde = num/(W*D^3),  rStar = S/W
```

`skew(gradU)` alone is MINUS the paper's W_ij; using it without the minus sign inverts r~ (pressure and suction
sides swap). The basis library writes the same absolute tensor as `-(skew(gradU) + frameTensor)` with
`frameTensor_ij = eps_ijk Omega_k` (kOmegaSSTBasis.C, `What`).

## 5. Consequences for verification

* In the one-cell-thick, fully developed channel, u = (U(y), 0, 0) and every field depends on y only, so
  U . grad(S) == 0: the DS/Dt part of r~ vanishes identically and only the frame terms act. Locally, with
  G = dU/dy and D = S: r* = |G|/|G - 2 W|, r~ = -W sign(G - 2 W)/|G| (closed form, `analytic_cases.py`,
  family `rotshear`); r~ < 0 and f_r1 > 1 near the bottom (pressure) wall, r~ > 0 and f_r1 < 1 near the top wall.
* The DS/Dt implementation (Smirnov and Menter Eqs. (12)-(15), `fvc::div(phi, S)` or `U & fvc::grad(S)`) is not
  exercised by the channels; it is checked pointwise by the `azimuthal_*` and `general3d_*` states of
  `manufactured_states.txt` and only weakly by the flat plate (target f_r1 = 1, friction within 0.1 % of SST).
* The published channel targets (`published_targets.json`) are expressed in the observables above:
  `wall_friction_ratio` (bottom/top) and the velocity profile with y_case/h = 2 y/H - 1.
