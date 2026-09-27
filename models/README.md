# Closures

Each correction of the paper is a coefficient dictionary for `kOmegaSSTBasis`, the OpenFOAM turbulence model in
[`closures/`](../closures). The model is the k–ω SST model of OpenFOAM with two added terms, both algebraic
functions of the local flow: an anisotropic Reynolds stress in the momentum equations and a source in the
turbulent-kinetic-energy equation (Appendix A of the paper). A closure is therefore data: the solver is not
recompiled for a new closure, and the dictionaries below are the complete definitions of the models in the paper.

| Closure | Paper name | Section | Description |
|---|---|---|---|
| `sst` | SST | — | Stock k–ω SST of OpenFOAM v2312 (`kOmegaSST`), the baseline of every error ratio |
| `invariant` | Invariant correction | II | Final equation of the invariant search; inputs are the strain- and rotation-rate invariants, the SST blending function, the turbulence Reynolds number and the production-to-dissipation ratio |
| `flow_state` | Flow-state correction | III | Selected equation of the flow-state search; adds the pressure and turbulent-kinetic-energy gradients and the frame rotation to the inputs |
| `rotation_limited` | Rotation-limited correction | III B 3 | The flow-state correction with its rotation term multiplied by tanh(R_f/0.01); adopted |
| `shielded_S2`, `shielded_S4`, `shielded_S8`, `shielded_T1` | Shielded variants | III B 3 | The rotation-limited correction with its excess-production term multiplied by (1−F₁)², (1−F₁)⁴ or (1−F₁)⁸, which removes it in the inner part of boundary layers, or with the threshold at which that term acts raised by 1 (T1); fitted to the near-stall airfoil and not adopted |

## Files

- `<closure>/turbulenceProperties` is the dictionary a case reads, with the fitted coefficients written into
  the expressions. `reproduce/make_case.py` copies it into each case it builds; `index.json` lists its SHA-256.
- `<closure>.json` holds the same equation as the search recorded it: the expressions with the symbolic
  coefficients `c0`…`c7`, their fitted values, and the campaign it came from. Table VII of the paper lists the
  coefficients.

In `kOmegaSSTBasisCoeffs`, `bDelta` lists the terms g_n T^(n) of the anisotropic stress and `rSource` the terms
h_n T^(n) of the source, each as a basis tensor (`T1`, `T2`, `T3`, `T4`, `T6`) and the expression of its
coefficient function. The other entries are the settings of Appendix A, identical for every closure:

| Entry | Value | Meaning |
|---|---|---|
| `gMax` | 10 | Each coefficient function is limited to [−10, 10] |
| `rMaxFactor` | 5 | The source is limited to ±5β\*kω |
| `bDeltaRelax` | 0.5 | Under-relaxation of both added terms |
| `rampIterations` | 200 | Both terms are introduced linearly over the first 200 iterations |
| `realizabilityClip` | on | The anisotropic stress is scaled down where the total anisotropy would leave the realizable range |
| `nonlinearProduction` | on | The work of the anisotropic stress is added to the SST production before the production limiter |
| `coupledStressBoundary` | true | The stress is exchanged across processor boundaries before its divergence is taken |

## Names in the expressions

| Name | Symbol in Appendix A | Meaning |
|---|---|---|
| `I1`, `I2`, `I5` | I₁, I₂, I₅ | tr Ŝ², tr Ŵ², tr(Ŵ²Ŝ²), with the strain and rotation rates scaled by the turbulence frequency |
| `F1` | F₁ | SST blending function |
| `Ret` | Re_t | Turbulence Reynolds number k/(νω) |
| `PoE` | Π | Ratio of SST production to dissipation, limited to 10 |
| `Gp`, `Gk` | G_p, G_k | Pressure and turbulent-kinetic-energy gradients on the turbulence scales, between 0 and 1 |
| `Ksn` | K_sn | Normal strain rate along the gradient of k |
| `Rf` | R_f | Frame rotation rate relative to the turbulence frequency |
| `Rw` | R_w | Cosine of the angle between the frame rotation and the relative vorticity |

The functions are `sqrt`, `exp`, `tanh`, `abs` and `max`. The invariant correction uses `I1`, `I2`, `I5`, `F1`,
`Ret` and `PoE`; the other closures use `I1`, `I2`, `F1`, `Ret` and `PoE` with the five flow-state inputs `Gp`,
`Gk`, `Ksn`, `Rf` and `Rw`. Appendix A gives the exact definitions, including the floors of k and ω.

## Using a closure in another case

1. Build the library once in an OpenFOAM v2312 environment: `./closures/Allwmake`.
2. Load it in `system/controlDict`: `libs ("libforgeClosures.so");`.
3. Copy `models/<closure>/turbulenceProperties` to `constant/turbulenceProperties`.
4. In a rotating frame, add the frame's angular velocity (rad/s) to `kOmegaSSTBasisCoeffs`, for example
   `frameOmega (0 0 0.05);`. The model stops with an error if a rotating-frame source is configured without it.

The closures were fitted and tested with steady `simpleFoam` and the numerics that `reproduce/make_case.py`
writes for each case (Appendix B of the paper); a case built with it is a working example of every setting.
