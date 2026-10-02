# BSL-EARSM tensor basis against the repository basis

What the EARSM comparator (closures/comparators/earsm/) may reuse from closures/kOmegaSSTBasis/basisTensors/integrityBasis.H
(Python twin: `tedp.tensors.integrity_basis`). Checked numerically on 200 random states, including frame
rotation and states where the Kolmogorov limiter is active, by
`test_earsm_reference.py::test_mapping_to_the_kOmegaSSTBasis_integrity_basis` (agreement to round-off).

Sources: M12 = Menter, Garbaruk and Egorov (2012), Progress in Flight Physics 3:89-104, printed page numbers;
M09 = the 2009 EUCASS paper, PDF page numbers.

## Verdict

- The ten repository tensors span everything the BSL-EARSM needs. Each Menter tensor is a single repository
  tensor, or a combination of two, times a power of r = tau*omega and a sign (table below).
- **The inputs differ in three ways.** (1) The time scale: the repository uses 1/omega, while the EARSM uses
  tau = max(1/(0.09 omega), 6 sqrt(nu/(0.09 k omega))). (2) The sign of the rotation tensor: the repository's
  What is minus the Wallin-Johansson W. (3) The frame term: What already contains the frame rotation, so it
  is the absolute rotation. Do not feed Shat and What from the basis library into the EARSM formulas
  unchanged. With 1/omega, every invariant is too small by a factor 0.09^2 = 0.0081 (IIS, IIW) or 0.09^3 (IV),
  and N, the betas and the anisotropy are then all wrong.
- The comparator directory has to be self-contained (handoff/harness_interface.md section 1: nothing outside
  `<dir>` may be included except OpenFOAM). The formulas may therefore be copied, but not `#include`d.
  The simplest correct reuse is to build S and W with the EARSM time scale and the WJ sign, and write the six
  Menter tensors directly (six lines, M12 eq. 3). The mapping below is mainly a cross-check.

## Inputs

| | Repository (kOmegaSSTBasis.C, lines ~266-306) | BSL-EARSM (M12 p. 92; M09 p. 3 eqs. 4-6) |
|---|---|---|
| time scale | 1/omegaSafe | tau = max(1/(Cmu omega), 6 sqrt(nu/(Cmu k omega))), Cmu = 0.09 |
| strain | Shat = dev(symm(grad U))/omega | S_ij = (tau/2)(dU_i/dx_j + dU_j/dx_i) = tau dev(symm(grad U)) |
| rotation | What = (skew(grad U) + F)/omega, F_ij = eps_ijk Omega_k | W_ij = (tau/2)(dU_i/dx_j - dU_j/dx_i) = -tau skew(grad U) |
| frame | absolute rotation, built in (What = -W_abs/omega) | not specified in M12/M09 (see "Frame rotation") |

OpenFOAM's `fvc::grad(U)` has components g_ij = dU_j/dx_i, the transpose of the velocity gradient. Hence
`skew(grad U)` = -W/tau, which is the sign flip. Define r = tau*omega (r = 1/0.09 = 11.11 when the limiter is
inactive, and larger where it is active). Then

    S = r Shat,    W_abs = -r What.

## Tensors

Menter's regrouped basis is M12 p. 92, eq. 3 (M09 p. 3, eq. 3). The repository basis T1..T10 is defined in
integrityBasis.H. I1..I5 are the repository invariants of (Shat, What): I1 = tr(Shat^2), I2 = tr(What^2),
I3 = tr(Shat^3), I4 = tr(What^2 Shat), I5 = tr(What^2 Shat^2).

| Menter (M12 eq. 3) | Definition | Repository equivalent |
|---|---|---|
| T1 | S | r T1 |
| T2 | S^2 - (1/3) IIS I | r^2 T3 (T3 = dev(S^2)) |
| T3 | W^2 - (1/3) IIW I | r^2 T4 (T4 = dev(W^2)) |
| T4 | S W - W S | **-** r^2 T2 (T2 = S W - W S, odd in W, so the sign flips) |
| T6 | S W^2 + W^2 S - (2/3) IV I - IIW S | r^3 (T6 - I2 T1) |
| T9 | W S W^2 - W^2 S W + (1/2) IIW (S W - W S) | **-** r^4 (T7 + (1/2) I2 T2) |
| (not used) | | T5, T8, T9, T10 of the repository: beta = 0 in the explicit WJ solution |

| Menter invariant (M12 p. 92) | Repository |
|---|---|
| IIS = S_ij S_ji | r^2 I1 |
| IIW = W_ij W_ji (<= 0) | r^2 I2 |
| IV = S_ik W_kj W_ji = tr(S W^2) | r^3 I4 |

Notes:
- The Menter tensors T6 and T9 differ from Wallin and Johansson's own T6 and T9. M12 p. 92 says the
  regrouping only separates the 2D and 3D parts, so that T6 = T9 = 0 in two-dimensional mean flow. With this
  regrouping, beta_1 = -N/Q and beta_4 = -1/Q (M09 p. 3, eq. 7). M12 p. 92 misprints beta_4 as -N/Q; the
  value -1/Q is proved by test_m12_printed_beta4_is_a_misprint and by the implicit-relation tests.
- The repository T6, written dev(symm(W W S + S W W)), is the same tensor as S W^2 + W^2 S - (2/3) tr(S W^2) I.
  Menter's T6 subtracts IIW S in addition. That part is proportional to S but carries beta_6, so it belongs
  in the nonlinear stress and not in nut.
- Odd powers of W (Menter T4 and T9; repository T2, T5, T7, T8 and T10) change sign with the convention. If
  W is built from `skew(grad U)` without the minus sign, the secondary flows reverse direction. The check is
  simple shear U = (G y, 0, 0) with G > 0: it must give a11 > 0 > a22 and a12 < 0
  (test_shear_equilibrium_matches_the_closed_form).

## Trace handling

- The two bases remove the traces in the same way. Repository T3, T4, T6 and T9 apply dev(); Menter subtracts
  (1/3) IIS I, (1/3) IIW I and (2/3) IV I. These are identical for traceless S. Every Menter tensor is
  symmetric and traceless (test_basis_is_symmetric_and_traceless), so the anisotropy a is traceless.
- Strain: use dev(symm(grad U)), as the repository does. The WJ definition has no dev(), but it assumes
  div U = 0, and dev() removes the discrete divergence error from the invariants.
- The `symm()` wrappers in integrityBasis.H do nothing in exact arithmetic, because S W - W S, W S W^2 - W^2 S W
  and the rest are already symmetric.
- Stress split in OpenFOAM: R = (2/3) k I - nut twoSymm(grad U) + nonlinearStress, with
  nut = -(1/2) beta_1 k tau and nonlinearStress = k (beta_3 T3 + beta_4 T4 + beta_6 T6 [+ beta_9 T9]).
  The nonlinear stress is traceless, so k is unchanged (test_earsm_state_openfoam_split).

## Frame rotation (outside M12/M09)

M12 and M09 give no system-rotation term (M12 p. 90). The basis library's What is the absolute rotation.
For the EARSM there are three choices: W_rel (as printed), W_abs = W_rel - F, or the form consistent with the
LRR model behind WJ, W* = W_rel - (13/4) F = W_abs - (9/4) F. Here F_ij = tau eps_ijk Omega_k. The last form
is derived in `reference.lrr_frame_factor` and checked in test_lrr_frame_rotation_derivation; it uses the
Coriolis term of the Reynolds-stress equation and the LRR rapid pressure-strain with C2 = 5/9 acting on the
absolute rotation. `reference.strain_rotation(..., frame_mode="relative"|"absolute"|"lrr")` implements all
three, so any of them can be verified. The comparator's choice must be stated in its handoff
(`frame_rotation`).

## Reuse checklist for closures/comparators/earsm

1. Compute tau with the limiter, S = tau dev(symm(grad U)) and W = -tau (skew(grad U) [+ frame term]).
2. Write T1, T2, T3, T4, T6 (and T9 for the full WJ variant) as in M12 eq. 3, or take the repository
   formulas with Shat := S and What := -W (that is, r = 1). Then T4 = -T2_repo and
   T9 = -(T7_repo + (1/2) I2 T2_repo).
3. Take the invariants from the same S and W: IIS = tr(S S), IIW = tr(W W), IV = tr(S W W).
4. The basis library's own Shat and What (time scale 1/omega) must not be reused for N or for the betas.
