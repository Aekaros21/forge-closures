/*---------------------------------------------------------------------------*\
File
    sstrcKernel.H

Description
    Pointwise Spalart-Shur rotation/curvature function of the SST-RC (SST-CC)
    model of Smirnov & Menter (2009), J. Turbomach. 131, 041010, p. 041010-2:

      fRotation = (1 + cr1) 2 r* / (1 + r*) [1 - cr3 atan(cr2 rTilde)] - cr1   Eq.(1)
      fr1       = max(min(fRotation, 1.25), 0)                               Eq.(4)
      r*        = S/W                                                        Eq.(5)
      rTilde    = 2 W_ik S_jk [DS_ij/Dt + (eps_imn S_jn + eps_jmn S_in)
                  Omega_m] / (W D^3)                                         Eq.(6)
      S_ij      = (du_i/dx_j + du_j/dx_i)/2                                  Eq.(7)
      W_ij      = (du_i/dx_j - du_j/dx_i)/2 + eps_mji Omega_m                Eq.(8)
      S^2 = 2 S_ij S_ij, W^2 = 2 W_ij W_ij, D^2 = max(S^2, 0.09 omega^2)     Eqs.(9-11)
      cr1 = 1, cr2 = 2, cr3 = 1                                              (Sec. 2)

    (Smirnov & Menter write Omega_ij for W_ij and Omega for W; W is used here
    to keep the frame angular velocity Omega_m distinct.)  Velocities and
    derivatives are those of the calculation frame, which rotates at the
    angular velocity Omega_m (Spalart & Shur 1997; Shur et al. 2000, p. 785).
    W_ij is then the ABSOLUTE rotation tensor and the bracket of Eq.(6) the
    inertial-frame rate of change of S_ij.

    Optional scaling of the ANSYS implementation (not in the paper):
      fScaled = max(0, 1 + Cscale (fr1 - 1)),  Cscale = 1  ->  fScaled = fr1
      (evaluated as fr1 + (Cscale - 1)(fr1 - 1), exact for Cscale = 1).

    Argument conventions
      gradU : OpenFOAM fvc::grad(U), gradU_ij = du_j/dx_i (the TRANSPOSE of
              the paper's du_i/dx_j); converted here.
      DSDt  : Lagrangian derivative of S_ij in the calculation frame, without
              the frame terms (steady RANS: U_k dS_ij/dx_k).
      Omega : frame angular velocity vector [rad/s].
      omega : SST specific dissipation rate [1/s].

    Singular states are evaluated without division by zero:
      2 r* / (1 + r*) is computed as 2 S/(S + W)  (identical for W > 0),
      rTilde uses max(W D^3, VSMALL)            (numerator is 0 when W = 0).
    For S = W, rTilde = 0 the result is exactly fr1 = 1 in floating point.

    The same functions are called by the RAS model (kOmegaSSTRC.C) and by
    the pointwise unit executable (src/comparators/sstrc_unit), so the unit
    test exercises the production code.
\*---------------------------------------------------------------------------*/

#ifndef sstrcKernel_H
#define sstrcKernel_H

#include "tensor.H"
#include "symmTensor.H"
#include "vector.H"
#include "scalar.H"

#include <cmath>

namespace Foam
{
namespace sstrc
{

//- Model constants (published values as defaults)
struct Coeffs
{
    scalar cr1 = 1.0;       // Smirnov & Menter (2009) Sec. 2
    scalar cr2 = 2.0;       // Smirnov & Menter (2009) Sec. 2 (12 in SARC)
    scalar cr3 = 1.0;       // Smirnov & Menter (2009) Sec. 2
    scalar fr1Max = 1.25;   // Smirnov & Menter (2009) Eq. (4)
    scalar Cscale = 1.0;    // ANSYS scaling; 1 = paper
};

//- Fixed numbers of the formulation
//  lower limit of Eq. (4) and the 0.09 of Eq. (11)
static constexpr scalar fr1Min = 0.0;
static constexpr scalar Comega2 = 0.09;

//- Result of one evaluation (all diagnostics)
struct Result
{
    scalar S;           // sqrt(2 S_ij S_ij)
    scalar W;           // sqrt(2 W_ij W_ij), absolute rotation
    scalar D;           // sqrt(max(S^2, 0.09 omega^2))
    scalar rStar;       // S/W (diagnostic; VSMALL-guarded)
    scalar rTilde;      // Eq. (6)
    scalar fRotation;   // Eq. (1), unlimited
    scalar fr1;         // Eq. (4)
    scalar fScaled;     // max(0, 1 + Cscale (fr1 - 1)), multiplies production
};


//- Levi-Civita symbol, indices 0..2
inline scalar eps(const label i, const label j, const label k)
{
    return scalar((i - j)*(j - k)*(k - i))/2;
}


//- Absolute rotation tensor W_ij of Eq. (8) in the paper's index order
//  A_ij = du_i/dx_j = gradU_ji
inline tensor absoluteRotation(const tensor& gradU, const vector& Omega)
{
    const tensor A(gradU.T());
    tensor W(Zero);
    for (label i = 0; i < 3; ++i)
    {
        for (label j = 0; j < 3; ++j)
        {
            scalar frame = 0;
            for (label m = 0; m < 3; ++m)
            {
                frame += eps(m, j, i)*Omega[m];
            }
            W(i, j) = 0.5*(A(i, j) - A(j, i)) + frame;
        }
    }
    return W;
}


//- Frame term of Eq. (6): (eps_imn S_jn + eps_jmn S_in) Omega_m
inline symmTensor frameTerm(const symmTensor& S, const vector& Omega)
{
    const tensor Sf(S);
    tensor T(Zero);
    for (label i = 0; i < 3; ++i)
    {
        for (label j = 0; j < 3; ++j)
        {
            scalar sum = 0;
            for (label m = 0; m < 3; ++m)
            {
                for (label n = 0; n < 3; ++n)
                {
                    sum +=
                        (eps(i, m, n)*Sf(j, n) + eps(j, m, n)*Sf(i, n))
                       *Omega[m];
                }
            }
            T(i, j) = sum;
        }
    }
    return symm(T);
}


//- Evaluate the rotation/curvature function at one point
inline Result evaluate
(
    const tensor& gradU,
    const symmTensor& DSDt,
    const vector& Omega,
    const scalar omega,
    const Coeffs& c
)
{
    Result r;

    // Eq. (7): symmetric, so the gradient convention does not matter
    const symmTensor S(symm(gradU));
    // Eq. (8)
    const tensor W(absoluteRotation(gradU, Omega));

    // Eqs. (9)-(11)
    const scalar S2 = 2*(S && S);
    const scalar W2 = 2*(W && W);
    const scalar D2 = max(S2, Comega2*sqr(omega));
    r.S = std::sqrt(S2);
    r.W = std::sqrt(W2);
    r.D = std::sqrt(D2);

    // Eq. (5)
    r.rStar = r.S/max(r.W, VSMALL);

    // Eq. (6): 2 W_ik S_jk X_ij with X = DS/Dt + frame term (symmetric)
    const symmTensor X(DSDt + frameTerm(S, Omega));
    const tensor Sf(S);
    const tensor Xf(X);
    scalar num = 0;
    for (label i = 0; i < 3; ++i)
    {
        for (label j = 0; j < 3; ++j)
        {
            for (label k = 0; k < 3; ++k)
            {
                num += W(i, k)*Sf(j, k)*Xf(i, j);
            }
        }
    }
    num *= 2;
    r.rTilde = num/max(r.W*D2*r.D, VSMALL);

    // Eq. (1), with 2 r* / (1 + r*) = 2 S/(S + W)
    const scalar ratio = 2*r.S/max(r.S + r.W, VSMALL);
    r.fRotation =
        (1 + c.cr1)*ratio*(1 - c.cr3*std::atan(c.cr2*r.rTilde)) - c.cr1;

    // Eq. (4)
    r.fr1 = max(min(r.fRotation, c.fr1Max), fr1Min);

    // ANSYS scaling max(0, 1 + Cscale (fr1 - 1)), written as
    // fr1 + (Cscale - 1)(fr1 - 1) so that Cscale = 1 returns fr1 bit for bit
    r.fScaled = max(scalar(0), r.fr1 + (c.Cscale - 1)*(r.fr1 - 1));

    return r;
}


} // End namespace sstrc
} // End namespace Foam

#endif

// ************************************************************************* //
