/*---------------------------------------------------------------------------*\
    autoTurbProduction.H

    One-point algebra of the production correction selected by AutoTurb
    (Y. Zhang, K. Zheng, F. Liu, Q. Zhang, Z. Wang, arXiv:2410.10657v1,
    Phys. Fluids 37, 015211 (2025)), "Model-LLMs", Eq. (11), p. 15:

        b^Delta_ij = 0,
        R = 2 k (dU_i/dx_j) [ (sin(lambda1) + 0.5) T1_ij + T2_ij ]

    with (Sec. 2.2, Eqs. (6)-(9), pp. 6-7) the time scale 1/omega,
        S~ = S/omega,  Omega~ = Omega/omega,
        S_ij = 1/2 (dU_i/dx_j + dU_j/dx_i)   (trace-less, incompressible),
        Omega_ij = 1/2 (dU_i/dx_j - dU_j/dx_i),
        T1 = S~,  T2 = S~ Omega~ - Omega~ S~,  lambda1 = S~_mn S~_nm.

    Written with the published constants as coefficients,
        alpha1(lambda1) = alpha1Sin*sin(lambda1) + alpha1Const   (1, 0.5)
        alpha2          = alpha2                                  (1)
    so that alpha1Sin = alpha1Const = alpha2 = 0 gives R = 0 exactly.

    OpenFOAM stores grad(U)_ij = dU_j/dx_i; the paper's dU_i/dx_j is its
    transpose. T2 is symmetric and T2:S = tr(S~ Omega~ S - Omega~ S~ S) = 0
    identically, so the T2 term contributes only round-off; it is evaluated
    as published, not dropped.

    Shared by the RAS model kOmegaSSTAutoTurb and its unit test, so the test
    exercises the code the solver runs.
\*---------------------------------------------------------------------------*/

#ifndef autoTurbProduction_H
#define autoTurbProduction_H

#include "tensor.H"
#include "symmTensor.H"
#include "scalar.H"

namespace Foam
{
namespace autoTurb
{

//- The three published constants of Eq. (11)
struct Coeffs
{
    scalar alpha1Sin;
    scalar alpha1Const;
    scalar alpha2;

    bool zero() const
    {
        return alpha1Sin == 0 && alpha1Const == 0 && alpha2 == 0;
    }
};


//- One-point result; R = k*RbyK
struct Point
{
    scalar lambda1;   // tr(S~^2)
    scalar alpha1;    // alpha1Sin*sin(lambda1) + alpha1Const
    scalar t1Work;    // T1_ij dU_i/dx_j          [1/s]
    scalar t2Work;    // T2_ij dU_i/dx_j          [1/s], zero up to round-off
    scalar RbyK;      // 2 (alpha1 t1Work + alpha2 t2Work)   [1/s]
};


//- gradU in OpenFOAM order (gradU_ij = dU_j/dx_i); omegaSafe > 0
inline Point evaluate
(
    const tensor& gradU,
    const scalar omegaSafe,
    const Coeffs& c
)
{
    // paper order: L_ij = dU_i/dx_j
    const tensor L(gradU.T());
    const symmTensor S(dev(symm(L)));
    const tensor Om(skew(L));
    const scalar tau = 1.0/omegaSafe;
    const symmTensor Sh(tau*S);
    const tensor Wh(tau*Om);

    Point p;
    p.lambda1 = Sh && Sh;
    p.alpha1 = c.alpha1Sin*Foam::sin(p.lambda1) + c.alpha1Const;

    const tensor T1(Sh);
    const tensor T2((Sh & Wh) - (Wh & Sh));
    p.t1Work = T1 && L;
    p.t2Work = T2 && L;
    p.RbyK = 2.0*(p.alpha1*p.t1Work + c.alpha2*p.t2Work);
    return p;
}

} // End namespace autoTurb
} // End namespace Foam

#endif

// ************************************************************************* //
