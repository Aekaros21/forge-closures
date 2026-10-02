/*---------------------------------------------------------------------------*\
File
    sstrcLagrangian.H

Description
    Steady Lagrangian derivative of the strain-rate tensor for the SST-RC
    model, DS_ij/Dt = U_k dS_ij/dx_k, in the face-flux form of Smirnov &
    Menter (2009), Sec. 3, Eqs. (13)-(15):

        DS_ij/Dt = [sum_f phi_f S_ij,f - S_ij,P sum_f phi_f] / V_P

    with phi_f the face volume flux and S_ij,f the linear interpolate of S
    (no fvSchemes lookup).  The second term removes the discrete continuity
    defect, so a uniform S gives DS/Dt = 0 exactly and the result is the
    convective (non-conservative) derivative also before convergence.  The
    local time derivative is omitted: steady RANS only.

    Used by kOmegaSSTRC.C and by the unit executable sstrc_dsdt.
\*---------------------------------------------------------------------------*/

#ifndef sstrcLagrangian_H
#define sstrcLagrangian_H

#include "volFields.H"
#include "surfaceFields.H"
#include "fvcSurfaceIntegrate.H"
#include "linear.H"

namespace Foam
{
namespace sstrc
{

inline tmp<volSymmTensorField> steadyLagrangianDerivative
(
    const surfaceScalarField& phi,
    const volSymmTensorField& S
)
{
    return
        fvc::surfaceIntegrate(phi*linearInterpolate(S))
      - fvc::surfaceIntegrate(phi)*S;
}

} // End namespace sstrc
} // End namespace Foam

#endif

// ************************************************************************* //
