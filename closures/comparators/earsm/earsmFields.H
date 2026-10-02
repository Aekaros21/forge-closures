/*---------------------------------------------------------------------------*\
    Field assembly of the BSL-EARSM eddy viscosity and explicit stress
    (shared by kOmegaEARSM and the unit test src/testEARSM).

      nut             = -beta1 k tau/2
      nonlinearStress = k (beta2 T2 + beta3 T3 + beta4 T4 + beta6 T6
                           + beta9 T9)          (beta2 = 0, beta9 = 0)

    Cells: from the cell values. Physical (non-constraint, non-coupled)
    patches: from the boundary values of grad U, k, omega and nu; the
    Reynolds stress vanishes on walls. Constraint patches (symmetry, wedge,
    empty) evaluate the stress from the cell values; coupled patches
    exchange the neighbour cell values (evaluateCoupled), so a decomposed
    case reproduces the serial face fluxes. nut is assigned (not forced) on
    physical patches, so fixed-value and wall-function patches keep their
    own values; nut boundary conditions (wall functions, coupled) are left
    to nut.correctBoundaryConditions() of the caller.
\*---------------------------------------------------------------------------*/

#ifndef earsmFields_H
#define earsmFields_H

#include "earsmAlgebra.H"
#include "volFields.H"
#include "wallFvPatch.H"
#include "coupledFvPatch.H"
#include "polyPatch.H"

namespace Foam
{
namespace earsm
{

struct Stats
{
    scalar Nmin = GREAT;
    scalar Nmax = -GREAT;
    scalar Cmin = GREAT;
    scalar Cmax = -GREAT;
    label nNegP2 = 0;
    label nKolmogorov = 0;
};


inline void assemble
(
    const volTensorField& gradU,
    const volScalarField& k,
    const volScalarField& omega,
    const volScalarField& nu,
    const Coefficients& c,
    const scalar kMin,
    const scalar omegaMin,
    volScalarField& nut,
    volSymmTensorField& ns,
    Stats& stats
)
{
    // One point: tau from the bounded k, omega; nut and the stress scale
    // with the actual k (zero where k is zero)
    auto point = [&]
    (
        const tensor& g, const scalar kv, const scalar wv, const scalar nv,
        scalar& nutOut, symmTensor& nsOut, const bool count
    )
    {
        const scalar kS = max(kv, kMin);
        const scalar wS = max(wv, omegaMin);
        const State st(evaluate(g, kS, wS, nv, c));
        const scalar kPos = max(kv, scalar(0));
        nutOut = st.CmuEff*kPos*st.tau;
        nsOut = kPos*st.aNL;
        if (count)
        {
            stats.Nmin = min(stats.Nmin, st.N);
            stats.Nmax = max(stats.Nmax, st.N);
            stats.Cmin = min(stats.Cmin, st.CmuEff);
            stats.Cmax = max(stats.Cmax, st.CmuEff);
            if (st.branch == Branch::P2negative) ++stats.nNegP2;
            if (st.tau*c.Cmu*wS > 1.0 + 1e-12) ++stats.nKolmogorov;
        }
    };

    // Cells
    {
        scalarField& nutI = nut.primitiveFieldRef();
        symmTensorField& nsI = ns.primitiveFieldRef();
        const tensorField& gI = gradU.primitiveField();
        const scalarField& kI = k.primitiveField();
        const scalarField& wI = omega.primitiveField();
        const scalarField& nuI = nu.primitiveField();
        forAll(nutI, celli)
        {
            point(gI[celli], kI[celli], wI[celli], nuI[celli],
                  nutI[celli], nsI[celli], true);
        }
    }

    // Patches
    const fvMesh& mesh = k.mesh();
    const polyBoundaryMesh& pbm = mesh.boundaryMesh();
    auto& nutB = nut.boundaryFieldRef();
    auto& nsB = ns.boundaryFieldRef();
    forAll(pbm, patchi)
    {
        const polyPatch& pp = pbm[patchi];
        if (pp.coupled())
        {
            continue;
        }
        if (polyPatch::constraintType(pp.type()))
        {
            nsB[patchi].evaluate();
            continue;
        }

        const fvPatch& fp = mesh.boundary()[patchi];
        scalarField nutP(fp.size(), Zero);
        symmTensorField nsP(fp.size(), Zero);
        if (!isA<wallFvPatch>(fp))
        {
            const tensorField& gP = gradU.boundaryField()[patchi];
            const scalarField& kP = k.boundaryField()[patchi];
            const scalarField& wP = omega.boundaryField()[patchi];
            const scalarField& nuP = nu.boundaryField()[patchi];
            forAll(nutP, facei)
            {
                point(gP[facei], kP[facei], wP[facei], nuP[facei],
                      nutP[facei], nsP[facei], false);
            }
        }
        // plain assignment for nut: fixed-value and wall-function patches
        // keep their own values (as nut_ = expression in stock models)
        nutB[patchi] = nutP;
        nsB[patchi] == nsP;
    }
    nsB.template evaluateCoupled<coupledFvPatch>();
}

} // End namespace earsm
} // End namespace Foam

#endif
