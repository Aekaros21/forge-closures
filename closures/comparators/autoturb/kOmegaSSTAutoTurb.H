/*---------------------------------------------------------------------------*\
Class
    Foam::RASModels::kOmegaSSTAutoTurb

Description
    Stock OpenFOAM v2312 k-omega SST with the production correction selected
    by AutoTurb ("Model-LLMs"):

        Y. Zhang, K. Zheng, F. Liu, Q. Zhang, Z. Wang (2025),
        "AutoTurb: Using large language models for automatic algebraic
        turbulence model discovery", Phys. Fluids 37, 015211;
        preprint arXiv:2410.10657v1, Eq. (11), p. 15.

    The selected model keeps the linear stress (b^Delta = 0) and adds

        R = 2 k b^R_ij dU_i/dx_j,
        b^R = (alpha1Sin sin(lambda1) + alpha1Const) T1 + alpha2 T2,

    with T1 = S/omega, T2 = (S Omega - Omega S)/omega^2 and
    lambda1 = tr(S^2)/omega^2 (preprint Eqs. (6)-(9), pp. 6-7). As in the
    SpaRTA formulation that AutoTurb follows (Schmelzer et al. 2020,
    Eqs. (4)-(5); Saidi et al. 2022, Eqs. (10)-(11)), R is added to the
    production of both transport equations:

        k:      ... = P_k + R - beta* k omega + ...
        omega:  ... = gamma (P_k + R)/nu_t - beta omega^2 + ...

    i.e. +R in the k equation, after the SST production limiter, and
    +gamma(F1) R/nu_t in the omega equation, beside its limited production.
    No bound or under-relaxation is applied to R: the publication states
    none. R is evaluated once per iteration from the current k, omega and
    grad(U), before the SST equations are assembled. An optional start-up
    ramp (rampIterations > 0) multiplies R by min(1, iteration/
    rampIterations), as the harness does for the paper's corrections; it
    ends after rampIterations iterations and leaves the converged equations
    unchanged. rampIterations = 0 (the default) applies R in full from the
    first iteration.

    Published coefficients (code defaults, every one read from
    kOmegaSSTAutoTurbCoeffs):
        alpha1Sin    1.0
        alpha1Const  0.5
        alpha2       1.0
    and the start-up control
        rampIterations 0   (iterations; 0 = no ramp)
    With alpha1Sin = alpha1Const = alpha2 = 0 the model adds nothing and the
    assembled equations are those of stock kOmegaSST, term for term.

    frameOmega is read (the harness always writes it) and ignored: the
    publication has no frame rotation, and the only rotation-dependent term,
    T2, does no work on the mean flow (T2:S = 0 for any rotation tensor).

SourceFiles
    kOmegaSSTAutoTurb.C
    autoTurbProduction.H
\*---------------------------------------------------------------------------*/

#ifndef kOmegaSSTAutoTurb_H
#define kOmegaSSTAutoTurb_H

#include "kOmegaSSTBase.H"
#include "RASModel.H"
#include "eddyViscosity.H"
#include "autoTurbProduction.H"

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                      Class kOmegaSSTAutoTurb Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class kOmegaSSTAutoTurb
:
    public kOmegaSSTBase<eddyViscosity<RASModel<BasicTurbulenceModel>>>
{
    // Private Member Functions

        //- No copy construct
        kOmegaSSTAutoTurb(const kOmegaSSTAutoTurb&) = delete;

        //- No copy assignment
        void operator=(const kOmegaSSTAutoTurb&) = delete;


protected:

    typedef kOmegaSSTBase
    <
        eddyViscosity<RASModel<BasicTurbulenceModel>>
    > BaseType;

    // Protected data

        //- Amplitude of sin(lambda1) in the T1 coefficient (published 1)
        dimensionedScalar alpha1Sin_;

        //- Constant of the T1 coefficient (published 0.5)
        dimensionedScalar alpha1Const_;

        //- Coefficient of T2 (published 1)
        dimensionedScalar alpha2_;

        //- Start-up ramp length in iterations (0: none); not a model constant
        dimensionedScalar rampIterations_;

        //- Frame rotation written by the harness; read and not used
        vector frameOmega_;

        //- Calls of correct() since construction, for the ramp
        label iter_;

        //- Production correction R of the current iteration [m2/s3]
        volScalarField::Internal R_;

        //- gamma(F1) of the current iteration, for the omega source
        volScalarField::Internal gammaF1_;


    // Protected Member Functions

        //- The coefficients as plain scalars
        autoTurb::Coeffs coeffs() const;

        //- Evaluate R_ and gammaF1_ from the current state
        void updateR();

        //- + R in the k equation
        virtual tmp<fvScalarMatrix> kSource() const;

        //- + gamma(F1) R/nut in the omega equation
        virtual tmp<fvScalarMatrix> omegaSource() const;

        //- Stock SST eddy viscosity
        virtual void correctNut(const volScalarField& S2);
        virtual void correctNut();


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("kOmegaSSTAutoTurb");


    // Constructors

        kOmegaSSTAutoTurb
        (
            const alphaField& alpha,
            const rhoField& rho,
            const volVectorField& U,
            const surfaceScalarField& alphaRhoPhi,
            const surfaceScalarField& phi,
            const transportModel& transport,
            const word& propertiesName = turbulenceModel::propertiesName,
            const word& type = typeName
        );


    //- Destructor
    virtual ~kOmegaSSTAutoTurb() = default;


    // Member Functions

        //- Re-read the SST and AutoTurb coefficients
        virtual bool read();

        //- Evaluate R, then solve the SST equations with it
        virtual void correct();
};


} // End namespace RASModels
} // End namespace Foam

#ifdef NoRepository
    #include "kOmegaSSTAutoTurb.C"
#endif

#endif

// ************************************************************************* //
