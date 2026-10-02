/*---------------------------------------------------------------------------*\
Class
    Foam::RASModels::kOmegaEARSM

Description
    BSL-EARSM of Menter, Garbaruk and Egorov (2012): the Wallin-Johansson
    explicit algebraic Reynolds-stress relation (A1 recalibrated to 1.245,
    two-dimensional N equation, T9 omitted) closed with the k-omega BSL
    equations of Menter (1994).

    F. R. Menter, A. V. Garbaruk, Y. Egorov, Explicit algebraic Reynolds
    stress models for anisotropic wall-bounded flows, Progress in Flight
    Physics 3 (2012) 89-104, doi:10.1051/eucass/201203089.

    Stress (pp. 91-93):
        R = k (a + 2/3 I) = 2/3 k I - nut twoSymm(gradU) + nonlinearStress
        nut             = -beta1 k tau/2        (replaces the SST nut and its
                                                 a1 limiter)
        nonlinearStress = k (beta3 T3 + beta4 T4 + beta6 T6 + beta9 T9)
    Transport equations (p. 93):
        Dk/Dt     = div((nu + sigmaK k/omega) grad k) + Pk~ - betaStar k omega
        Domega/Dt = div((nu + sigmaOmega k/omega) grad omega)
                  + gamma omega/k Pk~ - beta omega^2
                  + sigmaD/omega grad k . grad omega
        Pk~       = min(-R : grad U, c1 betaStar k omega),  c1 = 10
    Diffusion uses nu_t = k/omega (p. 93), not the EARSM nut.
    Constants (p. 94): BSL set 1 (k-omega) and set 2 (k-epsilon) blended
    with F1 of Menter (1994); sigmaD = 2 (1 - F1) sigmaOmega2;
    gamma_i = beta_i/betaStar - sigmaOmega_i kappa^2/sqrt(betaStar).

    frameOmega is read (required by the FORGE harness) and ignored: the
    published model has no system-rotation term.

    Start-up ramp (FORGE harness, not part of the published model):
    rampIterations > 0 blends the stress from the SST state to the EARSM
    state over the first rampIterations calls of correct(),
        r = min(1, iteration/rampIterations),
        nut = (1 - r) nut_SST + r nut_EARSM,
        nonlinearStress = r k (beta3 T3 + ...),
    with nut_SST = a1 k/max(a1 omega, b1 F2 sqrt(2) |symm(grad U)|),
    a1 = 0.31, b1 = 1 (stock kOmegaSST, Menter 2003), so the run starts from
    the SST stress of the SST initial fields; the k and omega equations are
    the BSL-EARSM ones throughout. r = 1 after the ramp (the published
    model); rampIterations = 0 (the code default) applies the published
    model from the first iteration.

    Point-wise algebra: earsmAlgebra.H (shared with the unit tests).

SourceFiles
    kOmegaEARSM.C
\*---------------------------------------------------------------------------*/

#ifndef kOmegaEARSM_H
#define kOmegaEARSM_H

#include "RASModel.H"
#include "nonlinearEddyViscosity.H"
#include "earsmAlgebra.H"

namespace Foam
{
namespace RASModels
{

template<class BasicTurbulenceModel>
class kOmegaEARSM
:
    public nonlinearEddyViscosity<RASModel<BasicTurbulenceModel>>
{
    // Private Member Functions

        kOmegaEARSM(const kOmegaEARSM&) = delete;
        void operator=(const kOmegaEARSM&) = delete;


protected:

    // Protected data

        // Stress-strain relation (pp. 92-93)
        dimensionedScalar A1_;
        dimensionedScalar C1_;
        dimensionedScalar Ctau_;
        Switch includeT9_;

        // BSL constants (p. 94)
        dimensionedScalar sigmaK1_;
        dimensionedScalar sigmaK2_;
        dimensionedScalar sigmaOmega1_;
        dimensionedScalar sigmaOmega2_;
        dimensionedScalar beta1_;
        dimensionedScalar beta2_;
        dimensionedScalar betaStar_;
        dimensionedScalar kappa_;
        dimensionedScalar c1_;

        //- Start-up ramp length in iterations (0: none); not a model constant
        dimensionedScalar rampIterations_;

        //- Read and ignored (the published model has no rotation term)
        vector frameOmega_;

        //- Wall distance
        const volScalarField& y_;

        // Fields
        volScalarField k_;
        volScalarField omega_;

        //- Calls of correct() since construction (ramp, diagnostic line)
        label iter_;


    // Protected Member Functions

        earsm::Coefficients algebraCoeffs() const;

        tmp<volScalarField> F1(const volScalarField& CDkOmega) const;

        //- Start-up ramp factor r = min(1, iteration/rampIterations)
        scalar rampFactor() const;

        //- SST eddy viscosity of the start-up ramp (stock kOmegaSST form)
        tmp<volScalarField> nutSST(const volTensorField& gradU) const;

        tmp<volScalarField> blend
        (
            const volScalarField& F1,
            const dimensionedScalar& psi1,
            const dimensionedScalar& psi2
        ) const
        {
            return F1*(psi1 - psi2) + psi2;
        }

        tmp<volScalarField::Internal> blend
        (
            const volScalarField::Internal& F1,
            const dimensionedScalar& psi1,
            const dimensionedScalar& psi2
        ) const
        {
            return F1*(psi1 - psi2) + psi2;
        }

        //- gamma_i = beta_i/betaStar - sigmaOmega_i kappa^2/sqrt(betaStar)
        dimensionedScalar gamma1() const;
        dimensionedScalar gamma2() const;

        //- Diffusivities with nu_t = k/omega (p. 93)
        tmp<volScalarField> DkEff(const volScalarField& F1) const;
        tmp<volScalarField> DomegaEff(const volScalarField& F1) const;

        //- nut and nonlinearStress from k, omega, grad U
        virtual void correctNut();
        virtual void correctNonlinearStress(const volTensorField& gradU);


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("kOmegaEARSM");


    // Constructors

        kOmegaEARSM
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
    virtual ~kOmegaEARSM() = default;


    // Member Functions

        virtual bool read();

        virtual tmp<volScalarField> k() const
        {
            return k_;
        }

        virtual tmp<volScalarField> epsilon() const;

        virtual tmp<volScalarField> omega() const
        {
            return omega_;
        }

        //- Solve the omega and k equations, then update nut and the stress
        virtual void correct();
};

} // End namespace RASModels
} // End namespace Foam

#ifdef NoRepository
    #include "kOmegaEARSM.C"
#endif

#endif
