/*---------------------------------------------------------------------------*\
Class
    Foam::RASModels::kOmegaSSTRC

Description
    k-omega SST (stock OpenFOAM v2312 kOmegaSSTBase, coefficients untouched)
    with the Spalart-Shur rotation/curvature correction in the form of
    Smirnov & Menter (2009), "Sensitization of the SST turbulence model to
    rotation and curvature by applying the Spalart-Shur correction term",
    J. Turbomach. 131(4), 041010 (the SST-CC model, here SST-RC):

        k:      P_k            ->  P_k fr1                        Eq. (2)
        omega:  alpha P_k/nu_t ->  alpha P_k/nu_t fr1             Eq. (3)

    with fr1 from Eqs. (1) and (4)-(11) (see sstrcKernel.H).  In OpenFOAM
    the k production is Pk(G) = min(G, c1 betaStar k omega) and the omega
    production gamma*GbyNu(...) with GbyNu = min(G/nut, (c1/a1) betaStar
    omega max(a1 omega, b1 F2 S)); both are multiplied by fr1.

    Lagrangian derivative (steady RANS, Smirnov & Menter Sec. 3, Eqs. 12-15):
        DS_ij/Dt = U_k dS_ij/dx_k
                 = [sum_faces phi_f S_ij,f - S_ij sum_faces phi_f]/V
    i.e. fvc::surfaceIntegrate(phi*linearInterpolate(S)) - S*surfaceIntegrate(phi):
    the face-flux form of Eq. (15) with the continuity defect removed, so a
    uniform S has DS/Dt = 0 exactly.  The time derivative is omitted (steady
    RANS only, as in all cases of this study).

    System rotation: the angular velocity of the calculation frame is the
    required coefficient frameOmega (rad/s), written by the harness.  It
    enters the absolute rotation tensor (Eq. 8) and the frame term of
    Eq. (6).  At the first correct() the model checks it against the
    rotating-frame momentum source: if fvOptions contains a
    tabulatedAccelerationSource, its (constant) tabulated angular velocity
    must equal frameOmega; a non-zero frameOmega without such a source, or
    any other rotating-frame source type, is a fatal error.

    Reduction to SST:
      rotationCurvatureCorrection off: Pk and GbyNu are the base-class
      values untouched (bit-identical to kOmegaSST);
      fr1 = 1: the products Pk*1 and GbyNu*1 are exact in floating point.

    Start-up ramp (rampIterations > 0), as the harness applies to the paper's
    corrections and as src/comparators/autoturb does: the departure of the
    multiplier from one is scaled by r = min(1, iteration/rampIterations),
    f = 1 + r (fr1 - 1), over the first rampIterations calls of correct().
    For r = 1 the multiplier is fr1 itself (bit for bit), so the ramp
    changes nothing after it ends; rampIterations = 0 (the default) applies
    fr1 in full from the first iteration, as published.

    Coefficients (kOmegaSSTRCCoeffs, published values as defaults):
        cr1 1; cr2 2; cr3 1; fr1Max 1.25; Cscale 1;
        rampIterations 0 (iterations; 0 = no ramp, not a model constant);
        rotationCurvatureCorrection true; frameOmega (0 0 0) [required]

    The multiplier field is registered as "fr1" and written at write times.

SourceFiles
    kOmegaSSTRC.C
    sstrcKernel.H
\*---------------------------------------------------------------------------*/

#ifndef kOmegaSSTRC_H
#define kOmegaSSTRC_H

#include "kOmegaSSTBase.H"
#include "RASModel.H"
#include "eddyViscosity.H"
#include "sstrcKernel.H"

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                         Class kOmegaSSTRC Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class kOmegaSSTRC
:
    public kOmegaSSTBase<eddyViscosity<RASModel<BasicTurbulenceModel>>>
{
    // Private Member Functions

        //- No copy construct
        kOmegaSSTRC(const kOmegaSSTRC&) = delete;

        //- No copy assignment
        void operator=(const kOmegaSSTRC&) = delete;


protected:

    typedef kOmegaSSTBase<eddyViscosity<RASModel<BasicTurbulenceModel>>>
        BaseType;

    // Protected Data

        dimensionedScalar cr1_;
        dimensionedScalar cr2_;
        dimensionedScalar cr3_;
        dimensionedScalar fr1Max_;
        dimensionedScalar Cscale_;

        //- Start-up ramp length in iterations (0: none); not a model constant
        dimensionedScalar rampIterations_;

        Switch rotationCurvatureCorrection_;

        //- Angular velocity of the calculation frame [rad/s]
        vector frameOmega_;

        bool frameChecked_;

        //- Calls of correct() since construction, for the ramp and the log
        label iter_;

        //- Production multiplier max(0, 1 + Cscale (fr1 - 1)) [-]
        volScalarField fr1_;


    // Protected Member Functions

        //- Read the coefficients (frameOmega is required)
        void readCoeffs();

        //- Compare frameOmega with the rotating-frame momentum source
        void checkFrameConfiguration();

        //- Update fr1_ from the current U and omega
        void updateFr1();

        virtual void correctNut(const volScalarField& S2);
        virtual void correctNut();

        //- k production: base value times fr1 (Eq. 2)
        virtual tmp<volScalarField::Internal> Pk
        (
            const volScalarField::Internal& G
        ) const;

        //- omega production per gamma: base value times fr1 (Eq. 3)
        virtual tmp<volScalarField::Internal> GbyNu
        (
            const volScalarField::Internal& GbyNu0,
            const volScalarField::Internal& F2,
            const volScalarField::Internal& S2
        ) const;


public:

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("kOmegaSSTRC");


    // Constructors

        //- Construct from components
        kOmegaSSTRC
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
    virtual ~kOmegaSSTRC() = default;


    // Member Functions

        //- Re-read model coefficients if they have changed
        virtual bool read();

        //- The production multiplier
        const volScalarField& fr1() const
        {
            return fr1_;
        }

        //- Solve the turbulence equations and correct the turbulent viscosity
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "kOmegaSSTRC.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
