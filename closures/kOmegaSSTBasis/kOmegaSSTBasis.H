/*---------------------------------------------------------------------------*\
Class
    Foam::RASModels::kOmegaSSTBasis

Description
    k-omega SST (Menter 2003, stock coefficients untouched) plus two
    candidate correction channels expressed in the integrity basis of
    Shat = S/omega, What = W/omega:

      1. constitutive:  b^Delta = sum_n g_n(I1..I5,Ret,F1,PoE) T^(n)
         injected as nonlinearStress = 2 k b^Delta (momentum, R(), Pk)
      2. production:    R = 2 k (sum_m h_m(...) T^(m)) : grad(U)
         added to the k equation and gamma(F1)*R/nut to the omega equation

    The scalar functions g_n, h_m are expression strings in the frozen
    basisExpr grammar, read from kOmegaSSTBasisCoeffs and parsed ONCE at
    construction — candidates are dictionaries, never recompiled code.
    Free constants c0..c7 must already be substituted with literals by the
    driver; a remaining c-token is a fatal startup error.

    mode expressions:  evaluate the expressions each iteration (discovery)
    mode frozenFields: read bDeltaField / RField from the start time and
                       hold them fixed (Phase-2 propagation test)

    With both channels empty the assembled equations are term-for-term
    identical to stock kOmegaSST (verified to 1e-6 by tedp verify-recovery).

    A third channel modulating the a1 limiter is declared (a1Expr) but
    switched off; a non-empty a1Expr is a fatal error in this phase.

SourceFiles
    kOmegaSSTBasis.C
\*---------------------------------------------------------------------------*/

#ifndef kOmegaSSTBasis_H
#define kOmegaSSTBasis_H

#include "kOmegaSSTBase.H"
#include "RASModel.H"
#include "nonlinearEddyViscosity.H"

#include "basisExpr/exprParser.H"

#include <memory>
#include <vector>

namespace Foam
{
namespace RASModels
{

/*---------------------------------------------------------------------------*\
                       Class kOmegaSSTBasis Declaration
\*---------------------------------------------------------------------------*/

template<class BasicTurbulenceModel>
class kOmegaSSTBasis
:
    public kOmegaSSTBase
    <
        nonlinearEddyViscosity<RASModel<BasicTurbulenceModel>>
    >
{
    // Private Member Functions

        //- No copy construct / assignment
        kOmegaSSTBasis(const kOmegaSSTBasis&) = delete;
        void operator=(const kOmegaSSTBasis&) = delete;


protected:

    typedef kOmegaSSTBase
    <
        nonlinearEddyViscosity<RASModel<BasicTurbulenceModel>>
    > BaseType;

    // Protected data

        //- One correction channel: parallel lists of basis-tensor index
        //  (1..10) and parsed expression ASTs
        struct Channel
        {
            List<label> tensorIndex;
            std::vector<std::unique_ptr<tedp::expr::ExprNode>> g;

            bool empty() const noexcept { return tensorIndex.empty(); }
        };

        enum class Mode { expressions, frozenFields };

        Mode mode_;
        Channel bDelta_;
        Channel rSource_;

        //- Clamp on every evaluated g_n / h_m
        scalar gMax_;

        //- Frame angular velocity added to the rotation tensor (absolute
        //  vorticity in a rotating frame); zero for inertial-frame cases
        vector frameOmega_;

        //- frameOmega was declared explicitly in the coefficient dictionary.
        //  FORGE V3: a configured rotating-frame source (fvOptions
        //  tabulatedAccelerationSource) without an explicit, matching
        //  frameOmega is a fatal configuration error, never a silent zero.
        bool frameOmegaDeclared_;
        bool frameConfigurationChecked_;

        //- Write the FORGE V3 state features (v3Gp ... v3Rw) at write times
        //  and the dimensionless feature inputs, for cross-checks.
        Switch writeV3Features_;

        //- |R| <= rMaxFactor * betaStar * k * omega
        scalar rMaxFactor_;

        //- Deferred-correction relaxation on the correction fields
        scalar bDeltaRelax_;

        //- Linear ramp-in of both channels over this many iterations
        label rampIterations_;

        //- Fold -(nonlinearStress && gradU) into the limited production Pk
        Switch nonlinearProduction_;

        //- Per-cell eigenvalue clip of b toward the Lumley triangle
        Switch realizabilityClip_;

        //- Exchange processor/cyclic stress data before the momentum flux.
        Switch coupledStressBoundary_;

        //- Final Buchanan et al. model: never differentiate the binary gate.
        Switch publishedRITA_;
        Switch exactOmegaSource_;
        volScalarField ritaSigma_;
        volSymmTensorField ritaUngatedStress_;

        //- Iteration counter for the ramp
        label iter_;

        //- Fraction of cells clipped by the realizability guard (reported)
        scalar clipFraction_;

        //- Fractions of g evaluations clamped at gMax (bDelta and R
        //  channels) and of cells where |R| hit the rMaxFactor bound;
        //  global over all processors, printed every 100 iterations and
        //  at write times as "kOmegaSSTBasis clamps: ..." (reported only)
        scalar gClampFractionB_;
        scalar gClampFractionR_;
        scalar rBoundFraction_;

        // Cached per-iteration correction fields
        volScalarField::Internal Rfield_;   // R source           [m2/s3]
        volScalarField::Internal Gnl_;      // -(nlStress:gradU)  [m2/s3]
        volScalarField::Internal gammaF1_;  // gamma blended with cached F1

        //- Frozen-field inputs (mode frozenFields only)
        autoPtr<volSymmTensorField> bDeltaFieldPtr_;
        autoPtr<volScalarField> RFieldPtr_;


    // Protected Member Functions

        //- Parse one channel sub-dictionary ("bDelta" / "rSource")
        void parseChannel(const dictionary& coeffs, const word& name, Channel&);

        //- Evaluate channels, clip, ramp, relax; refresh
        //  nonlinearStress_, Rfield_, Gnl_, gammaF1_
        void updateCorrections(const volTensorField& gradU);

        //- FORGE V3 physical-state features (tedp/features.py is the twin):
        //  Gp Gk Apk Psn Ksn Rf Rw from the current iterate's p, k, omega,
        //  U and the declared frame rotation. Fatal on a non-kinematic
        //  pressure or a missing pressure field.
        void computeV3Features
        (
            const volScalarField& omegaSafe,
            const volSymmTensorField& Shat,
            const volVectorField& gradK,
            PtrList<volScalarField>& out
        ) const;

        //- Fail explicitly when the solver's rotating-frame configuration
        //  (fvOptions) and frameOmega disagree or frameOmega is missing.
        void checkFrameConfiguration();

        //- Production limited with the nonlinear contribution folded in
        virtual tmp<volScalarField::Internal> Pk
        (
            const volScalarField::Internal& G
        ) const;

        //- + R in the k equation
        virtual tmp<fvScalarMatrix> kSource() const;

        //- + gamma(F1) (Gnl + R)/nut in the omega equation
        virtual tmp<fvScalarMatrix> omegaSource() const;

        //- Stock SST a1 limiter (untouched)
        virtual void correctNut(const volScalarField& S2);
        virtual void correctNut();

        //- Required by nonlinearEddyViscosity; work happens in
        //  updateCorrections (called from correct() with the same gradU)
        virtual void correctNonlinearStress(const volTensorField& gradU);


public:

    virtual tmp<fvVectorMatrix> divDevRhoReff(volVectorField& U) const;
    virtual tmp<fvVectorMatrix> divDevRhoReff
    (
        const volScalarField& rho,
        volVectorField& U
    ) const;

    typedef typename BasicTurbulenceModel::alphaField alphaField;
    typedef typename BasicTurbulenceModel::rhoField rhoField;
    typedef typename BasicTurbulenceModel::transportModel transportModel;


    //- Runtime type information
    TypeName("kOmegaSSTBasis");


    // Constructors

        kOmegaSSTBasis
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
    virtual ~kOmegaSSTBasis() = default;


    // Member Functions

        //- Re-read the scalar controls (expressions are fixed at start-up)
        virtual bool read();

        //- Update corrections, then solve the stock SST equations
        virtual void correct();
};

} // End namespace RASModels
} // End namespace Foam

#ifdef NoRepository
    #include "kOmegaSSTBasis.C"
#endif

#endif
