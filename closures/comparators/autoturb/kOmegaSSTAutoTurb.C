/*---------------------------------------------------------------------------*\
    kOmegaSSTAutoTurb: stock v2312 SST + AutoTurb production correction.
    See kOmegaSSTAutoTurb.H for the equations and sources.
\*---------------------------------------------------------------------------*/

#include "kOmegaSSTAutoTurb.H"
#include "fvcGrad.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
autoTurb::Coeffs kOmegaSSTAutoTurb<BasicTurbulenceModel>::coeffs() const
{
    return autoTurb::Coeffs
    {
        alpha1Sin_.value(),
        alpha1Const_.value(),
        alpha2_.value()
    };
}


template<class BasicTurbulenceModel>
void kOmegaSSTAutoTurb<BasicTurbulenceModel>::updateR()
{
    const autoTurb::Coeffs c(coeffs());
    if (c.zero())
    {
        R_ = dimensionedScalar(R_.dimensions(), Zero);
        return;
    }

    const volScalarField& k = this->k_;
    const volScalarField& omega = this->omega_;

    // gamma(F1) from the current k and omega, formed as kOmegaSSTBase does
    {
        const volScalarField CDkOmega
        (
            (2*this->alphaOmega2_)*(fvc::grad(k) & fvc::grad(omega))/omega
        );
        const volScalarField F1(this->F1(CDkOmega));
        gammaF1_ = this->gamma(F1());
    }

    const tmp<volTensorField> tgradU(fvc::grad(this->U_));
    const tensorField& gradU = tgradU().primitiveField();
    const scalarField& kf = k.primitiveField();
    const scalarField& omf = omega.primitiveField();
    const scalar omegaMin = this->omegaMin_.value();
    const scalar betaStar = this->betaStar_.value();
    const scalar nRamp = rampIterations_.value();
    const scalar ramp =
        nRamp > 0 ? min(scalar(1), scalar(iter_)/nRamp) : scalar(1);
    scalarField& R = R_.field();

    scalar alpha1Min = GREAT;
    scalar alpha1Max = -GREAT;
    scalar ratioMax = 0;
    scalar negativeVolume = 0;
    const scalarField& V = this->mesh_.V();

    forAll(R, celli)
    {
        const scalar omegaSafe = max(omf[celli], omegaMin);
        const autoTurb::Point p =
            autoTurb::evaluate(gradU[celli], omegaSafe, c);
        R[celli] = ramp*kf[celli]*p.RbyK;

        alpha1Min = min(alpha1Min, p.alpha1);
        alpha1Max = max(alpha1Max, p.alpha1);
        ratioMax = max(ratioMax, ramp*mag(p.RbyK)/(betaStar*omegaSafe));
        if (R[celli] < 0)
        {
            negativeVolume += V[celli];
        }
    }

    if (this->runTime_.timeIndex() % 100 == 0)
    {
        reduce(alpha1Min, minOp<scalar>());
        reduce(alpha1Max, maxOp<scalar>());
        reduce(ratioMax, maxOp<scalar>());
        reduce(negativeVolume, sumOp<scalar>());
        Info<< "kOmegaSSTAutoTurb: ramp " << ramp
            << ", alpha1 [" << alpha1Min << ", " << alpha1Max
            << "], max |R|/(betaStar k omega) " << ratioMax
            << ", R < 0 in " << 100.0*negativeVolume/gSum(V)
            << "% of the volume" << endl;
    }
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> kOmegaSSTAutoTurb<BasicTurbulenceModel>::kSource() const
{
    tmp<fvScalarMatrix> tSrc(BaseType::kSource());
    if (!coeffs().zero())
    {
        tSrc.ref() += this->alpha_()*this->rho_()*R_;
    }
    return tSrc;
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> kOmegaSSTAutoTurb<BasicTurbulenceModel>::omegaSource() const
{
    tmp<fvScalarMatrix> tSrc(BaseType::omegaSource());
    if (coeffs().zero())
    {
        return tSrc;
    }

    // gamma (P_k + R)/nu_t: the R part, divided by the eddy viscosity
    // without a floor (preprint Eq. (4) with P_k -> P_k + R)
    volScalarField::Internal RbyNut
    (
        IOobject
        (
            IOobject::scopedName(this->type(), "RbyNut"),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        ),
        this->mesh_,
        dimensionedScalar(dimless/sqr(dimTime), Zero)
    );
    const scalarField& nutf = this->nut_.primitiveField();
    forAll(RbyNut, celli)
    {
        const scalar R = R_[celli];
        if (R == 0)
        {
            continue;
        }
        if (nutf[celli] <= 0)
        {
            FatalErrorInFunction
                << "nonzero AutoTurb omega source with nonpositive nut "
                << nutf[celli] << " in cell " << celli
                << exit(FatalError);
        }
        RbyNut[celli] = R/nutf[celli];
    }
    tSrc.ref() += this->alpha_()*this->rho_()*gammaF1_*RbyNut;
    return tSrc;
}


template<class BasicTurbulenceModel>
void kOmegaSSTAutoTurb<BasicTurbulenceModel>::correctNut
(
    const volScalarField& S2
)
{
    BaseType::correctNut(S2);
    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void kOmegaSSTAutoTurb<BasicTurbulenceModel>::correctNut()
{
    correctNut(2*magSqr(symm(fvc::grad(this->U_))));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
kOmegaSSTAutoTurb<BasicTurbulenceModel>::kOmegaSSTAutoTurb
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    BaseType
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),
    alpha1Sin_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alpha1Sin",
            this->coeffDict_,
            1.0
        )
    ),
    alpha1Const_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alpha1Const",
            this->coeffDict_,
            0.5
        )
    ),
    alpha2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "alpha2",
            this->coeffDict_,
            1.0
        )
    ),
    rampIterations_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "rampIterations",
            this->coeffDict_,
            0.0
        )
    ),
    frameOmega_(this->coeffDict_.template get<vector>("frameOmega")),
    iter_(0),
    R_
    (
        IOobject
        (
            IOobject::scopedName(typeName, "R"),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        ),
        this->mesh_,
        dimensionedScalar(sqr(dimLength)/pow3(dimTime), Zero)
    ),
    gammaF1_
    (
        IOobject
        (
            IOobject::scopedName(typeName, "gammaF1"),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            IOobject::NO_REGISTER
        ),
        this->mesh_,
        dimensionedScalar(dimless, Zero)
    )
{
    if (type == typeName)
    {
        this->printCoeffs(type);
    }

    Info<< "kOmegaSSTAutoTurb: R = 2k[(" << alpha1Sin_.value()
        << " sin(lambda1) + " << alpha1Const_.value() << ") T1 + "
        << alpha2_.value() << " T2] : grad(U)^T"
        << (coeffs().zero() ? " (zero: stock SST)" : "")
        << "; start-up ramp " << rampIterations_.value() << " iterations"
        << "; frameOmega " << frameOmega_ << " not used" << nl;
}


// * * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool kOmegaSSTAutoTurb<BasicTurbulenceModel>::read()
{
    if (BaseType::read())
    {
        alpha1Sin_.readIfPresent(this->coeffDict());
        alpha1Const_.readIfPresent(this->coeffDict());
        alpha2_.readIfPresent(this->coeffDict());
        rampIterations_.readIfPresent(this->coeffDict());
        frameOmega_ = this->coeffDict().template get<vector>("frameOmega");
        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
void kOmegaSSTAutoTurb<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    ++iter_;
    updateR();

    BaseType::correct();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
