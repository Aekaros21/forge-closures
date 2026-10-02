/*---------------------------------------------------------------------------*\
    kOmegaSSTRC: SST with the Spalart-Shur rotation/curvature correction
    (Smirnov & Menter 2009).  See kOmegaSSTRC.H and sstrcKernel.H.
\*---------------------------------------------------------------------------*/

#include "kOmegaSSTRC.H"
#include "fvc.H"
#include "sstrcLagrangian.H"
#include "IFstream.H"
#include "Tuple2.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void kOmegaSSTRC<BasicTurbulenceModel>::readCoeffs()
{
    cr1_.readIfPresent(this->coeffDict());
    cr2_.readIfPresent(this->coeffDict());
    cr3_.readIfPresent(this->coeffDict());
    fr1Max_.readIfPresent(this->coeffDict());
    Cscale_.readIfPresent(this->coeffDict());
    rampIterations_.readIfPresent(this->coeffDict());
    rotationCurvatureCorrection_.readIfPresent
    (
        "rotationCurvatureCorrection",
        this->coeffDict()
    );
    frameOmega_ = this->coeffDict().template get<vector>("frameOmega");
    frameChecked_ = false;
}


template<class BasicTurbulenceModel>
void kOmegaSSTRC<BasicTurbulenceModel>::checkFrameConfiguration()
{
    if (frameChecked_)
    {
        return;
    }
    frameChecked_ = true;

    bool frameSource = false;
    bool tableRead = false;
    vector tableOmega(Zero);

    const objectRegistry& db = this->mesh_.thisDb();
    if (db.template foundObject<IOdictionary>("fvOptions"))
    {
        const dictionary& options =
            db.template lookupObject<IOdictionary>("fvOptions");

        for (const entry& e : options)
        {
            if (!e.isDict())
            {
                continue;
            }
            const dictionary& d = e.dict();
            const word sourceType(d.getOrDefault<word>("type", word::null));

            if (sourceType == "tabulatedAccelerationSource")
            {
                if (frameSource)
                {
                    FatalErrorInFunction
                        << "kOmegaSSTRC: more than one rotating-frame "
                        << "source in fvOptions" << exit(FatalError);
                }
                frameSource = true;

                fileName table(d.get<fileName>("timeDataFileName"));
                table.expand();
                if (!table.isAbsolute())
                {
                    // The source resolves the name against the case
                    // directory (the solver's working directory)
                    table = this->runTime_.globalPath()/table;
                }
                IFstream is(table);
                if (!is.good())
                {
                    FatalErrorInFunction
                        << "kOmegaSSTRC: cannot read the rotating-frame table "
                        << table << exit(FatalError);
                }
                List<Tuple2<scalar, Vector<vector>>> rows(is);
                if (rows.empty())
                {
                    FatalErrorInFunction
                        << "kOmegaSSTRC: empty rotating-frame table " << table
                        << exit(FatalError);
                }
                tableRead = true;
                // (linear acceleration, angular velocity, angular acceleration)
                tableOmega = rows.first().second().y();
                for (const auto& row : rows)
                {
                    if
                    (
                        mag(row.second().y() - tableOmega)
                      > 1.0e-9*max(mag(tableOmega), scalar(1))
                     || mag(row.second().z()) > 0
                    )
                    {
                        FatalErrorInFunction
                            << "kOmegaSSTRC: the frame angular velocity in "
                            << table << " is not constant; a time-varying "
                            << "frame is not supported" << exit(FatalError);
                    }
                }
            }
            else if
            (
                sourceType.find("SRF") != std::string::npos
             || sourceType.find("Coriolis") != std::string::npos
             || sourceType.find("coriolis") != std::string::npos
             || sourceType.find("otating") != std::string::npos
            )
            {
                FatalErrorInFunction
                    << "kOmegaSSTRC: rotating-frame source type " << sourceType
                    << " is not supported; only tabulatedAccelerationSource "
                    << "is checked against frameOmega" << exit(FatalError);
            }
        }
    }

    if
    (
        tableRead
     && mag(tableOmega - frameOmega_)
      > 1.0e-9*max(mag(tableOmega), scalar(1))
    )
    {
        FatalErrorInFunction
            << "kOmegaSSTRC: frameOmega " << frameOmega_
            << " differs from the rotating-frame source angular velocity "
            << tableOmega << exit(FatalError);
    }
    if (!frameSource && mag(frameOmega_) > 0)
    {
        FatalErrorInFunction
            << "kOmegaSSTRC: frameOmega " << frameOmega_ << " is declared "
            << "but no rotating-frame momentum source is configured in fvOptions"
            << exit(FatalError);
    }

    Info<< "kOmegaSSTRC: frame configuration checked; frameOmega "
        << frameOmega_
        << (frameSource ? " (rotating-frame source present)" : " (inertial)")
        << "; rotationCurvatureCorrection " << rotationCurvatureCorrection_
        << "; start-up ramp " << rampIterations_.value() << " iterations"
        << nl;
}


template<class BasicTurbulenceModel>
void kOmegaSSTRC<BasicTurbulenceModel>::updateFr1()
{
    const volVectorField& U = this->U_;
    const volScalarField& omega = this->omega_;
    const surfaceScalarField& phi = this->phi_;

    tmp<volTensorField> tgradU = fvc::grad(U);
    const volTensorField& gradU = tgradU();

    // Eq. (7)
    const volSymmTensorField S(symm(gradU));

    // Steady Lagrangian derivative U_k dS_ij/dx_k, face-flux form of
    // Smirnov & Menter Eq. (15) minus S div(phi) (sstrcLagrangian.H)
    const volSymmTensorField DSDt(sstrc::steadyLagrangianDerivative(phi, S));

    sstrc::Coeffs c;
    c.cr1 = cr1_.value();
    c.cr2 = cr2_.value();
    c.cr3 = cr3_.value();
    c.fr1Max = fr1Max_.value();
    c.Cscale = Cscale_.value();

    // Start-up ramp of the departure from one (as src/comparators/autoturb)
    const scalar nRamp = rampIterations_.value();
    const scalar ramp =
        nRamp > 0 ? min(scalar(1), scalar(iter_)/nRamp) : scalar(1);

    scalarField& f = fr1_.primitiveFieldRef();
    const tensorField& gU = gradU.primitiveField();
    const symmTensorField& dS = DSDt.primitiveField();
    const scalarField& om = omega.primitiveField();

    label nLow = 0;
    label nHigh = 0;
    forAll(f, celli)
    {
        const sstrc::Result r =
            sstrc::evaluate(gU[celli], dS[celli], frameOmega_, om[celli], c);
        f[celli] = ramp < 1 ? 1 + ramp*(r.fScaled - 1) : r.fScaled;
        if (r.fRotation <= sstrc::fr1Min)
        {
            ++nLow;
        }
        else if (r.fRotation >= c.fr1Max)
        {
            ++nHigh;
        }
    }
    fr1_.correctBoundaryConditions();

    if (iter_ % 100 == 1)
    {
        reduce(nLow, sumOp<label>());
        reduce(nHigh, sumOp<label>());
        Info<< "kOmegaSSTRC: ramp " << ramp
            << "; fr1 min " << gMin(f) << " mean " << gAverage(f)
            << " max " << gMax(f) << "; cells at lower/upper limit "
            << nLow << "/" << nHigh << " of " << returnReduce(f.size(), sumOp<label>())
            << nl;
    }
}


template<class BasicTurbulenceModel>
void kOmegaSSTRC<BasicTurbulenceModel>::correctNut(const volScalarField& S2)
{
    // As stock RASModels::kOmegaSST
    BaseType::correctNut(S2);
    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void kOmegaSSTRC<BasicTurbulenceModel>::correctNut()
{
    // As stock RASModels::kOmegaSST
    correctNut(2*magSqr(symm(fvc::grad(this->U_))));
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> kOmegaSSTRC<BasicTurbulenceModel>::Pk
(
    const volScalarField::Internal& G
) const
{
    if (!rotationCurvatureCorrection_)
    {
        return BaseType::Pk(G);
    }
    return BaseType::Pk(G)*fr1_();
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> kOmegaSSTRC<BasicTurbulenceModel>::GbyNu
(
    const volScalarField::Internal& GbyNu0,
    const volScalarField::Internal& F2,
    const volScalarField::Internal& S2
) const
{
    if (!rotationCurvatureCorrection_)
    {
        return BaseType::GbyNu(GbyNu0, F2, S2);
    }
    return BaseType::GbyNu(GbyNu0, F2, S2)*fr1_();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
kOmegaSSTRC<BasicTurbulenceModel>::kOmegaSSTRC
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
    cr1_(dimensioned<scalar>::getOrAddToDict("cr1", this->coeffDict_, 1.0)),
    cr2_(dimensioned<scalar>::getOrAddToDict("cr2", this->coeffDict_, 2.0)),
    cr3_(dimensioned<scalar>::getOrAddToDict("cr3", this->coeffDict_, 1.0)),
    fr1Max_
    (
        dimensioned<scalar>::getOrAddToDict("fr1Max", this->coeffDict_, 1.25)
    ),
    Cscale_
    (
        dimensioned<scalar>::getOrAddToDict("Cscale", this->coeffDict_, 1.0)
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
    rotationCurvatureCorrection_
    (
        Switch::getOrAddToDict
        (
            "rotationCurvatureCorrection",
            this->coeffDict_,
            true
        )
    ),
    frameOmega_(this->coeffDict_.template get<vector>("frameOmega")),
    frameChecked_(false),
    iter_(0),
    fr1_
    (
        IOobject
        (
            "fr1",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_,
        dimensionedScalar(dimless, 1.0),
        fvPatchFieldBase::zeroGradientType()
    )
{
    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool kOmegaSSTRC<BasicTurbulenceModel>::read()
{
    if (BaseType::read())
    {
        readCoeffs();
        return true;
    }

    return false;
}


template<class BasicTurbulenceModel>
void kOmegaSSTRC<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    ++iter_;

    checkFrameConfiguration();

    if (rotationCurvatureCorrection_)
    {
        updateFr1();
    }

    BaseType::correct();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
