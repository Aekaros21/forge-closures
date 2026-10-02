/*---------------------------------------------------------------------------*\
    BSL-EARSM (Menter, Garbaruk and Egorov 2012). See kOmegaEARSM.H for the
    equations with page references and earsmAlgebra.H for the point-wise
    stress-strain relation.
\*---------------------------------------------------------------------------*/

#include "kOmegaEARSM.H"
#include "earsmFields.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
earsm::Coefficients kOmegaEARSM<BasicTurbulenceModel>::algebraCoeffs() const
{
    earsm::Coefficients c;
    c.A1 = A1_.value();
    c.C1 = C1_.value();
    c.Ctau = Ctau_.value();
    c.Cmu = betaStar_.value();      // p. 94: betaStar = Cmu = 0.09
    c.includeT9 = bool(includeT9_);
    return c;
}


template<class BasicTurbulenceModel>
dimensionedScalar kOmegaEARSM<BasicTurbulenceModel>::gamma1() const
{
    return beta1_/betaStar_ - sigmaOmega1_*sqr(kappa_)/sqrt(betaStar_);
}


template<class BasicTurbulenceModel>
dimensionedScalar kOmegaEARSM<BasicTurbulenceModel>::gamma2() const
{
    return beta2_/betaStar_ - sigmaOmega2_*sqr(kappa_)/sqrt(betaStar_);
}


template<class BasicTurbulenceModel>
tmp<volScalarField> kOmegaEARSM<BasicTurbulenceModel>::F1
(
    const volScalarField& CDkOmega
) const
{
    // Menter (1994) BSL blending, as in the SST baseline (kOmegaSSTBase::F1):
    // 4 sigmaOmega2 k/(CDkOmega+ y^2) = 2 k omega/(y^2 grad k . grad omega)
    tmp<volScalarField> CDkOmegaPlus = max
    (
        CDkOmega,
        dimensionedScalar(dimless/sqr(dimTime), 1.0e-10)
    );

    tmp<volScalarField> arg1 = min
    (
        min
        (
            max
            (
                (scalar(1)/betaStar_)*sqrt(k_)/(omega_*y_),
                scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_)
            ),
            (4*sigmaOmega2_)*k_/(CDkOmegaPlus*sqr(y_))
        ),
        scalar(10)
    );

    return tanh(pow4(arg1));
}


template<class BasicTurbulenceModel>
scalar kOmegaEARSM<BasicTurbulenceModel>::rampFactor() const
{
    const scalar nRamp = rampIterations_.value();
    return nRamp > 0 ? min(scalar(1), scalar(iter_)/nRamp) : scalar(1);
}


template<class BasicTurbulenceModel>
tmp<volScalarField> kOmegaEARSM<BasicTurbulenceModel>::nutSST
(
    const volTensorField& gradU
) const
{
    // stock kOmegaSST (v2312) eddy viscosity: a1 = 0.31, b1 = 1, F2 of
    // Menter (2003); used only during the start-up ramp
    const dimensionedScalar a1(dimless, 0.31);
    const dimensionedScalar b1(dimless, 1.0);
    const volScalarField arg2
    (
        min
        (
            max
            (
                (scalar(2)/betaStar_)*sqrt(k_)/(omega_*y_),
                scalar(500)*(this->mu()/this->rho_)/(sqr(y_)*omega_)
            ),
            scalar(100)
        )
    );
    const volScalarField F2(tanh(sqr(arg2)));
    const volScalarField S2(2*magSqr(symm(gradU)));
    return a1*k_/max(a1*omega_, b1*F2*sqrt(S2));
}


template<class BasicTurbulenceModel>
tmp<volScalarField> kOmegaEARSM<BasicTurbulenceModel>::DkEff
(
    const volScalarField& F1
) const
{
    return tmp<volScalarField>::New
    (
        "DkEff",
        blend(F1, sigmaK1_, sigmaK2_)*k_/omega_ + this->nu()
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> kOmegaEARSM<BasicTurbulenceModel>::DomegaEff
(
    const volScalarField& F1
) const
{
    return tmp<volScalarField>::New
    (
        "DomegaEff",
        blend(F1, sigmaOmega1_, sigmaOmega2_)*k_/omega_ + this->nu()
    );
}


template<class BasicTurbulenceModel>
void kOmegaEARSM<BasicTurbulenceModel>::correctNut()
{
    correctNonlinearStress(fvc::grad(this->U_));
}


template<class BasicTurbulenceModel>
void kOmegaEARSM<BasicTurbulenceModel>::correctNonlinearStress
(
    const volTensorField& gradU
)
{
    tmp<volScalarField> tnu = this->nu();
    earsm::Stats stats;
    earsm::assemble
    (
        gradU, k_, omega_, tnu(), algebraCoeffs(),
        this->kMin_.value(), this->omegaMin_.value(),
        this->nut_, this->nonlinearStress_, stats
    );

    // start-up ramp from the SST stress (rampIterations > 0 only)
    const scalar r = rampFactor();
    if (r < 1)
    {
        this->nut_ = (1 - r)*nutSST(gradU) + r*this->nut_;
        this->nonlinearStress_ *= r;
    }

    // wall functions (nutLowReWallFunction etc.) and coupled patches
    this->nut_.correctBoundaryConditions();

    if (iter_ % 100 == 0 || (r < 1 && iter_ % 50 == 0))
    {
        label nCells = this->mesh_.nCells();
        reduce(stats.Nmin, minOp<scalar>());
        reduce(stats.Nmax, maxOp<scalar>());
        reduce(stats.Cmin, minOp<scalar>());
        reduce(stats.Cmax, maxOp<scalar>());
        reduce(stats.nNegP2, sumOp<label>());
        reduce(stats.nKolmogorov, sumOp<label>());
        reduce(nCells, sumOp<label>());
        const scalar nC = max(scalar(nCells), scalar(1));
        Info<< "kOmegaEARSM: iteration " << iter_ << " ramp " << r
            << " N [" << stats.Nmin << ", " << stats.Nmax << "]"
            << " CmuEff [" << stats.Cmin << ", " << stats.Cmax << "]"
            << " fraction P2<0 " << scalar(stats.nNegP2)/nC
            << " fraction Kolmogorov-limited " << scalar(stats.nKolmogorov)/nC
            << endl;
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
kOmegaEARSM<BasicTurbulenceModel>::kOmegaEARSM
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
    nonlinearEddyViscosity<RASModel<BasicTurbulenceModel>>
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

    A1_(dimensioned<scalar>::getOrAddToDict("A1", this->coeffDict_, 1.245)),
    C1_(dimensioned<scalar>::getOrAddToDict("C1", this->coeffDict_, 1.8)),
    Ctau_(dimensioned<scalar>::getOrAddToDict("Ctau", this->coeffDict_, 6.0)),
    includeT9_(Switch::getOrAddToDict("includeT9", this->coeffDict_, false)),

    sigmaK1_
    (
        dimensioned<scalar>::getOrAddToDict("sigmaK1", this->coeffDict_, 0.5)
    ),
    sigmaK2_
    (
        dimensioned<scalar>::getOrAddToDict("sigmaK2", this->coeffDict_, 1.0)
    ),
    sigmaOmega1_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "sigmaOmega1", this->coeffDict_, 0.5
        )
    ),
    sigmaOmega2_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "sigmaOmega2", this->coeffDict_, 0.856
        )
    ),
    beta1_
    (
        dimensioned<scalar>::getOrAddToDict("beta1", this->coeffDict_, 0.075)
    ),
    beta2_
    (
        dimensioned<scalar>::getOrAddToDict("beta2", this->coeffDict_, 0.0828)
    ),
    betaStar_
    (
        dimensioned<scalar>::getOrAddToDict("betaStar", this->coeffDict_, 0.09)
    ),
    kappa_
    (
        dimensioned<scalar>::getOrAddToDict("kappa", this->coeffDict_, 0.41)
    ),
    c1_(dimensioned<scalar>::getOrAddToDict("c1", this->coeffDict_, 10.0)),
    rampIterations_
    (
        dimensioned<scalar>::getOrAddToDict
        (
            "rampIterations", this->coeffDict_, 0.0
        )
    ),

    frameOmega_(this->coeffDict_.template get<vector>("frameOmega")),

    y_(wallDist::New(this->mesh_).y()),

    k_
    (
        IOobject
        (
            IOobject::groupName("k", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    omega_
    (
        IOobject
        (
            IOobject::groupName("omega", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),
    iter_(0)
{
    bound(k_, this->kMin_);
    bound(omega_, this->omegaMin_);

    this->nonlinearStress_.writeOpt(IOobject::AUTO_WRITE);

    Info<< "kOmegaEARSM: BSL-EARSM (Menter, Garbaruk and Egorov 2012);"
        << " frameOmega " << frameOmega_
        << " read and ignored (no system-rotation term in the published"
        << " model); gamma1 " << gamma1().value()
        << " gamma2 " << gamma2().value()
        << "; start-up ramp " << rampIterations_.value() << " iterations"
        << endl;

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool kOmegaEARSM<BasicTurbulenceModel>::read()
{
    if (nonlinearEddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        const dictionary& d = this->coeffDict();
        A1_.readIfPresent(d);
        C1_.readIfPresent(d);
        Ctau_.readIfPresent(d);
        includeT9_.readIfPresent("includeT9", d);
        sigmaK1_.readIfPresent(d);
        sigmaK2_.readIfPresent(d);
        sigmaOmega1_.readIfPresent(d);
        sigmaOmega2_.readIfPresent(d);
        beta1_.readIfPresent(d);
        beta2_.readIfPresent(d);
        betaStar_.readIfPresent(d);
        kappa_.readIfPresent(d);
        c1_.readIfPresent(d);
        rampIterations_.readIfPresent(d);
        frameOmega_ = d.template get<vector>("frameOmega");
        return true;
    }
    return false;
}


template<class BasicTurbulenceModel>
tmp<volScalarField> kOmegaEARSM<BasicTurbulenceModel>::epsilon() const
{
    return tmp<volScalarField>::New
    (
        IOobject
        (
            IOobject::groupName("epsilon", this->alphaRhoPhi_.group()),
            this->runTime_.timeName(),
            this->mesh_
        ),
        betaStar_*k_*omega_,
        omega_.boundaryField().types()
    );
}


template<class BasicTurbulenceModel>
void kOmegaEARSM<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    const volVectorField& U = this->U_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    nonlinearEddyViscosity<RASModel<BasicTurbulenceModel>>::correct();
    ++iter_;

    const volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(this->phi(), U))
    );

    tmp<volTensorField> tgradU = fvc::grad(U);
    const volTensorField& gradU = tgradU();

    // -tau_ij dU_i/dx_j (p. 93): linear part with nut, nonlinear part
    // explicit; the 2/3 k div(U) part is the SuSp term below, as in SST
    volScalarField::Internal G
    (
        this->GName(),
        this->nut_()*(gradU() && dev(twoSymm(gradU())))
      - (this->nonlinearStress_() && gradU())
    );

    // Update omega and G at the wall (omega wall functions), then push the
    // changed cell values to coupled neighbours (as kOmegaSSTBase)
    omega_.boundaryFieldRef().updateCoeffs();
    omega_.boundaryFieldRef().template evaluateCoupled<coupledFvPatch>();

    const volScalarField CDkOmega
    (
        (2*sigmaOmega2_)*(fvc::grad(k_) & fvc::grad(omega_))/omega_
    );

    const volScalarField F1(this->F1(CDkOmega));

    {
        const volScalarField::Internal gamma(blend(F1(), gamma1(), gamma2()));
        const volScalarField::Internal beta(blend(F1(), beta1_, beta2_));

        // gamma omega/k Pk~ with Pk~ = min(G, c1 betaStar k omega)
        const volScalarField::Internal kSafe(max(k_(), this->kMin_));
        const volScalarField::Internal PkTilde
        (
            min(G, (c1_*betaStar_)*k_()*omega_())
        );

        tmp<fvScalarMatrix> omegaEqn
        (
            fvm::ddt(alpha, rho, omega_)
          + fvm::div(alphaRhoPhi, omega_)
          - fvm::laplacian(alpha*rho*DomegaEff(F1), omega_)
         ==
            alpha()*rho()*gamma*PkTilde*omega_()/kSafe
          - fvm::SuSp((2.0/3.0)*alpha()*rho()*gamma*divU, omega_)
          - fvm::Sp(alpha()*rho()*beta*omega_(), omega_)
          - fvm::SuSp
            (
                alpha()*rho()*(F1() - scalar(1))*CDkOmega()/omega_(),
                omega_
            )
          + fvOptions(alpha, rho, omega_)
        );

        omegaEqn.ref().relax();
        fvOptions.constrain(omegaEqn.ref());
        omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
        solve(omegaEqn);
        fvOptions.correct(omega_);
        bound(omega_, this->omegaMin_);
    }

    {
        tmp<fvScalarMatrix> kEqn
        (
            fvm::ddt(alpha, rho, k_)
          + fvm::div(alphaRhoPhi, k_)
          - fvm::laplacian(alpha*rho*DkEff(F1), k_)
         ==
            alpha()*rho()*min(G, (c1_*betaStar_)*k_()*omega_())
          - fvm::SuSp((2.0/3.0)*alpha()*rho()*divU, k_)
          - fvm::Sp(alpha()*rho()*betaStar_*omega_(), k_)
          + fvOptions(alpha, rho, k_)
        );

        kEqn.ref().relax();
        fvOptions.constrain(kEqn.ref());
        solve(kEqn);
        fvOptions.correct(k_);
        bound(k_, this->kMin_);
    }

    correctNonlinearStress(gradU);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
