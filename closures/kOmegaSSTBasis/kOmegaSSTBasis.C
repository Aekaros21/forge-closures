#include "kOmegaSSTBasis.H"
#include "basisTensors/integrityBasis.H"
#include "fvcGrad.H"
#include "fvcCurl.H"
#include "fvcDiv.H"
#include "coupledFvPatch.H"
#include "publishedRITA.H"
#include "bound.H"
#include "IFstream.H"
#include "Tuple2.H"

#include <cstdlib>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions * * * * * * * * * * * //

template<class BasicTurbulenceModel>
void kOmegaSSTBasis<BasicTurbulenceModel>::parseChannel
(
    const dictionary& coeffs,
    const word& name,
    Channel& channel
)
{
    if (!coeffs.found(name))
    {
        return;
    }
    const dictionary& sub = coeffs.subDict(name);
    List<Tuple2<word, string>> terms;
    sub.readIfPresent("terms", terms);

    for (const auto& term : terms)
    {
        const word& tname = term.first();
        if (tname.size() < 2 || tname[0] != 'T')
        {
            FatalIOErrorInFunction(sub)
                << "bad basis tensor name " << tname
                << " in channel " << name << exit(FatalIOError);
        }
        const label n = atoi(tname.substr(1).c_str());
        if (n < 1 || n > 10)
        {
            FatalIOErrorInFunction(sub)
                << "basis tensor " << tname << " outside T1..T10"
                << exit(FatalIOError);
        }
        try
        {
            // free constants forbidden: driver must substitute literals
            channel.g.push_back
            (
                tedp::expr::parse(std::string(term.second()), false)
            );
        }
        catch (const tedp::expr::GrammarError& e)
        {
            FatalIOErrorInFunction(sub)
                << "channel " << name << ", term " << tname
                << ": " << e.what() << exit(FatalIOError);
        }
        channel.tensorIndex.append(n);
    }
}


template<class BasicTurbulenceModel>
void kOmegaSSTBasis<BasicTurbulenceModel>::computeV3Features
(
    const volScalarField& omegaSafe,
    const volSymmTensorField& Shat,
    const volVectorField& gradK,
    PtrList<volScalarField>& out
) const
{
    // Definitions and conventions: src/tedp/features.py (FEATURE_VERSION
    // forge-v3-state-features-1). Every quantity is dimensionless; OpenFOAM's
    // dimension checking enforces the construction at run time.
    const scalar EPS = 1.0e-3;
    const objectRegistry& db = this->mesh_.thisDb();
    if (!db.template foundObject<volScalarField>("p"))
    {
        FatalErrorInFunction
            << "FORGE V3 features need the solver's kinematic pressure field 'p'; "
            << "no such registered field" << exit(FatalError);
    }
    const volScalarField& p = db.template lookupObject<volScalarField>("p");
    if (p.dimensions() != sqr(dimVelocity))
    {
        FatalErrorInFunction
            << "FORGE V3 features are defined for the kinematic pressure p/rho "
            << "[m^2/s^2]; the registered 'p' has dimensions " << p.dimensions()
            << ". A dimensional pressure is refused rather than silently rescaled."
            << exit(FatalError);
    }
    const volScalarField& k = this->k_;
    // A = omega_s*sqrt(k_s) [m/s^2] with the solver's own floors
    const volScalarField A(omegaSafe*sqrt(max(k, this->kMin_)));
    const volVectorField a(fvc::grad(p)/A);
    const volVectorField g(gradK/A);
    const volScalarField ma(mag(a));
    const volScalarField mg(mag(g));
    const dimensionedVector Omega(dimless/dimTime, frameOmega_);
    const volScalarField tau(dimensionedScalar(dimless, 1.0)/omegaSafe);
    const volVectorField ohat(tau*Omega);
    const volVectorField zhat(tau*fvc::curl(this->U_));
    const volScalarField mo(mag(ohat));
    const volScalarField mz(mag(zhat));

    auto named = [&](const word& name, const tmp<volScalarField>& value)
    {
        return new volScalarField
        (
            IOobject(name, this->runTime_.timeName(), this->mesh_,
                     IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER),
            value
        );
    };
    out.setSize(tedp::expr::N_V3_VARS);
    out.set(0, named("v3Gp", ma/(1.0 + ma)));
    out.set(1, named("v3Gk", mg/(1.0 + mg)));
    out.set(2, named("v3Apk", (a & g)/(ma*mg + EPS)));
    out.set(3, named("v3Psn", (a & (Shat & a))/(magSqr(a) + EPS)));
    out.set(4, named("v3Ksn", (g & (Shat & g))/(magSqr(g) + EPS)));
    out.set(5, named("v3Rf", mo));
    out.set(6, named("v3Rw", (ohat & zhat)/(mo*mz + EPS)));

    if (writeV3Features_ && this->runTime_.writeTime())
    {
        forAll(out, i)
        {
            out[i].write();
        }
        // the dimensionless inputs, so an external check can rebuild the
        // features from the written fields without the solver's internals
        volVectorField(IOobject("v3a", this->runTime_.timeName(), this->mesh_,
            IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER), a).write();
        volVectorField(IOobject("v3g", this->runTime_.timeName(), this->mesh_,
            IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER), g).write();
        volVectorField(IOobject("v3ohat", this->runTime_.timeName(), this->mesh_,
            IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER), ohat).write();
        volVectorField(IOobject("v3zhat", this->runTime_.timeName(), this->mesh_,
            IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER), zhat).write();
        volSymmTensorField(IOobject("v3Shat", this->runTime_.timeName(), this->mesh_,
            IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER), Shat).write();
    }
}


template<class BasicTurbulenceModel>
void kOmegaSSTBasis<BasicTurbulenceModel>::checkFrameConfiguration()
{
    if (frameConfigurationChecked_)
    {
        return;
    }
    frameConfigurationChecked_ = true;
    bool frameSource = false;
    bool tableRead = false;
    vector tableOmega(Zero);
    const objectRegistry& db = this->mesh_.thisDb();
    if (db.template foundObject<IOdictionary>("fvOptions"))
    {
        const dictionary& options = db.template lookupObject<IOdictionary>("fvOptions");
        for (const entry& e : options)
        {
            if (!e.isDict())
            {
                continue;
            }
            const dictionary& d = e.dict();
            const word sourceType(d.getOrDefault<word>("type", ""));
            if (sourceType == "tabulatedAccelerationSource")
            {
                frameSource = true;
                fileName table(d.get<fileName>("timeDataFileName"));
                table.expand();
                if (!table.isAbsolute())
                {
                    table = this->runTime_.path()/table;
                }
                IFstream is(table);
                if (!is.good())
                {
                    FatalErrorInFunction
                        << "FORGE V3: cannot read the rotating-frame table "
                        << table << exit(FatalError);
                }
                List<Tuple2<scalar, Vector<vector>>> rows(is);
                if (rows.empty())
                {
                    FatalErrorInFunction
                        << "FORGE V3: empty rotating-frame table " << table
                        << exit(FatalError);
                }
                tableRead = true;
                tableOmega = rows.first().second().y();
                for (const auto& row : rows)
                {
                    if (mag(row.second().y() - tableOmega) > 1.0e-9*max(mag(tableOmega), scalar(1)))
                    {
                        FatalErrorInFunction
                            << "FORGE V3: the frame angular velocity in " << table
                            << " is not constant; a time-varying frame is not supported"
                            << exit(FatalError);
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
                frameSource = true;
            }
        }
    }
    if (frameSource && !frameOmegaDeclared_)
    {
        FatalErrorInFunction
            << "FORGE V3: a rotating-frame momentum source is configured in fvOptions "
            << "but kOmegaSSTBasisCoeffs declares no frameOmega. The system rotation "
            << "must be declared explicitly; it is never replaced by zero."
            << exit(FatalError);
    }
    if (tableRead && mag(tableOmega - frameOmega_) > 1.0e-9*max(mag(tableOmega), scalar(1)))
    {
        FatalErrorInFunction
            << "FORGE V3: frameOmega " << frameOmega_ << " differs from the "
            << "rotating-frame source angular velocity " << tableOmega
            << exit(FatalError);
    }
    if (!frameSource && mag(frameOmega_) > 0)
    {
        FatalErrorInFunction
            << "FORGE V3: frameOmega " << frameOmega_ << " is declared but no "
            << "rotating-frame momentum source is configured in fvOptions"
            << exit(FatalError);
    }
    Info<< "kOmegaSSTBasis: frame configuration checked; frameOmega "
        << frameOmega_ << (frameSource ? " (rotating-frame source present)" : " (inertial)")
        << nl;
}


template<class BasicTurbulenceModel>
void kOmegaSSTBasis<BasicTurbulenceModel>::updateCorrections
(
    const volTensorField& gradU
)
{
    const volScalarField& k = this->k_;
    const volScalarField& nut = this->nut_;
    const label nCells = this->mesh_.nCells();

    const volScalarField omegaSafe(max(this->omega_, this->omegaMin_));
    const volSymmTensorField Sdim(dev(symm(gradU)));

    // cached blending for omegaSource: gamma(F1) from the current state
    {
        const volVectorField gradK(fvc::grad(k));
        const volScalarField CDkOmega
        (
            (2*this->alphaOmega2_)
           *(gradK & fvc::grad(this->omega_))/this->omega_
        );
        const volScalarField F1(this->F1(CDkOmega));
        gammaF1_ = this->gamma(F1());

        // ---- variable fields for the expression grammar
        const volScalarField tau
        (
            dimensionedScalar(dimless, 1.0)/omegaSafe
        );
        const volSymmTensorField Shat(tau*Sdim);
        // absolute rotation tensor: W_abs,ij = W_rel,ij - eps_ijk Omega_k,
        // so that a rotating-frame solution sees the absolute vorticity
        const tensor frameTensor
        (
            0.0,           frameOmega_.z(), -frameOmega_.y(),
           -frameOmega_.z(), 0.0,            frameOmega_.x(),
            frameOmega_.y(), -frameOmega_.x(), 0.0
        );
        // Absolute rotation rate in a rotating frame. In the convention
        // W_ij = 1/2 (dU_i/dx_j - dU_j/dx_i) the absolute tensor is
        // W_ij - eps_ijk Omega_k, but OpenFOAM's grad(U)_ij = dU_j/dx_i, so
        // skew(gradU) = -W and the frame term is ADDED here. Verified
        // physically: with Omega = +z and the flow along +x the Coriolis
        // acceleration -2 Omega x u points to -y, so the -y wall is the
        // pressure (destabilised, higher-friction) side, which a
        // rotation-sensitive model must reproduce.
        const volTensorField What
        (
            tau*(skew(gradU) + dimensionedTensor(dimless/dimTime, frameTensor))
        );

        const scalar ramp =
            rampIterations_ > 0
          ? min(scalar(1), scalar(iter_)/scalar(rampIterations_))
          : scalar(1);
        const scalar relax = bDeltaRelax_;

        // frozen-field mode: assign stored fields (still ramped/relaxed)
        if (mode_ == Mode::frozenFields)
        {
            // the same safeguards a candidate gets: realizability clip of
            // the total b (the injected bDelta was inferred against the
            // frozen nut; with the evolving nut the total can leave the
            // Lumley triangle and run away) and the rMaxFactor bound on R
            const symmTensorField& bDin = bDeltaFieldPtr_().primitiveField();
            const scalarField& Rin = RFieldPtr_().primitiveField();
            symmTensorField& ns = this->nonlinearStress_.primitiveFieldRef();
            scalarField& Rf = Rfield_.field();
            const scalarField& kf = k.primitiveField();
            const scalarField& omf = omegaSafe.primitiveField();
            const symmTensorField& Sd = Sdim.primitiveField();
            const scalarField& nutf = nut.primitiveField();
            const scalar tol = 1.0e-6;
            auto inTriangle = [tol](const symmTensor& b) -> bool
            {
                const vector ev = eigenValues(b);
                return ev.x() >= -1.0/3.0 - tol && ev.z() <= 2.0/3.0 + tol;
            };
            label nClipped = 0;
            for (label c = 0; c < nCells; ++c)
            {
                symmTensor bD = bDin[c];
                if (realizabilityClip_)
                {
                    const symmTensor bB = -(nutf[c]/max(kf[c], SMALL))*Sd[c];
                    if (!inTriangle(bB + bD))
                    {
                        ++nClipped;
                        scalar lo = 0.0, hi = 1.0;
                        for (int it = 0; it < 6; ++it)
                        {
                            const scalar mid = 0.5*(lo + hi);
                            if (inTriangle(bB + mid*bD)) lo = mid; else hi = mid;
                        }
                        bD *= lo;
                    }
                }
                const scalar bnd = rMaxFactor_*this->betaStar_.value()*kf[c]*omf[c];
                const scalar Rb = max(-bnd, min(bnd, Rin[c]));
                ns[c] = (1.0 - relax)*ns[c] + relax*ramp*(2.0*kf[c]*bD);
                Rf[c] = (1.0 - relax)*Rf[c] + relax*ramp*Rb;
            }
            clipFraction_ =
                scalar(returnReduce(nClipped, sumOp<label>()))
               /max(returnReduce(nCells, sumOp<label>()), label(1));
        }
        else if (publishedRITA_ || !bDelta_.empty() || !rSource_.empty())
        {
            const volScalarField Ret(k/(this->nu()*omegaSafe));
            const volScalarField G2S
            (
                nut*2.0*magSqr(symm(gradU))
            );
            const volScalarField PoE
            (
                min
                (
                    G2S/max
                    (
                        this->betaStar_*k*omegaSafe,
                        dimensionedScalar(G2S.dimensions(), SMALL)
                    ),
                    dimensionedScalar(dimless, 10.0)
                )
            );

            // ---- RITA comparator variables (Buchanan et al. 2025,
            // arXiv:2504.06758). Built only when an expression references one,
            // so no discovered model pays for them and no existing result can
            // shift. sigmaSL is the paper's binary shear-layer classifier;
            // the phi ratios are relative magnitudes of terms in the k budget.
            bool needsRita = publishedRITA_;
            for (const Channel* ch : {&bDelta_, &rSource_})
            {
                for (const auto& gexpr : ch->g)
                {
                    for (std::size_t vi = tedp::expr::FIRST_RITA_VAR;
                         vi < tedp::expr::FIRST_V3_VAR; ++vi)
                    {
                        if (gexpr->usesVar(vi)) needsRita = true;
                    }
                }
            }
            autoPtr<volScalarField> sigmaSL, phiDkPk, phiDkCk, phik, ReOmega;
            if (needsRita)
            {
                const volScalarField Dk(this->betaStar_*k*omegaSafe);
                const dimensionedScalar tiny(Dk.dimensions(), SMALL);
                phiDkPk.reset(new volScalarField(Dk/max(G2S + Dk, tiny)));
                phiDkCk.reset(new volScalarField
                (
                    Dk/max(mag(this->U_ & fvc::grad(k)) + Dk, tiny)
                ));
                phik.reset(new volScalarField
                (
                    k/max(k + 0.5*magSqr(this->U_),
                          dimensionedScalar(k.dimensions(), SMALL))
                ));
                // Eq. 24: the criterion uses the MIN-MAX NORMALISED vorticity
                // Reynolds number, phi_ReOmega = (Re - Re_min)/(Re_max - Re_min)
                // in [0,1], not the raw Re_Omega (which is O(1e3-1e6) and would
                // make the 0.02 threshold vacuous). Normalised per case over the
                // whole field.
                volScalarField reRaw(sqr(this->y_)*mag(fvc::curl(this->U_))/this->nu());
                const scalar reMin = gMin(reRaw.primitiveField());
                const scalar reMax = gMax(reRaw.primitiveField());
                ReOmega.reset(new volScalarField
                (
                    (reRaw - dimensionedScalar(reRaw.dimensions(), reMin))
                   /dimensionedScalar(reRaw.dimensions(), max(reMax - reMin, SMALL))
                ));
                sigmaSL.reset(new volScalarField
                (
                    IOobject
                    (
                        "sigmaSL", this->runTime_.timeName(), this->mesh_,
                        IOobject::NO_READ, IOobject::NO_WRITE
                    ),
                    pos0(0.55 - phiDkPk()) * pos0(phik() - 0.12)
                  * pos0(ReOmega() - 0.02)   // phi_ReOmega, normalised above
                ));
                if (publishedRITA_)
                {
                    const volScalarField convection(mag(this->U_ & fvc::grad(k)));
                    const volScalarField nativeProduction
                    (
                        nut*(gradU && devTwoSymm(gradU))
                    );
                    for (label c = 0; c < nCells; ++c)
                    {
                        ReOmega()[c] = tedpRita::normalizedRe(reRaw[c], reMin, reMax);
                        phiDkPk()[c] = tedpRita::phiDkPk
                        (
                            nativeProduction[c], Dk[c], Rfield_[c]
                        );
                        phiDkCk()[c] = tedpRita::phiDkCk(Dk[c], convection[c]);
                        phik()[c] = tedpRita::phiK(k[c], magSqr(this->U_[c]));
                        sigmaSL()[c] = tedpRita::sigmaSL
                        (
                            phiDkPk()[c], phik()[c], ReOmega()[c]
                        );
                    }
                }
                if (iter_ % 100 == 0 || this->runTime_.writeTime())
                {
                    if (this->runTime_.writeTime())
                    {
                        sigmaSL().write();
                        // the three criteria separately, so the zone can be
                        // attributed to the term that actually bounds it
                        volScalarField(IOobject("phiDkPk", this->runTime_.timeName(),
                            this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE), phiDkPk()).write();
                        volScalarField(IOobject("phik", this->runTime_.timeName(),
                            this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE), phik()).write();
                        volScalarField(IOobject("phiReOmega", this->runTime_.timeName(),
                            this->mesh_, IOobject::NO_READ, IOobject::NO_WRITE), ReOmega()).write();
                    }
                    Info<< "RITA classifier: sigmaSL = 1 in "
                        << 100.0*gSum(sigmaSL().primitiveField()
                                     *this->mesh_.V())/gSum(this->mesh_.V())
                        << "% of the volume" << endl;
                }
            }

            // ---- FORGE V3 physical-state features: built only when an
            // expression references one (or the write switch is on), so a
            // legacy expression's cost and result are exactly unchanged.
            bool needsV3 = writeV3Features_;
            for (const Channel* ch : {&bDelta_, &rSource_})
            {
                for (const auto& gexpr : ch->g)
                {
                    for (std::size_t vi = tedp::expr::FIRST_V3_VAR;
                         vi < tedp::expr::N_STATE_VARS; ++vi)
                    {
                        if (gexpr->usesVar(vi)) needsV3 = true;
                    }
                }
            }
            PtrList<volScalarField> v3;
            if (needsV3)
            {
                computeV3Features(omegaSafe, Shat, gradK, v3);
            }

            // gather grammar variables (indices as in basisExpr)
            PtrList<volScalarField> invs(5);
            for (label i = 0; i < 5; ++i)
            {
                invs.set(i, tedpBasis::invariant(i + 1, Shat, What).ptr());
            }
            const double* vars[tedp::expr::N_ALL_VARS] = {nullptr};
            for (label i = 0; i < 5; ++i)
            {
                vars[i] = invs[i].primitiveField().cdata();
            }
            vars[5] = Ret.primitiveField().cdata();
            vars[6] = F1.primitiveField().cdata();
            vars[7] = PoE.primitiveField().cdata();
            if (needsRita)
            {
                vars[8] = sigmaSL().primitiveField().cdata();
                vars[9] = phiDkPk().primitiveField().cdata();
                vars[10] = phiDkCk().primitiveField().cdata();
                vars[11] = phik().primitiveField().cdata();
                vars[12] = ReOmega().primitiveField().cdata();
            }
            if (needsV3)
            {
                for (std::size_t i = 0; i < tedp::expr::N_V3_VARS; ++i)
                {
                    vars[tedp::expr::FIRST_V3_VAR + i] = v3[i].primitiveField().cdata();
                }
            }

            const scalarField& kf = k.primitiveField();
            const scalarField& omf = omegaSafe.primitiveField();

            auto evalChannel =
                [&](const Channel& channel, symmTensorField& out) -> label
            {
                out = symmTensor::zero;
                label nClamped = 0;
                scalarField gn(nCells);
                forAll(channel.tensorIndex, t)
                {
                    channel.g[t]->eval(gn.data(), vars, nCells);
                    const tmp<volSymmTensorField> tT
                    (
                        tedpBasis::basisTensor
                        (
                            channel.tensorIndex[t], Shat, What
                        )
                    );
                    const symmTensorField& T = tT().primitiveField();
                    for (label c = 0; c < nCells; ++c)
                    {
                        if (mag(gn[c]) > gMax_)
                        {
                            ++nClamped;
                        }
                        const scalar g =
                            max(-gMax_, min(gMax_, gn[c]));
                        out[c] += g*T[c];
                    }
                }
                return nClamped;
            };
            const label nCellsGlobal = returnReduce(nCells, sumOp<label>());
            label nClampB = 0;
            label nClampR = 0;
            label nBoundR = 0;

            if (publishedRITA_)
            {
                // Final-paper source is a scalar, not an inverse-I1 tensor
                // surrogate. T2 is converted to OpenFOAM convention once.
                const tmp<volSymmTensorField> tT2
                (
                    tedpBasis::basisTensor(2, Shat, What)
                );
                const symmTensorField& T2 = tT2().primitiveField();
                symmTensorField& raw = ritaUngatedStress_.primitiveFieldRef();
                symmTensorField& ns = this->nonlinearStress_.primitiveFieldRef();
                scalarField& Rf = Rfield_.field();
                for (label c = 0; c < nCells; ++c)
                {
                    const scalar gate = sigmaSL()[c];
                    ritaSigma_[c] = gate;
                    const scalar g = tedpRita::bDeltaT2Solver(phiDkPk()[c], invs[1][c]);
                    raw[c] = (1-relax)*raw[c]
                           + relax*ramp*(2*kf[c]*g*T2[c]);
                    ns[c] = gate*raw[c];
                    const scalar source = tedpRita::rOverEpsilon(phiDkCk()[c])
                                        *this->betaStar_.value()*kf[c]*omf[c];
                    Rf[c] = gate*((1-relax)*Rf[c] + relax*ramp*source);
                }
                clipFraction_ = gClampFractionB_ = gClampFractionR_ = rBoundFraction_ = 0;
                if (iter_ % 100 == 0 || this->runTime_.writeTime())
                {
                    Info<< "kOmegaSSTBasis clamps: realizability 0 gMax(bDelta) 0 gMax(R) 0 rBound 0" << endl;
                }
            }
            else
            {

            // ---- channel 1: b^Delta -> nonlinearStress
            symmTensorField bDeltaNew(nCells, symmTensor::zero);
            if (!bDelta_.empty())
            {
                nClampB = evalChannel(bDelta_, bDeltaNew);

                if (realizabilityClip_)
                {
                    const symmTensorField& Sd = Sdim.primitiveField();
                    const scalarField& nutf = nut.primitiveField();
                    label nClipped = 0;
                    const scalar tol = 1.0e-6;
                    auto inTriangle = [tol](const symmTensor& b) -> bool
                    {
                        const vector ev = eigenValues(b);
                        return ev.x() >= -1.0/3.0 - tol
                            && ev.z() <= 2.0/3.0 + tol;
                    };
                    for (label c = 0; c < nCells; ++c)
                    {
                        const symmTensor bB =
                            -(nutf[c]/max(kf[c], SMALL))*Sd[c];
                        if (inTriangle(bB + bDeltaNew[c]))
                        {
                            continue;
                        }
                        ++nClipped;
                        scalar lo = 0.0, hi = 1.0;
                        for (int it = 0; it < 6; ++it)
                        {
                            const scalar mid = 0.5*(lo + hi);
                            if (inTriangle(bB + mid*bDeltaNew[c]))
                            {
                                lo = mid;
                            }
                            else
                            {
                                hi = mid;
                            }
                        }
                        bDeltaNew[c] *= lo;
                    }
                    clipFraction_ =
                        scalar(returnReduce(nClipped, sumOp<label>()))
                       /max(nCellsGlobal, label(1));
                }
            }
            gClampFractionB_ =
                scalar(returnReduce(nClampB, sumOp<label>()))
               /max(nCellsGlobal*label(bDelta_.tensorIndex.size()), label(1));

            symmTensorField& ns = this->nonlinearStress_.primitiveFieldRef();
            for (label c = 0; c < nCells; ++c)
            {
                ns[c] =
                    (1.0 - relax)*ns[c]
                  + relax*ramp*(2.0*kf[c]*bDeltaNew[c]);
            }

            // ---- channel 2: R source
            scalarField rNew(nCells, 0.0);
            if (!rSource_.empty())
            {
                symmTensorField bR(nCells, symmTensor::zero);
                nClampR = evalChannel(rSource_, bR);
                // bR is traceless, so bR : gradU == bR : dev(symm(gradU))
                const symmTensorField& Sd = Sdim.primitiveField();
                for (label c = 0; c < nCells; ++c)
                {
                    rNew[c] = 2.0*kf[c]*(bR[c] && Sd[c]);
                    const scalar bnd =
                        rMaxFactor_*this->betaStar_.value()*kf[c]*omf[c];
                    if (mag(rNew[c]) > bnd)
                    {
                        ++nBoundR;
                    }
                    rNew[c] = max(-bnd, min(bnd, rNew[c]));
                }
            }
            gClampFractionR_ =
                scalar(returnReduce(nClampR, sumOp<label>()))
               /max(nCellsGlobal*label(rSource_.tensorIndex.size()), label(1));
            rBoundFraction_ =
                scalar(returnReduce(nBoundR, sumOp<label>()))
               /max(nCellsGlobal, label(1));
            scalarField& Rf = Rfield_.field();
            for (label c = 0; c < nCells; ++c)
            {
                Rf[c] = (1.0 - relax)*Rf[c] + relax*ramp*rNew[c];
            }
            if (iter_ % 100 == 0 || this->runTime_.writeTime())
            {
                Info<< "kOmegaSSTBasis clamps: realizability "
                    << clipFraction_
                    << " gMax(bDelta) " << gClampFractionB_
                    << " gMax(R) " << gClampFractionR_
                    << " rBound " << rBoundFraction_
                    << endl;
            }
            }
        }
    }

    // Extrapolate on physical patches and exchange across processor patches.
    // Owner-only processor data silently change the nonlinear face flux.
    auto refreshStress = [&](volSymmTensorField& field)
    {
        auto& patches = field.boundaryFieldRef();
        forAll(patches, patchi)
        {
            if (!coupledStressBoundary_ || !patches[patchi].coupled())
                patches[patchi] == patches[patchi].patchInternalField()();
        }
        if (coupledStressBoundary_)
            patches.template evaluateCoupled<coupledFvPatch>();
    };
    refreshStress(this->nonlinearStress_);
    if (publishedRITA_)
    {
        refreshStress(ritaUngatedStress_);
        auto& sb = ritaSigma_.boundaryFieldRef();
        forAll(sb, patchi)
            if (!sb[patchi].coupled()) sb[patchi] == sb[patchi].patchInternalField()();
        sb.template evaluateCoupled<coupledFvPatch>();
    }

    // production from the nonlinear stress (for Pk and omegaSource);
    // the stress is traceless, so contracting with dev(symm(gradU)) is exact
    {
        const symmTensorField& ns = this->nonlinearStress_.primitiveField();
        const symmTensorField& Sd = Sdim.primitiveField();
        scalarField& gf = Gnl_.field();
        for (label c = 0; c < nCells; ++c)
        {
            // T2:S is identically zero, so avoid injecting roundoff production
            // into the final-paper RITA classifier/source feedback.
            gf[c] = publishedRITA_ ? 0.0 : -(ns[c] && Sd[c]);
        }
    }
}


template<class BasicTurbulenceModel>
tmp<volScalarField::Internal> kOmegaSSTBasis<BasicTurbulenceModel>::Pk
(
    const volScalarField::Internal& G
) const
{
    if (nonlinearProduction_)
    {
        return min
        (
            G + Gnl_,
            (this->c1_*this->betaStar_)*this->k_()*this->omega_()
        );
    }
    return BaseType::Pk(G);
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> kOmegaSSTBasis<BasicTurbulenceModel>::kSource() const
{
    tmp<fvScalarMatrix> tSrc(BaseType::kSource());
    tSrc.ref() += this->alpha_()*this->rho_()*Rfield_;
    return tSrc;
}


template<class BasicTurbulenceModel>
tmp<fvScalarMatrix> kOmegaSSTBasis<BasicTurbulenceModel>::omegaSource() const
{
    tmp<fvScalarMatrix> tSrc(BaseType::omegaSource());

    if (exactOmegaSource_)
    {
        volScalarField::Internal source
        (
            IOobject("publishedOmegaCorrection", this->runTime_.timeName(), this->mesh_),
            this->mesh_, dimensionedScalar(dimless/sqr(dimTime), Zero)
        );
        forAll(source, c)
        {
            const scalar value = Rfield_[c] + (nonlinearProduction_ ? Gnl_[c] : 0.0);
            if (value == 0.0) continue;
            if (this->nut_[c] <= 0.0)
            {
                FatalErrorInFunction
                    << "Nonzero published omega source with nonpositive nut in cell " << c
                    << ". Initialise from a converged positive-viscosity SST solution."
                    << exit(FatalError);
            }
            source[c] = value/this->nut_[c];
        }
        tSrc.ref() += this->alpha_()*this->rho_()*gammaF1_*source;
        return tSrc;
    }

    // floor nut by a viscosity scale: SMALL would let (Gnl+R)/nut detonate
    // the omega equation wherever nut ~ 0 (startup, laminarizing regions)
    const volScalarField::Internal nutSafe
    (
        max(this->nut_(), 1.0e-3*this->nu()()())
    );
    if (nonlinearProduction_)
    {
        tSrc.ref() +=
            this->alpha_()*this->rho_()*gammaF1_*(Gnl_ + Rfield_)/nutSafe;
    }
    else
    {
        tSrc.ref() +=
            this->alpha_()*this->rho_()*gammaF1_*Rfield_/nutSafe;
    }
    return tSrc;
}


template<class BasicTurbulenceModel>
void kOmegaSSTBasis<BasicTurbulenceModel>::correctNut
(
    const volScalarField& S2
)
{
    // stock SST a1 limiter, untouched
    BaseType::correctNut(S2);
    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void kOmegaSSTBasis<BasicTurbulenceModel>::correctNut()
{
    correctNut(2*magSqr(symm(fvc::grad(this->U_))));
}


template<class BasicTurbulenceModel>
void kOmegaSSTBasis<BasicTurbulenceModel>::correctNonlinearStress
(
    const volTensorField& gradU
)
{
    updateCorrections(gradU);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
kOmegaSSTBasis<BasicTurbulenceModel>::kOmegaSSTBasis
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
    mode_(Mode::expressions),
    gMax_(10.0),
    rMaxFactor_(5.0),
    bDeltaRelax_(0.5),
    rampIterations_(200),
    nonlinearProduction_(true),
    realizabilityClip_(true),
    coupledStressBoundary_(true),
    publishedRITA_(false),
    exactOmegaSource_(false),
    ritaSigma_
    (
        IOobject("ritaSigmaSL", this->runTime_.timeName(), this->mesh_),
        this->mesh_, dimensionedScalar(dimless, Zero)
    ),
    ritaUngatedStress_
    (
        IOobject("ritaUngatedStress", this->runTime_.timeName(), this->mesh_),
        this->mesh_, dimensionedSymmTensor(sqr(dimVelocity), Zero)
    ),
    iter_(0),
    clipFraction_(0.0),
    gClampFractionB_(0.0),
    gClampFractionR_(0.0),
    rBoundFraction_(0.0),
    Rfield_
    (
        IOobject
        (
            IOobject::scopedName(typeName, "R"),
            this->runTime_.timeName(),
            this->mesh_
        ),
        this->mesh_,
        dimensionedScalar(sqr(dimLength)/pow3(dimTime), Zero)
    ),
    Gnl_
    (
        IOobject
        (
            IOobject::scopedName(typeName, "Gnl"),
            this->runTime_.timeName(),
            this->mesh_
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
            this->mesh_
        ),
        this->mesh_,
        dimensionedScalar(dimless, Zero)
    )
{
    // the base constructs nonlinearStress_ NO_WRITE; scoring needs it
    this->nonlinearStress_.writeOpt(IOobject::AUTO_WRITE);

    const dictionary& coeffs = this->coeffDict();

    const word modeName(coeffs.getOrDefault<word>("mode", "expressions"));
    if (modeName == "frozenFields")
    {
        mode_ = Mode::frozenFields;
    }
    else if (modeName != "expressions")
    {
        FatalIOErrorInFunction(coeffs)
            << "mode must be 'expressions' or 'frozenFields', got "
            << modeName << exit(FatalIOError);
    }

    gMax_ = coeffs.getOrDefault<scalar>("gMax", 10.0);
    rMaxFactor_ = coeffs.getOrDefault<scalar>("rMaxFactor", 5.0);
    frameOmegaDeclared_ = coeffs.found("frameOmega");
    frameOmega_ = coeffs.getOrDefault<vector>("frameOmega", Zero);
    frameConfigurationChecked_ = false;
    writeV3Features_ = coeffs.getOrDefault<Switch>("writeV3Features", false);
    if (mag(frameOmega_) > 0)
    {
        Info<< "kOmegaSSTBasis: rotation tensor includes frame omega "
            << frameOmega_ << nl;
    }
    bDeltaRelax_ = coeffs.getOrDefault<scalar>("bDeltaRelax", 0.5);
    rampIterations_ = coeffs.getOrDefault<label>("rampIterations", 200);
    nonlinearProduction_ =
        coeffs.getOrDefault<Switch>("nonlinearProduction", true);
    realizabilityClip_ =
        coeffs.getOrDefault<Switch>("realizabilityClip", true);
    coupledStressBoundary_ =
        coeffs.getOrDefault<Switch>("coupledStressBoundary", true);
    publishedRITA_ = coeffs.getOrDefault<Switch>("publishedRITA", false);
    exactOmegaSource_ = coeffs.getOrDefault<Switch>("exactOmegaSource", publishedRITA_);
    if (publishedRITA_)
    {
        if (mode_ != Mode::expressions || !nonlinearProduction_)
        {
            FatalIOErrorInFunction(coeffs)
                << "publishedRITA requires expressions mode and nonlinearProduction on"
                << exit(FatalIOError);
        }
        ritaSigma_.writeOpt(IOobject::AUTO_WRITE);
        ritaUngatedStress_.writeOpt(IOobject::AUTO_WRITE);
        Info<< "RITA: final journal equations; correction-aware classifier; native scalar source; gate outside divergence" << nl;
    }

    // the a1 channel is declared but OFF in this phase
    const string a1Expr(coeffs.getOrDefault<string>("a1Expr", ""));
    if (!a1Expr.empty())
    {
        FatalIOErrorInFunction(coeffs)
            << "the a1 limiter channel (a1Expr) is defined but switched off"
            << " in this phase; it must be empty" << exit(FatalIOError);
    }

    if (mode_ == Mode::expressions)
    {
        parseChannel(coeffs, "bDelta", bDelta_);
        parseChannel(coeffs, "rSource", rSource_);

        Info<< "kOmegaSSTBasis: " << bDelta_.tensorIndex.size()
            << " bDelta term(s), " << rSource_.tensorIndex.size()
            << " rSource term(s)" << nl;
    }
    else
    {
        bDeltaFieldPtr_.reset
        (
            new volSymmTensorField
            (
                IOobject
                (
                    "bDeltaField",
                    this->runTime_.timeName(),
                    this->mesh_,
                    IOobject::MUST_READ
                ),
                this->mesh_
            )
        );
        RFieldPtr_.reset
        (
            new volScalarField
            (
                IOobject
                (
                    "RField",
                    this->runTime_.timeName(),
                    this->mesh_,
                    IOobject::MUST_READ
                ),
                this->mesh_
            )
        );
        Info<< "kOmegaSSTBasis: frozenFields mode (bDeltaField, RField)"
            << nl;
    }

    if (type == typeName)
    {
        this->printCoeffs(type);
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
tmp<fvVectorMatrix> kOmegaSSTBasis<BasicTurbulenceModel>::divDevRhoReff
(
    volVectorField& U
) const
{
    tmp<fvVectorMatrix> result(BaseType::divDevRhoReff(U));
    if (publishedRITA_)
    {
        result.ref() -= fvc::div(this->rho_*this->nonlinearStress_);
        result.ref() += ritaSigma_*fvc::div(this->rho_*ritaUngatedStress_);
    }
    return result;
}

template<class BasicTurbulenceModel>
tmp<fvVectorMatrix> kOmegaSSTBasis<BasicTurbulenceModel>::divDevRhoReff
(
    const volScalarField& rho,
    volVectorField& U
) const
{
    tmp<fvVectorMatrix> result(BaseType::divDevRhoReff(rho, U));
    if (publishedRITA_)
    {
        result.ref() -= fvc::div(rho*this->nonlinearStress_);
        result.ref() += ritaSigma_*fvc::div(rho*ritaUngatedStress_);
    }
    return result;
}

template<class BasicTurbulenceModel>
bool kOmegaSSTBasis<BasicTurbulenceModel>::read()
{
    if (BaseType::read())
    {
        const dictionary& coeffs = this->coeffDict();
        gMax_ = coeffs.getOrDefault<scalar>("gMax", gMax_);
        rMaxFactor_ = coeffs.getOrDefault<scalar>("rMaxFactor", rMaxFactor_);
        frameOmegaDeclared_ = frameOmegaDeclared_ || coeffs.found("frameOmega");
        frameOmega_ = coeffs.getOrDefault<vector>("frameOmega", frameOmega_);
        writeV3Features_ = coeffs.getOrDefault<Switch>("writeV3Features", writeV3Features_);
        bDeltaRelax_ =
            coeffs.getOrDefault<scalar>("bDeltaRelax", bDeltaRelax_);
        rampIterations_ =
            coeffs.getOrDefault<label>("rampIterations", rampIterations_);
        return true;
    }
    return false;
}


template<class BasicTurbulenceModel>
void kOmegaSSTBasis<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    ++iter_;

    const bool active =
        mode_ == Mode::frozenFields
     || publishedRITA_
     || !bDelta_.empty()
     || !rSource_.empty();

    if (active)
    {
        checkFrameConfiguration();
        tmp<volTensorField> tgradU = fvc::grad(this->U_);
        updateCorrections(tgradU());
        tgradU.clear();
    }

    BaseType::correct();
}


} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
