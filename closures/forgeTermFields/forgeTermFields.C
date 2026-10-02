/*---------------------------------------------------------------------------*\
Application
    forgeTermFields

Description
    Post-processing of a converged kOmegaSSTBasis case: the field of every
    term of the correction, for the figure "where the terms act".

    Reads U, p, k, omega and nut at the selected times, rebuilds the inputs
    of the correction exactly as libkOmegaSSTBasis does (basis tensors and
    invariants from src/kOmegaSSTBasis/basisTensors, expressions parsed and
    evaluated by src/kOmegaSSTBasis/basisExpr, F1 and the production of
    OpenFOAM v2312 kOmegaSSTBase, the FORGE V3 state features of
    kOmegaSSTBasis::computeV3Features), and splits the converged correction
    into the terms listed in system/termFieldsDict (termAlgebra.H).

    The library and the solver are not touched: this is a separate
    executable that only includes their headers and expression sources.
    The full coefficient functions of termFieldsDict must equal, term by
    term, those in the case's constant/turbulenceProperties; otherwise the
    utility stops. The recomputed full stress is compared with the
    nonlinearStress field the solver wrote (convergence consistency).

    Writes, at each selected time (prefix tf):
      tfF1 tfWallDist tfG tfPkSST tfDk                      SST quantities
      tfPi tfRf tfRw tfGk tfGp tfKsn                        inputs
      tfLambda tfClamp                                      limits
      tfTauDelta tfR tfGnl tfDeltaPk                        full correction
      tfTauDelta<j> tfR<j> tfGnl<j> tfNet<j>                term j
      tfDnut<j> tfNet<j>ByDk tfR<j>ByDk tfR<j>ByPkSST       term j
      tfGnl<j>ByDk tfDnut<j>ByNut
    and postProcessing/forgeTermFields/<time>/summary.json.

Usage
    forgeTermFields -latestTime        (system/termFieldsDict required)
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "wallDist.H"
#include "timeSelector.H"
#include "Tuple2.H"
#include "OFstream.H"
#include "OSspecific.H"

#include "basisTensors/integrityBasis.H"
#include "basisExpr/exprParser.H"
#include "termAlgebra.H"
#include "termDict.H"

#include <memory>
#include <type_traits>
#include <algorithm>
#include <cctype>
#include <vector>
#include <iomanip>
#include <sstream>

using namespace Foam;
using Foam::forgeTermFields::ExprList;
using Foam::forgeTermFields::readList;
using Foam::forgeTermFields::squeeze;

namespace
{

//- Same terms in any order (a candidate is rendered in its canonical order,
//  a revision scan in its source order); whitespace is ignored.
void requireSame(const ExprList& a, const ExprList& b, const word& what)
{
    auto keys = [](const ExprList& l)
    {
        std::vector<std::string> k;
        for (label t = 0; t < l.size(); ++t)
        {
            k.push_back(std::to_string(l.tensorIndex[t]) + ":" + squeeze(l.text[t]));
        }
        std::sort(k.begin(), k.end());
        return k;
    };
    const bool same = keys(a) == keys(b);
    if (!same)
    {
        FatalErrorInFunction
            << what << ": the coefficient functions of system/termFieldsDict "
            << "differ from those the solver ran (constant/turbulenceProperties)."
            << " Generate the dictionary for this model."
            << exit(FatalError);
    }
}

void requireTensors(const ExprList& full, const ExprList& term, const word& what)
{
    bool same = full.size() == term.size();
    for (label t = 0; same && t < full.size(); ++t)
    {
        same = full.tensorIndex[t] == term.tensorIndex[t];
    }
    if (!same)
    {
        FatalErrorInFunction
            << what << ": a term must list the same basis tensors, in the same"
            << " order, as the full correction" << exit(FatalError);
    }
}

//- A JSON number (unquoted: an OpenFOAM word), or null
word num(const scalar x)
{
    if (!std::isfinite(x)) return word("null");
    std::ostringstream os;
    os << std::setprecision(12) << x;
    return word(os.str(), false);
}

} // namespace


int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Fields of every term of a kOmegaSSTBasis correction at a converged"
        " state (system/termFieldsDict)"
    );
    timeSelector::addOptions();
    argList::addBoolOption
    (
        "writeInputs",
        "also write the discretised inputs (tfC, tfV, tfGradU, tfGradK,"
        " tfGradOmega, tfGradP) for an independent evaluation"
    );
    argList::addBoolOption
    (
        "noConsistency",
        "do not require the recomputed stress to match nonlinearStress"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    instantList timeDirs = timeSelector::select0(runTime, args);
    #include "createMesh.H"

    // ---- the model the solver ran
    IOdictionary turbProps
    (
        IOobject("turbulenceProperties", runTime.constant(), mesh,
                 IOobject::MUST_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER)
    );
    const dictionary& RASDict = turbProps.subDict("RAS");
    const word modelName
    (
        RASDict.getCompat<word>("model", {{"RASModel", -2006}})
    );
    if (modelName != "kOmegaSSTBasis")
    {
        FatalErrorInFunction
            << "the case ran " << modelName << ", not kOmegaSSTBasis"
            << exit(FatalError);
    }
    const dictionary& coeffs = RASDict.subDict("kOmegaSSTBasisCoeffs");
    if (coeffs.getOrDefault<word>("mode", "expressions") != "expressions")
    {
        FatalErrorInFunction << "only expressions mode is supported" << exit(FatalError);
    }
    if (coeffs.getOrDefault<Switch>("publishedRITA", false))
    {
        FatalErrorInFunction << "publishedRITA is not supported" << exit(FatalError);
    }
    if (coeffs.getOrDefault<Switch>("F3", false))
    {
        FatalErrorInFunction << "F3 is not supported" << exit(FatalError);
    }
    const ExprList caseB(readList(coeffs, "bDelta", "turbulenceProperties"));
    const ExprList caseR(readList(coeffs, "rSource", "turbulenceProperties"));

    forgeTermFields::Controls ctl;
    ctl.gMax = coeffs.getOrDefault<scalar>("gMax", 10.0);
    ctl.rMaxFactor = coeffs.getOrDefault<scalar>("rMaxFactor", 5.0);
    ctl.betaStar = coeffs.getOrDefault<scalar>("betaStar", 0.09);
    ctl.realizabilityClip = coeffs.getOrDefault<Switch>("realizabilityClip", true);
    const bool nonlinearProduction =
        coeffs.getOrDefault<Switch>("nonlinearProduction", true);
    const scalar alphaOmega2 = coeffs.getOrDefault<scalar>("alphaOmega2", 0.856);
    const scalar c1 = coeffs.getOrDefault<scalar>("c1", 10.0);
    const vector frameOmega = coeffs.getOrDefault<vector>("frameOmega", Zero);
    const scalar kMin = RASDict.getOrDefault<scalar>("kMin", SMALL);
    const scalar omegaMin = RASDict.getOrDefault<scalar>("omegaMin", SMALL);

    IOdictionary transportProperties
    (
        IOobject("transportProperties", runTime.constant(), mesh,
                 IOobject::MUST_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER)
    );
    const scalar nu =
        dimensionedScalar("nu", dimViscosity, transportProperties).value();

    // ---- the terms
    IOdictionary termDict
    (
        IOobject("termFieldsDict", runTime.system(), mesh,
                 IOobject::MUST_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER)
    );
    const dictionary& fullDict = termDict.subDict("full");
    const ExprList fullB(readList(fullDict, "bDelta", "full"));
    const ExprList fullR(readList(fullDict, "rSource", "full"));
    requireSame(caseB, fullB, "bDelta");
    requireSame(caseR, fullR, "rSource");
    if (termDict.found("gMax") && mag(termDict.get<scalar>("gMax") - ctl.gMax) > 0)
    {
        FatalErrorInFunction << "gMax differs from the case" << exit(FatalError);
    }
    if
    (
        termDict.found("rMaxFactor")
     && mag(termDict.get<scalar>("rMaxFactor") - ctl.rMaxFactor) > 0
    )
    {
        FatalErrorInFunction << "rMaxFactor differs from the case" << exit(FatalError);
    }

    const wordList termNames(termDict.get<wordList>("termOrder"));
    const dictionary& termsDict = termDict.subDict("terms");
    std::vector<ExprList> termB, termR;
    for (const word& name : termNames)
    {
        const dictionary& d = termsDict.subDict(name);
        termB.push_back(readList(d, "bDelta", name));
        termR.push_back(readList(d, "rSource", name));
        requireTensors(fullB, termB.back(), name + "/bDelta");
        requireTensors(fullR, termR.back(), name + "/rSource");
    }
    const label nTerms = termNames.size();
    label t1Index = -1;
    forAll(fullB.tensorIndex, t)
    {
        if (fullB.tensorIndex[t] == 1) { t1Index = t; break; }
    }

    Info<< "forgeTermFields: model " << termDict.getOrDefault<word>("label", "?")
        << ", terms " << termNames << ", frameOmega " << frameOmega
        << ", nu " << nu << ", gMax " << ctl.gMax << ", rMaxFactor "
        << ctl.rMaxFactor << nl << endl;

    const scalar EPS = 1.0e-3;     // computeV3Features
    const tensor frameTensor
    (
        0.0,             frameOmega.z(), -frameOmega.y(),
       -frameOmega.z(),  0.0,             frameOmega.x(),
        frameOmega.y(), -frameOmega.x(),  0.0
    );
    const dimensionedTensor frameT(dimless/dimTime, frameTensor);

    forAll(timeDirs, timei)
    {
        runTime.setTime(timeDirs[timei], timei);
        Info<< "Time = " << runTime.timeName() << endl;
        mesh.readUpdate();

        auto readField = [&](const word& name, auto* typeTag)
        {
            using FieldType = std::remove_pointer_t<decltype(typeTag)>;
            return FieldType
            (
                IOobject(name, runTime.timeName(), mesh,
                         IOobject::MUST_READ, IOobject::NO_WRITE),
                mesh
            );
        };
        const volVectorField U(readField("U", static_cast<volVectorField*>(nullptr)));
        const volScalarField p(readField("p", static_cast<volScalarField*>(nullptr)));
        const volScalarField k(readField("k", static_cast<volScalarField*>(nullptr)));
        const volScalarField omega(readField("omega", static_cast<volScalarField*>(nullptr)));
        const volScalarField nut(readField("nut", static_cast<volScalarField*>(nullptr)));
        if (p.dimensions() != sqr(dimVelocity))
        {
            FatalErrorInFunction
                << "p must be the kinematic pressure, as in the solver"
                << exit(FatalError);
        }

        const label nCells = mesh.nCells();
        const scalarField& kf = k.primitiveField();
        const scalarField& omf = omega.primitiveField();
        const scalarField& nutf = nut.primitiveField();

        // ---- velocity gradient and normalised tensors (library order)
        const volTensorField gradU(fvc::grad(U));
        const volSymmTensorField Sdim(dev(symm(gradU)));
        const volScalarField omegaSafe
        (
            max(omega, dimensionedScalar(dimless/dimTime, omegaMin))
        );
        const volScalarField tau(dimensionedScalar(dimless, 1.0)/omegaSafe);
        const volSymmTensorField Shat(tau*Sdim);
        const volTensorField What(tau*(skew(gradU) + frameT));
        const scalarField& oms = omegaSafe.primitiveField();
        const symmTensorField& Sd = Sdim.primitiveField();

        // ---- F1 of kOmegaSSTBase (v2312), wall distance as the model
        const volVectorField gradK(fvc::grad(k));
        const volVectorField gradOmega(fvc::grad(omega));
        const volScalarField& y = wallDist::New(mesh).y();
        scalarField F1(nCells), CDkOmega(nCells);
        for (label c = 0; c < nCells; ++c)
        {
            CDkOmega[c] = (2*alphaOmega2)*(gradK[c] & gradOmega[c])/omf[c];
            const scalar CDplus = max(CDkOmega[c], 1.0e-10);
            const scalar yc = y[c];
            const scalar arg1 = min
            (
                min
                (
                    max
                    (
                        (scalar(1)/ctl.betaStar)*std::sqrt(kf[c])/(omf[c]*yc),
                        scalar(500)*nu/(sqr(yc)*omf[c])
                    ),
                    (4*alphaOmega2)*kf[c]/(CDplus*sqr(yc))
                ),
                scalar(10)
            );
            F1[c] = std::tanh(Foam::pow4(arg1));
        }

        // ---- grammar variables (library definitions)
        scalarField Ret(nCells), PoE(nCells);
        const volScalarField magSqrSymm(magSqr(symm(gradU)));
        for (label c = 0; c < nCells; ++c)
        {
            Ret[c] = kf[c]/(nu*oms[c]);
            const scalar G2S = nutf[c]*2.0*magSqrSymm[c];
            PoE[c] = min(G2S/max(ctl.betaStar*kf[c]*oms[c], SMALL), 10.0);
        }
        // FORGE V3 state features (kOmegaSSTBasis::computeV3Features)
        const volVectorField gradP(fvc::grad(p));
        const volVectorField curlU(fvc::curl(U));
        List<scalarField> v3(tedp::expr::N_V3_VARS, scalarField(nCells, 0.0));
        for (label c = 0; c < nCells; ++c)
        {
            const scalar A = oms[c]*std::sqrt(Foam::max(kf[c], kMin));
            const vector a = gradP[c]/A;
            const vector g = gradK[c]/A;
            const scalar ma = mag(a), mg = mag(g);
            const scalar tc = 1.0/oms[c];
            const vector ohat = tc*frameOmega;
            const vector zhat = tc*curlU[c];
            const scalar mo = mag(ohat), mz = mag(zhat);
            const symmTensor& Sh = Shat[c];
            v3[0][c] = ma/(1.0 + ma);
            v3[1][c] = mg/(1.0 + mg);
            v3[2][c] = (a & g)/(ma*mg + EPS);
            v3[3][c] = (a & (Sh & a))/(magSqr(a) + EPS);
            v3[4][c] = (g & (Sh & g))/(magSqr(g) + EPS);
            v3[5][c] = mo;
            v3[6][c] = (ohat & zhat)/(mo*mz + EPS);
        }
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
        vars[5] = Ret.cdata();
        vars[6] = F1.cdata();
        vars[7] = PoE.cdata();
        for (std::size_t i = 0; i < tedp::expr::N_V3_VARS; ++i)
        {
            vars[tedp::expr::FIRST_V3_VAR + i] = v3[i].cdata();
        }
        auto refuseRita = [&](const ExprList& l)
        {
            for (const auto& node : l.node)
            {
                for (std::size_t vi = tedp::expr::FIRST_RITA_VAR;
                     vi < tedp::expr::FIRST_V3_VAR; ++vi)
                {
                    if (node->usesVar(vi))
                    {
                        FatalErrorInFunction
                            << "RITA variables are not supported" << exit(FatalError);
                    }
                }
            }
        };
        refuseRita(fullB); refuseRita(fullR);

        // ---- coefficient functions
        auto evalList = [&](const ExprList& l)
        {
            std::vector<scalarField> vals;
            for (label t = 0; t < l.size(); ++t)
            {
                vals.emplace_back(nCells);
                l.node[t]->eval(vals.back().data(), vars, nCells);
            }
            return vals;
        };
        const std::vector<scalarField> gF(evalList(fullB)), hF(evalList(fullR));
        std::vector<std::vector<scalarField>> gT, hT;
        for (label j = 0; j < nTerms; ++j)
        {
            gT.push_back(evalList(termB[j]));
            hT.push_back(evalList(termR[j]));
        }
        // additivity of the coefficient functions (the split is exact only
        // if the terms add up to the full correction)
        scalar additivity = 0;
        auto checkSum = [&](const std::vector<scalarField>& full,
                            const std::vector<std::vector<scalarField>>& parts)
        {
            for (std::size_t t = 0; t < full.size(); ++t)
            {
                for (label c = 0; c < nCells; ++c)
                {
                    scalar s = 0;
                    for (label j = 0; j < nTerms; ++j) s += parts[j][t][c];
                    additivity = max
                    (
                        additivity, mag(s - full[t][c])/max(scalar(1), mag(full[t][c]))
                    );
                }
            }
        };
        checkSum(gF, gT);
        checkSum(hF, hT);
        reduce(additivity, maxOp<scalar>());
        Info<< "  coefficient additivity residual " << additivity << endl;
        if (additivity > 1.0e-9)
        {
            FatalErrorInFunction
                << "the terms of termFieldsDict do not add up to the full"
                << " correction (residual " << additivity << ")" << exit(FatalError);
        }

        // ---- basis tensors
        auto tensors = [&](const ExprList& l)
        {
            PtrList<volSymmTensorField> T(l.size());
            for (label t = 0; t < l.size(); ++t)
            {
                T.set(t, tedpBasis::basisTensor(l.tensorIndex[t], Shat, What).ptr());
            }
            return T;
        };
        const PtrList<volSymmTensorField> Tb(tensors(fullB)), Tr(tensors(fullR));

        // ---- output fields
        auto sField = [&](const word& name, const dimensionSet& dims)
        {
            return new volScalarField
            (
                IOobject(name, runTime.timeName(), mesh,
                         IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER),
                mesh, dimensionedScalar(dims, Zero),
                fvPatchFieldBase::zeroGradientType()
            );
        };
        auto tField = [&](const word& name)
        {
            return new volSymmTensorField
            (
                IOobject(name, runTime.timeName(), mesh,
                         IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER),
                mesh, dimensionedSymmTensor(sqr(dimVelocity), Zero),
                fvPatchFieldBase::zeroGradientType()
            );
        };
        const dimensionSet dimP(sqr(dimVelocity)/dimTime);
        autoPtr<volSymmTensorField> tauF(tField("tfTauDelta"));
        autoPtr<volScalarField> RF(sField("tfR", dimP));
        autoPtr<volScalarField> GnlF(sField("tfGnl", dimP));
        autoPtr<volScalarField> dPkF(sField("tfDeltaPk", dimP));
        autoPtr<volScalarField> lambdaF(sField("tfLambda", dimless));
        autoPtr<volScalarField> clampF(sField("tfClamp", dimless));
        PtrList<volSymmTensorField> tauJ(nTerms);
        PtrList<volScalarField> RJ(nTerms), GnlJ(nTerms), netJ(nTerms), dnutJ(nTerms);
        PtrList<volScalarField> netJByDk(nTerms), RJByDk(nTerms), RJByPk(nTerms);
        PtrList<volScalarField> GnlJByDk(nTerms), dnutJByNut(nTerms);
        for (label j = 0; j < nTerms; ++j)
        {
            const word& n = termNames[j];
            tauJ.set(j, tField("tfTauDelta" + n));
            RJ.set(j, sField("tfR" + n, dimP));
            GnlJ.set(j, sField("tfGnl" + n, dimP));
            netJ.set(j, sField("tfNet" + n, dimP));
            dnutJ.set(j, sField("tfDnut" + n, dimViscosity));
            netJByDk.set(j, sField("tfNet" + n + "ByDk", dimless));
            RJByDk.set(j, sField("tfR" + n + "ByDk", dimless));
            RJByPk.set(j, sField("tfR" + n + "ByPkSST", dimless));
            GnlJByDk.set(j, sField("tfGnl" + n + "ByDk", dimless));
            dnutJByNut.set(j, sField("tfDnut" + n + "ByNut", dimless));
        }
        autoPtr<volScalarField> F1F(sField("tfF1", dimless));
        autoPtr<volScalarField> yF(sField("tfWallDist", dimLength));
        autoPtr<volScalarField> GF(sField("tfG", dimP));
        autoPtr<volScalarField> PkF(sField("tfPkSST", dimP));
        autoPtr<volScalarField> DkF(sField("tfDk", dimP));
        autoPtr<volScalarField> PiF(sField("tfPi", dimless));
        autoPtr<volScalarField> RfF(sField("tfRf", dimless));
        autoPtr<volScalarField> RwF(sField("tfRw", dimless));
        autoPtr<volScalarField> GkF(sField("tfGk", dimless));
        autoPtr<volScalarField> GpF(sField("tfGp", dimless));
        autoPtr<volScalarField> KsnF(sField("tfKsn", dimless));

        // SST production of v2312: G = nut (gradU && devTwoSymm(gradU)),
        // Pk = min(G, c1 betaStar k omega)
        const volScalarField GbyNu0(gradU && devTwoSymm(gradU));

        label nRealiz = 0, nGClamp = 0, nHClamp = 0, nRBound = 0;
        const scalar tiny = VSMALL;
        for (label c = 0; c < nCells; ++c)
        {
            forgeTermFields::CellInput in;
            in.k = kf[c];
            in.omegaSafe = oms[c];
            in.nut = nutf[c];
            in.Sd = Sd[c];
            for (label t = 0; t < fullB.size(); ++t)
            {
                in.Tb.push_back(Tb[t][c]);
                in.gFull.push_back(gF[t][c]);
            }
            for (label t = 0; t < fullR.size(); ++t)
            {
                in.Tr.push_back(Tr[t][c]);
                in.hFull.push_back(hF[t][c]);
            }
            in.gTerm.resize(nTerms);
            in.hTerm.resize(nTerms);
            for (label j = 0; j < nTerms; ++j)
            {
                for (label t = 0; t < fullB.size(); ++t) in.gTerm[j].push_back(gT[j][t][c]);
                for (label t = 0; t < fullR.size(); ++t) in.hTerm[j].push_back(hT[j][t][c]);
            }
            const forgeTermFields::CellOutput out =
                forgeTermFields::evaluateCell(in, ctl, t1Index);

            const scalar G = nutf[c]*GbyNu0[c];
            const scalar Dk = ctl.betaStar*kf[c]*omf[c];
            const scalar PkSST = min(G, c1*Dk);
            const scalar Gnl = nonlinearProduction ? out.Gnl : 0.0;
            const scalar PkCorr = nonlinearProduction ? min(G + Gnl, c1*Dk) : PkSST;

            tauF()[c] = out.tau;
            RF()[c] = out.R;
            GnlF()[c] = out.Gnl;
            dPkF()[c] = PkCorr + out.R - PkSST;
            lambdaF()[c] = out.lambda;
            clampF()[c] = (out.gClamped ? 1 : 0) + (out.hClamped ? 2 : 0) + (out.rBound ? 4 : 0);
            nRealiz += out.realizabilityActive;
            nGClamp += out.gClamped;
            nHClamp += out.hClamped;
            nRBound += out.rBound;
            for (label j = 0; j < nTerms; ++j)
            {
                const scalar GnlJc = nonlinearProduction ? out.GnlTerm[j] : 0.0;
                tauJ[j][c] = out.tauTerm[j];
                RJ[j][c] = out.RTerm[j];
                GnlJ[j][c] = out.GnlTerm[j];
                netJ[j][c] = GnlJc + out.RTerm[j];
                dnutJ[j][c] = -out.g1Term[j]*kf[c]/oms[c];
                netJByDk[j][c] = netJ[j][c]/max(Dk, tiny);
                RJByDk[j][c] = out.RTerm[j]/max(Dk, tiny);
                RJByPk[j][c] = out.RTerm[j]/max(PkSST, tiny);
                GnlJByDk[j][c] = out.GnlTerm[j]/max(Dk, tiny);
                dnutJByNut[j][c] = dnutJ[j][c]/max(nutf[c], tiny);
            }
            F1F()[c] = F1[c];
            yF()[c] = y[c];
            GF()[c] = G;
            PkF()[c] = PkSST;
            DkF()[c] = Dk;
            PiF()[c] = PoE[c];
            RfF()[c] = v3[5][c];
            RwF()[c] = v3[6][c];
            GkF()[c] = v3[1][c];
            GpF()[c] = v3[0][c];
            KsnF()[c] = v3[4][c];
        }

        // ---- consistency with the stress the solver wrote
        const scalar V = gSum(mesh.V());
        scalar consistencyMax = -1, consistencyRms = -1, storedMax = 0;
        IOobject nsHeader
        (
            "nonlinearStress", runTime.timeName(), mesh,
            IOobject::MUST_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER
        );
        if (nsHeader.typeHeaderOk<volSymmTensorField>(true))
        {
            const volSymmTensorField ns(nsHeader, mesh);
            const symmTensorField diff(tauF().primitiveField() - ns.primitiveField());
            storedMax = gMax(mag(ns.primitiveField())());
            consistencyMax = gMax(mag(diff)()) / max(storedMax, VSMALL);
            consistencyRms = std::sqrt
            (
                gSum(magSqr(diff)*mesh.V())
               /max(gSum(magSqr(ns.primitiveField())*mesh.V()), VSMALL)
            );
            Info<< "  recomputed stress against nonlinearStress: max "
                << consistencyMax << ", rms " << consistencyRms
                << " (relative)" << endl;
            if (!args.found("noConsistency") && consistencyRms > 1.0e-2)
            {
                FatalErrorInFunction
                    << "the recomputed stress differs from the converged"
                    << " nonlinearStress by " << consistencyRms
                    << " (rms, relative); the inputs are not those the solver"
                    << " used. Use -noConsistency to write the fields anyway."
                    << exit(FatalError);
            }
        }
        else
        {
            Info<< "  no nonlinearStress at this time: consistency not checked" << endl;
        }

        // ---- write
        auto writeS = [](volScalarField& f) { f.correctBoundaryConditions(); f.write(); };
        auto writeT = [](volSymmTensorField& f) { f.correctBoundaryConditions(); f.write(); };
        writeT(tauF()); writeS(RF()); writeS(GnlF()); writeS(dPkF());
        writeS(lambdaF()); writeS(clampF());
        for (label j = 0; j < nTerms; ++j)
        {
            writeT(tauJ[j]); writeS(RJ[j]); writeS(GnlJ[j]); writeS(netJ[j]);
            writeS(dnutJ[j]); writeS(netJByDk[j]); writeS(RJByDk[j]);
            writeS(RJByPk[j]); writeS(GnlJByDk[j]); writeS(dnutJByNut[j]);
        }
        writeS(F1F()); writeS(yF()); writeS(GF()); writeS(PkF()); writeS(DkF());
        writeS(PiF()); writeS(RfF()); writeS(RwF()); writeS(GkF()); writeS(GpF());
        writeS(KsnF());
        if (args.found("writeInputs"))
        {
            auto writeAs = [&](const word& name, const auto& f)
            {
                using FieldType = std::decay_t<decltype(f)>;
                FieldType out
                (
                    IOobject(name, runTime.timeName(), mesh,
                             IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER),
                    f
                );
                out.write();
            };
            writeAs("tfGradU", gradU);
            writeAs("tfGradK", gradK);
            writeAs("tfGradOmega", gradOmega);
            writeAs("tfGradP", gradP);
            volVectorField C
            (
                IOobject("tfC", runTime.timeName(), mesh,
                         IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER),
                mesh.C()
            );
            C.write();
            autoPtr<volScalarField> VF(sField("tfV", dimVolume));
            VF().primitiveFieldRef() = mesh.V();
            writeS(VF());
        }

        // ---- summary (volume integrals and extremes, for the metadata)
        const scalarField& Vc = mesh.V();
        const label nGlobal = returnReduce(nCells, sumOp<label>());
        reduce(nRealiz, sumOp<label>());
        reduce(nGClamp, sumOp<label>());
        reduce(nHClamp, sumOp<label>());
        reduce(nRBound, sumOp<label>());
        // every reduction on every rank, before the master writes
        const scalar PkInt = gSum(PkF().primitiveField()*Vc);
        const scalar DkInt = gSum(DkF().primitiveField()*Vc);
        const scalar RInt = gSum(RF().primitiveField()*Vc);
        const scalar GnlInt = gSum(GnlF().primitiveField()*Vc);
        const scalar dPkInt = gSum(dPkF().primitiveField()*Vc);
        List<scalarList> termStats(nTerms, scalarList(8, Zero));
        for (label j = 0; j < nTerms; ++j)
        {
            termStats[j][0] = gSum(RJ[j].primitiveField()*Vc);
            termStats[j][1] = gSum(GnlJ[j].primitiveField()*Vc);
            termStats[j][2] = gSum(netJ[j].primitiveField()*Vc);
            termStats[j][3] = gMax(netJByDk[j].primitiveField());
            termStats[j][4] = gMin(netJByDk[j].primitiveField());
            termStats[j][5] = gMax(mag(tauJ[j].primitiveField())());
            termStats[j][6] = gMax(dnutJByNut[j].primitiveField());
            termStats[j][7] = gMin(dnutJByNut[j].primitiveField());
        }
        const word statNames[8] =
        {
            "integral_R", "integral_Gnl", "integral_net", "max_net_by_Dk",
            "min_net_by_Dk", "max_abs_tau", "max_dnut_by_nut", "min_dnut_by_nut"
        };
        fileName dir(runTime.globalPath()/"postProcessing"/"forgeTermFields"/runTime.timeName());
        if (Pstream::master())
        {
            mkDir(dir);
            OFstream os(dir/"summary.json");
            os  << "{\n"
                << "  \"utility\": \"forgeTermFields\",\n"
                << "  \"label\": \"" << termDict.getOrDefault<word>("label", "?") << "\",\n"
                << "  \"time\": \"" << runTime.timeName() << "\",\n"
                << "  \"cells\": " << nGlobal << ",\n"
                << "  \"volume\": " << num(V) << ",\n"
                << "  \"nu\": " << num(nu) << ",\n"
                << "  \"frame_omega\": [" << num(frameOmega.x()) << ", "
                << num(frameOmega.y()) << ", " << num(frameOmega.z()) << "],\n"
                << "  \"coefficient_additivity_residual\": " << num(additivity) << ",\n"
                << "  \"consistency_vs_nonlinearStress\": {\"max_relative\": "
                << num(consistencyMax) << ", \"rms_relative\": " << num(consistencyRms)
                << ", \"stored_max\": " << num(storedMax) << "},\n"
                << "  \"cells_realizability_active\": " << nRealiz << ",\n"
                << "  \"cells_stress_coefficient_clamped\": " << nGClamp << ",\n"
                << "  \"cells_source_coefficient_clamped\": " << nHClamp << ",\n"
                << "  \"cells_R_bound_active\": " << nRBound << ",\n"
                << "  \"integral_PkSST\": " << num(PkInt) << ",\n"
                << "  \"integral_Dk\": " << num(DkInt) << ",\n"
                << "  \"integral_R\": " << num(RInt) << ",\n"
                << "  \"integral_Gnl\": " << num(GnlInt) << ",\n"
                << "  \"integral_DeltaPk\": " << num(dPkInt) << ",\n"
                << "  \"terms\": {";
            for (label j = 0; j < nTerms; ++j)
            {
                os  << (j ? "," : "") << "\n    \"" << termNames[j] << "\": {";
                for (label s = 0; s < 8; ++s)
                {
                    os  << (s ? ", " : "") << "\"" << statNames[s] << "\": "
                        << num(termStats[j][s]);
                }
                os  << "}";
            }
            os  << "\n  }\n}\n";
        }
        Info<< "  realizability active in " << nRealiz << " cells, coefficient clamp "
            << nGClamp << "/" << nHClamp << ", R bound " << nRBound << " of " << nGlobal
            << nl << endl;
    }

    Info<< "End" << nl << endl;
    return 0;
}
