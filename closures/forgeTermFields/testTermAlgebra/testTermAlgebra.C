/*---------------------------------------------------------------------------*\
    testTermAlgebra -- the per-cell algebra of forgeTermFields on given states,
    without a mesh or a case.

    Usage: testTermAlgebra <termFieldsDict> <states.txt> <out.txt>

    Parses the coefficient functions of termFieldsDict with the library's
    grammar (termDict.H), evaluates them at each state (ExprNode::evalPoint)
    and splits the correction with termAlgebra.H, the code forgeTermFields
    uses. scripts/termfields_dict.py writes the states and compares the
    output with its NumPy twin.

    states.txt: header "nStates nVars nb nr nTerms", then
                "gMax rMaxFactor betaStar realizabilityClip t1Index", then one
                line per state: vars[nVars] k omegaSafe nut Sd[6] Tb[nb][6]
                Tr[nr][6] (symmTensor order xx xy xz yy yz zz).
    out.txt:    one line per state: lambda realizability gClamp hClamp rBound
                tau[6] R Gnl gFull[nb] hFull[nr], then per term:
                gTerm[nb] hTerm[nr] tauTerm[6] RTerm GnlTerm g1Term.
\*---------------------------------------------------------------------------*/

#include "IFstream.H"
#include "OFstream.H"
#include "dictionary.H"
#include "wordList.H"

#include "../exprSources.C"
#include "termAlgebra.H"
#include "termDict.H"

#include <fstream>
#include <iomanip>

using namespace Foam;
using namespace Foam::forgeTermFields;

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cerr << "usage: testTermAlgebra <termFieldsDict> <states> <out>\n";
        return 2;
    }
    IFstream dictStream(argv[1]);
    const dictionary dict(dictStream);
    const ExprList fullB(readList(dict.subDict("full"), "bDelta", "full"));
    const ExprList fullR(readList(dict.subDict("full"), "rSource", "full"));
    const wordList names(dict.get<wordList>("termOrder"));
    std::vector<ExprList> termB, termR;
    for (const word& n : names)
    {
        termB.push_back(readList(dict.subDict("terms").subDict(n), "bDelta", n));
        termR.push_back(readList(dict.subDict("terms").subDict(n), "rSource", n));
    }

    std::ifstream in(argv[2]);
    long nStates, nVars, nb, nr, nTerms;
    in >> nStates >> nVars >> nb >> nr >> nTerms;
    Controls ctl;
    int clip;
    long t1;
    in >> ctl.gMax >> ctl.rMaxFactor >> ctl.betaStar >> clip >> t1;
    ctl.realizabilityClip = clip != 0;
    if
    (
        nb != fullB.size() || nr != fullR.size() || nTerms != long(names.size())
     || nVars != long(tedp::expr::N_STATE_VARS)
    )
    {
        std::cerr << "state file does not match the dictionary\n";
        return 3;
    }

    auto readT = [&](symmTensor& T)
    {
        in >> T.xx() >> T.xy() >> T.xz() >> T.yy() >> T.yz() >> T.zz();
    };
    std::ofstream out(argv[3]);
    out << std::setprecision(17);
    auto writeT = [&](const symmTensor& T)
    {
        out << ' ' << T.xx() << ' ' << T.xy() << ' ' << T.xz() << ' '
            << T.yy() << ' ' << T.yz() << ' ' << T.zz();
    };
    for (long s = 0; s < nStates; ++s)
    {
        double vars[tedp::expr::N_ALL_VARS] = {0};
        for (long v = 0; v < nVars; ++v) in >> vars[v];
        CellInput c;
        in >> c.k >> c.omegaSafe >> c.nut;
        readT(c.Sd);
        c.Tb.resize(nb);
        c.Tr.resize(nr);
        for (auto& T : c.Tb) readT(T);
        for (auto& T : c.Tr) readT(T);
        for (long t = 0; t < nb; ++t) c.gFull.push_back(fullB.node[t]->evalPoint(vars));
        for (long t = 0; t < nr; ++t) c.hFull.push_back(fullR.node[t]->evalPoint(vars));
        c.gTerm.resize(nTerms);
        c.hTerm.resize(nTerms);
        for (long j = 0; j < nTerms; ++j)
        {
            for (long t = 0; t < nb; ++t) c.gTerm[j].push_back(termB[j].node[t]->evalPoint(vars));
            for (long t = 0; t < nr; ++t) c.hTerm[j].push_back(termR[j].node[t]->evalPoint(vars));
        }
        const CellOutput o = evaluateCell(c, ctl, label(t1));
        out << o.lambda << ' ' << int(o.realizabilityActive) << ' ' << int(o.gClamped)
            << ' ' << int(o.hClamped) << ' ' << int(o.rBound);
        writeT(o.tau);
        out << ' ' << o.R << ' ' << o.Gnl;
        for (const scalar g : c.gFull) out << ' ' << g;
        for (const scalar h : c.hFull) out << ' ' << h;
        for (long j = 0; j < nTerms; ++j)
        {
            for (const scalar g : c.gTerm[j]) out << ' ' << g;
            for (const scalar h : c.hTerm[j]) out << ' ' << h;
            writeT(o.tauTerm[j]);
            out << ' ' << o.RTerm[j] << ' ' << o.GnlTerm[j] << ' ' << o.g1Term[j];
        }
        out << '\n';
    }
    if (!in)
    {
        std::cerr << "state file ended early\n";
        return 4;
    }
    return 0;
}
