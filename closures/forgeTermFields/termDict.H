/*---------------------------------------------------------------------------*\
    termDict.H -- coefficient-function lists of system/termFieldsDict and of
    kOmegaSSTBasisCoeffs, parsed with the library's frozen grammar (the same
    "terms ( (Tn "expr") ... );" layout as kOmegaSSTBasis::parseChannel).
\*---------------------------------------------------------------------------*/

#ifndef forgeTermDict_H
#define forgeTermDict_H

#include "dictionary.H"
#include "Tuple2.H"
#include "basisExpr/exprParser.H"

#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace Foam
{
namespace forgeTermFields
{

struct ExprList
{
    List<label> tensorIndex;
    List<string> text;
    std::vector<std::unique_ptr<tedp::expr::ExprNode>> node;
    label size() const { return tensorIndex.size(); }
};

inline ExprList readList(const dictionary& parent, const word& name, const word& where)
{
    ExprList out;
    if (!parent.found(name))
    {
        return out;
    }
    List<Tuple2<word, string>> terms;
    const dictionary& sub = parent.subDict(name);
    sub.readIfPresent("terms", terms);
    for (const auto& term : terms)
    {
        const word& tname = term.first();
        const label n = (tname.size() >= 2 && tname[0] == 'T')
            ? label(std::atoi(tname.substr(1).c_str())) : label(-1);
        if (n < 1 || n > 10)
        {
            FatalIOErrorInFunction(sub)
                << where << ": bad basis tensor " << tname << exit(FatalIOError);
        }
        try
        {
            out.node.push_back(tedp::expr::parse(std::string(term.second()), false));
        }
        catch (const tedp::expr::GrammarError& e)
        {
            FatalIOErrorInFunction(sub)
                << where << "/" << name << ", term " << tname << ": " << e.what()
                << exit(FatalIOError);
        }
        out.tensorIndex.append(n);
        out.text.append(term.second());
    }
    return out;
}

inline std::string squeeze(const std::string& s)
{
    std::string r;
    for (const char c : s)
    {
        if (!std::isspace(static_cast<unsigned char>(c))) r += c;
    }
    return r;
}


} // namespace forgeTermFields
} // namespace Foam

#endif
