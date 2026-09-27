#ifndef tedp_basisExpr_exprParser_H
#define tedp_basisExpr_exprParser_H

#include "exprNode.H"

#include <stdexcept>

namespace tedp
{
namespace expr
{

//- Thrown on any token or construct outside the frozen grammar.
//  The turbulence model converts this into a Foam::FatalError at startup.
class GrammarError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

//- Recursive-descent parser for grammar v1 (see tedp/expr.py).
//  allowFreeConstants: accept c0..c7 as variables 8..15 (testBasisExpr);
//  the turbulence model parses with false — candidates arrive with literals
//  already substituted by the Python driver.
std::unique_ptr<ExprNode> parse
(
    const std::string& source,
    bool allowFreeConstants = false
);

} // namespace expr
} // namespace tedp

#endif
