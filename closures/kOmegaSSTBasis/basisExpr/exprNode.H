/*---------------------------------------------------------------------------*\
    basisExpr — the frozen candidate-expression grammar (C++ side).

    Plain C++17, no OpenFOAM dependencies: the turbulence model evaluates on
    scalarField::data() pointers; testBasisExpr compiles with bare g++ to emit
    the golden files that pin this implementation to the Python twin
    (src/tedp/expr.py) at 1e-12.

    Grammar v1 and guarded semantics are documented in tedp/expr.py — the two
    files change together or not at all (frozen at end of Phase 1).
\*---------------------------------------------------------------------------*/

#ifndef tedp_basisExpr_exprNode_H
#define tedp_basisExpr_exprNode_H

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace tedp
{
namespace expr
{

constexpr double GUARD_EPS = 1.0e-12;
constexpr double EXP_CLAMP = 50.0;

// Variable indexing shared with the parser, testBasisExpr and tedp/expr.py:
//   0..4  I1..I5,  5 Ret,  6 F1,  7 PoE,
//   8..12 RITA comparator variables (sigmaSL, phiDkPk, phiDkCk, phik,
//         ReOmega) -- these exist only so the zonally-augmented model of
//         Buchanan et al. (arXiv:2504.06758) can be expressed in the same
//         grammar; no discovered model references them and their fields are
//         built only when an expression asks for one,
//   13..19 FORGE V3 physical-state features (Gp, Gk, Apk, Psn, Ksn, Rf, Rw),
//         defined in tedp/features.py and computed by
//         kOmegaSSTBasis::computeV3Features only when an expression uses one,
//   20..27 c0..c7 (free constants, accepted only when the parser is built
//         with allowFreeConstants=true).
constexpr std::size_t N_STATE_VARS = 20;
constexpr std::size_t FIRST_RITA_VAR = 8;
constexpr std::size_t FIRST_V3_VAR = 13;
constexpr std::size_t N_V3_VARS = 7;
constexpr std::size_t N_ALL_VARS = 28;

class ExprNode
{
public:
    enum class Kind
    {
        Num, Var, Neg, Add, Sub, Mul, Div, Tanh, Exp, Sqrt, Abs, Min, Max
    };

    Kind kind;
    double value = 0.0;          // Kind::Num
    std::size_t varIndex = 0;    // Kind::Var
    std::unique_ptr<ExprNode> a; // first child
    std::unique_ptr<ExprNode> b; // second child (binary ops, min, max)

    explicit ExprNode(Kind k) : kind(k) {}

    //- Evaluate on n contiguous points; vars[i] may be nullptr only if the
    //  expression never references variable i. out must hold n doubles.
    void eval
    (
        double* out,
        const double* const vars[N_ALL_VARS],
        std::size_t n
    ) const;

    //- Single-point evaluation (golden files, spot checks).
    double evalPoint(const double vars[N_ALL_VARS]) const;

    //- True if the subtree references variable index i.
    bool usesVar(std::size_t i) const;
};

} // namespace expr
} // namespace tedp

#endif
