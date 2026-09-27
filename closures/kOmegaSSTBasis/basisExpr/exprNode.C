#include "exprNode.H"

#include <algorithm>

namespace tedp
{
namespace expr
{

double ExprNode::evalPoint(const double vars[N_ALL_VARS]) const
{
    switch (kind)
    {
        case Kind::Num:  return value;
        case Kind::Var:  return vars[varIndex];
        case Kind::Neg:  return -a->evalPoint(vars);
        case Kind::Add:  return a->evalPoint(vars) + b->evalPoint(vars);
        case Kind::Sub:  return a->evalPoint(vars) - b->evalPoint(vars);
        case Kind::Mul:  return a->evalPoint(vars) * b->evalPoint(vars);
        case Kind::Div:
        {
            const double x = a->evalPoint(vars);
            const double y = b->evalPoint(vars);
            return (x*y)/(y*y + GUARD_EPS);
        }
        case Kind::Tanh: return std::tanh(a->evalPoint(vars));
        case Kind::Exp:  return std::exp(std::min(a->evalPoint(vars), EXP_CLAMP));
        case Kind::Sqrt: return std::sqrt(std::max(a->evalPoint(vars), 0.0));
        case Kind::Abs:  return std::abs(a->evalPoint(vars));
        case Kind::Min:  return std::min(a->evalPoint(vars), b->evalPoint(vars));
        case Kind::Max:  return std::max(a->evalPoint(vars), b->evalPoint(vars));
    }
    return 0.0;
}

void ExprNode::eval
(
    double* out,
    const double* const vars[N_ALL_VARS],
    std::size_t n
) const
{
    switch (kind)
    {
        case Kind::Num:
            std::fill(out, out + n, value);
            return;
        case Kind::Var:
        {
            const double* v = vars[varIndex];
            std::copy(v, v + n, out);
            return;
        }
        case Kind::Neg:
            a->eval(out, vars, n);
            for (std::size_t i = 0; i < n; ++i) out[i] = -out[i];
            return;
        case Kind::Tanh:
            a->eval(out, vars, n);
            for (std::size_t i = 0; i < n; ++i) out[i] = std::tanh(out[i]);
            return;
        case Kind::Exp:
            a->eval(out, vars, n);
            for (std::size_t i = 0; i < n; ++i)
                out[i] = std::exp(std::min(out[i], EXP_CLAMP));
            return;
        case Kind::Sqrt:
            a->eval(out, vars, n);
            for (std::size_t i = 0; i < n; ++i)
                out[i] = std::sqrt(std::max(out[i], 0.0));
            return;
        case Kind::Abs:
            a->eval(out, vars, n);
            for (std::size_t i = 0; i < n; ++i) out[i] = std::abs(out[i]);
            return;
        default:
            break;
    }

    // binary nodes: evaluate right child into scratch
    std::vector<double> rhs(n);
    a->eval(out, vars, n);
    b->eval(rhs.data(), vars, n);
    switch (kind)
    {
        case Kind::Add:
            for (std::size_t i = 0; i < n; ++i) out[i] += rhs[i];
            return;
        case Kind::Sub:
            for (std::size_t i = 0; i < n; ++i) out[i] -= rhs[i];
            return;
        case Kind::Mul:
            for (std::size_t i = 0; i < n; ++i) out[i] *= rhs[i];
            return;
        case Kind::Div:
            for (std::size_t i = 0; i < n; ++i)
                out[i] = (out[i]*rhs[i])/(rhs[i]*rhs[i] + GUARD_EPS);
            return;
        case Kind::Min:
            for (std::size_t i = 0; i < n; ++i) out[i] = std::min(out[i], rhs[i]);
            return;
        case Kind::Max:
            for (std::size_t i = 0; i < n; ++i) out[i] = std::max(out[i], rhs[i]);
            return;
        default:
            return;
    }
}

bool ExprNode::usesVar(std::size_t i) const
{
    if (kind == Kind::Var) return varIndex == i;
    if (a && a->usesVar(i)) return true;
    if (b && b->usesVar(i)) return true;
    return false;
}

} // namespace expr
} // namespace tedp
