/*---------------------------------------------------------------------------*\
    termAlgebra.H -- per-cell split of the correction into its terms.

    Post-processing only. This header repeats, for one cell, the operations
    that libkOmegaSSTBasis (src/kOmegaSSTBasis/kOmegaSSTBasis.C,
    updateCorrections) applies to the full coefficient functions at a
    converged state (ramp = 1, relaxation at its fixed point):

      g_t   = clamp(g_t, -gMax, gMax)               every stress term t
      b     = sum_t g_t T_t
      lambda: 1 if -nut S/k + b lies in [-1/3, 2/3] (tolerance 1e-6), else
              six bisection steps on [0, 1], lower endpoint kept
      tau   = 2 k lambda b
      h_t   = clamp(h_t, -gMax, gMax)               every source term t
      Rraw  = 2 k (sum_t h_t T_t) : S
      R     = clamp(Rraw, -rMax betaStar k omega_s, +rMax betaStar k omega_s)
      Gnl   = -tau : S

    and splits tau and R exactly into the contributions of the terms
    j = 1..nTerms, whose coefficient functions add up to the full ones
    (g_t = sum_j g_t^j). Where a clamp or the R bound is active, each term
    keeps its share of the unclamped value, so that the contributions always
    sum to the full correction:

      tau^j = 2 k lambda sum_t s_t g_t^j T_t,   s_t = clamp(g_t)/g_t (else 1)
      R^j   = sigma 2 k (sum_t q_t h_t^j T_t) : S,
              q_t = clamp(h_t)/h_t, sigma = R/Rraw (else 1)

    Plain C++ on OpenFOAM primitive types (symmTensor, eigenValues); no mesh,
    so testTermAlgebra can exercise it on random states.
\*---------------------------------------------------------------------------*/

#ifndef forgeTermAlgebra_H
#define forgeTermAlgebra_H

#include "symmTensor.H"
#include "vector.H"

#include <algorithm>
#include <cmath>
#include <vector>

namespace Foam
{
namespace forgeTermFields
{

struct Controls
{
    scalar gMax = 10.0;
    scalar rMaxFactor = 5.0;
    scalar betaStar = 0.09;
    bool realizabilityClip = true;
};

struct CellInput
{
    scalar k = 0;
    scalar omegaSafe = 0;      // max(omega, omegaMin), as the library
    scalar nut = 0;
    symmTensor Sd = Zero;      // dev(symm(gradU))
    // basis tensors of the stress terms and of the source terms, in the
    // order of the full coefficient lists
    std::vector<symmTensor> Tb;
    std::vector<symmTensor> Tr;
    // full coefficient functions (unclamped), one per term of the lists
    std::vector<scalar> gFull;
    std::vector<scalar> hFull;
    // coefficient functions of each physical term: gTerm[j][t], hTerm[j][t]
    std::vector<std::vector<scalar>> gTerm;
    std::vector<std::vector<scalar>> hTerm;
};

struct CellOutput
{
    scalar lambda = 1;
    bool realizabilityActive = false;
    bool gClamped = false;
    bool hClamped = false;
    bool rBound = false;
    symmTensor tau = Zero;               // 2 k lambda b, full correction
    scalar R = 0;                        // bounded explicit source, full
    scalar Gnl = 0;                      // -tau : S, full
    std::vector<symmTensor> tauTerm;     // per physical term
    std::vector<scalar> RTerm;
    std::vector<scalar> GnlTerm;
    std::vector<scalar> g1Term;          // effective T1 stress coefficient
                                         // lambda s_1 g_1^j (0 if no T1)
};

inline bool inTriangle(const symmTensor& b, const scalar tol = 1.0e-6)
{
    const vector ev = eigenValues(b);
    return ev.x() >= -1.0/3.0 - tol && ev.z() <= 2.0/3.0 + tol;
}

inline scalar clampTo(const scalar x, const scalar bound)
{
    return std::max(-bound, std::min(bound, x));
}

//- Split one cell. t1Index: position of the T1 term in the stress list
//  (-1 if none), for the eddy-viscosity form of the T1 stress channel.
inline CellOutput evaluateCell
(
    const CellInput& in,
    const Controls& ctl,
    const label t1Index = -1
)
{
    const label nb = label(in.gFull.size());
    const label nr = label(in.hFull.size());
    const label nTerms = label(in.gTerm.size());

    CellOutput out;
    out.tauTerm.assign(nTerms, symmTensor::zero);
    out.RTerm.assign(nTerms, 0.0);
    out.GnlTerm.assign(nTerms, 0.0);
    out.g1Term.assign(nTerms, 0.0);

    // ---- stress: clamp each coefficient, share of the clamp per term
    std::vector<scalar> sb(nb, 1.0);
    symmTensor b(Zero);
    for (label t = 0; t < nb; ++t)
    {
        const scalar g = in.gFull[t];
        const scalar gc = clampTo(g, ctl.gMax);
        if (std::abs(g) > ctl.gMax)
        {
            out.gClamped = true;
            sb[t] = gc/g;
        }
        b += gc*in.Tb[t];
    }

    // ---- realizability factor of the library (same test, same bisection)
    scalar lambda = 1.0;
    if (ctl.realizabilityClip && nb > 0)
    {
        const symmTensor bB = -(in.nut/std::max(in.k, SMALL))*in.Sd;
        if (!inTriangle(bB + b))
        {
            out.realizabilityActive = true;
            scalar lo = 0.0, hi = 1.0;
            for (int it = 0; it < 6; ++it)
            {
                const scalar mid = 0.5*(lo + hi);
                if (inTriangle(bB + mid*b)) lo = mid; else hi = mid;
            }
            lambda = lo;
        }
    }
    out.lambda = lambda;
    out.tau = 2.0*in.k*(lambda*b);
    out.Gnl = -(out.tau && in.Sd);

    for (label j = 0; j < nTerms; ++j)
    {
        symmTensor bj(Zero);
        for (label t = 0; t < nb; ++t)
        {
            bj += (sb[t]*in.gTerm[j][t])*in.Tb[t];
        }
        out.tauTerm[j] = 2.0*in.k*(lambda*bj);
        out.GnlTerm[j] = -(out.tauTerm[j] && in.Sd);
        if (t1Index >= 0 && t1Index < nb)
        {
            out.g1Term[j] = lambda*sb[t1Index]*in.gTerm[j][t1Index];
        }
    }

    // ---- explicit source: clamp each coefficient, then the R bound
    if (nr > 0)
    {
        std::vector<scalar> sr(nr, 1.0);
        symmTensor bR(Zero);
        for (label t = 0; t < nr; ++t)
        {
            const scalar h = in.hFull[t];
            const scalar hc = clampTo(h, ctl.gMax);
            if (std::abs(h) > ctl.gMax)
            {
                out.hClamped = true;
                sr[t] = hc/h;
            }
            bR += hc*in.Tr[t];
        }
        const scalar Rraw = 2.0*in.k*(bR && in.Sd);
        const scalar bnd = ctl.rMaxFactor*ctl.betaStar*in.k*in.omegaSafe;
        out.R = clampTo(Rraw, bnd);
        scalar sigma = 1.0;
        if (std::abs(Rraw) > bnd)
        {
            out.rBound = true;
            sigma = out.R/Rraw;
        }
        for (label j = 0; j < nTerms; ++j)
        {
            symmTensor bj(Zero);
            for (label t = 0; t < nr; ++t)
            {
                bj += (sr[t]*in.hTerm[j][t])*in.Tr[t];
            }
            out.RTerm[j] = sigma*2.0*in.k*(bj && in.Sd);
        }
    }

    return out;
}

} // namespace forgeTermFields
} // namespace Foam

#endif
