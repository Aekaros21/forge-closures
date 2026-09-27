// Scalar algebra of Buchanan et al., Computers & Fluids 305 (2026) 106899.
// DOI: 10.1016/j.compfluid.2025.106899, pp.4,6,8, Eqs.7-14 and 23-32.
// No OpenFOAM or frozen-expression-parser dependency. These functions do not
// implement transport, momentum discretization, ramps, or numerical limiters.
#ifndef TEDP_PUBLISHED_RITA_H
#define TEDP_PUBLISHED_RITA_H

#include <algorithm>
#include <cmath>

namespace tedpRita
{
    // Continuous magnitude ratio; the physically empty 0/0 budget is assigned 0.
    inline double magnitudeRatio(double numerator, double other)
    {
        const double a = std::abs(numerator), b = std::abs(other);
        const double scale = std::max(a, b);
        if (scale == 0.0) return 0.0;
        // Scaling avoids overflow without perturbing small, nonzero budgets.
        return (a/scale)/(a/scale + b/scale);
    }

    // Pk before the additive learned source, using the standard SST 10*epsilon
    // production ceiling. destruction is betaStar*k*omega, not omega*k.
    inline double limitedProduction(double rawPk, double destruction)
    {
        return std::min(rawPk, 10.0*destruction);
    }

    // Final paper p.4: positive R augments production; negative R augments
    // destruction in the classifier. previousAppliedR is the ACTUALLY APPLIED
    // source from the preceding update, including its prior sigmaSL gate.
    // This explicit lag resolves otherwise circular dynamic classification.
    inline double phiDkPk
    (
        double rawPk, double destruction, double previousAppliedR = 0.0
    )
    {
        const double production = limitedProduction(rawPk, destruction)
                                + std::max(previousAppliedR, 0.0);
        const double destructionBudget = destruction
                                       + std::max(-previousAppliedR, 0.0);
        return magnitudeRatio(destructionBudget, production);
    }

    // Ck = U dot grad(k) for the steady RANS setting. The source feedback above
    // is specified for phiDkPk only; this feature retains the base destruction.
    inline double phiDkCk(double destruction, double convection)
    {
        return magnitudeRatio(destruction, convection);
    }

    // velocitySquared = |U|^2. This feature is not Galilean invariant, as the
    // final paper explicitly acknowledges; comparisons need one reference frame.
    inline double phiK(double k, double velocitySquared)
    {
        return magnitudeRatio(k, 0.5*velocitySquared);
    }

    // The min and max MUST be case-wide extrema, including MPI reductions.
    // A constant ReOmega field has no distinguished high-vorticity region.
    inline double normalizedRe(double rawRe, double caseMin, double caseMax)
    {
        const double span = caseMax - caseMin;
        return span == 0.0 ? 0.0 : (rawRe - caseMin)/span;
    }

    inline double sigmaSL(double destructionProductionRatio, double tkeRatio,
                          double normalizedVorticityRe)
    {
        return destructionProductionRatio <= 0.55 && tkeRatio >= 0.12
            && normalizedVorticityRe >= 0.02 ? 1.0 : 0.0;
    }

    // Algebraically x/(1+x*x), evaluated without overflow for large |x|.
    // This is ordinary arithmetic, without an added epsilon in the denominator.
    inline double rationalPeak(double x)
    {
        if (std::abs(x) <= 1.0) return x/(1.0 + x*x);
        const double reciprocal = 1.0/x;
        return reciprocal/(1.0 + reciprocal*reciprocal);
    }

    // Eq.26; T2 = Shat*Omegahat - Omegahat*Shat, I2 = tr(Omegahat^2).
    // Ungated coefficient: multiply sigmaSL only at the appropriate equation.
    inline double bDeltaT2Paper(double destructionProductionRatio, double I2)
    {
        return 3.69*destructionProductionRatio
             - 5.092*rationalPeak(I2/0.01247);
    }

    // OpenFOAM's grad(U) index ordering reverses Omega relative to the paper.
    inline double bDeltaT2Solver(double destructionProductionRatio, double I2)
    {
        return -bDeltaT2Paper(destructionProductionRatio, I2);
    }

    // Eq.25; multiply by epsilon=betaStar*k*omega and sigmaSL for applied R.
    // In the physical feature range 0<=phiDkCk<=1, 0<=R/epsilon<=0.213.
    inline double rOverEpsilon(double destructionConvectionRatio)
    {
        return 0.426*rationalPeak(destructionConvectionRatio/0.1248);
    }
}

#endif
