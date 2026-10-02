/*---------------------------------------------------------------------------*\
    testAutoTurbSource: unit check of the AutoTurb production correction on
    manufactured states, without a mesh (no CFD).

    Usage: testAutoTurbSource <states.txt> <alpha1Sin> <alpha1Const> <alpha2>

    states.txt: first line N, then N lines of 13 numbers
        gradU_xx gradU_xy gradU_xz gradU_yx ... gradU_zz  k  omega  nut  F1
    in OpenFOAM order, gradU_ij = dU_j/dx_i.

    Calls autoTurb::evaluate, the function kOmegaSSTAutoTurb::updateR runs per
    cell, forms R = k*RbyK (as updateR), gamma(F1) = F1*gamma1 + (1-F1)*gamma2
    with the stock SST constants (as kOmegaSSTBase::gamma, blend()) and the
    omega source gamma*R/nut (as kOmegaSSTAutoTurb::omegaSource), and prints
    one line per state with 17 significant digits:
        lambda1 alpha1 t1Work t2Work RbyK R omegaSource
\*---------------------------------------------------------------------------*/

#include "autoTurbProduction.H"

#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace Foam;

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::fprintf(stderr, "usage: %s states.txt alpha1Sin alpha1Const alpha2\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    const autoTurb::Coeffs c
    {
        std::strtod(argv[2], nullptr),
        std::strtod(argv[3], nullptr),
        std::strtod(argv[4], nullptr)
    };
    // stock kOmegaSSTBase coefficients
    const scalar gamma1 = 5.0/9.0;
    const scalar gamma2 = 0.44;
    // RASModel default omegaMin (SMALL) as the floor of omegaSafe
    const scalar omegaMin = SMALL;

    long n = 0;
    in >> n;
    for (long i = 0; i < n; ++i)
    {
        double g[9];
        for (int j = 0; j < 9; ++j)
        {
            in >> g[j];
        }
        double k, omega, nut, F1;
        in >> k >> omega >> nut >> F1;
        if (!in)
        {
            std::fprintf(stderr, "read error at state %ld\n", i);
            return 1;
        }
        const tensor gradU(g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8]);
        const autoTurb::Point p =
            autoTurb::evaluate(gradU, max(omega, omegaMin), c);
        const scalar R = k*p.RbyK;
        const scalar gamma = F1*(gamma1 - gamma2) + gamma2;
        const scalar omegaSource = (R == 0) ? 0.0 : gamma*R/nut;
        std::printf
        (
            "%.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
            p.lambda1, p.alpha1, p.t1Work, p.t2Work, p.RbyK, R, omegaSource
        );
    }
    return 0;
}
