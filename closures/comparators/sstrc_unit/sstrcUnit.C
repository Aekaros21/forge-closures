/*---------------------------------------------------------------------------*\
Application
    sstrcUnit

Description
    Pointwise unit executable for the SST-RC rotation/curvature function.
    No mesh and no CFD: it evaluates the production kernel
    src/comparators/sstrc/sstrcKernel.H (the code kOmegaSSTRC calls cell by
    cell) on manufactured states read from a text file.

    Usage:  sstrcUnit <input> <output> [cr1 cr2 cr3 fr1Max Cscale]

    Input (one state per line, '#' lines ignored, 20 columns):
        id A11 A12 A13 A21 A22 A23 A31 A32 A33 D11 D12 D13 D22 D23 D33
           Om1 Om2 Om3 omega
      A_ij  = du_i/dx_j (paper convention; transposed here to OpenFOAM's
              fvc::grad(U)_ij = du_j/dx_i before the kernel is called)
      D_ij  = DS_ij/Dt in the calculation frame, without the frame terms
      Om_m  = frame angular velocity [rad/s]
      omega = SST specific dissipation rate [1/s]

    Output (header line starting with '#', %.17g):
        id S W rStar rTilde fRotation fr1 fScaled
\*---------------------------------------------------------------------------*/

#include "sstrcKernel.H"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace Foam;

int main(int argc, char* argv[])
{
    if (argc != 3 && argc != 8)
    {
        std::cerr
            << "usage: sstrcUnit <input> <output> [cr1 cr2 cr3 fr1Max Cscale]\n";
        return 2;
    }

    sstrc::Coeffs c;
    if (argc == 8)
    {
        c.cr1 = std::strtod(argv[3], nullptr);
        c.cr2 = std::strtod(argv[4], nullptr);
        c.cr3 = std::strtod(argv[5], nullptr);
        c.fr1Max = std::strtod(argv[6], nullptr);
        c.Cscale = std::strtod(argv[7], nullptr);
    }

    std::ifstream in(argv[1]);
    if (!in.good())
    {
        std::cerr << "sstrcUnit: cannot read " << argv[1] << "\n";
        return 2;
    }
    std::ofstream out(argv[2]);
    if (!out.good())
    {
        std::cerr << "sstrcUnit: cannot write " << argv[2] << "\n";
        return 2;
    }

    out << "# sstrcUnit kernel=src/comparators/sstrc/sstrcKernel.H"
        << " cr1=" << c.cr1 << " cr2=" << c.cr2 << " cr3=" << c.cr3
        << " fr1Max=" << c.fr1Max << " Cscale=" << c.Cscale << "\n"
        << "# id S W rStar rTilde fRotation fr1 fScaled\n";
    out << std::setprecision(17);

    std::string line;
    long nStates = 0;
    long lineNo = 0;
    while (std::getline(in, line))
    {
        ++lineNo;
        const std::size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#')
        {
            continue;
        }
        std::istringstream is(line);
        std::string id;
        double v[19];
        is >> id;
        for (int i = 0; i < 19; ++i)
        {
            is >> v[i];
        }
        std::string extra;
        if (is.fail() || (is >> extra))
        {
            std::cerr
                << "sstrcUnit: line " << lineNo
                << " does not have exactly 20 columns\n";
            return 3;
        }

        // Paper gradient A_ij = du_i/dx_j, row-major
        const tensor A(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
        // OpenFOAM fvc::grad(U)_ij = du_j/dx_i
        const tensor gradU(A.T());
        const symmTensor DSDt(v[9], v[10], v[11], v[12], v[13], v[14]);
        const vector Omega(v[15], v[16], v[17]);
        const scalar omega = v[18];

        const sstrc::Result r = sstrc::evaluate(gradU, DSDt, Omega, omega, c);

        out << id << ' ' << r.S << ' ' << r.W << ' ' << r.rStar << ' '
            << r.rTilde << ' ' << r.fRotation << ' ' << r.fr1 << ' '
            << r.fScaled << '\n';
        ++nStates;
    }

    std::cout << "sstrcUnit: evaluated " << nStates << " states -> "
              << argv[2] << "\n";
    return 0;
}


// ************************************************************************* //
