/*---------------------------------------------------------------------------*\
    testAutoTurbLoad: load libkOmegaSSTAutoTurb.so at run time, as a case's
    controlDict libs entry does, and check that kOmegaSSTAutoTurb is in the
    incompressible RAS model selection table (no mesh, no CFD).

    Usage: testAutoTurbLoad <path to libkOmegaSSTAutoTurb.so>
\*---------------------------------------------------------------------------*/

#include "turbulentTransportModel.H"
#include "dlLibraryTable.H"
#include "IOstreams.H"

using namespace Foam;

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        Info<< "usage: testAutoTurbLoad <library>" << endl;
        return 2;
    }
    const auto* table = incompressible::RASModel::dictionaryConstructorTablePtr_;
    const bool before = table && table->found("kOmegaSSTAutoTurb");
    if (!dlLibraryTable::libs().open(fileName(argv[1]), true))
    {
        Info<< "FAILED to open " << argv[1] << endl;
        return 1;
    }
    table = incompressible::RASModel::dictionaryConstructorTablePtr_;
    const bool after = table && table->found("kOmegaSSTAutoTurb");
    const bool stock = table && table->found("kOmegaSST");
    Info<< "registered before load: " << before
        << ", after load: " << after
        << ", stock kOmegaSST present: " << stock << endl;
    return (!before && after && stock) ? 0 : 1;
}
