/*---------------------------------------------------------------------------*\
    Register kOmegaSSTAutoTurb with the incompressible RAS model table.
    Load in a case with:  libs ("libkOmegaSSTAutoTurb.so");
\*---------------------------------------------------------------------------*/

#include "turbulentTransportModels.H"
#include "addToRunTimeSelectionTable.H"
#include "kOmegaSSTAutoTurb.H"
makeRASModel(kOmegaSSTAutoTurb);

// ************************************************************************* //
