/*---------------------------------------------------------------------------*\
    Register kOmegaSSTRC with the incompressible RAS model table.
    Load in a case with:  libs ("libkOmegaSSTRC.so");
\*---------------------------------------------------------------------------*/

#include "turbulentTransportModels.H"
#include "addToRunTimeSelectionTable.H"

#include "kOmegaSSTRC.H"
makeRASModel(kOmegaSSTRC);

// ************************************************************************* //
