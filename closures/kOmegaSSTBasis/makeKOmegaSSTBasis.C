/*---------------------------------------------------------------------------*\
    Register kOmegaSSTBasis with the incompressible RAS model table.
    Load in a case with:  libs ("libkOmegaSSTBasis.so");
\*---------------------------------------------------------------------------*/

#include "turbulentTransportModels.H"
#include "addToRunTimeSelectionTable.H"

#include "kOmegaSSTBasis.H"
makeRASModel(kOmegaSSTBasis);

// ************************************************************************* //
