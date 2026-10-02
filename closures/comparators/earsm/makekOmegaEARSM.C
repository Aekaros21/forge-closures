/*---------------------------------------------------------------------------*\
    Register kOmegaEARSM (BSL-EARSM, Menter, Garbaruk and Egorov 2012) with
    the incompressible RAS model table.
    Load in a case with:  libs ("libkOmegaEARSM.so");
\*---------------------------------------------------------------------------*/

#include "turbulentTransportModels.H"
#include "addToRunTimeSelectionTable.H"

#include "kOmegaEARSM.H"
makeRASModel(kOmegaEARSM);

// ************************************************************************* //
