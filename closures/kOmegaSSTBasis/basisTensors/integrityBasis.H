/*---------------------------------------------------------------------------*\
    Pope's integrity basis T1..T10 and invariants I1..I5 of the normalized
    strain/rotation pair (Shat, What) as OpenFOAM fields.

    Python twin: tedp.tensors.integrity_basis — formulas must match exactly.
\*---------------------------------------------------------------------------*/

#ifndef tedp_integrityBasis_H
#define tedp_integrityBasis_H

#include "volFields.H"

namespace Foam
{
namespace tedpBasis
{

//- Basis tensor n (1..10). Every result is symmetric and traceless.
inline tmp<volSymmTensorField> basisTensor
(
    const label n,
    const volSymmTensorField& S,
    const volTensorField& W
)
{
    switch (n)
    {
        case 1:
            return tmp<volSymmTensorField>::New("T1", S);
        case 2:
            return symm((S & W) - (W & S));
        case 3:
            return dev(symm(S & S));
        case 4:
            return dev(symm(W & W));
        case 5:
            return symm((W & S & S) - (S & S & W));
        case 6:
            return dev(symm((W & W & S) + (S & W & W)));
        case 7:
            return symm((W & S & W & W) - (W & W & S & W));
        case 8:
            return symm((S & W & S & S) - (S & S & W & S));
        case 9:
            return dev(symm((W & W & S & S) + (S & S & W & W)));
        case 10:
            return symm((W & S & S & W & W) - (W & W & S & S & W));
        default:
            FatalErrorInFunction
                << "basis tensor index " << n << " outside 1..10"
                << exit(FatalError);
            return nullptr;
    }
}

//- Invariant i (1..5): I1=tr(S^2) I2=tr(W^2) I3=tr(S^3) I4=tr(W^2 S)
//  I5=tr(W^2 S^2)
inline tmp<volScalarField> invariant
(
    const label i,
    const volSymmTensorField& S,
    const volTensorField& W
)
{
    switch (i)
    {
        case 1: return tr(S & S);
        case 2: return tr(W & W);
        case 3: return tr(S & S & S);
        case 4: return tr(W & W & S);
        case 5: return tr(W & W & S & S);
        default:
            FatalErrorInFunction
                << "invariant index " << i << " outside 1..5"
                << exit(FatalError);
            return nullptr;
    }
}

} // namespace tedpBasis
} // namespace Foam

#endif
