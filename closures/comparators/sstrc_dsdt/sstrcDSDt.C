/*---------------------------------------------------------------------------*\
Application
    sstrcDSDt

Description
    Unit test of the SST-RC steady Lagrangian derivative operator
    (src/comparators/sstrc/sstrcLagrangian.H, the function kOmegaSSTRC calls)
    on manufactured fields.  No case directory, no files read, no solver:
    a uniform two-dimensional Cartesian grid is built in memory, the strain
    rate S_ij and the face fluxes phi_f are set from an analytic,
    divergence-free velocity field, and the discrete DS_ij/Dt is compared
    with the exact U_k dS_ij/dx_k at the cell centres on four grids.

    Velocity (divergence-free):
        u = x^2 y + y^2/2 + 0.2 x
        v = -x y^2 + 0.3 x^3 - 0.2 y + 0.1 x
    on x in [0.3, 1.3], y in [0.2, 1.2].

    Checks
      1. uniform S with the same fluxes: |DS/Dt| = 0 to round-off;
      2. manufactured S: max error relative to max |DS/Dt| on 16^2..128^2
         grids and observed order of convergence: 2 in the interior, 1 in
         the cells with a face on the through-flow boundary (exact boundary
         value against linear interior interpolate; zero on walls).

    Usage:  sstrcDSDt <output>
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "emptyPolyPatch.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "sstrcLagrangian.H"

#include <fstream>
#include <iomanip>

using namespace Foam;

namespace
{

const scalar x0 = 0.3, x1 = 1.3, y0 = 0.2, y1 = 1.2, dz = 0.1;

vector velocity(const point& p)
{
    const scalar x = p.x(), y = p.y();
    return vector
    (
        x*x*y + 0.5*y*y + 0.2*x,
       -x*y*y + 0.3*x*x*x - 0.2*y + 0.1*x,
        0
    );
}

symmTensor strain(const point& p)
{
    const scalar x = p.x(), y = p.y();
    const scalar s11 = 2*x*y + 0.2;
    const scalar s12 = 0.5*(1.9*x*x + y - y*y + 0.1);
    return symmTensor(s11, s12, 0, -s11, 0, 0);
}

symmTensor exactDSDt(const point& p)
{
    const scalar x = p.x(), y = p.y();
    const vector U(velocity(p));
    const scalar d11 = U.x()*2*y + U.y()*2*x;
    const scalar d12 = U.x()*1.9*x + U.y()*(0.5 - y);
    return symmTensor(d11, d12, 0, -d11, 0, 0);
}


autoPtr<fvMesh> makeMesh(const Time& runTime, const label n)
{
    const label nx = n, ny = n;
    const scalar dx = (x1 - x0)/nx, dy = (y1 - y0)/ny;
    auto pid = [&](label i, label j, label k)
    {
        return i + (nx + 1)*(j + (ny + 1)*k);
    };
    auto cid = [&](label i, label j) { return i + nx*j; };

    pointField points(2*(nx + 1)*(ny + 1));
    for (label k = 0; k < 2; ++k)
    {
        for (label j = 0; j <= ny; ++j)
        {
            for (label i = 0; i <= nx; ++i)
            {
                points[pid(i, j, k)] = point(x0 + i*dx, y0 + j*dy, k*dz);
            }
        }
    }

    DynamicList<face> faces;
    DynamicList<label> owner;
    DynamicList<label> neighbour;

    // Internal faces, upper-triangular order (owner, then neighbour)
    for (label j = 0; j < ny; ++j)
    {
        for (label i = 0; i < nx; ++i)
        {
            const label c = cid(i, j);
            if (i + 1 < nx)
            {
                faces.append(face{pid(i+1, j, 0), pid(i+1, j+1, 0),
                                  pid(i+1, j+1, 1), pid(i+1, j, 1)});
                owner.append(c);
                neighbour.append(cid(i+1, j));
            }
            if (j + 1 < ny)
            {
                faces.append(face{pid(i, j+1, 0), pid(i, j+1, 1),
                                  pid(i+1, j+1, 1), pid(i+1, j+1, 0)});
                owner.append(c);
                neighbour.append(cid(i, j+1));
            }
        }
    }
    const label nInternal = faces.size();

    // Patch "sides": the four lateral boundaries, outward normals
    for (label j = 0; j < ny; ++j)
    {
        faces.append(face{pid(0, j, 0), pid(0, j, 1), pid(0, j+1, 1), pid(0, j+1, 0)});
        owner.append(cid(0, j));
        faces.append(face{pid(nx, j, 0), pid(nx, j+1, 0), pid(nx, j+1, 1), pid(nx, j, 1)});
        owner.append(cid(nx-1, j));
    }
    for (label i = 0; i < nx; ++i)
    {
        faces.append(face{pid(i, 0, 0), pid(i+1, 0, 0), pid(i+1, 0, 1), pid(i, 0, 1)});
        owner.append(cid(i, 0));
        faces.append(face{pid(i, ny, 0), pid(i, ny, 1), pid(i+1, ny, 1), pid(i+1, ny, 0)});
        owner.append(cid(i, ny-1));
    }
    const label nSides = faces.size() - nInternal;

    // Patch "frontAndBack": empty
    for (label j = 0; j < ny; ++j)
    {
        for (label i = 0; i < nx; ++i)
        {
            faces.append(face{pid(i, j, 0), pid(i, j+1, 0), pid(i+1, j+1, 0), pid(i+1, j, 0)});
            owner.append(cid(i, j));
            faces.append(face{pid(i, j, 1), pid(i+1, j, 1), pid(i+1, j+1, 1), pid(i, j+1, 1)});
            owner.append(cid(i, j));
        }
    }
    const label nEmpty = faces.size() - nInternal - nSides;

    autoPtr<fvMesh> meshPtr
    (
        new fvMesh
        (
            IOobject
            (
                "grid" + Foam::name(n),
                runTime.timeName(),
                runTime,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            std::move(points),
            faceList(std::move(faces)),
            labelList(std::move(owner)),
            labelList(std::move(neighbour))
        )
    );
    fvMesh& mesh = meshPtr();

    List<polyPatch*> patches(2);
    patches[0] = new polyPatch
    (
        "sides", nSides, nInternal, 0, mesh.boundaryMesh(), polyPatch::typeName
    );
    patches[1] = new emptyPolyPatch
    (
        "frontAndBack", nEmpty, nInternal + nSides, 1, mesh.boundaryMesh(),
        emptyPolyPatch::typeName
    );
    mesh.addFvPatches(patches);

    return meshPtr;
}


struct Errors
{
    scalar maxErr;       // all cells
    scalar maxErrInt;    // cells without a face on the through-flow boundary
    scalar maxRef;
    scalar uniformMax;
};


Errors evaluate(const Time& runTime, const label n)
{
    autoPtr<fvMesh> meshPtr(makeMesh(runTime, n));
    const fvMesh& mesh = meshPtr();

    // Face fluxes from the analytic velocity at the face centres
    surfaceScalarField phi
    (
        IOobject("phi", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedScalar(dimVolume/dimTime, Zero)
    );
    {
        const surfaceVectorField& Sf = mesh.Sf();
        const surfaceVectorField& Cf = mesh.Cf();
        forAll(phi, facei)
        {
            phi[facei] = velocity(Cf[facei]) & Sf[facei];
        }
        auto& pbf = phi.boundaryFieldRef();
        forAll(pbf, patchi)
        {
            forAll(pbf[patchi], facei)
            {
                pbf[patchi][facei] =
                    velocity(Cf.boundaryField()[patchi][facei])
                  & Sf.boundaryField()[patchi][facei];
            }
        }
    }

    // Manufactured strain rate at the cell centres and boundary faces
    volSymmTensorField S
    (
        IOobject("S", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedSymmTensor(dimless/dimTime, Zero)
    );
    const volVectorField& C = mesh.C();
    forAll(S, celli)
    {
        S[celli] = strain(C[celli]);
    }
    {
        auto& sbf = S.boundaryFieldRef();
        forAll(sbf, patchi)
        {
            forAll(sbf[patchi], facei)
            {
                sbf[patchi][facei] = strain(C.boundaryField()[patchi][facei]);
            }
        }
    }

    const volSymmTensorField DSDt(sstrc::steadyLagrangianDerivative(phi, S));

    // Cells with a face on the through-flow boundary: there the exact
    // boundary value meets the linear interior interpolate, an O(h) term
    // proportional to the boundary flux (zero on walls, phi_f = 0)
    boolList nearBoundary(mesh.nCells(), false);
    for (const label celli : mesh.boundaryMesh()["sides"].faceCells())
    {
        nearBoundary[celli] = true;
    }

    Errors e{0, 0, 0, 0};
    forAll(DSDt, celli)
    {
        const symmTensor exact(exactDSDt(C[celli]));
        const scalar err = mag(DSDt[celli] - exact);
        e.maxErr = max(e.maxErr, err);
        if (!nearBoundary[celli])
        {
            e.maxErrInt = max(e.maxErrInt, err);
        }
        e.maxRef = max(e.maxRef, mag(exact));
    }

    // Uniform S: the derivative must vanish
    volSymmTensorField S0
    (
        IOobject("S0", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh,
        dimensionedSymmTensor(dimless/dimTime, symmTensor(0.7, -0.3, 0.1, 0.2, 0.05, -0.9))
    );
    const volSymmTensorField DS0(sstrc::steadyLagrangianDerivative(phi, S0));
    e.uniformMax = gMax(mag(DS0.primitiveField())());

    return e;
}

} // End anonymous namespace


int main(int argc, char *argv[])
{
    argList::noParallel();
    argList::noBanner();
    argList::addArgument("output");
    argList args(argc, argv, false, false, false);

    autoPtr<Time> runTimePtr(Time::New());
    const Time& runTime = runTimePtr();

    std::ofstream out(args.get<fileName>(1));
    out << "# sstrcDSDt operator=src/comparators/sstrc/sstrcLagrangian.H\n"
        << "# n maxErrAll maxErrInterior maxRef relErrAll relErrInterior"
        << " uniformMaxAbs orderAll orderInterior\n"
        << std::setprecision(10);

    scalar prev = -1;
    scalar prevInt = -1;
    for (const label n : {16, 32, 64, 128})
    {
        const Errors e = evaluate(runTime, n);
        const scalar rel = e.maxErr/e.maxRef;
        const scalar relInt = e.maxErrInt/e.maxRef;
        const scalar order = prev > 0 ? std::log(prev/rel)/std::log(2.0) : 0;
        const scalar orderInt =
            prevInt > 0 ? std::log(prevInt/relInt)/std::log(2.0) : 0;
        out << n << ' ' << e.maxErr << ' ' << e.maxErrInt << ' ' << e.maxRef
            << ' ' << rel << ' ' << relInt << ' ' << e.uniformMax << ' '
            << order << ' ' << orderInt << '\n';
        Info<< "n " << n << "  rel. max error all/interior " << rel << " / "
            << relInt << "  uniform-S max |DS/Dt| " << e.uniformMax
            << "  observed order all/interior " << order << " / " << orderInt
            << nl;
        prev = rel;
        prevInt = relInt;
    }

    return 0;
}


// ************************************************************************* //
