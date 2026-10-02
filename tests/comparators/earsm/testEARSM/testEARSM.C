/*---------------------------------------------------------------------------*\
    Unit tests of the BSL-EARSM comparator (src/comparators/earsm); no CFD.

    testEARSM -algebra
        (i)  homogeneous-shear equilibrium states, 2D closed forms,
             3D implicit-equation residual, P/eps consistency
        (ii) closed-form root N of the cubic on a sweep of invariants
        Lines "SWEEP IIS IIW N branch residual" and "RESULT key value" are
        compared with an independent Python reference by
        scripts/qualify_earsm.py.

    testEARSM -states <in.csv> -out <out.csv>
        earsm::evaluate (the function kOmegaEARSM calls per cell) on shared
        states (columns id,gxx,...,gzz,k,omega,nu,Ox,Oy,Oz; g = OpenFOAM
        gradU, g_ij = dU_j/dx_i); writes tau, invariants, N, betas, the full
        anisotropy a, nut and nonlinearStress for the independent reference
        (src/comparators/earsm_reference). Frame columns are echoed only.

    testEARSM -divergence -n <cells per side> -case <dir>
        (iii) divergence of the explicit stress of a manufactured field on
              an in-memory box mesh (no mesh files, no flow solve): the
              stress is assembled by the model's own routine
              (earsmFields.H) from fvc::grad(U), and fvc::div of it is
              compared with the exact divergence of the point-wise stress
              (fourth-order central differences of the algebra).
              Also: R = 2/3 k I - nut dev(twoSymm(grad U)) + nonlinearStress
              against k (a + 2/3 I), and Menter's regrouped tensors against
              the repository's integrity basis (tedpBasis::basisTensor).
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "earsmAlgebra.H"
#include "earsmFields.H"
#include "basisTensors/integrityBasis.H"

#include <cstdio>
#include <random>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <vector>

using namespace Foam;

static void result(const char* key, const double v)
{
    std::printf("RESULT %s %.17g\n", key, v);
}


// ---------------------------------------------------------------- (i), (ii)

static earsm::State shearState(const double sigma, const earsm::Coefficients& c)
{
    // U = (G y, 0, 0), normalized: S12 = W12 = sigma/2 (W12 = tau/2 dU1/dx2)
    const double s = 0.5*sigma;
    const symmTensor S(0, s, 0, 0, 0, 0);
    const tensor W(0, s, 0, -s, 0, 0, 0, 0, 0);
    return earsm::anisotropy(S, W, c);
}


static double PoE(const earsm::State& st, const double sigma)
{
    // P/eps = -a_ij S_ij (tau = k/eps)
    const double s = 0.5*sigma;
    return -(st.a.xy()*s*2.0);
}


static double solveSigma(const double target, const earsm::Coefficients& c)
{
    double lo = 1e-3, hi = 100.0;
    for (int it = 0; it < 200; ++it)
    {
        const double mid = 0.5*(lo + hi);
        if (PoE(shearState(mid, c), mid) < target) lo = mid; else hi = mid;
    }
    return 0.5*(lo + hi);
}


static int algebra()
{
    earsm::Coefficients pub;            // published defaults
    earsm::Coefficients wj;
    wj.A1 = 1.2;                        // Wallin-Johansson value

    result("A1", pub.A1);
    result("C1prime", pub.C1prime());

    // (i) equilibrium states: P/eps = 1 (log layer), and the asymptotic
    // homogeneous-shear P/eps of the BSL omega equation with set 1 and set 2
    // (beta_i/(gamma_i betaStar))
    const double kap = 0.41, bs = 0.09;
    const double g1 = 0.075/bs - 0.5*kap*kap/std::sqrt(bs);
    const double g2 = 0.0828/bs - 0.856*kap*kap/std::sqrt(bs);
    const double targets[3] = {1.0, 0.075/(g1*bs), 0.0828/(g2*bs)};
    const char* tname[3] = {"logLayer", "homShearSet1", "homShearSet2"};
    const earsm::Coefficients* cs[2] = {&pub, &wj};
    const char* cname[2] = {"pub", "wj"};
    for (int ic = 0; ic < 2; ++ic)
    {
        for (int it = 0; it < 3; ++it)
        {
            const double sig = solveSigma(targets[it], *cs[ic]);
            const earsm::State st(shearState(sig, *cs[ic]));
            char key[128];
            auto put = [&](const char* q, const double v)
            {
                std::snprintf(key, sizeof key, "shear.%s.%s.%s", cname[ic], tname[it], q);
                result(key, v);
            };
            put("PoE", PoE(st, sig));
            put("sigma", sig);
            put("N", st.N);
            put("branch", double(int(st.branch)));
            put("CmuEff", st.CmuEff);
            put("a11", st.a.xx());
            put("a22", st.a.yy());
            put("a33", st.a.zz());
            put("a12", st.a.xy());
            put("a23", st.a.yz());
            if (it == 0) put("kappaEff", kap*std::pow(st.CmuEff/bs, 0.75));
            // 2D closed forms: beta1 = -A1 N/(N^2 + 2 IIS'), a11 = -2 beta4 s^2
            const double s = 0.5*sig;
            const double Q = (st.N*st.N - 2.0*st.IIW)/cs[ic]->A1;
            put("closedForm.a12", std::fabs(st.a.xy() - (-st.N/Q)*s));
            put("closedForm.a11", std::fabs(st.a.xx() - 2.0/Q*s*s));
            put("closedForm.a22", std::fabs(st.a.yy() + 2.0/Q*s*s));
        }
    }

    // (i) 3D implicit equation N a = -A1 S + (a W - W a) (WJ 2000 with A2 = 0):
    // with beta9 = 1/Q1 the explicit solution is exact for any N; residual
    // relative to max(|N a|, A1 |S|). With N from the cubic, the size of
    // the omitted beta9 T9 term relative to |a| (published: beta9 = 0).
    std::mt19937_64 gen(20261001);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    double resT9 = 0, resT9cubic = 0, t9max = 0, t9sum = 0;
    double betaDenMin = GREAT;
    const int nSamples = 2000;
    for (int t = 0; t < nSamples; ++t)
    {
        const double scale = std::pow(10.0, -2.0 + 3.0*ud(gen));
        const tensor A(nd(gen), nd(gen), nd(gen), nd(gen), nd(gen), nd(gen),
                       nd(gen), nd(gen), nd(gen));
        const tensor B(nd(gen), nd(gen), nd(gen), nd(gen), nd(gen), nd(gen),
                       nd(gen), nd(gen), nd(gen));
        const symmTensor S(scale*dev(symm(A)));
        const tensor W(scale*skew(B));
        const earsm::Basis Bs(earsm::basis(S, W));
        const double C1p = pub.C1prime();
        const double Nany = C1p + 20.0*ud(gen);
        const double Ncub = earsm::rootN(Bs.IIS, Bs.IIW, C1p);
        for (int pass = 0; pass < 2; ++pass)
        {
            const double N = (pass == 1 ? Ncub : Nany);
            const earsm::Betas b(earsm::betas(N, Bs.IIW, Bs.IV, pub.A1, true));
            const symmTensor a =
                b.b1*Bs.T1 + b.b2*Bs.T2 + b.b3*Bs.T3 + b.b4*Bs.T4
              + b.b6*Bs.T6 + b.b9*Bs.T9;
            const tensor at(a);
            const tensor r(N*at + pub.A1*tensor(S) - ((at & W) - (W & at)));
            const double rel = cmptMax(cmptMag(r))
               /max(max(N*cmptMax(cmptMag(a)), pub.A1*cmptMax(cmptMag(S))), VSMALL);
            if (pass == 0) resT9 = max(resT9, rel);
            if (pass == 1)
            {
                resT9cubic = max(resT9cubic, rel);
                const double f = mag(b.b9*Bs.T9)/max(mag(a), VSMALL);
                t9max = max(t9max, f);
                t9sum += f;
            }
            const double Q = (N*N - 2.0*Bs.IIW)/pub.A1;
            const double Q1 = Q/6.0*(2.0*N*N - Bs.IIW);
            betaDenMin = min(betaDenMin, min(min(Q, Q1), N));
        }
    }
    result("implicit3D.anyN.withT9.maxRelativeResidual", resT9);
    result("implicit3D.cubicN.withT9.maxRelativeResidual", resT9cubic);
    result("implicit3D.cubicN.omittedT9.maxRelativeMagnitude", t9max);
    result("implicit3D.cubicN.omittedT9.meanRelativeMagnitude", t9sum/nSamples);
    result("denominators.min", betaDenMin);

    // (i) 2D consistency of the cubic: N = C1' + 9/4 P/eps with P/eps = -a:S
    double cons12 = 0, cons1245 = 0;
    for (int t = 0; t < 2000; ++t)
    {
        const double g = std::pow(10.0, -3.0 + 5.0*ud(gen));
        const double r = ud(gen);   // mix of shear and plane strain/rotation
        // plane flow: dU1/dx2 = g, dU2/dx1 = (2r - 1) g
        tensor L(Zero);
        L.xy() = g;
        L.yx() = (2.0*r - 1.0)*g;
        const symmTensor S(dev(symm(L)));
        const tensor W(skew(L));
        for (int ic = 0; ic < 2; ++ic)
        {
            const earsm::State st(earsm::anisotropy(S, W, *cs[ic]));
            const double pe = -(st.a && S);
            const double d = std::fabs(st.N - (cs[ic]->C1prime() + 2.25*pe))/st.N;
            if (ic == 0) cons1245 = max(cons1245, d); else cons12 = max(cons12, d);
        }
    }
    result("consistency2D.A1_1.2.maxRelative", cons12);
    result("consistency2D.A1_1.245.maxRelative", cons1245);

    // OpenFOAM convention: fvc::grad(U)_ij = dU_j/dx_i. Simple shear
    // U = (G y, 0, 0) has gradU.yx() = G; evaluate() must reproduce the
    // paper-convention state built directly (W12 = +tau G/2), and for random
    // gradients normalized() must equal tau/2 (L + L^T), tau/2 (L - L^T)
    // with L_ij = dU_i/dx_j written out component by component.
    {
        const double G = 7.0, kk = 0.3, ww = 2.0, nn = 1e-6;
        tensor g(Zero);
        g.yx() = G;
        const earsm::State st(earsm::evaluate(g, kk, ww, nn, pub));
        const double sig = st.tau*G;
        const earsm::State ref(shearState(sig, pub));
        result("convention.shear.maxAbsDiff", cmptMax(cmptMag(st.a - ref.a)));
        result("convention.shear.a11", st.a.xx());
        result("convention.shear.a12", st.a.xy());
        double e = 0;
        for (int t = 0; t < 200; ++t)
        {
            tensor gr;
            for (direction i = 0; i < 9; ++i) gr[i] = nd(gen);
            const double tau = 0.5 + ud(gen);
            symmTensor S; tensor W;
            earsm::normalized(gr, tau, S, W);
            const tensor St(S);
            const double trL = gr.xx() + gr.yy() + gr.zz();
            for (direction i = 0; i < 3; ++i)
            {
                for (direction j = 0; j < 3; ++j)
                {
                    const double Lij = gr(j, i), Lji = gr(i, j);
                    const double Sij = 0.5*tau*(Lij + Lji) - (i == j ? tau*trL/3.0 : 0.0);
                    const double Wij = 0.5*tau*(Lij - Lji);
                    e = max(e, max(std::fabs(St(i, j) - Sij), std::fabs(W(i, j) - Wij)));
                }
            }
        }
        result("convention.normalized.maxAbsDiff", e);
    }

    // (ii) sweep of invariants
    const double C1p = pub.C1prime();
    std::vector<double> vals{0.0};
    for (int i = 0; i <= 40; ++i) vals.push_back(std::pow(10.0, -6.0 + 0.25*i));
    for (const double IIS : vals)
    {
        for (const double mIIW : vals)
        {
            earsm::Branch br;
            const double N = earsm::rootN(IIS, -mIIW, C1p, &br);
            std::printf("SWEEP %.17g %.17g %.17g %d %.17g\n", IIS, -mIIW, N,
                        int(br), earsm::cubicResidual(N, IIS, -mIIW, C1p));
        }
    }
    // the branch boundary P2 = 0: along IIW = -lambda IIS, bisect for P2 = 0
    for (int i = 0; i < 25; ++i)
    {
        const double lam = 0.1*i;
        for (int j = 0; j < 9; ++j)
        {
            // P2 as a function of IIS along the ray
            auto P2of = [&](const double x)
            {
                double P1, P2;
                earsm::cubicP(x, -lam*x, C1p, P1, P2);
                return P2;
            };
            double lo = std::pow(10.0, -4.0 + j), hi = 10.0*lo;
            if (P2of(lo)*P2of(hi) > 0) continue;
            for (int it = 0; it < 200; ++it)
            {
                const double mid = 0.5*(lo + hi);
                if (P2of(lo)*P2of(mid) <= 0) hi = mid; else lo = mid;
            }
            for (const double f : {1.0 - 1e-9, 1.0 - 1e-12, 1.0, 1.0 + 1e-12, 1.0 + 1e-9})
            {
                const double IIS = lo*f;
                earsm::Branch br;
                const double N = earsm::rootN(IIS, -lam*IIS, C1p, &br);
                std::printf("SWEEP %.17g %.17g %.17g %d %.17g\n", IIS, -lam*IIS,
                            N, int(br), earsm::cubicResidual(N, IIS, -lam*IIS, C1p));
            }
        }
    }
    return 0;
}


// ---------------------------------------------------------------- (iii)

namespace mf
{
const double pi = constant::mathematical::pi;

// solenoidal: Taylor-Green part with equal wavenumbers (1 + 1 - 2 = 0) plus
// components independent of their own coordinate; wavenumber q = pi keeps
// the cubic-in-gradient stress resolved on 16^3 to 64^3 cells
const double q = pi;

inline vector U(const vector& p)
{
    const double x = p.x(), y = p.y(), z = p.z();
    return vector
    (
        std::sin(q*x)*std::cos(q*y)*std::cos(q*z) + 0.8*std::sin(pi*y)*std::cos(0.5*pi*z),
        std::cos(q*x)*std::sin(q*y)*std::cos(q*z) + 0.3*std::sin(pi*z),
       -2.0*std::cos(q*x)*std::cos(q*y)*std::sin(q*z) + 0.2*std::cos(pi*x)
    );
}

// OpenFOAM convention gradU_ij = dU_j/dx_i
inline tensor gradU(const vector& p)
{
    const double x = p.x(), y = p.y(), z = p.z();
    const double sx = std::sin(q*x), cx = std::cos(q*x);
    const double sy = std::sin(q*y), cy = std::cos(q*y);
    const double sz = std::sin(q*z), cz = std::cos(q*z);
    tensor g;
    // dUx/dx_i
    g.xx() = q*cx*cy*cz;
    g.yx() = -q*sx*sy*cz + 0.8*pi*std::cos(pi*y)*std::cos(0.5*pi*z);
    g.zx() = -q*sx*cy*sz - 0.4*pi*std::sin(pi*y)*std::sin(0.5*pi*z);
    // dUy/dx_i
    g.xy() = -q*sx*sy*cz;
    g.yy() = q*cx*cy*cz;
    g.zy() = -q*cx*sy*sz + 0.3*pi*std::cos(pi*z);
    // dUz/dx_i
    g.xz() = 2.0*q*sx*cy*sz - 0.2*pi*std::sin(pi*x);
    g.yz() = 2.0*q*cx*sy*sz;
    g.zz() = -2.0*q*cx*cy*cz;
    return g;
}

inline double k(const vector& p)
{
    return 0.02*(1.5 + 0.5*std::sin(2*pi*p.x() + 0.3)*std::cos(pi*p.y())
               + 0.3*std::cos(pi*p.z()));
}

inline double omega(const vector& p)
{
    return 20.0*(1.5 + 0.4*std::cos(pi*p.x())*std::sin(2*pi*p.y() + 0.5)
               + 0.3*std::sin(pi*p.z()));
}

const double nu = 1e-6;

inline symmTensor stress(const vector& p, const earsm::Coefficients& c)
{
    const earsm::State st(earsm::evaluate(gradU(p), k(p), omega(p), nu, c));
    return k(p)*st.aNL;
}

// exact divergence d tau_ij/dx_i by fourth-order central differences
inline vector divStress(const vector& p, const earsm::Coefficients& c)
{
    const double h = 1e-4;
    vector d(Zero);
    for (direction i = 0; i < 3; ++i)
    {
        vector e(Zero);
        e[i] = h;
        const symmTensor t =
            (-stress(p + 2*e, c) + 8*stress(p + e, c)
             - 8*stress(p - e, c) + stress(p - 2*e, c))/(12*h);
        // row i of the derivative of tau along x_i
        const tensor tt(t);
        d += vector(tt(i, 0), tt(i, 1), tt(i, 2));
    }
    return d;
}
} // namespace mf


static autoPtr<fvMesh> boxMesh(const Time& runTime, const label n)
{
    const label np = n + 1;
    auto P = [np](label i, label j, label k) { return i + np*(j + np*k); };
    auto C = [n](label i, label j, label k) { return i + n*(j + n*k); };

    pointField points(np*np*np);
    for (label k = 0; k < np; ++k)
        for (label j = 0; j < np; ++j)
            for (label i = 0; i < np; ++i)
                points[P(i, j, k)] = vector(scalar(i)/n, scalar(j)/n, scalar(k)/n);

    DynamicList<face> faces;
    DynamicList<label> own, nei;
    // internal faces, upper-triangular order
    for (label k = 0; k < n; ++k)
        for (label j = 0; j < n; ++j)
            for (label i = 0; i < n; ++i)
            {
                const label c = C(i, j, k);
                if (i + 1 < n)
                {
                    faces.append(face(labelList({P(i+1,j,k), P(i+1,j+1,k), P(i+1,j+1,k+1), P(i+1,j,k+1)})));
                    own.append(c); nei.append(C(i+1, j, k));
                }
                if (j + 1 < n)
                {
                    faces.append(face(labelList({P(i,j+1,k), P(i,j+1,k+1), P(i+1,j+1,k+1), P(i+1,j+1,k)})));
                    own.append(c); nei.append(C(i, j+1, k));
                }
                if (k + 1 < n)
                {
                    faces.append(face(labelList({P(i,j,k+1), P(i+1,j,k+1), P(i+1,j+1,k+1), P(i,j+1,k+1)})));
                    own.append(c); nei.append(C(i, j, k+1));
                }
            }
    const label nInternal = faces.size();
    // boundary faces, outward normals
    for (label k = 0; k < n; ++k)
        for (label j = 0; j < n; ++j)
        {
            faces.append(face(labelList({P(0,j,k), P(0,j,k+1), P(0,j+1,k+1), P(0,j+1,k)})));
            own.append(C(0, j, k));
            faces.append(face(labelList({P(n,j,k), P(n,j+1,k), P(n,j+1,k+1), P(n,j,k+1)})));
            own.append(C(n-1, j, k));
        }
    for (label k = 0; k < n; ++k)
        for (label i = 0; i < n; ++i)
        {
            faces.append(face(labelList({P(i,0,k), P(i+1,0,k), P(i+1,0,k+1), P(i,0,k+1)})));
            own.append(C(i, 0, k));
            faces.append(face(labelList({P(i,n,k), P(i,n,k+1), P(i+1,n,k+1), P(i+1,n,k)})));
            own.append(C(i, n-1, k));
        }
    for (label j = 0; j < n; ++j)
        for (label i = 0; i < n; ++i)
        {
            faces.append(face(labelList({P(i,j,0), P(i,j+1,0), P(i+1,j+1,0), P(i+1,j,0)})));
            own.append(C(i, j, 0));
            faces.append(face(labelList({P(i,j,n), P(i+1,j,n), P(i+1,j+1,n), P(i,j+1,n)})));
            own.append(C(i, j, n-1));
        }
    const label nBoundary = faces.size() - nInternal;

    autoPtr<fvMesh> meshPtr
    (
        new fvMesh
        (
            IOobject
            (
                polyMesh::defaultRegion,
                runTime.timeName(),
                runTime,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE
            ),
            std::move(points),
            faceList(std::move(faces)),
            labelList(std::move(own)),
            labelList(std::move(nei))
        )
    );
    List<polyPatch*> patches(1);
    patches[0] = new polyPatch
    (
        "box", nBoundary, nInternal, 0, meshPtr->boundaryMesh(), "patch"
    );
    meshPtr->addFvPatches(patches);
    return meshPtr;
}


static int divergence(const Time& runTime, const label n)
{
    autoPtr<fvMesh> meshPtr(boxMesh(runTime, n));
    const fvMesh& mesh = meshPtr();
    const earsm::Coefficients c;

    auto vf = [&](const word& name, const dimensionSet& dims)
    {
        return volScalarField
        (
            IOobject(name, runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dims, Zero), fixedValueFvPatchScalarField::typeName
        );
    };
    volScalarField k(vf("k", sqr(dimVelocity)));
    volScalarField omega(vf("omega", inv(dimTime)));
    volScalarField nu(vf("nu", sqr(dimLength)/dimTime));
    volScalarField nut(vf("nut", sqr(dimLength)/dimTime));
    volVectorField U
    (
        IOobject("U", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh, dimensionedVector(dimVelocity, Zero), fixedValueFvPatchVectorField::typeName
    );
    volSymmTensorField ns
    (
        IOobject("nonlinearStress", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
        mesh, dimensionedSymmTensor(sqr(dimVelocity), Zero)
    );

    const vectorField& Cc = mesh.C().primitiveField();
    forAll(Cc, ci)
    {
        U[ci] = mf::U(Cc[ci]);
        k[ci] = mf::k(Cc[ci]);
        omega[ci] = mf::omega(Cc[ci]);
        nu[ci] = mf::nu;
    }
    forAll(mesh.boundary(), pi)
    {
        const vectorField& Cf = mesh.boundary()[pi].Cf();
        forAll(Cf, fi)
        {
            U.boundaryFieldRef()[pi][fi] = mf::U(Cf[fi]);
            k.boundaryFieldRef()[pi][fi] = mf::k(Cf[fi]);
            omega.boundaryFieldRef()[pi][fi] = mf::omega(Cf[fi]);
            nu.boundaryFieldRef()[pi][fi] = mf::nu;
        }
    }

    // the model's assembly routine, with gradU from fvc::grad as in correct()
    const volTensorField gradU(fvc::grad(U));
    earsm::Stats stats;
    earsm::assemble(gradU, k, omega, nu, c, 1e-15, 1e-15, nut, ns, stats);
    nut.correctBoundaryConditions();

    const volVectorField divNs(fvc::div(ns));

    double eMaxAll = 0, eMaxIn = 0, e2 = 0, ref2 = 0, refMax = 0;
    label nIn = 0;
    const label nn = n;
    forAll(Cc, ci)
    {
        const vector ex(mf::divStress(Cc[ci], c));
        const double e = mag(divNs[ci] - ex);
        eMaxAll = max(eMaxAll, e);
        refMax = max(refMax, mag(ex));
        const label i = ci % nn, j = (ci/nn) % nn, kk = ci/(nn*nn);
        const bool interior = i > 1 && j > 1 && kk > 1 && i < nn-2 && j < nn-2 && kk < nn-2;
        if (interior)
        {
            eMaxIn = max(eMaxIn, e);
            e2 += sqr(e);
            ref2 += magSqr(ex);
            ++nIn;
        }
    }
    std::printf("RESULT div.n %d\n", int(n));
    result("div.maxErrorAllCells", eMaxAll);
    result("div.maxErrorInterior", eMaxIn);
    result("div.rmsErrorInterior", std::sqrt(e2/max(nIn, 1)));
    result("div.rmsExactInterior", std::sqrt(ref2/max(nIn, 1)));
    result("div.maxExact", refMax);
    result("div.NminCells", stats.Nmin);
    result("div.NmaxCells", stats.Nmax);
    result("div.fractionP2negative", scalar(stats.nNegP2)/mesh.nCells());
    result("div.fractionKolmogorovActive", scalar(stats.nKolmogorov)/mesh.nCells());

    // exact point-wise stress at the cell centres with the exact gradient:
    // isolates the assembly from the gradient discretization
    double eStress = 0, sStress = 0;
    forAll(Cc, ci)
    {
        eStress = max(eStress, cmptMax(cmptMag(ns[ci] - mf::stress(Cc[ci], c))));
        sStress = max(sStress, cmptMax(cmptMag(mf::stress(Cc[ci], c))));
    }
    result("stress.maxAbsVsExactGradient", eStress);
    result("stress.maxAbs", sStress);

    // boundary faces: assembled from the boundary values
    double eFace = 0;
    {
        const vectorField& Cf = mesh.boundary()[0].Cf();
        const symmTensorField& nsb = ns.boundaryField()[0];
        forAll(Cf, fi)
        {
            eFace = max(eFace, cmptMax(cmptMag(nsb[fi] - mf::stress(Cf[fi], c))));
        }
    }
    result("stress.boundaryMaxAbsVsExactGradient", eFace);

    // Reynolds stress decomposition: -nut dev(twoSymm(gradU)) + ns = k a
    double eR = 0;
    {
        const volSymmTensorField Rlin(-nut*dev(twoSymm(gradU)) + ns);
        forAll(Cc, ci)
        {
            const earsm::State st(earsm::evaluate(gradU[ci], k[ci], omega[ci], nu[ci], c));
            eR = max(eR, cmptMax(cmptMag(Rlin[ci] - k[ci]*st.a)));
        }
    }
    result("reynoldsStress.maxAbsDecompositionError", eR);

    // Menter's regrouped tensors from the repository's integrity basis
    // (tedpBasis::basisTensor, Pope numbering): T1 = P1, T2 = P3, T3 = P4,
    // T4 = P2, T6 = P6 - IIW P1, T9 = P7 + IIW/2 P2, with W = +Omega
    double eB = 0;
    {
        volSymmTensorField Sh
        (
            IOobject("Sh", runTime.timeName(), mesh), mesh,
            dimensionedSymmTensor(dimless, Zero)
        );
        volTensorField Wh
        (
            IOobject("Wh", runTime.timeName(), mesh), mesh,
            dimensionedTensor(dimless, Zero)
        );
        forAll(Cc, ci)
        {
            const scalar tau = earsm::timeScale(k[ci], omega[ci], nu[ci], c);
            symmTensor S; tensor W;
            earsm::normalized(gradU[ci], tau, S, W);
            Sh[ci] = S; Wh[ci] = W;
        }
        const volScalarField IIW(tedpBasis::invariant(2, Sh, Wh));
        const volSymmTensorField P1(tedpBasis::basisTensor(1, Sh, Wh));
        const volSymmTensorField P2(tedpBasis::basisTensor(2, Sh, Wh));
        const volSymmTensorField P3(tedpBasis::basisTensor(3, Sh, Wh));
        const volSymmTensorField P4(tedpBasis::basisTensor(4, Sh, Wh));
        const volSymmTensorField P6(tedpBasis::basisTensor(6, Sh, Wh));
        const volSymmTensorField P7(tedpBasis::basisTensor(7, Sh, Wh));
        forAll(Cc, ci)
        {
            const earsm::Basis B(earsm::basis(Sh[ci], Wh[ci]));
            const scalar w = IIW[ci];
            eB = max(eB, cmptMax(cmptMag(B.T1 - P1[ci])));
            eB = max(eB, cmptMax(cmptMag(B.T2 - P3[ci])));
            eB = max(eB, cmptMax(cmptMag(B.T3 - P4[ci])));
            eB = max(eB, cmptMax(cmptMag(B.T4 - P2[ci])));
            eB = max(eB, cmptMax(cmptMag(B.T6 - (P6[ci] - w*P1[ci]))));
            eB = max(eB, cmptMax(cmptMag(B.T9 - (P7[ci] + 0.5*w*P2[ci]))));
            eB = max(eB, std::fabs(B.IIW - w));
        }
    }
    result("basis.maxAbsDifferenceToIntegrityBasis", eB);
    return 0;
}


static std::vector<std::string> splitCsv(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        while (!item.empty() && (item.back() == '\r' || item.back() == ' ')) item.pop_back();
        while (!item.empty() && item.front() == ' ') item.erase(item.begin());
        out.push_back(item);
    }
    return out;
}


static int states(const std::string& inPath, const std::string& outPath)
{
    std::ifstream in(inPath);
    if (!in)
    {
        std::fprintf(stderr, "cannot open %s\n", inPath.c_str());
        return 2;
    }
    std::string line;
    while (std::getline(in, line) && (line.empty() || line[0] == '#')) {}
    const std::vector<std::string> head(splitCsv(line));
    std::map<std::string, size_t> col;
    for (size_t i = 0; i < head.size(); ++i) col[head[i]] = i;
    const char* need[] =
        {"id", "gxx", "gxy", "gxz", "gyx", "gyy", "gyz", "gzx", "gzy", "gzz",
         "k", "omega", "nu", "Ox", "Oy", "Oz"};
    for (const char* n : need)
    {
        if (!col.count(n))
        {
            std::fprintf(stderr, "missing column %s\n", n);
            return 2;
        }
    }

    FILE* out = std::fopen(outPath.c_str(), "w");
    if (!out)
    {
        std::fprintf(stderr, "cannot write %s\n", outPath.c_str());
        return 2;
    }
    std::fprintf(out, "id,gxx,gxy,gxz,gyx,gyy,gyz,gzx,gzy,gzz,k,omega,nu,Ox,Oy,Oz,"
        "tau,IIS,IIW,IV,N,beta1,beta3,beta4,beta6,beta9,"
        "axx,axy,axz,ayy,ayz,azz,nut,nsxx,nsxy,nsxz,nsyy,nsyz,nszz,branch\n");

    const earsm::Coefficients c;        // published defaults, beta9 = 0
    const double kMin = 1e-15, omegaMin = 1e-15;   // RASModel kMin, omegaMin
    label rows = 0;
    while (std::getline(in, line))
    {
        if (line.find_first_not_of(" \r\n") == std::string::npos || line[0] == '#') continue;
        const std::vector<std::string> v(splitCsv(line));
        auto num = [&](const char* n) { return std::stod(v.at(col[n])); };
        tensor g
        (
            num("gxx"), num("gxy"), num("gxz"),
            num("gyx"), num("gyy"), num("gyz"),
            num("gzx"), num("gzy"), num("gzz")
        );
        const double k = num("k"), w = num("omega"), nu = num("nu");
        // as earsm::assemble: tau from the floored k, omega; nut and the
        // stress scale with max(k, 0)
        const double kS = max(k, kMin), wS = max(w, omegaMin);
        const earsm::State st(earsm::evaluate(g, kS, wS, nu, c));
        const double kPos = max(k, 0.0);
        const double nut = st.CmuEff*kPos*st.tau;
        const symmTensor ns(kPos*st.aNL);
        std::fprintf(out, "%s", v.at(col["id"]).c_str());
        for (direction i = 0; i < 9; ++i) std::fprintf(out, ",%.17g", g[i]);
        std::fprintf(out, ",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
                     k, w, nu, num("Ox"), num("Oy"), num("Oz"));
        std::fprintf(out, ",%.17g,%.17g,%.17g,%.17g,%.17g", st.tau, st.IIS, st.IIW, st.IV, st.N);
        std::fprintf(out, ",%.17g,%.17g,%.17g,%.17g,%.17g",
                     st.beta.b1, st.beta.b3, st.beta.b4, st.beta.b6, st.beta.b9);
        std::fprintf(out, ",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
                     st.a.xx(), st.a.xy(), st.a.xz(), st.a.yy(), st.a.yz(), st.a.zz());
        std::fprintf(out, ",%.17g", nut);
        std::fprintf(out, ",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
                     ns.xx(), ns.xy(), ns.xz(), ns.yy(), ns.yz(), ns.zz());
        std::fprintf(out, ",%d\n", int(st.branch));
        ++rows;
    }
    std::fclose(out);
    std::printf("RESULT states.rows %d\n", int(rows));
    return 0;
}


int main(int argc, char *argv[])
{
    argList::noParallel();
    argList::addBoolOption("algebra", "algebra tests (i), (ii)");
    argList::addBoolOption("divergence", "divergence test (iii)");
    argList::addOption("n", "label", "cells per side (divergence)");
    argList::noCheckProcessorDirectories();

    if (argc > 1 && std::string(argv[1]) == "-algebra")
    {
        return algebra();
    }
    if (argc == 5 && std::string(argv[1]) == "-states" && std::string(argv[3]) == "-out")
    {
        return states(argv[2], argv[4]);
    }

    #include "setRootCase.H"
    #include "createTime.H"

    const label n = args.getOrDefault<label>("n", 16);
    return divergence(runTime, n);
}
