/*---------------------------------------------------------------------------*\
    Point-wise algebra of the BSL-EARSM stress-strain relation
    (Wallin and Johansson 2000 relation as regrouped and recalibrated by
    Menter, Garbaruk and Egorov 2012).

    Reference: F. R. Menter, A. V. Garbaruk, Y. Egorov, "Explicit algebraic
    Reynolds stress models for anisotropic wall-bounded flows", Progress in
    Flight Physics 3 (2012) 89-104, doi:10.1051/eucass/201203089
    (identical formulation in the EUCASS 2009 paper of the same title).
    Equation numbers below are those of the 2012 paper; page numbers are the
    journal pages.

      tau_ij = k (a_ij + 2/3 delta_ij)                              Eq. (1)
      a = beta1 T1 + beta2 T2 + beta3 T3 + beta4 T4
        + beta6 T6 + beta9 T9                                       Eq. (2)
      T1 = S
      T2 = S.S - IIS/3 I
      T3 = W.W - IIW/3 I
      T4 = S.W - W.S
      T6 = S.W.W + W.W.S - 2/3 IV I - IIW S
      T9 = W.S.W.W - W.W.S.W + 1/2 IIW (S.W - W.S)                  Eq. (3)
      S  = tau/2 (dU_i/dx_j + dU_j/dx_i),
      W  = tau/2 (dU_i/dx_j - dU_j/dx_i)                            p. 92
      tau = max(1/(Cmu omega), Ctau sqrt(nu/(Cmu k omega)))         p. 92
      IIS = tr(S.S), IIW = tr(W.W), IV = tr(S.W.W)                  p. 92
      beta1 = -N/Q, beta2 = 0, beta3 = -2 IV/(N Q1), beta4 = -1/Q,
      beta6 = -N/Q1, beta9 = 1/Q1                                   p. 92
        (beta4 = -1/Q as in the 2009 paper, Eq. (7); the 2012 paper
        prints -N/Q, a misprint: only -1/Q solves the implicit relation
        N a = -A1 S + (a W - W a), which the unit tests check in 3D)
      Q = (N^2 - 2 IIW)/A1, Q1 = Q/6 (2 N^2 - IIW)                  Eq. (4)
      N = C1' + 9/4 P/eps,  C1' = 9/4 (C1 - 1), C1 = 1.8            Eq. (5)
      N^3 - C1' N^2 - (2.7 IIS + 2 IIW) N + 2 C1' IIW = 0           Eq. (6)
      N from Eqs. (7)-(8) (the two-dimensional N equation)          p. 92-93
      T9 is not included in the published model (beta9 = 0)        p. 93
        (printed "beta0 = 0" in 2012; "beta9 = 0" in 2009, p. 4)
      A1 = 1.245 for the BSL-EARSM (WJ: 1.2)                        p. 93

    The cubic (6)-(8) is used exactly as printed (coefficients 2.7, 9/20,
    9/10, i.e. the WJ value A1 = 1.2); the recalibrated A1 = 1.245 enters
    Q and Q1 only, Eq. (4), as stated on p. 93. Eq. (7) gives the largest
    real root of (6); its term sign(P1 - sqrt P2)|P1 - sqrt P2|^(1/3) is
    evaluated as B/(P1 + sqrt P2)^(1/3), an exact identity that avoids
    cancellation.

    The strain is taken traceless (dev); for solenoidal velocity this is the
    published definition.

    OpenFOAM convention: fvc::grad(U)_ij = dU_j/dx_i, so the paper's
    velocity-gradient tensor is L = gradU.T(): S = tau symm(L),
    W = tau skew(L) (skew(T) = (T - T^T)/2).
\*---------------------------------------------------------------------------*/

#ifndef earsmAlgebra_H
#define earsmAlgebra_H

#include "scalar.H"
#include "tensor.H"
#include "symmTensor.H"

#include <cmath>

namespace Foam
{
namespace earsm
{

//- Published constants of the stress-strain relation
struct Coefficients
{
    scalar A1 = 1.245;      // p. 93 (BSL-EARSM recalibration)
    scalar C1 = 1.8;        // p. 92
    scalar Ctau = 6.0;      // p. 92 (Kolmogorov limiter, Durbin 1991)
    scalar Cmu = 0.09;      // p. 94: beta* = Cmu = 0.09
    bool includeT9 = false; // p. 93: beta9 = 0 in the published model

    scalar C1prime() const { return 2.25*(C1 - 1.0); }
};


//- Which closed-form branch of Eq. (7) was taken
enum class Branch { P2nonNegative = 0, P2negative = 1 };


//- Eq. (8): P1 and P2
inline void cubicP(const scalar IIS, const scalar IIW, const scalar C1p,
                   scalar& P1, scalar& P2)
{
    P1 = C1p*(C1p*C1p/27.0 + (9.0/20.0)*IIS - (2.0/3.0)*IIW);
    const scalar r = C1p*C1p/9.0 + (9.0/10.0)*IIS + (2.0/3.0)*IIW;
    P2 = P1*P1 - r*r*r;
}


//- Eq. (7): the root N of the cubic Eq. (6), closed form as published
inline scalar rootN
(
    const scalar IIS,
    const scalar IIW,
    const scalar C1p,
    Branch* branch = nullptr
)
{
    scalar P1, P2;
    cubicP(IIS, IIW, C1p, P1, P2);

    if (P2 >= 0)
    {
        // sign(P1 - sqrt P2)|P1 - sqrt P2|^(1/3) evaluated without
        // cancellation: (P1 + sqrt P2)(P1 - sqrt P2) = P1^2 - P2 = B^3,
        // B = C1'^2/9 + 9/10 IIS + 2/3 IIW, and P1 + sqrt P2 > 0 (P1 > 0)
        if (branch) *branch = Branch::P2nonNegative;
        const scalar B = C1p*C1p/9.0 + (9.0/10.0)*IIS + (2.0/3.0)*IIW;
        const scalar c = std::cbrt(P1 + std::sqrt(P2));
        return C1p/3.0 + c + B/c;
    }

    if (branch) *branch = Branch::P2negative;
    const scalar m = P1*P1 - P2;    // > P1^2 >= 0 on this branch
    scalar c = P1/std::sqrt(m);
    c = (c > 1.0 ? 1.0 : (c < -1.0 ? -1.0 : c));
    return C1p/3.0 + 2.0*std::pow(m, 1.0/6.0)*std::cos(std::acos(c)/3.0);
}


//- Residual of the cubic Eq. (6) at N
inline scalar cubicResidual
(
    const scalar N, const scalar IIS, const scalar IIW, const scalar C1p
)
{
    return N*N*N - C1p*N*N - (2.7*IIS + 2.0*IIW)*N + 2.0*C1p*IIW;
}


//- Coefficients beta_1 ... beta_9 (p. 92, Eq. (4))
struct Betas
{
    scalar b1, b2, b3, b4, b6, b9;
};

inline Betas betas
(
    const scalar N,
    const scalar IIW,
    const scalar IV,
    const scalar A1,
    const bool includeT9
)
{
    const scalar Q = (N*N - 2.0*IIW)/A1;
    const scalar Q1 = Q/6.0*(2.0*N*N - IIW);
    Betas b;
    b.b1 = -N/Q;
    b.b2 = 0.0;
    b.b3 = -2.0*IV/(N*Q1);
    b.b4 = -1.0/Q;
    b.b6 = -N/Q1;
    b.b9 = includeT9 ? 1.0/Q1 : 0.0;
    return b;
}


//- The tensor basis of Eq. (3)
struct Basis
{
    symmTensor T1, T2, T3, T4, T6, T9;
    scalar IIS, IIW, IV;
};

inline Basis basis(const symmTensor& S, const tensor& W)
{
    const tensor Wt(W);
    const tensor St(S);
    const tensor SW(St & Wt);
    const tensor WS(Wt & St);
    const tensor WW(Wt & Wt);

    Basis B;
    B.IIS = tr(St & St);
    B.IIW = tr(WW);
    B.IV = tr(St & WW);

    B.T1 = S;
    B.T2 = symm(St & St) - (B.IIS/3.0)*symmTensor::I;
    B.T3 = symm(WW) - (B.IIW/3.0)*symmTensor::I;
    B.T4 = symm(SW - WS);
    B.T6 =
        symm((SW & Wt) + (WW & St))
      - (2.0/3.0*B.IV)*symmTensor::I
      - B.IIW*S;
    B.T9 =
        symm((WS & WW) - (WW & SW))
      + (0.5*B.IIW)*symm(SW - WS);
    return B;
}


//- Everything the model needs at one point
struct State
{
    scalar tau;         // time scale [s]
    scalar IIS, IIW, IV;
    scalar N;
    Branch branch;
    Betas beta;
    symmTensor a;       // full anisotropy, Eq. (2)
    symmTensor aNL;     // a - beta1 T1 (the part not carried by nut)
    scalar CmuEff;      // -beta1/2: nut = CmuEff k tau
    scalar nut;         // -beta1 k tau/2 [m2/s]
};


//- Time scale with the Kolmogorov limiter (p. 92)
inline scalar timeScale
(
    const scalar k, const scalar omega, const scalar nu, const Coefficients& c
)
{
    const scalar t1 = 1.0/(c.Cmu*omega);
    const scalar t2 = c.Ctau*std::sqrt(nu/(c.Cmu*k*omega));
    return t1 > t2 ? t1 : t2;
}


//- Normalized strain and rotation from OpenFOAM's gradU (dU_j/dx_i) and tau
inline void normalized
(
    const tensor& gradU, const scalar tau, symmTensor& S, tensor& W
)
{
    const tensor L(gradU.T());          // L_ij = dU_i/dx_j
    S = tau*dev(symm(L));
    W = tau*skew(L);
}


//- Anisotropy from normalized S, W (any time scale)
inline State anisotropy
(
    const symmTensor& S, const tensor& W, const Coefficients& c
)
{
    const Basis B(basis(S, W));
    State st;
    st.tau = 0;
    st.IIS = B.IIS;
    st.IIW = B.IIW;
    st.IV = B.IV;
    st.N = rootN(B.IIS, B.IIW, c.C1prime(), &st.branch);
    st.beta = betas(st.N, B.IIW, B.IV, c.A1, c.includeT9);
    st.aNL =
        st.beta.b2*B.T2 + st.beta.b3*B.T3 + st.beta.b4*B.T4
      + st.beta.b6*B.T6 + st.beta.b9*B.T9;
    st.a = st.beta.b1*B.T1 + st.aNL;
    st.CmuEff = -0.5*st.beta.b1;
    st.nut = 0;
    return st;
}


//- Full evaluation at one point (k, omega already bounded by the caller)
inline State evaluate
(
    const tensor& gradU,
    const scalar k,
    const scalar omega,
    const scalar nu,
    const Coefficients& c
)
{
    const scalar tau = timeScale(k, omega, nu, c);
    symmTensor S;
    tensor W;
    normalized(gradU, tau, S, W);
    State st(anisotropy(S, W, c));
    st.tau = tau;
    st.nut = st.CmuEff*k*tau;
    return st;
}

} // End namespace earsm
} // End namespace Foam

#endif
