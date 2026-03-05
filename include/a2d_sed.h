/**
 * @file a2d_sed.h
 * @brief SED-filtered operations for expression-level Hessian convexification.
 *
 * Provides modified versions of A2D operations that filter negative-semi-definite
 * (NSD) curvature during hreverse, guaranteeing a PSD assembled Hessian.
 *
 * == Unary operations (sin, cos, exp, sqrt, log, etc.) ==
 *
 * For a unary operation f(a), the standard hreverse accumulates:
 *   a.hvalue += f'(a) * hval + f''(a) * bval * a.pvalue
 *
 * The second term f''(a)*bval*pvalue is the curvature contribution.
 * If f''(a)*bval < 0, this contributes NSD curvature to the Hessian.
 * SED filtering drops this term, keeping only PSD curvature.
 *
 * == Binary operations (multiply, divide) ==
 *
 * For a binary operation g(a,b), the cross-derivative contributes a rank-2
 * term to the Hessian via the 2x2 local Hessian:
 *
 *   M = bval * [[g_aa, g_ab], [g_ab, g_bb]]
 *
 * SED filtering PSD-projects this 2x2 matrix: eigendecompose, clip negative
 * eigenvalues to zero. The projected contribution is PSD by construction.
 *
 * == PSD Guarantee Theorem ==
 *
 * The assembled Hessian H_SED = sum_k J_k^T M_k^+ J_k where M_k^+ >= 0
 * is the PSD projection of each operation's local Hessian. Since congruence
 * preserves PSD (J^T M^+ J >= 0) and sum of PSD is PSD, H_SED >= 0.
 *
 * Reference: Griewank & Toint (1984), element-wise convexification.
 * Extension to binary operations and IPM: this work.
 */

#ifndef A2D_SED_H
#define A2D_SED_H

#include <cmath>

#include "a2dcore.h"

namespace A2D_SED {

// ============================================================================
// Helper: 2x2 symmetric PSD projection
// ============================================================================

/**
 * @brief PSD-project a 2x2 symmetric matrix and apply to a vector.
 *
 * Given M = [[m11, m12], [m12, m22]], compute M^+ * [pa, pb]
 * where M^+ is the PSD part of M (negative eigenvalues clipped to zero).
 *
 * @param m11, m12, m22  Entries of the symmetric 2x2 matrix
 * @param pa, pb         Input vector (perturbation values)
 * @param ha, hb         Output: M^+ * [pa, pb]
 */
template <typename T>
A2D_FUNCTION inline void psd_project_2x2(T m11, T m12, T m22, T pa, T pb,
                                          T& ha, T& hb) {
  T trace = m11 + m22;
  T det = m11 * m22 - m12 * m12;

  if (det >= T(0.0) && trace >= T(0.0)) {
    // Both eigenvalues >= 0: already PSD, keep full contribution
    ha = m11 * pa + m12 * pb;
    hb = m12 * pa + m22 * pb;
    return;
  }

  if (det >= T(0.0) && trace < T(0.0)) {
    // Both eigenvalues <= 0: fully NSD, drop everything
    ha = T(0.0);
    hb = T(0.0);
    return;
  }

  // det < 0: one positive and one negative eigenvalue.
  // Keep only the positive eigenvalue direction.
  T half_diff = T(0.5) * (m11 - m22);
  T disc = std::sqrt(half_diff * half_diff + m12 * m12);
  T lam_pos = T(0.5) * trace + disc;  // the positive eigenvalue

  // Eigenvector for lam_pos: v = [m12, lam_pos - m11]
  T v0 = m12;
  T v1 = lam_pos - m11;
  T vnorm2 = v0 * v0 + v1 * v1;

  if (vnorm2 < T(1e-30)) {
    ha = T(0.0);
    hb = T(0.0);
    return;
  }

  // M^+ * p = lam_pos * (v^T p / |v|^2) * v
  T coeff = lam_pos * (v0 * pa + v1 * pb) / vnorm2;
  ha = coeff * v0;
  hb = coeff * v1;
}

// ============================================================================
// Macro for SED-filtered second-order unary operations
// ============================================================================

/**
 * Identical to A2D_2ND_UNARY except hreverse checks sign(f''*bval)
 * and drops the curvature term when it would be NSD.
 */
#define A2D_2ND_UNARY_SED(OBJNAME, OPERNAME, FUNCBODY, TEMPBODY, DERIVBODY, \
                          DERIV2BODY)                                        \
                                                                             \
  template <class A, class Ta, class T, bool CA>                             \
  class OBJNAME : public A2D::A2DExpr<OBJNAME<A, Ta, T, CA>, T> {           \
   public:                                                                   \
    using expr_t = typename std::conditional<CA, const A2D::A2DExpr<A, Ta>,  \
                                             A2D::A2DExpr<A, Ta>>::type;     \
    using A_t = typename std::conditional<CA, A, A &>::type;                 \
    A2D_FUNCTION OBJNAME(expr_t &a0)                                         \
        : a(a0.self()),                                                      \
          val(0.0),                                                          \
          bval(0.0),                                                         \
          pval(0.0),                                                         \
          hval(0.0),                                                         \
          tmp(0.0) {}                                                        \
    A2D_FUNCTION void eval() {                                               \
      a.eval();                                                              \
      val = (FUNCBODY);                                                      \
      tmp = (TEMPBODY);                                                      \
    }                                                                        \
    A2D_FUNCTION void reverse() {                                            \
      a.bvalue() += (DERIVBODY)*bval;                                        \
      a.reverse();                                                           \
    }                                                                        \
    A2D_FUNCTION void hforward() {                                           \
      a.hforward();                                                          \
      pval = (DERIVBODY)*a.pvalue();                                         \
    }                                                                        \
    A2D_FUNCTION void hreverse() {                                           \
      T curv = (DERIV2BODY)*bval;                                            \
      if (curv >= T(0.0)) {                                                  \
        a.hvalue() += (DERIVBODY)*hval + curv * a.pvalue();                  \
      } else {                                                               \
        a.hvalue() += (DERIVBODY)*hval;                                      \
      }                                                                      \
      a.hreverse();                                                          \
    }                                                                        \
    A2D_FUNCTION void bzero() {                                              \
      bval = T(0.0);                                                         \
      a.bzero();                                                             \
    }                                                                        \
    A2D_FUNCTION void hzero() {                                              \
      hval = T(0.0);                                                         \
      a.hzero();                                                             \
    }                                                                        \
    A2D_FUNCTION T &value() { return val; }                                  \
    A2D_FUNCTION const T &value() const { return val; }                      \
    A2D_FUNCTION T &bvalue() { return bval; }                                \
    A2D_FUNCTION const T &bvalue() const { return bval; }                    \
    A2D_FUNCTION T &pvalue() { return pval; }                                \
    A2D_FUNCTION const T &pvalue() const { return pval; }                    \
    A2D_FUNCTION T &hvalue() { return hval; }                                \
    A2D_FUNCTION const T &hvalue() const { return hval; }                    \
                                                                             \
   private:                                                                  \
    A_t a;                                                                   \
    T val, bval, pval, hval, tmp;                                            \
  };                                                                         \
  template <class A, class Ta>                                               \
  A2D_FUNCTION auto OPERNAME(const A2D::A2DExpr<A, Ta> &a) {                 \
    using T = typename A2D::remove_const_and_refs<Ta>::type;                 \
    return OBJNAME<A, Ta, T, true>(a);                                       \
  }                                                                          \
  template <class A, class Ta>                                               \
  A2D_FUNCTION auto OPERNAME(A2D::A2DExpr<A, Ta> &a) {                       \
    using T = typename A2D::remove_const_and_refs<Ta>::type;                 \
    return OBJNAME<A, Ta, T, false>(a);                                      \
  }

// Scalar passthrough overloads for unary ops
inline double exp(double x) { return std::exp(x); }
inline double sin(double x) { return std::sin(x); }
inline double cos(double x) { return std::cos(x); }
inline double sqrt(double x) { return std::sqrt(x); }
inline double log(double x) { return std::log(x); }
inline double acos(double x) { return std::acos(x); }
inline double asin(double x) { return std::asin(x); }

// SED-filtered unary operations
// For each: FUNCBODY, TEMPBODY, DERIVBODY, DERIV2BODY

// exp(a): f' = e^a, f'' = e^a
A2D_2ND_UNARY_SED(ExpExpr2SED, exp, exp(a.value()), val, tmp, tmp)

// sin(a): f' = cos(a), f'' = -sin(a)
A2D_2ND_UNARY_SED(SinExpr2SED, sin, sin(a.value()), cos(a.value()), tmp, -val)

// cos(a): f' = -sin(a), f'' = -cos(a)
A2D_2ND_UNARY_SED(CosExpr2SED, cos, cos(a.value()), sin(a.value()), -tmp, -val)

// sqrt(a): f' = 1/(2*sqrt(a)), f'' = -1/(4*a^(3/2))
A2D_2ND_UNARY_SED(SqrtExpr2SED, sqrt, sqrt(a.value()), 1.0 / val, 0.5 * tmp,
                   -0.25 * tmp * tmp * tmp)

// log(a): f' = 1/a, f'' = -1/a^2
A2D_2ND_UNARY_SED(LogExpr2SED, log, log(a.value()), 1.0 / a.value(), tmp,
                   -tmp * tmp)

// acos(a): f' = -1/sqrt(1-a^2), f'' = -a/(1-a^2)^(3/2)
A2D_2ND_UNARY_SED(ACosExpr2SED, acos, acos(a.value()),
                   -1.0 / sqrt(1.0 - a.value() * a.value()), tmp,
                   -a.value() / pow(1.0 - a.value() * a.value(), 1.5))

// asin(a): f' = 1/sqrt(1-a^2), f'' = a/(1-a^2)^(3/2)
A2D_2ND_UNARY_SED(ASinExpr2SED, asin, asin(a.value()),
                   1.0 / sqrt(1.0 - a.value() * a.value()), tmp,
                   a.value() / pow(1.0 - a.value() * a.value(), 1.5))

// atan(a): f' = 1/(1+a^2), f'' = -2a/(1+a^2)^2
inline double atan(double x) { return std::atan(x); }
A2D_2ND_UNARY_SED(ATanExpr2SED, atan, atan(a.value()),
                   1.0 / (1.0 + a.value() * a.value()), tmp,
                   -2.0 * a.value() /
                       ((1.0 + a.value() * a.value()) *
                        (1.0 + a.value() * a.value())))

#undef A2D_2ND_UNARY_SED

// ============================================================================
// SED-filtered MULTIPLICATION: v = a * b
// ============================================================================
//
// Local Hessian of (a*b): [[0, 1], [1, 0]]
// Scaled by bval: M = bval * [[0,1],[1,0]], eigenvalues +bval and -bval.
//
// PSD projection (closed-form, no sqrt needed):
//   bval >= 0: keep eigenvalue +bval along (1,1)/sqrt(2)
//     M^+ * p = (bval/2) * [pa+pb, pa+pb]
//   bval < 0:  keep eigenvalue |bval| along (1,-1)/sqrt(2)
//     M^+ * p = (|bval|/2) * [pa-pb, -(pa-pb)]

template <class A, class Ta, class B, class Tb, class T, bool CA, bool CB>
class MultExpr2SED
    : public A2D::A2DExpr<MultExpr2SED<A, Ta, B, Tb, T, CA, CB>, T> {
 public:
  using expr_a_t = typename std::conditional<CA, const A2D::A2DExpr<A, Ta>,
                                             A2D::A2DExpr<A, Ta>>::type;
  using expr_b_t = typename std::conditional<CB, const A2D::A2DExpr<B, Tb>,
                                             A2D::A2DExpr<B, Tb>>::type;
  using A_t = typename std::conditional<CA, A, A &>::type;
  using B_t = typename std::conditional<CB, B, B &>::type;

  A2D_FUNCTION MultExpr2SED(expr_a_t &a0, expr_b_t &b0)
      : a(a0.self()), b(b0.self()), val(0.0), bval(0.0), pval(0.0),
        hval(0.0) {}

  A2D_FUNCTION void eval() {
    a.eval();
    b.eval();
    val = a.value() * b.value();
  }
  A2D_FUNCTION void reverse() {
    a.bvalue() += b.value() * bval;
    b.bvalue() += a.value() * bval;
    a.reverse();
    b.reverse();
  }
  A2D_FUNCTION void hforward() {
    a.hforward();
    b.hforward();
    pval = a.pvalue() * b.value() + a.value() * b.pvalue();
  }
  A2D_FUNCTION void hreverse() {
    // First-order chain rule (always kept)
    a.hvalue() += b.value() * hval;
    b.hvalue() += a.value() * hval;

    // Cross-derivative PSD projection
    // M = bval * [[0,1],[1,0]], eigenvalues +-|bval|
    if (bval >= T(0.0)) {
      // Keep positive eigenvalue along (1,1)/sqrt(2)
      T sp = bval * T(0.5) * (a.pvalue() + b.pvalue());
      a.hvalue() += sp;
      b.hvalue() += sp;
    } else {
      // Keep positive eigenvalue |bval| along (1,-1)/sqrt(2)
      T dp = (-bval) * T(0.5) * (a.pvalue() - b.pvalue());
      a.hvalue() += dp;
      b.hvalue() -= dp;
    }

    a.hreverse();
    b.hreverse();
  }
  A2D_FUNCTION void bzero() {
    bval = T(0.0);
    a.bzero();
    b.bzero();
  }
  A2D_FUNCTION void hzero() {
    hval = T(0.0);
    a.hzero();
    b.hzero();
  }
  A2D_FUNCTION T &value() { return val; }
  A2D_FUNCTION const T &value() const { return val; }
  A2D_FUNCTION T &bvalue() { return bval; }
  A2D_FUNCTION const T &bvalue() const { return bval; }
  A2D_FUNCTION T &pvalue() { return pval; }
  A2D_FUNCTION const T &pvalue() const { return pval; }
  A2D_FUNCTION T &hvalue() { return hval; }
  A2D_FUNCTION const T &hvalue() const { return hval; }

 private:
  A_t a;
  B_t b;
  T val, bval, pval, hval;
};

// Four overloads for const/non-const combinations
template <class A, class Ta, class B, class Tb>
A2D_FUNCTION auto multiply(const A2D::A2DExpr<A, Ta> &a,
                            const A2D::A2DExpr<B, Tb> &b) {
  using T = typename A2D::remove_const_and_refs<Ta>::type;
  return MultExpr2SED<A, Ta, B, Tb, T, true, true>(a, b);
}
template <class A, class Ta, class B, class Tb>
A2D_FUNCTION auto multiply(A2D::A2DExpr<A, Ta> &a,
                            const A2D::A2DExpr<B, Tb> &b) {
  using T = typename A2D::remove_const_and_refs<Ta>::type;
  return MultExpr2SED<A, Ta, B, Tb, T, false, true>(a, b);
}
template <class A, class Ta, class B, class Tb>
A2D_FUNCTION auto multiply(const A2D::A2DExpr<A, Ta> &a,
                            A2D::A2DExpr<B, Tb> &b) {
  using T = typename A2D::remove_const_and_refs<Ta>::type;
  return MultExpr2SED<A, Ta, B, Tb, T, true, false>(a, b);
}
template <class A, class Ta, class B, class Tb>
A2D_FUNCTION auto multiply(A2D::A2DExpr<A, Ta> &a, A2D::A2DExpr<B, Tb> &b) {
  using T = typename A2D::remove_const_and_refs<Ta>::type;
  return MultExpr2SED<A, Ta, B, Tb, T, false, false>(a, b);
}

// ============================================================================
// SED-filtered DIVISION: v = a / b
// ============================================================================
//
// Local Hessian of (a/b) with tmp = 1/b:
//   [[0,       -1/b^2    ],      = [[0,       -tmp^2    ],
//    [-1/b^2,   2a/b^3   ]]         [-tmp^2,   2a*tmp^3 ]]
//
// Scaled by bval, this always has det < 0 (one positive, one negative
// eigenvalue). Uses the general 2x2 PSD projection helper.

template <class A, class Ta, class B, class Tb, class T, bool CA, bool CB>
class DivideExpr2SED
    : public A2D::A2DExpr<DivideExpr2SED<A, Ta, B, Tb, T, CA, CB>, T> {
 public:
  using expr_a_t = typename std::conditional<CA, const A2D::A2DExpr<A, Ta>,
                                             A2D::A2DExpr<A, Ta>>::type;
  using expr_b_t = typename std::conditional<CB, const A2D::A2DExpr<B, Tb>,
                                             A2D::A2DExpr<B, Tb>>::type;
  using A_t = typename std::conditional<CA, A, A &>::type;
  using B_t = typename std::conditional<CB, B, B &>::type;

  A2D_FUNCTION DivideExpr2SED(expr_a_t &a0, expr_b_t &b0)
      : a(a0.self()), b(b0.self()), val(0.0), bval(0.0), pval(0.0),
        hval(0.0), tmp(0.0) {}

  A2D_FUNCTION void eval() {
    a.eval();
    b.eval();
    val = a.value() / b.value();
    tmp = T(1.0) / b.value();
  }
  A2D_FUNCTION void reverse() {
    a.bvalue() += tmp * bval;
    b.bvalue() += -tmp * tmp * a.value() * bval;
    a.reverse();
    b.reverse();
  }
  A2D_FUNCTION void hforward() {
    a.hforward();
    b.hforward();
    pval = tmp * (a.pvalue() - tmp * a.value() * b.pvalue());
  }
  A2D_FUNCTION void hreverse() {
    // First-order chain rule (always kept)
    // da/dv = 1/b = tmp,  db/dv = -a/b^2 = -a*tmp^2
    a.hvalue() += tmp * hval;
    b.hvalue() += -a.value() * tmp * tmp * hval;

    // Second-order: PSD project the 2x2 curvature matrix
    // M = bval * [[0, -tmp^2], [-tmp^2, 2*a*tmp^3]]
    T m12 = -bval * tmp * tmp;
    T m22 = T(2.0) * a.value() * bval * tmp * tmp * tmp;
    T ha, hb;
    psd_project_2x2(T(0.0), m12, m22, a.pvalue(), b.pvalue(), ha, hb);
    a.hvalue() += ha;
    b.hvalue() += hb;

    a.hreverse();
    b.hreverse();
  }
  A2D_FUNCTION void bzero() {
    bval = T(0.0);
    a.bzero();
    b.bzero();
  }
  A2D_FUNCTION void hzero() {
    hval = T(0.0);
    a.hzero();
    b.hzero();
  }
  A2D_FUNCTION T &value() { return val; }
  A2D_FUNCTION const T &value() const { return val; }
  A2D_FUNCTION T &bvalue() { return bval; }
  A2D_FUNCTION const T &bvalue() const { return bval; }
  A2D_FUNCTION T &pvalue() { return pval; }
  A2D_FUNCTION const T &pvalue() const { return pval; }
  A2D_FUNCTION T &hvalue() { return hval; }
  A2D_FUNCTION const T &hvalue() const { return hval; }

 private:
  A_t a;
  B_t b;
  T val, bval, pval, hval, tmp;
};

// Four overloads for const/non-const combinations
template <class A, class Ta, class B, class Tb>
A2D_FUNCTION auto divide(const A2D::A2DExpr<A, Ta> &a,
                          const A2D::A2DExpr<B, Tb> &b) {
  using T = typename A2D::remove_const_and_refs<Ta>::type;
  return DivideExpr2SED<A, Ta, B, Tb, T, true, true>(a, b);
}
template <class A, class Ta, class B, class Tb>
A2D_FUNCTION auto divide(A2D::A2DExpr<A, Ta> &a,
                          const A2D::A2DExpr<B, Tb> &b) {
  using T = typename A2D::remove_const_and_refs<Ta>::type;
  return DivideExpr2SED<A, Ta, B, Tb, T, false, true>(a, b);
}
template <class A, class Ta, class B, class Tb>
A2D_FUNCTION auto divide(const A2D::A2DExpr<A, Ta> &a,
                          A2D::A2DExpr<B, Tb> &b) {
  using T = typename A2D::remove_const_and_refs<Ta>::type;
  return DivideExpr2SED<A, Ta, B, Tb, T, true, false>(a, b);
}
template <class A, class Ta, class B, class Tb>
A2D_FUNCTION auto divide(A2D::A2DExpr<A, Ta> &a, A2D::A2DExpr<B, Tb> &b) {
  using T = typename A2D::remove_const_and_refs<Ta>::type;
  return DivideExpr2SED<A, Ta, B, Tb, T, false, false>(a, b);
}

// ============================================================================
// SED-filtered POWER: v = a^b (b is constant scalar)
// ============================================================================
//
// When b is a constant exponent, this is effectively unary:
//   f(a) = a^b,  f' = b*a^(b-1),  f'' = b*(b-1)*a^(b-2)
// SED filter: check sign of f''(a)*bval, drop curvature if NSD.

template <class A, class Ta, class T, bool CA>
class PowExpr2SED : public A2D::A2DExpr<PowExpr2SED<A, Ta, T, CA>, T> {
 public:
  using expr_t = typename std::conditional<CA, const A2D::A2DExpr<A, Ta>,
                                           A2D::A2DExpr<A, Ta>>::type;
  using A_t = typename std::conditional<CA, A, A &>::type;

  A2D_FUNCTION PowExpr2SED(expr_t &a0, T b0)
      : a(a0.self()), b_exp(b0), val(0.0), bval(0.0), pval(0.0), hval(0.0),
        deriv1(0.0) {}

  A2D_FUNCTION void eval() {
    a.eval();
    val = std::pow(a.value(), b_exp);
    // Cache first derivative for reuse
    deriv1 = b_exp * std::pow(a.value(), b_exp - T(1.0));
  }
  A2D_FUNCTION void reverse() {
    a.bvalue() += deriv1 * bval;
    a.reverse();
  }
  A2D_FUNCTION void hforward() {
    a.hforward();
    pval = deriv1 * a.pvalue();
  }
  A2D_FUNCTION void hreverse() {
    T deriv2 = b_exp * (b_exp - T(1.0)) * std::pow(a.value(), b_exp - T(2.0));
    T curv = deriv2 * bval;
    if (curv >= T(0.0)) {
      a.hvalue() += deriv1 * hval + curv * a.pvalue();
    } else {
      a.hvalue() += deriv1 * hval;
    }
    a.hreverse();
  }
  A2D_FUNCTION void bzero() {
    bval = T(0.0);
    a.bzero();
  }
  A2D_FUNCTION void hzero() {
    hval = T(0.0);
    a.hzero();
  }
  A2D_FUNCTION T &value() { return val; }
  A2D_FUNCTION const T &value() const { return val; }
  A2D_FUNCTION T &bvalue() { return bval; }
  A2D_FUNCTION const T &bvalue() const { return bval; }
  A2D_FUNCTION T &pvalue() { return pval; }
  A2D_FUNCTION const T &pvalue() const { return pval; }
  A2D_FUNCTION T &hvalue() { return hval; }
  A2D_FUNCTION const T &hvalue() const { return hval; }

 private:
  A_t a;
  T b_exp;  // constant exponent
  T val, bval, pval, hval, deriv1;
};

template <class A, class Ta>
A2D_FUNCTION auto pow(const A2D::A2DExpr<A, Ta> &a,
                       typename A2D::remove_const_and_refs<Ta>::type b) {
  using T = typename A2D::remove_const_and_refs<Ta>::type;
  return PowExpr2SED<A, Ta, T, true>(a, b);
}
template <class A, class Ta>
A2D_FUNCTION auto pow(A2D::A2DExpr<A, Ta> &a,
                       typename A2D::remove_const_and_refs<Ta>::type b) {
  using T = typename A2D::remove_const_and_refs<Ta>::type;
  return PowExpr2SED<A, Ta, T, false>(a, b);
}

// Scalar passthrough for pow (when base is not an A2DExpr)
inline double pow(double x, double y) { return std::pow(x, y); }

}  // namespace A2D_SED

#endif  // A2D_SED_H
