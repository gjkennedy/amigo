#ifndef AMIGO_INTERIOR_POINT_BACKEND_H
#define AMIGO_INTERIOR_POINT_BACKEND_H

/*
  Primal-dual interior-point backend for the full-space solver.

  Problem formulation (after slack introduction for inequalities):

    min  f(x)
    s.t. c(x) = 0          (equalities, including c_k(x) - s_k = 0 for ineq)
         xL <= x <= xU      (bounds on all primals: design vars + slacks)

  Full 8-block Newton system:

    Block 1 (x stationarity):     rhs_x  = grad_f + J^T y - z_L + z_U
    Block 2 (s stationarity):     rhs_s  = -y_d - v_L + v_U
    Block 3 (eq feasibility):     rhs_yc = c(x)
    Block 4 (ineq feasibility):   rhs_yd = d(x) - s
    Block 5 (lower compl, x):     rhs_zL = gap_xL * z_L - mu
    Block 6 (upper compl, x):     rhs_zU = gap_xU * z_U - mu
    Block 7 (lower compl, s):     rhs_vL = gap_sL * v_L - mu
    Block 8 (upper compl, s):     rhs_vU = gap_sU * v_U - mu

  Condensation to 4-block augmented system:

    augRhs_x = rhs_x + rhs_zL / gap_xL - rhs_zU / gap_xU
    augRhs_s = rhs_s + rhs_vL / gap_sL - rhs_vU / gap_sU
    (= simplifies to: -grad + mu/gap_L - mu/gap_U, as z terms cancel)

  In amigo, x and s share the primal vector, and y_c and y_d share
  the constraint vector, so the 4-block reduces to a 2-block solve:

    [ W + Sigma + delta_w*I    J^T      ] [ dx   ]   [ -augRhs ]
    [       J              -delta_c*I   ] [ dlam ] = [ -rhs_c  ]

  Back-substitution for bound dual steps:
    dz_L = -(rhs_zL + z_L * dx) / gap_xL
    dz_U = -(rhs_zU - z_U * dx) / gap_xU
*/

#include <cmath>
#include <memory>

#include "a2dcore.h"
#include "amigo.h"

namespace amigo {

template <typename T>
class OptVector;

namespace detail {

/**
 * @brief OptProblemInfo contains the information about the primal variables and
 * constraints on the host or device.
 */
template <typename T>
struct OptProblemInfo {
  int num_primals = 0;
  int num_constraints = 0;
  const int* primal_indices = nullptr;
  const int* constraint_indices = nullptr;
  const T *lbx = nullptr, *ubx = nullptr;
  const T* lbh = nullptr;
};

// Pointers into OptVector storage. T may be const-qualified for read-only
// access.
template <typename T>
struct OptState {
  T* x = nullptr;
  T* zl = nullptr;
  T* zu = nullptr;

  template <ExecPolicy policy, typename R>
  static OptState make(std::shared_ptr<OptVector<R>> vars) {
    OptState<T> state{};
    state.x = vars->template get_solution_array<policy>();
    vars->template get_bound_duals<policy>(&state.zl, &state.zu);
    return state;
  }
};

// Project all primals into the strict interior of their bounds (Section 3.6).
// For one-sided bounds: x <- max(x, lb + kappa1 * max(1, |lb|))  or similar.
// For two-sided bounds: x is projected into [lb + p_l, ub - p_u] where
//   p_l = min(kappa1 * max(1, |lb|), kappa2 * (ub - lb))
//   p_u = min(kappa1 * max(1, |ub|), kappa2 * (ub - lb))
// Defaults: kappa1 = 0.01, kappa2 = 0.01.
// Must be called before initialize_bound_duals.
template <typename T>
void project_primals_into_interior(const OptProblemInfo<T>& info, T* xlam,
                                   T kappa1 = 1e-2, T kappa2 = 1e-2) {
  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    T x = xlam[idx];
    T lb = info.lbx[i];
    T ub = info.ubx[i];
    bool has_lb = !std::isinf(lb);
    bool has_ub = !std::isinf(ub);

    if (has_lb && has_ub) {
      T range = ub - lb;
      T pl = A2D::min2(kappa1 * A2D::max2(T(1), std::abs(lb)), kappa2 * range);
      T pu = A2D::min2(kappa1 * A2D::max2(T(1), std::abs(ub)), kappa2 * range);
      xlam[idx] = A2D::max2(A2D::min2(x, ub - pu), lb + pl);
    } else if (has_lb) {
      xlam[idx] = A2D::max2(x, lb + kappa1 * A2D::max2(T(1), std::abs(lb)));
    } else if (has_ub) {
      xlam[idx] = A2D::min2(x, ub - kappa1 * A2D::max2(T(1), std::abs(ub)));
    }
  }
}

// Initialize bound duals to 1.0 for all finite bounds (Section 3.6).
// Must be called after project_primals_into_interior.
template <typename T>
void initialize_bound_duals(T mu, const OptProblemInfo<T>& info, const T* xlam,
                            T* zl, T* zu) {
  for (int i = 0; i < info.num_primals; i++) {
    zl[i] = mu;
    zu[i] = mu;
    if (std::isinf(info.lbx[i])) {
      zl[i] = 0.0;
    }
    if (std::isinf(info.ubx[i])) {
      zu[i] = 0.0;
    }
  }
}

// Augmented system RHS via 8-block to 4-block condensation.
//
// For each primal i, the 8-block RHS has three relevant blocks:
//   rhs_stat   = grad[i] - zl[i] + zu[i]       (stationarity, Blocks 1-2)
//   rhs_complL = gap_L * zl - mu                (complementarity, Block 5/7)
//   rhs_complU = gap_U * zu - mu                (complementarity, Block 6/8)
//
// Condensation folds complementarity into stationarity:
//   augRhs = rhs_stat + rhs_complL/gap_L - rhs_complU/gap_U
//
// Output is negated: res = -augRhs  (convention: K * px = res gives Newton
// step)
template <typename T>
void compute_residual(T mu, const OptProblemInfo<T>& info,
                      OptState<const T>& current, const T* grad, T* res) {
  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];

    // Stationarity residual (Blocks 1-2: rhs_x or rhs_s)
    T r = grad[idx] - current.zl[i] + current.zu[i];

    // Condense complementarity into stationarity
    if (!std::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      r += (gap * current.zl[i] - mu) / gap;  // +rhs_complL / gap_L
    }
    if (!std::isinf(info.ubx[i])) {
      T gap = info.ubx[i] - x;
      r -= (gap * current.zu[i] - mu) / gap;  // -rhs_complU / gap_U
    }
    res[idx] = -r;
  }

  // Constraint feasibility (Blocks 3-4: rhs_yc, rhs_yd)
  for (int j = 0; j < info.num_constraints; j++) {
    int idx = info.constraint_indices[j];
    res[idx] = -(grad[idx] - info.lbh[j]);
  }
}

// Barrier diagonal Sigma for the augmented system.
//
//   sigma_x[i] = z_L[i]/gap_xL[i] + z_U[i]/gap_xU[i]  (Block 1,1)
//   sigma_s[k] = v_L[k]/gap_sL[k] + v_U[k]/gap_sU[k]  (Block 2,2)
//
// In amigo, x and s share the primal vector, so both use the same looinfo.
// Constraint diagonal entries are zero (regularization delta_c added
// separately).
template <typename T>
void compute_diagonal(const OptProblemInfo<T>& info, OptState<const T>& current,
                      T* diag) {
  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];

    T sigma = 0.0;
    if (!std::isinf(info.lbx[i])) {
      T gap = (x - info.lbx[i]);
      sigma += current.zl[i] / gap;
    }
    if (!std::isinf(info.ubx[i])) {
      T gap = (info.ubx[i] - x);
      sigma += current.zu[i] / gap;
    }
    diag[idx] = sigma;
  }
}

// Bound dual back-substitution.
//
// After solving the augmented system for (dx, dlam), recover the bound
// dual steps that were eliminated during RHS condensation.
//
// The complementarity blocks give:
//   rhs_complL = gap_L * zl - mu     (Block 5/7)
//   rhs_complU = gap_U * zu - mu     (Block 6/8)
//
// Back-substitution with sign correction:
//   dzl = -(rhs_complL + zl * dx) / gap_L
//   dzu = -(rhs_complU - zu * dx) / gap_U
template <typename T>
void compute_bound_dual_step(T mu, const OptProblemInfo<T>& info,
                             OptState<const T>& current, const T* px, T* dzl,
                             T* dzu) {
  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];
    T dx = px[idx];
    dzl[i] = dzu[i] = 0.0;

    if (!std::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      T rhs = gap * current.zl[i] - mu;
      dzl[i] = -(rhs + current.zl[i] * dx) / gap;
    }
    if (!std::isinf(info.ubx[i])) {
      T gap = info.ubx[i] - x;
      T rhs = gap * current.zu[i] - mu;
      dzu[i] = -(rhs - current.zu[i] * dx) / gap;
    }
  }
}

// Fraction-to-the-boundary rule. Finds the largest step alpha in (0,1]
// such that all primals stay within bounds and all duals stay positive:
//   x + alpha*dx >= (1-tau)*(x - lb)  for each finite lower bound
//   ub - (x + alpha*dx) >= (1-tau)*(ub - x)  for each finite upper bound
//   zl + alpha*dzl >= (1-tau)*zl, zu + alpha*dzu >= (1-tau)*zu
template <typename T>
void compute_max_step(T tau, const OptProblemInfo<T>& info,
                      OptState<const T>& current, OptState<const T>& step,
                      T& ax, int& xi, T& az, int& zi) {
  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];
    T dx = step.x[idx];
    T dzl = step.zl[i];
    T dzu = step.zu[i];

    if (!std::isinf(info.lbx[i])) {
      if (dx < 0.0) {
        T gap = x - info.lbx[i];
        T a = -tau * gap / dx;
        if (a < ax) {
          ax = a;
          xi = idx;
        }
      }
      if (dzl < 0.0) {
        T a = -tau * current.zl[i] / dzl;
        if (a < az) {
          az = a;
          zi = idx;
        }
      }
    }
    if (!std::isinf(info.ubx[i])) {
      if (dx > 0.0) {
        T gap = info.ubx[i] - x;
        T a = tau * gap / dx;
        if (a < ax) {
          ax = a;
          xi = idx;
        }
      }
      if (dzu < 0.0) {
        T a = -tau * current.zu[i] / dzu;
        if (a < az) {
          az = a;
          zi = idx;
        }
      }
    }
  }
}

// Apply the full primal-dual-bound trial step (eq. 14-15).
//   xlam_new = xlam + alpha_x * dxlam   (primals + multipliers)
//   zl_new   = zl   + alpha_z * dzl     (lower bound duals)
//   zu_new   = zu   + alpha_z * dzu     (upper bound duals)
template <typename T>
void apply_step(T ax, T az, const OptProblemInfo<T>& info,
                OptState<const T>& current, OptState<const T>& step,
                OptState<T>& result) {
  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    result.x[idx] = current.x[idx] + ax * step.x[idx];
  }
  for (int i = 0; i < info.num_constraints; i++) {
    int idx = info.constraint_indices[i];
    result.x[idx] = current.x[idx] + az * step.x[idx];
  }
  for (int i = 0; i < info.num_primals; i++) {
    if (!std::isinf(info.lbx[i])) {
      result.zl[i] = current.zl[i] + az * step.zl[i];
    }
    if (!std::isinf(info.ubx[i])) {
      result.zu[i] = current.zu[i] + az * step.zu[i];
    }
  }
}

// Average complementarity mu_avg = sum(gap*z) / n_bounds, and
// minimum complementarity product (for uniformity measure xi).
template <typename T>
void compute_complementarity(const OptProblemInfo<T>& info,
                             OptState<const T>& current, T partial_sum[],
                             T& local_min) {
  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];
    if (!std::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      T comp = gap * current.zl[i];
      partial_sum[0] += comp;
      partial_sum[1] += 1.0;
      local_min = A2D::min2(local_min, comp);
    }
    if (!std::isinf(info.ubx[i])) {
      T gap = info.ubx[i] - x;
      T comp = gap * current.zu[i];
      partial_sum[0] += comp;
      partial_sum[1] += 1.0;
      local_min = A2D::min2(local_min, comp);
    }
  }
}

// Optimality error E_mu with three components (infinity norms):
//   dual    = max |grad_i - zl_i + zu_i|           (stationarity)
//   primal  = max |c_j(x) - target_j|              (feasibility)
//   comp    = max |gap_i * z_i - mu|                (complementarity)
template <typename T>
void compute_kkt_error(T mu, const OptProblemInfo<T>& info,
                       OptState<const T>& current, const T* grad, T& dual,
                       T& primal, T& comp) {
  dual = primal = comp = 0.0;

  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];

    dual = A2D::max2(dual, std::abs(grad[idx] - current.zl[i] + current.zu[i]));
    if (!std::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      comp = A2D::max2(comp, std::abs(gap * current.zl[i] - mu));
    }
    if (!std::isinf(info.ubx[i])) {
      T gap = info.ubx[i] - x;
      comp = A2D::max2(comp, std::abs(gap * current.zu[i] - mu));
    }
  }

  for (int j = 0; j < info.num_constraints; j++) {
    int idx = info.constraint_indices[j];
    primal = A2D::max2(primal, std::abs(grad[idx] - info.lbh[j]));
  }
}

// Barrier log-sum: -mu * sum_i ln(x_i - lb_i) - mu * sum_i ln(ub_i - x_i).
// Added to the objective f(x) to form the barrier objective phi_mu(x).
template <typename T>
T compute_log_barrier(T mu, const OptProblemInfo<T>& info,
                      OptState<const T>& current) {
  T barrier = 0.0;
  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];

    if (!std::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      if (gap > 0) {
        barrier -= mu * std::log(gap);
      }
    }
    if (!std::isinf(info.ubx[i])) {
      T gap = info.ubx[i] - x;
      if (gap > 0) {
        barrier -= mu * std::log(gap);
      }
    }
  }
  return barrier;
}

// Directional derivative of the log barrier function along the search
// direction:
//   dphi = sum_i (grad_i * dx_i - mu * dx_i / gap_l_i + mu * dx_i / gap_u_i)
// Used in the Armijo condition and switching condition of the filter line
// search.
template <typename T>
T compute_log_barrier_derivative(T mu, const OptProblemInfo<T>& info,
                                 OptState<const T>& current,
                                 OptState<const T>& step) {
  T deriv = 0.0;
  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];
    T dx = step.x[idx];
    if (!std::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      deriv -= mu * dx / gap;
    }
    if (!std::isinf(info.ubx[i])) {
      T gap = info.ubx[i] - x;
      deriv += mu * dx / gap;
    }
  }
  return deriv;
}

template <typename T>
T compute_infeasibility(const OptProblemInfo<T>& info, const T grad[]) {
  T result = 0.0;
  for (int j = 0; j < info.num_constraints; j++) {
    int idx = info.constraint_indices[j];
    result += std::abs(grad[idx] - info.lbh[j]);
  }
  return result;
}

// Compute the sum of the squared complementarity products (for quality function
// evaluation).
template <typename T>
T compute_sum_squared_complementarity(T mu, const OptProblemInfo<T>& info,
                                      OptState<const T>& current) {
  T result = 0.0;
  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    T x = current.x[idx];
    if (!std::isinf(info.lbx[i])) {
      T gap = x - info.lbx[i];
      T r = gap * current.zl[i] - mu;
      result += r * r;
    }
    if (!std::isinf(info.ubx[i])) {
      T gap = x - info.lbx[i];
      T r = gap * current.zu[i] - mu;
      result += r * r;
    }
  }
  return result;
}

// Dual residual vector: r_d[i] = grad[i] - zl[i] + zu[i] for primals, 0
// elsewhere. Used by the quality function to compute the cross term r_d^T *
// (Hessian_mod * dx).
template <typename T>
void compute_dual_residual(const OptProblemInfo<T>& info,
                           OptState<const T>& current, const T* grad, T* out,
                           int size) {
  for (int i = 0; i < info.num_primals; i++) {
    int idx = info.primal_indices[i];
    out[idx] = grad[idx] - current.zl[i] + current.zu[i];
  }
}

// Relax bounds by a small factor to avoid numerical issues at exact bounds.
//   x_L -= min(constr_viol_tol, factor * max(1, |x_L|))
//   x_U += min(constr_viol_tol, factor * max(1, |x_U|))
// Default: bound_relax_factor = 1e-8, constr_viol_tol = 1e-4.
template <typename T>
void relax_bounds(OptProblemInfo<T>& info, T* lbx_buf, T* ubx_buf,
                  T factor = 1e-8, T constr_viol_tol = 1e-4) {
  for (int i = 0; i < info.num_primals; i++) {
    if (!std::isinf(info.lbx[i])) {
      T delta = A2D::min2(constr_viol_tol,
                          factor * A2D::max2(T(1), std::abs(info.lbx[i])));
      lbx_buf[i] = info.lbx[i] - delta;
    } else {
      lbx_buf[i] = info.lbx[i];
    }

    if (!std::isinf(info.ubx[i])) {
      T delta = A2D::min2(constr_viol_tol,
                          factor * A2D::max2(T(1), std::abs(info.ubx[i])));
      ubx_buf[i] = info.ubx[i] + delta;
    } else {
      ubx_buf[i] = info.ubx[i];
    }
  }
}

// Template delarations
#ifdef AMIGO_USE_CUDA
template <typename T>
void set_dual_values_cuda(const OptProblemInfo<T>& info, const T value, T* d_x,
                          cudaStream_t stream = 0);

template <typename T>
void set_primal_values_cuda(const OptProblemInfo<T>& info, const T value,
                            T* d_x, cudaStream_t stream = 0);

template <typename T>
void copy_duals_cuda(const OptProblemInfo<T>& info, const T* d_src, T* d_dest,
                     cudaStream_t stream = 0);

template <typename T>
void copy_primals_cuda(const OptProblemInfo<T>& info, const T* d_src, T* d_dest,
                       cudaStream_t stream = 0);

template <typename T>
void initialize_multipliers_and_slacks_cuda(T barrier_param,
                                            const OptProblemInfo<T>& info,
                                            const T* d_g, OptState<T>& pt,
                                            cudaStream_t stream = 0);

template <typename T>
void compute_residual_cuda(T barrier_param, T gamma,
                           const OptProblemInfo<T>& info, OptState<const T>& pt,
                           const T* g, T* r, cudaStream_t stream = 0);

template <typename T>
void compute_update_cuda(T barrier_param, T gamma,
                         const OptProblemInfo<T>& info, OptState<const T>& pt,
                         OptState<T>& up, cudaStream_t stream = 0);

template <typename T>
void compute_diagonal_cuda(const OptProblemInfo<T>& info, OptState<const T>& pt,
                           T* diag, cudaStream_t stream = 0);

template <typename T>
void compute_max_step_cuda(const T tau, const OptProblemInfo<T>& info,
                           OptState<const T>& pt, OptState<const T>& up,
                           T& alpha_x_max, int& x_index, T& alpha_z_max,
                           int& z_index, cudaStream_t stream = 0);

template <typename T>
void apply_step_update_cuda(const T alpha_x, const T alpha_z,
                            const OptProblemInfo<T>& info,
                            OptState<const T>& pt, OptState<const T>& up,
                            OptState<T>& tmp, cudaStream_t stream = 0);

template <typename T>
void compute_affine_start_point_cuda(T beta_min, const OptProblemInfo<T>& info,
                                     OptState<const T>& pt,
                                     OptState<const T>& up, OptState<T>& tmp,
                                     cudaStream_t stream = 0);

template <typename T>
void compute_complementarity_pairs_cuda(const OptProblemInfo<T>& info,
                                        const OptState<const T>& pt,
                                        T partial_sum[], T& local_min);
#endif

}  // namespace detail

}  // namespace amigo

#endif  // AMIGO_INTERIOR_POINT_BACKEND_H
