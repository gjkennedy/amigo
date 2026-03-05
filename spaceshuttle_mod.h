#include "amigo.h"
#include "a2dcore.h"
#include "a2d_sed.h"
namespace amigo {
template<typename T__> 
class ShuttleCollocation__ {
 public:
  inline static constexpr double dt = static_cast<double>(20.0);
  inline static constexpr double Re = static_cast<double>(20902900.0);
  inline static constexpr double mu = static_cast<double>(1.4076539e+16);
  inline static constexpr double rho0 = static_cast<double>(0.002378);
  inline static constexpr double h_r = static_cast<double>(23800.0);
  inline static constexpr double S = static_cast<double>(2690.0);
  inline static constexpr double mass = static_cast<double>(6309.442406912414);
  inline static constexpr double a0 = static_cast<double>(-0.20704);
  inline static constexpr double a1 = static_cast<double>(0.029244);
  inline static constexpr double b0 = static_cast<double>(0.07854);
  inline static constexpr double b1 = static_cast<double>(-0.0061592);
  inline static constexpr double b2 = static_cast<double>(0.000621408);
  template <typename R__> using Input = A2D::VarTuple<R__, A2D::Vec<R__, 6>, A2D::Vec<R__, 6>, A2D::Vec<R__, 2>, A2D::Vec<R__, 2>, A2D::Vec<R__, 6>>;
  static constexpr int ncomp = Input<T__>::ncomp;
  template <typename R__> using Data = typename A2D::VarTuple<R__, R__>;
  static constexpr int ndata = 0;
  static constexpr bool is_compute_empty = false;
  static constexpr bool is_continuation_component = false;
  static constexpr bool is_output_empty = true;
  template<typename R__> using Output = typename A2D::VarTuple<R__, R__>;
  static constexpr int noutputs = 0;
  template <typename R__>
  AMIGO_HOST_DEVICE static R__ lagrange(R__ alpha__, Data<R__>& data__, Input<R__>& input__) {
    A2D::Vec<R__, 6>& q1 = A2D::get<0>(input__);
    A2D::Vec<R__, 6>& q2 = A2D::get<1>(input__);
    A2D::Vec<R__, 2>& u1 = A2D::get<2>(input__);
    A2D::Vec<R__, 2>& u2 = A2D::get<3>(input__);
    A2D::Vec<R__, 6>& lam_res__ = A2D::get<4>(input__);
    R__ t21__;
    R__ t24__;
    R__ t26__;
    R__ t31__;
    R__ t33__;
    R__ t44__;
    R__ t45__;
    R__ t46__;
    R__ t47__;
    R__ t48__;
    R__ t50__;
    R__ t53__;
    R__ t57__;
    R__ t58__;
    R__ t59__;
    R__ t60__;
    R__ t61__;
    R__ t63__;
    R__ t66__;
    R__ t96__;
    R__ t99__;
    R__ t109__;
    R__ t120__;
    R__ t122__;
    R__ t132__;
    R__ t145__;
    R__ t146__;
    R__ t158__;
    R__ t159__;
    R__ lagrangian__;
    t21__ = (0.5 * dt);
    t24__ = (10000.0 * q1[3]);
    t26__ = A2D::sin(q1[4]);
    t31__ = (10000.0 * q2[3]);
    t33__ = A2D::sin(q2[4]);
    t44__ = (100000.0 * q1[0]);
    t45__ = (Re + t44__);
    t46__ = (t24__ / t45__);
    t47__ = A2D::cos(q1[4]);
    t48__ = (t46__ * t47__);
    t50__ = A2D::sin(q1[5]);
    t53__ = A2D::cos(q1[2]);
    t57__ = (100000.0 * q2[0]);
    t58__ = (Re + t57__);
    t59__ = (t31__ / t58__);
    t60__ = A2D::cos(q2[4]);
    t61__ = (t59__ * t60__);
    t63__ = A2D::sin(q2[5]);
    t66__ = A2D::cos(q2[2]);
    t96__ = ((((0.5 * (rho0 * A2D::exp((-(t44__) / h_r)))) * t24__) * t24__) * S);
    t99__ = (u1[0] * 57.29577951308232);
    t109__ = (mu / (t45__ * t45__));
    t120__ = ((((0.5 * (rho0 * A2D::exp((-(t57__) / h_r)))) * t31__) * t31__) * S);
    t122__ = (u2[0] * 57.29577951308232);
    t132__ = (mu / (t58__ * t58__));
    t145__ = (t96__ * (a0 + (a1 * t99__)));
    t146__ = (mass * t24__);
    t158__ = (t120__ * (a0 + (a1 * t122__)));
    t159__ = (mass * t31__);
    lagrangian__ = ((((((((q2[0] - q1[0]) - (t21__ * (((t24__ * t26__) / 100000.0) + ((t31__ * t33__) / 100000.0)))) * lam_res__[0]) + (((q2[1] - q1[1]) - (t21__ * ((((t48__ * t50__) / t53__) / 1.0) + (((t61__ * t63__) / t66__) / 1.0)))) * lam_res__[1])) + (((q2[2] - q1[2]) - (t21__ * (((t48__ * A2D::cos(q1[5])) / 1.0) + ((t61__ * A2D::cos(q2[5])) / 1.0)))) * lam_res__[2])) + (((q2[3] - q1[3]) - (t21__ * (((-(((t96__ * ((b0 + (b1 * t99__)) + ((b2 * t99__) * t99__))) / mass)) - (t109__ * t26__)) / 10000.0) + ((-(((t120__ * ((b0 + (b1 * t122__)) + ((b2 * t122__) * t122__))) / mass)) - (t132__ * t33__)) / 10000.0)))) * lam_res__[3])) + (((q2[4] - q1[4]) - (t21__ * (((((t145__ / t146__) * A2D::cos(u1[1])) + (t47__ * (t46__ - (t109__ / t24__)))) / 1.0) + ((((t158__ / t159__) * A2D::cos(u2[1])) + (t60__ * (t59__ - (t132__ / t31__)))) / 1.0)))) * lam_res__[4])) + (((q2[5] - q1[5]) - (t21__ * (((((t145__ * A2D::sin(u1[1])) / (t146__ * t47__)) + ((((t24__ / (t45__ * t53__)) * t47__) * t50__) * A2D::sin(q1[2]))) / 1.0) + ((((t158__ * A2D::sin(u2[1])) / (t159__ * t60__)) + ((((t31__ / (t58__ * t66__)) * t60__) * t63__) * A2D::sin(q2[2]))) / 1.0)))) * lam_res__[5]));
    return lagrangian__;
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void gradient(R__ alpha__, Data<R__>& data__, Input<R__>& input__, Input<R__>& boutput__) {
    A2D::ADObj<A2D::Vec<R__, 6>&> q1(A2D::get<0>(input__), A2D::get<0>(boutput__));
    A2D::ADObj<A2D::Vec<R__, 6>&> q2(A2D::get<1>(input__), A2D::get<1>(boutput__));
    A2D::ADObj<A2D::Vec<R__, 2>&> u1(A2D::get<2>(input__), A2D::get<2>(boutput__));
    A2D::ADObj<A2D::Vec<R__, 2>&> u2(A2D::get<3>(input__), A2D::get<3>(boutput__));
    A2D::ADObj<A2D::Vec<R__, 6>&> lam_res__(A2D::get<4>(input__), A2D::get<4>(boutput__));
    R__ t21__;
    A2D::ADObj<R__> t24__;
    A2D::ADObj<R__> t26__;
    A2D::ADObj<R__> t31__;
    A2D::ADObj<R__> t33__;
    A2D::ADObj<R__> t44__;
    A2D::ADObj<R__> t45__;
    A2D::ADObj<R__> t46__;
    A2D::ADObj<R__> t47__;
    A2D::ADObj<R__> t48__;
    A2D::ADObj<R__> t50__;
    A2D::ADObj<R__> t53__;
    A2D::ADObj<R__> t57__;
    A2D::ADObj<R__> t58__;
    A2D::ADObj<R__> t59__;
    A2D::ADObj<R__> t60__;
    A2D::ADObj<R__> t61__;
    A2D::ADObj<R__> t63__;
    A2D::ADObj<R__> t66__;
    A2D::ADObj<R__> t96__;
    A2D::ADObj<R__> t99__;
    A2D::ADObj<R__> t109__;
    A2D::ADObj<R__> t120__;
    A2D::ADObj<R__> t122__;
    A2D::ADObj<R__> t132__;
    A2D::ADObj<R__> t145__;
    A2D::ADObj<R__> t146__;
    A2D::ADObj<R__> t158__;
    A2D::ADObj<R__> t159__;
    A2D::ADObj<R__> lagrangian__;
    t21__ = (0.5 * dt);
    auto stack__ = A2D::MakeStack(
      A2D::Eval((10000.0 * q1[3]), t24__),
      A2D::Eval(A2D::sin(q1[4]), t26__),
      A2D::Eval((10000.0 * q2[3]), t31__),
      A2D::Eval(A2D::sin(q2[4]), t33__),
      A2D::Eval((100000.0 * q1[0]), t44__),
      A2D::Eval((Re + t44__), t45__),
      A2D::Eval((t24__ / t45__), t46__),
      A2D::Eval(A2D::cos(q1[4]), t47__),
      A2D::Eval((t46__ * t47__), t48__),
      A2D::Eval(A2D::sin(q1[5]), t50__),
      A2D::Eval(A2D::cos(q1[2]), t53__),
      A2D::Eval((100000.0 * q2[0]), t57__),
      A2D::Eval((Re + t57__), t58__),
      A2D::Eval((t31__ / t58__), t59__),
      A2D::Eval(A2D::cos(q2[4]), t60__),
      A2D::Eval((t59__ * t60__), t61__),
      A2D::Eval(A2D::sin(q2[5]), t63__),
      A2D::Eval(A2D::cos(q2[2]), t66__),
      A2D::Eval(((((0.5 * (rho0 * A2D::exp((-(t44__) / h_r)))) * t24__) * t24__) * S), t96__),
      A2D::Eval((u1[0] * 57.29577951308232), t99__),
      A2D::Eval((mu / (t45__ * t45__)), t109__),
      A2D::Eval(((((0.5 * (rho0 * A2D::exp((-(t57__) / h_r)))) * t31__) * t31__) * S), t120__),
      A2D::Eval((u2[0] * 57.29577951308232), t122__),
      A2D::Eval((mu / (t58__ * t58__)), t132__),
      A2D::Eval((t96__ * (a0 + (a1 * t99__))), t145__),
      A2D::Eval((mass * t24__), t146__),
      A2D::Eval((t120__ * (a0 + (a1 * t122__))), t158__),
      A2D::Eval((mass * t31__), t159__),
      A2D::Eval(((((((((q2[0] - q1[0]) - (t21__ * (((t24__ * t26__) / 100000.0) + ((t31__ * t33__) / 100000.0)))) * lam_res__[0]) + (((q2[1] - q1[1]) - (t21__ * ((((t48__ * t50__) / t53__) / 1.0) + (((t61__ * t63__) / t66__) / 1.0)))) * lam_res__[1])) + (((q2[2] - q1[2]) - (t21__ * (((t48__ * A2D::cos(q1[5])) / 1.0) + ((t61__ * A2D::cos(q2[5])) / 1.0)))) * lam_res__[2])) + (((q2[3] - q1[3]) - (t21__ * (((-(((t96__ * ((b0 + (b1 * t99__)) + ((b2 * t99__) * t99__))) / mass)) - (t109__ * t26__)) / 10000.0) + ((-(((t120__ * ((b0 + (b1 * t122__)) + ((b2 * t122__) * t122__))) / mass)) - (t132__ * t33__)) / 10000.0)))) * lam_res__[3])) + (((q2[4] - q1[4]) - (t21__ * (((((t145__ / t146__) * A2D::cos(u1[1])) + (t47__ * (t46__ - (t109__ / t24__)))) / 1.0) + ((((t158__ / t159__) * A2D::cos(u2[1])) + (t60__ * (t59__ - (t132__ / t31__)))) / 1.0)))) * lam_res__[4])) + (((q2[5] - q1[5]) - (t21__ * (((((t145__ * A2D::sin(u1[1])) / (t146__ * t47__)) + ((((t24__ / (t45__ * t53__)) * t47__) * t50__) * A2D::sin(q1[2]))) / 1.0) + ((((t158__ * A2D::sin(u2[1])) / (t159__ * t60__)) + ((((t31__ / (t58__ * t66__)) * t60__) * t63__) * A2D::sin(q2[2]))) / 1.0)))) * lam_res__[5])), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void hessian(R__ alpha__, Data<R__>& data__, Input<R__>& input__, Input<R__>& pinput__, Input<R__>& boutput__, Input<R__>& houtput__) {
    A2D::A2DObj<A2D::Vec<R__, 6>&> q1(A2D::get<0>(input__), A2D::get<0>(boutput__), A2D::get<0>(pinput__), A2D::get<0>(houtput__));
    A2D::A2DObj<A2D::Vec<R__, 6>&> q2(A2D::get<1>(input__), A2D::get<1>(boutput__), A2D::get<1>(pinput__), A2D::get<1>(houtput__));
    A2D::A2DObj<A2D::Vec<R__, 2>&> u1(A2D::get<2>(input__), A2D::get<2>(boutput__), A2D::get<2>(pinput__), A2D::get<2>(houtput__));
    A2D::A2DObj<A2D::Vec<R__, 2>&> u2(A2D::get<3>(input__), A2D::get<3>(boutput__), A2D::get<3>(pinput__), A2D::get<3>(houtput__));
    A2D::A2DObj<A2D::Vec<R__, 6>&> lam_res__(A2D::get<4>(input__), A2D::get<4>(boutput__), A2D::get<4>(pinput__), A2D::get<4>(houtput__));
    R__ t21__;
    A2D::A2DObj<R__> t24__;
    A2D::A2DObj<R__> t26__;
    A2D::A2DObj<R__> t31__;
    A2D::A2DObj<R__> t33__;
    A2D::A2DObj<R__> t44__;
    A2D::A2DObj<R__> t45__;
    A2D::A2DObj<R__> t46__;
    A2D::A2DObj<R__> t47__;
    A2D::A2DObj<R__> t48__;
    A2D::A2DObj<R__> t50__;
    A2D::A2DObj<R__> t53__;
    A2D::A2DObj<R__> t57__;
    A2D::A2DObj<R__> t58__;
    A2D::A2DObj<R__> t59__;
    A2D::A2DObj<R__> t60__;
    A2D::A2DObj<R__> t61__;
    A2D::A2DObj<R__> t63__;
    A2D::A2DObj<R__> t66__;
    A2D::A2DObj<R__> t96__;
    A2D::A2DObj<R__> t99__;
    A2D::A2DObj<R__> t109__;
    A2D::A2DObj<R__> t120__;
    A2D::A2DObj<R__> t122__;
    A2D::A2DObj<R__> t132__;
    A2D::A2DObj<R__> t145__;
    A2D::A2DObj<R__> t146__;
    A2D::A2DObj<R__> t158__;
    A2D::A2DObj<R__> t159__;
    A2D::A2DObj<R__> lagrangian__;
    t21__ = (0.5 * dt);
    auto stack__ = A2D::MakeStack(
      A2D::Eval((10000.0 * q1[3]), t24__),
      A2D::Eval(A2D_SED::sin(q1[4]), t26__),
      A2D::Eval((10000.0 * q2[3]), t31__),
      A2D::Eval(A2D_SED::sin(q2[4]), t33__),
      A2D::Eval((100000.0 * q1[0]), t44__),
      A2D::Eval((Re + t44__), t45__),
      A2D::Eval((t24__ / t45__), t46__),
      A2D::Eval(A2D_SED::cos(q1[4]), t47__),
      A2D::Eval((t46__ * t47__), t48__),
      A2D::Eval(A2D_SED::sin(q1[5]), t50__),
      A2D::Eval(A2D_SED::cos(q1[2]), t53__),
      A2D::Eval((100000.0 * q2[0]), t57__),
      A2D::Eval((Re + t57__), t58__),
      A2D::Eval((t31__ / t58__), t59__),
      A2D::Eval(A2D_SED::cos(q2[4]), t60__),
      A2D::Eval((t59__ * t60__), t61__),
      A2D::Eval(A2D_SED::sin(q2[5]), t63__),
      A2D::Eval(A2D_SED::cos(q2[2]), t66__),
      A2D::Eval(((((0.5 * (rho0 * A2D_SED::exp((-(t44__) / h_r)))) * t24__) * t24__) * S), t96__),
      A2D::Eval((u1[0] * 57.29577951308232), t99__),
      A2D::Eval((mu / (t45__ * t45__)), t109__),
      A2D::Eval(((((0.5 * (rho0 * A2D_SED::exp((-(t57__) / h_r)))) * t31__) * t31__) * S), t120__),
      A2D::Eval((u2[0] * 57.29577951308232), t122__),
      A2D::Eval((mu / (t58__ * t58__)), t132__),
      A2D::Eval((t96__ * (a0 + (a1 * t99__))), t145__),
      A2D::Eval((mass * t24__), t146__),
      A2D::Eval((t120__ * (a0 + (a1 * t122__))), t158__),
      A2D::Eval((mass * t31__), t159__),
      A2D::Eval(((((((((q2[0] - q1[0]) - (t21__ * (((t24__ * t26__) / 100000.0) + ((t31__ * t33__) / 100000.0)))) * lam_res__[0]) + (((q2[1] - q1[1]) - (t21__ * ((((t48__ * t50__) / t53__) / 1.0) + (((t61__ * t63__) / t66__) / 1.0)))) * lam_res__[1])) + (((q2[2] - q1[2]) - (t21__ * (((t48__ * A2D_SED::cos(q1[5])) / 1.0) + ((t61__ * A2D_SED::cos(q2[5])) / 1.0)))) * lam_res__[2])) + (((q2[3] - q1[3]) - (t21__ * (((-(((t96__ * ((b0 + (b1 * t99__)) + ((b2 * t99__) * t99__))) / mass)) - (t109__ * t26__)) / 10000.0) + ((-(((t120__ * ((b0 + (b1 * t122__)) + ((b2 * t122__) * t122__))) / mass)) - (t132__ * t33__)) / 10000.0)))) * lam_res__[3])) + (((q2[4] - q1[4]) - (t21__ * (((((t145__ / t146__) * A2D_SED::cos(u1[1])) + (t47__ * (t46__ - (t109__ / t24__)))) / 1.0) + ((((t158__ / t159__) * A2D_SED::cos(u2[1])) + (t60__ * (t59__ - (t132__ / t31__)))) / 1.0)))) * lam_res__[4])) + (((q2[5] - q1[5]) - (t21__ * (((((t145__ * A2D_SED::sin(u1[1])) / (t146__ * t47__)) + ((((t24__ / (t45__ * t53__)) * t47__) * t50__) * A2D_SED::sin(q1[2]))) / 1.0) + ((((t158__ * A2D_SED::sin(u2[1])) / (t159__ * t60__)) + ((((t31__ / (t58__ * t66__)) * t60__) * t63__) * A2D_SED::sin(q2[2]))) / 1.0)))) * lam_res__[5])), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
    stack__.hforward();
    stack__.hreverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void compute_output(Data<R__>& data__, Input<R__>& input__, Output<R__>& output__) {
  }
};

template<typename T__> 
class InitialConditions__ {
 public:
  template <typename R__> using Input = A2D::VarTuple<R__, A2D::Vec<R__, 6>, A2D::Vec<R__, 6>>;
  static constexpr int ncomp = Input<T__>::ncomp;
  template <typename R__> using Data = typename A2D::VarTuple<R__, R__>;
  static constexpr int ndata = 0;
  static constexpr bool is_compute_empty = false;
  static constexpr bool is_continuation_component = false;
  static constexpr bool is_output_empty = true;
  template<typename R__> using Output = typename A2D::VarTuple<R__, R__>;
  static constexpr int noutputs = 0;
  template <typename R__>
  AMIGO_HOST_DEVICE static R__ lagrange(R__ alpha__, Data<R__>& data__, Input<R__>& input__) {
    A2D::Vec<R__, 6>& q = A2D::get<0>(input__);
    A2D::Vec<R__, 6>& lam_res__ = A2D::get<1>(input__);
    R__ lagrangian__;
    lagrangian__ = (((((((q[0] - 2.6) * lam_res__[0]) + ((q[1] - 0.0) * lam_res__[1])) + ((q[2] - 0.0) * lam_res__[2])) + ((q[3] - 2.56) * lam_res__[3])) + ((q[4] - -0.017453292519943295) * lam_res__[4])) + ((q[5] - 1.5707963267948966) * lam_res__[5]));
    return lagrangian__;
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void gradient(R__ alpha__, Data<R__>& data__, Input<R__>& input__, Input<R__>& boutput__) {
    A2D::ADObj<A2D::Vec<R__, 6>&> q(A2D::get<0>(input__), A2D::get<0>(boutput__));
    A2D::ADObj<A2D::Vec<R__, 6>&> lam_res__(A2D::get<1>(input__), A2D::get<1>(boutput__));
    A2D::ADObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(
      A2D::Eval((((((((q[0] - 2.6) * lam_res__[0]) + ((q[1] - 0.0) * lam_res__[1])) + ((q[2] - 0.0) * lam_res__[2])) + ((q[3] - 2.56) * lam_res__[3])) + ((q[4] - -0.017453292519943295) * lam_res__[4])) + ((q[5] - 1.5707963267948966) * lam_res__[5])), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void hessian(R__ alpha__, Data<R__>& data__, Input<R__>& input__, Input<R__>& pinput__, Input<R__>& boutput__, Input<R__>& houtput__) {
    A2D::A2DObj<A2D::Vec<R__, 6>&> q(A2D::get<0>(input__), A2D::get<0>(boutput__), A2D::get<0>(pinput__), A2D::get<0>(houtput__));
    A2D::A2DObj<A2D::Vec<R__, 6>&> lam_res__(A2D::get<1>(input__), A2D::get<1>(boutput__), A2D::get<1>(pinput__), A2D::get<1>(houtput__));
    A2D::A2DObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(
      A2D::Eval((((((((q[0] - 2.6) * lam_res__[0]) + ((q[1] - 0.0) * lam_res__[1])) + ((q[2] - 0.0) * lam_res__[2])) + ((q[3] - 2.56) * lam_res__[3])) + ((q[4] - -0.017453292519943295) * lam_res__[4])) + ((q[5] - 1.5707963267948966) * lam_res__[5])), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
    stack__.hforward();
    stack__.hreverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void compute_output(Data<R__>& data__, Input<R__>& input__, Output<R__>& output__) {
  }
};

template<typename T__> 
class FinalConditions__ {
 public:
  template <typename R__> using Input = A2D::VarTuple<R__, A2D::Vec<R__, 6>, A2D::Vec<R__, 3>>;
  static constexpr int ncomp = Input<T__>::ncomp;
  template <typename R__> using Data = typename A2D::VarTuple<R__, R__>;
  static constexpr int ndata = 0;
  static constexpr bool is_compute_empty = false;
  static constexpr bool is_continuation_component = false;
  static constexpr bool is_output_empty = true;
  template<typename R__> using Output = typename A2D::VarTuple<R__, R__>;
  static constexpr int noutputs = 0;
  template <typename R__>
  AMIGO_HOST_DEVICE static R__ lagrange(R__ alpha__, Data<R__>& data__, Input<R__>& input__) {
    A2D::Vec<R__, 6>& q = A2D::get<0>(input__);
    A2D::Vec<R__, 3>& lam_res__ = A2D::get<1>(input__);
    R__ lagrangian__;
    lagrangian__ = ((((q[0] - 0.8) * lam_res__[0]) + ((q[3] - 0.25) * lam_res__[1])) + ((q[4] - -0.08726646259971647) * lam_res__[2]));
    return lagrangian__;
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void gradient(R__ alpha__, Data<R__>& data__, Input<R__>& input__, Input<R__>& boutput__) {
    A2D::ADObj<A2D::Vec<R__, 6>&> q(A2D::get<0>(input__), A2D::get<0>(boutput__));
    A2D::ADObj<A2D::Vec<R__, 3>&> lam_res__(A2D::get<1>(input__), A2D::get<1>(boutput__));
    A2D::ADObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(
      A2D::Eval(((((q[0] - 0.8) * lam_res__[0]) + ((q[3] - 0.25) * lam_res__[1])) + ((q[4] - -0.08726646259971647) * lam_res__[2])), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void hessian(R__ alpha__, Data<R__>& data__, Input<R__>& input__, Input<R__>& pinput__, Input<R__>& boutput__, Input<R__>& houtput__) {
    A2D::A2DObj<A2D::Vec<R__, 6>&> q(A2D::get<0>(input__), A2D::get<0>(boutput__), A2D::get<0>(pinput__), A2D::get<0>(houtput__));
    A2D::A2DObj<A2D::Vec<R__, 3>&> lam_res__(A2D::get<1>(input__), A2D::get<1>(boutput__), A2D::get<1>(pinput__), A2D::get<1>(houtput__));
    A2D::A2DObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(
      A2D::Eval(((((q[0] - 0.8) * lam_res__[0]) + ((q[3] - 0.25) * lam_res__[1])) + ((q[4] - -0.08726646259971647) * lam_res__[2])), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
    stack__.hforward();
    stack__.hreverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void compute_output(Data<R__>& data__, Input<R__>& input__, Output<R__>& output__) {
  }
};

template<typename T__> 
class HeatingConstraint__ {
 public:
  inline static constexpr double rho0 = static_cast<double>(0.002378);
  inline static constexpr double h_r = static_cast<double>(23800.0);
  inline static constexpr double c0 = static_cast<double>(1.0672181);
  inline static constexpr double c1 = static_cast<double>(-0.019213774);
  inline static constexpr double c2 = static_cast<double>(0.00021286289);
  inline static constexpr double c3 = static_cast<double>(-1.0117249e-06);
  inline static constexpr double qU = static_cast<double>(70.0);
  template <typename R__> using Input = A2D::VarTuple<R__, R__, R__, R__, R__, A2D::Vec<R__, 1>>;
  static constexpr int ncomp = Input<T__>::ncomp;
  template <typename R__> using Data = typename A2D::VarTuple<R__, R__>;
  static constexpr int ndata = 0;
  static constexpr bool is_compute_empty = false;
  static constexpr bool is_continuation_component = false;
  static constexpr bool is_output_empty = true;
  template<typename R__> using Output = typename A2D::VarTuple<R__, R__>;
  static constexpr int noutputs = 0;
  template <typename R__>
  AMIGO_HOST_DEVICE static R__ lagrange(R__ alpha__, Data<R__>& data__, Input<R__>& input__) {
    R__& h = A2D::get<0>(input__);
    R__& v = A2D::get<1>(input__);
    R__& alpha = A2D::get<2>(input__);
    R__& slack = A2D::get<3>(input__);
    A2D::Vec<R__, 1>& lam_res__ = A2D::get<4>(input__);
    R__ t13__;
    R__ lagrangian__;
    t13__ = (alpha * 57.29577951308232);
    lagrangian__ = (((((((c0 + (c1 * t13__)) + ((c2 * t13__) * t13__)) + (((c3 * t13__) * t13__) * t13__)) * (((17700.0 * A2D::sqrt(rho0)) * A2D::exp((-((100000.0 * h)) / (2.0 * h_r)))) * A2D::pow((0.0001 * (10000.0 * v)), 3.07))) / qU) - slack) * lam_res__[0]);
    return lagrangian__;
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void gradient(R__ alpha__, Data<R__>& data__, Input<R__>& input__, Input<R__>& boutput__) {
    A2D::ADObj<R__&> h(A2D::get<0>(input__), A2D::get<0>(boutput__));
    A2D::ADObj<R__&> v(A2D::get<1>(input__), A2D::get<1>(boutput__));
    A2D::ADObj<R__&> alpha(A2D::get<2>(input__), A2D::get<2>(boutput__));
    A2D::ADObj<R__&> slack(A2D::get<3>(input__), A2D::get<3>(boutput__));
    A2D::ADObj<A2D::Vec<R__, 1>&> lam_res__(A2D::get<4>(input__), A2D::get<4>(boutput__));
    A2D::ADObj<R__> t13__;
    A2D::ADObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(
      A2D::Eval((alpha * 57.29577951308232), t13__),
      A2D::Eval((((((((c0 + (c1 * t13__)) + ((c2 * t13__) * t13__)) + (((c3 * t13__) * t13__) * t13__)) * (((17700.0 * A2D::sqrt(rho0)) * A2D::exp((-((100000.0 * h)) / (2.0 * h_r)))) * A2D::pow((0.0001 * (10000.0 * v)), 3.07))) / qU) - slack) * lam_res__[0]), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void hessian(R__ alpha__, Data<R__>& data__, Input<R__>& input__, Input<R__>& pinput__, Input<R__>& boutput__, Input<R__>& houtput__) {
    A2D::A2DObj<R__&> h(A2D::get<0>(input__), A2D::get<0>(boutput__), A2D::get<0>(pinput__), A2D::get<0>(houtput__));
    A2D::A2DObj<R__&> v(A2D::get<1>(input__), A2D::get<1>(boutput__), A2D::get<1>(pinput__), A2D::get<1>(houtput__));
    A2D::A2DObj<R__&> alpha(A2D::get<2>(input__), A2D::get<2>(boutput__), A2D::get<2>(pinput__), A2D::get<2>(houtput__));
    A2D::A2DObj<R__&> slack(A2D::get<3>(input__), A2D::get<3>(boutput__), A2D::get<3>(pinput__), A2D::get<3>(houtput__));
    A2D::A2DObj<A2D::Vec<R__, 1>&> lam_res__(A2D::get<4>(input__), A2D::get<4>(boutput__), A2D::get<4>(pinput__), A2D::get<4>(houtput__));
    A2D::A2DObj<R__> t13__;
    A2D::A2DObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(
      A2D::Eval((alpha * 57.29577951308232), t13__),
      A2D::Eval((((((((c0 + (c1 * t13__)) + ((c2 * t13__) * t13__)) + (((c3 * t13__) * t13__) * t13__)) * (((17700.0 * A2D_SED::sqrt(rho0)) * A2D_SED::exp((-((100000.0 * h)) / (2.0 * h_r)))) * A2D::pow((0.0001 * (10000.0 * v)), 3.07))) / qU) - slack) * lam_res__[0]), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
    stack__.hforward();
    stack__.hreverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void compute_output(Data<R__>& data__, Input<R__>& input__, Output<R__>& output__) {
  }
};

template<typename T__> 
class Objective__ {
 public:
  template <typename R__> using Input = A2D::VarTuple<R__, R__>;
  static constexpr int ncomp = Input<T__>::ncomp;
  template <typename R__> using Data = typename A2D::VarTuple<R__, R__>;
  static constexpr int ndata = 0;
  static constexpr bool is_compute_empty = false;
  static constexpr bool is_continuation_component = false;
  static constexpr bool is_output_empty = true;
  template<typename R__> using Output = typename A2D::VarTuple<R__, R__>;
  static constexpr int noutputs = 0;
  template <typename R__>
  AMIGO_HOST_DEVICE static R__ lagrange(R__ alpha__, Data<R__>& data__, Input<R__>& input__) {
    R__& theta_final = A2D::get<0>(input__);
    R__ lagrangian__;
    lagrangian__ = (alpha__ * -(theta_final));
    return lagrangian__;
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void gradient(R__ alpha__, Data<R__>& data__, Input<R__>& input__, Input<R__>& boutput__) {
    A2D::ADObj<R__&> theta_final(A2D::get<0>(input__), A2D::get<0>(boutput__));
    A2D::ADObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(
      A2D::Eval((alpha__ * -(theta_final)), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void hessian(R__ alpha__, Data<R__>& data__, Input<R__>& input__, Input<R__>& pinput__, Input<R__>& boutput__, Input<R__>& houtput__) {
    A2D::A2DObj<R__&> theta_final(A2D::get<0>(input__), A2D::get<0>(boutput__), A2D::get<0>(pinput__), A2D::get<0>(houtput__));
    A2D::A2DObj<R__> lagrangian__;
    auto stack__ = A2D::MakeStack(
      A2D::Eval((alpha__ * -(theta_final)), lagrangian__));
    lagrangian__.bvalue() = 1.0;
    stack__.reverse();
    stack__.hforward();
    stack__.hreverse();
  }
  template <typename R__>
  AMIGO_HOST_DEVICE static void compute_output(Data<R__>& data__, Input<R__>& input__, Output<R__>& output__) {
  }
};
}
