#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "component_group.h"
#include "spaceshuttle_mod.h"
namespace py = pybind11;
PYBIND11_MODULE(spaceshuttle_mod, mod) {
#ifdef AMIGO_USE_OPENMP
  constexpr amigo::ExecPolicy policy = amigo::ExecPolicy::OPENMP;
#elif defined(AMIGO_USE_CUDA)
  constexpr amigo::ExecPolicy policy = amigo::ExecPolicy::CUDA;
#else
  constexpr amigo::ExecPolicy policy = amigo::ExecPolicy::SERIAL;
#endif
py::class_<amigo::ComponentGroup<double, policy, amigo::ShuttleCollocation__<double>>, amigo::ComponentGroupBase<double, policy>, std::shared_ptr<amigo::ComponentGroup<double, policy, amigo::ShuttleCollocation__<double>>>>(mod, "ShuttleCollocation").def(py::init<int, std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>>());
py::class_<amigo::ComponentGroup<double, policy, amigo::InitialConditions__<double>>, amigo::ComponentGroupBase<double, policy>, std::shared_ptr<amigo::ComponentGroup<double, policy, amigo::InitialConditions__<double>>>>(mod, "InitialConditions").def(py::init<int, std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>>());
py::class_<amigo::ComponentGroup<double, policy, amigo::FinalConditions__<double>>, amigo::ComponentGroupBase<double, policy>, std::shared_ptr<amigo::ComponentGroup<double, policy, amigo::FinalConditions__<double>>>>(mod, "FinalConditions").def(py::init<int, std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>>());
py::class_<amigo::ComponentGroup<double, policy, amigo::HeatingConstraint__<double>>, amigo::ComponentGroupBase<double, policy>, std::shared_ptr<amigo::ComponentGroup<double, policy, amigo::HeatingConstraint__<double>>>>(mod, "HeatingConstraint").def(py::init<int, std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>>());
py::class_<amigo::ComponentGroup<double, policy, amigo::Objective__<double>>, amigo::ComponentGroupBase<double, policy>, std::shared_ptr<amigo::ComponentGroup<double, policy, amigo::Objective__<double>>>>(mod, "Objective").def(py::init<int, std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>, std::shared_ptr<amigo::Vector<int>>>());
}
