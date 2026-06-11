#ifndef COMPONENT_GROUP_CUDA_INSTANTIATION_H
#define COMPONENT_GROUP_CUDA_INSTANTIATION_H

#include "amigo.h"
#include "component_group.h"
#include "slack_coupling.h"

namespace amigo {

template class ComponentGroup<double, ExecPolicy::CUDA,
                              SlackComponent__<double>>;
template class ComponentGroup<float, ExecPolicy::CUDA, SlackComponent__<float>>;

}  // namespace amigo

#endif  // COMPONENT_GROUP_CUDA_INSTANTIATION_H