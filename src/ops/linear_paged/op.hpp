#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void linear_paged(std::byte** out_map, tensor_t in, tensor_t weight, tensor_t bias);
}
