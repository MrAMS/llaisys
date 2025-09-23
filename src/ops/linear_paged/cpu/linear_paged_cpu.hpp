#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void linear_paged(std::byte** out_map, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t m, size_t k, size_t n, llaisysDataType_t type);
}