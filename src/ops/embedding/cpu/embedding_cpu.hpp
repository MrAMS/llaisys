#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, size_t index_len, const std::byte *weight, const std::vector<size_t>& shape, const std::vector<ptrdiff_t>& strides, llaisysDataType_t type);
}