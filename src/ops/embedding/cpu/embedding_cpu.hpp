#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, size_t index_len, const std::byte *weight, const std::vector<size_t>& shape, llaisysDataType_t type);
}