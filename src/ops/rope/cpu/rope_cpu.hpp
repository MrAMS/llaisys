#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, size_t d_seq, size_t d_head, size_t d, float theta, llaisysDataType_t type);
}