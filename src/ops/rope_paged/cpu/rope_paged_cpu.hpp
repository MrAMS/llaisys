#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void rope_paged(std::byte **out_map, const std::byte * const* in_map, const std::byte *pos_ids, size_t d_seq, size_t d_head, size_t d, float theta, llaisysDataType_t type);
}