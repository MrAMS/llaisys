#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void self_attention_paged(std::byte *attn_val, const std::byte *q, const std::byte* const* k_map, const std::byte* const* v_map, 
    float scale, size_t d_seq, size_t d_head, size_t d_v, size_t d_qk, size_t d_kvhead, size_t d_tot,
    llaisysDataType_t type);
}