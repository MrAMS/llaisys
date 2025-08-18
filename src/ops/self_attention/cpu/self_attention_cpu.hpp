#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, 
    float scale, size_t d_seq, size_t d_head, size_t d_v, size_t d_qk, size_t d_kvhead, size_t d_tot,
    llaisysDataType_t type);
}