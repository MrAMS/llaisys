#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void self_attention_paged(tensor_t attn_val, tensor_t q, const std::byte* const* k_map, const std::byte* const* v_map, uint64_t d_kvhead, uint64_t d_tot, float scale);
}
