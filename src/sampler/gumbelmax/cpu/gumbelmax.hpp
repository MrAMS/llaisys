#pragma once
#include "llaisys.h"
#include <cstddef>
#include <cstdint>

namespace llaisys::sampler {
int64_t gumbelmax(std::byte* logits, size_t numel, float temperature, llaisysDataType_t type);
}