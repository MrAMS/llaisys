#pragma once
#include "llaisys.h"
#include <cstddef>
#include <cstdint>

namespace llaisys::sampler {
int64_t argmax(std::byte* logits, size_t numel, llaisysDataType_t type);
}