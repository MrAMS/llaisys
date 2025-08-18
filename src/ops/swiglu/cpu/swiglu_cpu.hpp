#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *up, const std::byte *gate, size_t len, llaisysDataType_t type);
}