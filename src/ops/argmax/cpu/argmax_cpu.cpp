#include "argmax_cpu.hpp"

#include "../../../utils.hpp"
#include <cstdint>
#include <cmath>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    float max_val_f = llaisys::utils::cast<float>(vals[0]);
    *max_idx = 0;
    for (size_t i = 1; i < numel; i++) {
        const float t = llaisys::utils::cast<float>(vals[i]);
        if(t > max_val_f){
            max_val_f = t;
            *max_idx = i;
        }
    }
    *max_val = llaisys::utils::cast<T>(max_val_f);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel, llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<float *>(max_val),
        reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
         reinterpret_cast<llaisys::bf16_t *>(max_val),
         reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<llaisys::fp16_t *>(max_val),
        reinterpret_cast<const llaisys::fp16_t *>(vals), numel);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}