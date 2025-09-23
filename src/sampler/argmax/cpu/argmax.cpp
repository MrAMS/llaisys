#include "argmax.hpp"

#include "../../../utils.hpp"
#include <cstdint>


template <typename T>
int64_t argmax_(const T *logits, size_t numel) {
    int64_t sampled=0;
    float max_val_f = llaisys::utils::cast<float>(logits[0]);
    for (size_t i = 1; i < numel; i++) {
        const float t = llaisys::utils::cast<float>(logits[i]);
        if(t > max_val_f){
            max_val_f = t;
            sampled = i;
        }
    }
    return sampled;
}

namespace llaisys::sampler {
int64_t argmax(std::byte* logits, size_t numel, llaisysDataType_t type){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<const float *>(logits), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<const llaisys::bf16_t *>(logits), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<const llaisys::fp16_t *>(logits), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}