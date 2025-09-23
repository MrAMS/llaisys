#include "div__cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

#define TOF(X) llaisys::utils::cast<float>(X)

template <typename T>
void div__(T *a, const T *b, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        a[i] = llaisys::utils::cast<T>(TOF(a[i])/TOF(*b));
    }
}

namespace llaisys::ops::cpu {
void div_(std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return div__(reinterpret_cast<float *>(a), reinterpret_cast<const float *>(b), numel);
    case LLAISYS_DTYPE_BF16:
        return div__(reinterpret_cast<llaisys::bf16_t *>(a), reinterpret_cast<const llaisys::bf16_t *>(b), numel);

    case LLAISYS_DTYPE_F16:
        return div__(reinterpret_cast<llaisys::fp16_t *>(a), reinterpret_cast<const llaisys::fp16_t *>(b), numel);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
