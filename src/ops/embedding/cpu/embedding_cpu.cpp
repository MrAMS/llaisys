#include "embedding_cpu.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include "../../../utils.hpp"


template <typename T>
void embedding_(T *out, const int64_t *index, size_t index_len, const T *weight, const std::vector<size_t>& shape, const std::vector<ptrdiff_t>& strides){
    for(size_t i=0;i<index_len;++i){
        size_t idx = index[i];
        CHECK_ARGUMENT(idx >= 0, "Index must be non-negative in embedding operation.");
        CHECK_ARGUMENT(idx < shape[0], "Index out of bounds in embedding operation.");
        auto row_sz = shape[0]*sizeof(T);
        memcpy(out+i*row_sz, weight+idx*strides[0], row_sz);
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, size_t index_len, const std::byte *weight, const std::vector<size_t>& shape, const std::vector<ptrdiff_t>& strides, llaisysDataType_t type){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float_t *>(out),
        reinterpret_cast<const int64_t *>(index),
        index_len,
        reinterpret_cast<const float *>(weight),
        shape, strides);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out),
        reinterpret_cast<const int64_t *>(index),
        index_len,
        reinterpret_cast<const llaisys::bf16_t *>(weight),
        shape, strides);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out),
        reinterpret_cast<const int64_t *>(index),
        index_len,
        reinterpret_cast<const llaisys::fp16_t *>(weight),
        shape, strides);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}