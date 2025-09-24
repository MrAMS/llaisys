#include "embedding_cpu.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include "../../../utils.hpp"

#include <cstdio>
#include <omp.h>


template <typename T>
void embedding_(T *out, const int64_t *index, size_t index_len, const T *weight, const std::vector<size_t>& shape){
    #pragma omp parallel for
    for(size_t i=0;i<index_len;++i){
        int64_t idx = index[i];
        CHECK_ARGUMENT(size_t(idx) < shape[0], "Index out of bounds in embedding operation.");
        std::memcpy(out+i*shape[1], weight+idx*shape[1], shape[1]*sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, size_t index_len, const std::byte *weight, const std::vector<size_t>& shape, llaisysDataType_t type){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float_t *>(out),
        reinterpret_cast<const int64_t *>(index),
        index_len,
        reinterpret_cast<const float *>(weight),
        shape);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out),
        reinterpret_cast<const int64_t *>(index),
        index_len,
        reinterpret_cast<const llaisys::bf16_t *>(weight),
        shape);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out),
        reinterpret_cast<const int64_t *>(index),
        index_len,
        reinterpret_cast<const llaisys::fp16_t *>(weight),
        shape);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}