#include "swiglu_cpu.hpp"
#include <cstddef>
#include <cmath>
#include "../../../utils.hpp"

#include <omp.h>

#define TOF(X) llaisys::utils::cast<float>(X)

template<typename T>
void swiglu_(T *out, const T *up, const T *gate, size_t len){
    #pragma omp parallel for
    for(size_t i = 0; i < len; ++i){
        const auto up_val = TOF(up[i]);
        const auto gate_val = TOF(gate[i]);
        out[i] = llaisys::utils::cast<T>(up_val*gate_val/(1.f+std::exp(-gate_val)));
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *up, const std::byte *gate, size_t len, llaisysDataType_t type){
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return swiglu_(reinterpret_cast<float_t *>(out),
                reinterpret_cast<const float_t *>(up),
                reinterpret_cast<const float_t *>(gate),
                len);
        case LLAISYS_DTYPE_BF16:
            return swiglu_(reinterpret_cast<bf16_t *>(out),
                reinterpret_cast<const bf16_t *>(up),
                reinterpret_cast<const bf16_t *>(gate),
                len);
        case LLAISYS_DTYPE_F16:
            return swiglu_(reinterpret_cast<fp16_t *>(out),
                reinterpret_cast<const fp16_t *>(up),
                reinterpret_cast<const fp16_t *>(gate),
                len);

        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}