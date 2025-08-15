#include "rms_norm_cpu.hpp"
#include <cstddef>
#include <cmath>
#include "../../../utils.hpp"

#define TOF(X) llaisys::utils::cast<float>(X)

template<typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t m, size_t n, float eps){
    // weight:1*n, in:m*n, out:m*n 
    for(size_t i=0;i<m;++i){
        // 计算每行的RMS
        float rms_row = 0;
        for(size_t j=0;j<n;++j){
            auto t = TOF(in[i*n+j]);
            rms_row += t*t;
        }
        rms_row = std::sqrt(rms_row/n) + eps;
        // 点乘W
        for(size_t j=0;j<n;++j){
            out[i*n+j] = llaisys::utils::cast<T>(TOF(weight[j]) * TOF(in[i*n+j]) / rms_row);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, size_t m, size_t n, float eps, llaisysDataType_t type){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float_t *>(out),
            reinterpret_cast<const float_t *>(in),
            reinterpret_cast<const float_t *>(weight),
            m, n, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(in),
            reinterpret_cast<const bf16_t *>(weight),
            m, n, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(in),
            reinterpret_cast<const fp16_t *>(weight),
            m, n, eps);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}