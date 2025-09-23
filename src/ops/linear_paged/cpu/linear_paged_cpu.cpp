#include "linear_paged_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cmath>
#include <cstddef>

#define TOF(X) llaisys::utils::cast<float>(X)

/* m*k times k*n add m*n = m*n */
template<typename T>
void linear_paged_(T **out_map, const T *in, const T *weight, const T *bias, size_t m, size_t k, size_t n){
    for(size_t i=0;i<m;++i)
        for(size_t j=0;j<n;++j){
            float psum = 0;
            if(bias) psum = TOF(bias[j]);
            for(size_t l=0;l<k;++l){
                psum += TOF(in[i*k+l]) * TOF(weight[j*k+l]);
            }
            out_map[i][j] = llaisys::utils::cast<T>(psum);
        }
}

namespace llaisys::ops::cpu {
void linear_paged(std::byte** out_map, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t m, size_t k, size_t n, llaisysDataType_t type){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_paged_(reinterpret_cast<float_t **>(out_map),
            reinterpret_cast<const float_t *>(in),
            reinterpret_cast<const float_t *>(weight),
            reinterpret_cast<const float_t *>(bias),
            m, k, n);
    case LLAISYS_DTYPE_BF16:
        return linear_paged_(reinterpret_cast<bf16_t **>(out_map),
            reinterpret_cast<const bf16_t *>(in),
            reinterpret_cast<const bf16_t *>(weight),
            reinterpret_cast<const bf16_t *>(bias),
            m, k, n);
    case LLAISYS_DTYPE_F16:
        return linear_paged_(reinterpret_cast<fp16_t **>(out_map),
            reinterpret_cast<const fp16_t *>(in),
            reinterpret_cast<const fp16_t *>(weight),
            reinterpret_cast<const fp16_t *>(bias),
            m, k, n);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}