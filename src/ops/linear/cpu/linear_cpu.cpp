#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cmath>
#include <cstddef>

#define TOF(X) llaisys::utils::cast<float>(X)

#define SIMD_ON 1

/* m*k times k*n add m*n = m*n */
#if SIMD_ON
#define SIMD_WIDTH  16
#define SIMD_NUM    (SIMD_WIDTH/sizeof(float))
typedef float vec __attribute__ (( vector_size(SIMD_WIDTH) ));
template<typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t k, size_t n) {
    // Check for 16-byte alignment to choose the most efficient load instruction.
    ASSERT(reinterpret_cast<uintptr_t>(in) % SIMD_WIDTH == 0, "in must be aligned");
    ASSERT(reinterpret_cast<uintptr_t>(weight) % SIMD_WIDTH == 0, "weight must be aligned");

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            vec sum_vec{};
            size_t l = 0;
            for (; l + SIMD_NUM-1 < k; l += SIMD_NUM) {
                vec* in_vec;
                vec* weight_vec;

                if constexpr (std::is_same<T, float>::value){
                    in_vec = (vec*)(&in[i*k+l]);
                    weight_vec = (vec*)(&weight[j*k+l]);
                    sum_vec += (*in_vec) * (*weight_vec);
                }else{
                    float arr_f32_in[SIMD_NUM], arr_f32_weight[SIMD_NUM];
                    for(size_t t=0;t<SIMD_NUM;++t){
                        arr_f32_in[t] = TOF(in[i*k+l +t]);
                    }
                    in_vec = (vec*)(arr_f32_in);
                    for(size_t t=0;t<SIMD_NUM;++t){
                        arr_f32_weight[t] = TOF(weight[j*k+l +t]);
                    }
                    weight_vec = (vec*)(arr_f32_weight);
                    sum_vec += (*in_vec) * (*weight_vec);
                }
            }

            float psum=0;
            
            for(size_t t=0;t<SIMD_NUM;++t)
                psum += sum_vec[t];
            
            for (; l < k; ++l) {
                psum += TOF(in[i * k + l]) * TOF(weight[j * k + l]);
            }

            if (bias) {
                psum += TOF(bias[j]);
            }
            
            out[i*n+j] = llaisys::utils::cast<T>(psum);
        }
    }
}
#else
template<typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t k, size_t n){
    for(size_t i=0;i<m;++i){
        #pragma omp parallel for
        for(size_t j=0;j<n;++j){
            float psum = 0;
            if(bias) psum = TOF(bias[j]);
            for(size_t l=0;l<k;++l){
                psum += TOF(in[i*k+l]) * TOF(weight[j*k+l]);
            }
            out[i*n+j] = llaisys::utils::cast<T>(psum);
        }
    }
}
#endif

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t m, size_t k, size_t n, llaisysDataType_t type){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float_t *>(out),
            reinterpret_cast<const float_t *>(in),
            reinterpret_cast<const float_t *>(weight),
            reinterpret_cast<const float_t *>(bias),
            m, k, n);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(in),
            reinterpret_cast<const bf16_t *>(weight),
            reinterpret_cast<const bf16_t *>(bias),
            m, k, n);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(in),
            reinterpret_cast<const fp16_t *>(weight),
            reinterpret_cast<const fp16_t *>(bias),
            m, k, n);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}