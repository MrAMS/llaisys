#include "linear_paged_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cmath>
#include <cstddef>

#define SIMD_ON 1

#if SIMD_ON
#include <emmintrin.h> // SSE2
#include <pmmintrin.h> // SSE3, _mm_hadd_ps
#endif

#include <omp.h>

#define TOF(X) llaisys::utils::cast<float>(X)

#if SIMD_ON
/* m*k times k*n add m*n = m*n */
template<typename T>
void linear_paged_(T **out_map, const T *in, const T *weight, const T *bias, size_t m, size_t k, size_t n) {
    // Check for 16-byte alignment to choose the most efficient load instruction.
    ASSERT(reinterpret_cast<uintptr_t>(in) % 16 == 0, "in must be aligned");
    ASSERT(reinterpret_cast<uintptr_t>(weight) % 16 == 0, "weight must be aligned");

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            /*
            float psum = 0;
            if(bias) psum = TOF(bias[j]);
            */
            __m128 sum_vec;
            sum_vec = _mm_setzero_ps();

            /*
            for(size_t l=0;l<k;++l){
                psum += TOF(in[i*k+l]) * TOF(weight[j*k+l]);
            }
            */
            size_t l = 0;
            for (; l + 3 < k; l += 4) {
                __m128 in_vec, weight_vec;

                if constexpr (std::is_same<T, float>::value){
                    in_vec = _mm_load_ps(&in[i*k+l]);
                    weight_vec = _mm_load_ps(&weight[j*k+l]);
                }else{
                    float arr_f32_in[4], arr_f32_weight[4];
                    for(int t=0;t<4;++t){
                        arr_f32_in[t] = TOF(in[i*k+l +t]);
                    }
                    in_vec = _mm_load_ps(arr_f32_in);
                    for(int t=0;t<4;++t){
                        arr_f32_weight[t] = TOF(weight[j*k+l +t]);
                    }
                    weight_vec = _mm_load_ps(arr_f32_weight);
                }
                
                // maybe can use FMA
                sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(in_vec, weight_vec));
            }

            float psum;
            
            __m128 hsum = _mm_hadd_ps(sum_vec, sum_vec);
            hsum = _mm_hadd_ps(hsum, hsum);
            _mm_store_ss(&psum, hsum);
            
            for (; l < k; ++l) {
                psum += TOF(in[i * k + l]) * TOF(weight[j * k + l]);
            }

            if (bias) {
                psum += TOF(bias[j]);
            }
            
            out_map[i][j] = llaisys::utils::cast<T>(psum);
        }
    }
}
#else
template<typename T>
void linear_paged_(T **out_map, const T *in, const T *weight, const T *bias, size_t m, size_t k, size_t n){
    for(size_t i=0;i<m;++i){
        #pragma omp parallel for
        for(size_t j=0;j<n;++j){
            float psum = 0;
            if(bias) psum = TOF(bias[j]);
            for(size_t l=0;l<k;++l){
                psum += TOF(in[i*k+l]) * TOF(weight[j*k+l]);
            }
            out_map[i][j] = llaisys::utils::cast<T>(psum);
        }
    }
}
#endif

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