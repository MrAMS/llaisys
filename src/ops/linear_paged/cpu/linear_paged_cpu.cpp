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

#define SIMD_ON 1

/* m*k times k*n add m*n = m*n */
#if SIMD_ON
#define SIMD_WIDTH  16
#define SIMD_NUM    (SIMD_WIDTH/sizeof(float))
// 块大小可根据CPU缓存调整
#define BLOCK_SIZE_I 32
#define BLOCK_SIZE_J 32
#define BLOCK_SIZE_K 256

typedef float vec __attribute__ (( vector_size(SIMD_WIDTH) ));

template<typename T>
void linear_paged_(T ** out_map, const T *in, const T *weight, const T *bias, size_t m, size_t k, size_t n) {
    ASSERT(reinterpret_cast<uintptr_t>(in) % SIMD_WIDTH == 0, "in must be aligned");
    ASSERT(reinterpret_cast<uintptr_t>(weight) % SIMD_WIDTH == 0, "weight must be aligned");

    #pragma omp parallel
    {
        // 为每个线程分配最大块尺寸的临时缓冲区
        float temp_sum[BLOCK_SIZE_I][BLOCK_SIZE_J];

        #pragma omp for collapse(2)
        for (size_t i_block = 0; i_block < m; i_block += BLOCK_SIZE_I) {
            for (size_t j_block = 0; j_block < n; j_block += BLOCK_SIZE_J) {
                // 计算当前块的实际边界
                const size_t i_start = i_block;
                const size_t i_end = std::min(i_block + BLOCK_SIZE_I, m);
                const size_t j_start = j_block;
                const size_t j_end = std::min(j_block + BLOCK_SIZE_J, n);
                const size_t i_count = i_end - i_start;
                const size_t j_count = j_end - j_start;

                for (size_t i = 0; i < i_count; ++i) {
                    for (size_t j = 0; j < j_count; ++j) {
                        temp_sum[i][j] = 0.0f;
                    }
                }

                // K维度分块计算
                for (size_t k_block = 0; k_block < k; k_block += BLOCK_SIZE_K) {
                    const size_t k_start = k_block;
                    const size_t k_end = std::min(k_block + BLOCK_SIZE_K, k);

                    // 处理当前块内元素
                    for (size_t i = i_start; i < i_end; ++i) {
                        const size_t ti = i - i_start;  // 临时缓冲区行索引
                        
                        for (size_t j = j_start; j < j_end; ++j) {
                            const size_t tj = j - j_start;  // 临时缓冲区列索引
                            vec sum_vec{};
                            size_t l = k_start;

                            // SIMD向量计算
                            for (; l + SIMD_NUM - 1 < k_end; l += SIMD_NUM) {
                                vec* in_vec;
                                vec* weight_vec;

                                if constexpr (std::is_same<T, float>::value) {
                                    in_vec = (vec*)(&in[i * k + l]);
                                    weight_vec = (vec*)(&weight[j * k + l]);
                                    sum_vec += (*in_vec) * (*weight_vec);
                                } else {
                                    alignas(SIMD_WIDTH) float arr_f32_in[SIMD_NUM], arr_f32_weight[SIMD_NUM];
                                    for (size_t t = 0; t < SIMD_NUM; ++t) {
                                        arr_f32_in[t] = TOF(in[i * k + l + t]);
                                        arr_f32_weight[t] = TOF(weight[j * k + l + t]);
                                    }
                                    in_vec = (vec*)arr_f32_in;
                                    weight_vec = (vec*)arr_f32_weight;
                                    sum_vec += (*in_vec) * (*weight_vec);
                                }
                            }

                            // 累加SIMD结果
                            float psum = 0.0f;
                            for (size_t t = 0; t < SIMD_NUM; ++t) {
                                psum += sum_vec[t];
                            }

                            // 处理剩余元素
                            for (; l < k_end; ++l) {
                                psum += TOF(in[i * k + l]) * TOF(weight[j * k + l]);
                            }

                            // 累加到临时缓冲区
                            temp_sum[ti][tj] += psum;
                        }
                    }
                }

                // 写入最终结果
                for (size_t i = i_start; i < i_end; ++i) {
                    const size_t ti = i - i_start;
                    for (size_t j = j_start; j < j_end; ++j) {
                        const size_t tj = j - j_start;
                        float total = temp_sum[ti][tj];
                        
                        if (bias) {
                            total += TOF(bias[j]);
                        }
                        
                        out_map[i][j] = llaisys::utils::cast<T>(total);
                    }
                }
            }
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