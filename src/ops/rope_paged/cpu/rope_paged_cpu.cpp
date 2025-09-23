#include "rope_paged_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>

#define TOF(X) llaisys::utils::cast<float>(X)

template<typename T>
void rope_paged_(T **out_map, const T* const* in_map, const int64_t *pos_ids, size_t d_seq, size_t d_head, size_t d, float theta){
    // 对每个token进行rope_paged编码
    for(size_t s=0;s<d_seq;++s){
        const auto pos_id = pos_ids[s];
        for(size_t h=0;h<d_head;++h){
            for(size_t i=0;i<d/2;++i){
                auto in = in_map[s];
                auto out = out_map[s];

                // rope_paged角度
                const auto phi = pos_id/std::pow(theta, 2.f*i/d);
                const auto cos_phi = std::cos(phi);
                const auto sin_phi = std::sin(phi);

                // 两两一组，[0, d/2], [1, d/2+1], ...
                const auto a_idx = h*d+i; 
                const auto b_idx = a_idx + d/2;
                const auto a = TOF(in[a_idx]);
                const auto b = TOF(in[b_idx]);

                // a_out = a cos(phi) - b sin(phi)
                out[a_idx] = llaisys::utils::cast<T>(a*cos_phi-b*sin_phi);
                // b_out = b cos(phi) + a sin(phi)
                out[b_idx] = llaisys::utils::cast<T>(b*cos_phi+a*sin_phi);
            }
        }
    }
}


namespace llaisys::ops::cpu {
void rope_paged(std::byte **out_map, const std::byte * const* in_map, const std::byte *pos_ids, size_t d_seq, size_t d_head, size_t d, float theta, llaisysDataType_t type){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_paged_(reinterpret_cast<float_t **>(out_map),
            reinterpret_cast<const float_t * const *>(in_map),
            reinterpret_cast<const int64_t *>(pos_ids),
            d_seq, d_head, d, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_paged_(reinterpret_cast<bf16_t **>(out_map),
            reinterpret_cast<const bf16_t * const*>(in_map),
            reinterpret_cast<const int64_t *>(pos_ids),
            d_seq, d_head, d, theta);
    case LLAISYS_DTYPE_F16:
        return rope_paged_(reinterpret_cast<fp16_t **>(out_map),
            reinterpret_cast<const fp16_t * const*>(in_map),
            reinterpret_cast<const int64_t *>(pos_ids),
            d_seq, d_head, d, theta);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}