#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#define TOF(X) llaisys::utils::cast<float>(X)

template<typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, 
    float scale, size_t d_seq, size_t d_head, size_t d_v, size_t d_qk, size_t d_kvhead, size_t d_tot){
        // attn_val: [d_seq, d_head, d_v], q: [d_seq, d_head, d_qk]
        // k: [d_tot, d_kvhead, d_qk], v: [d_tot, d_kvhead, d_v]
        for(size_t s=0;s<d_seq;++s){ // 遍历scores矩阵的每一行
            for(size_t h=0;h<d_head;++h){
                const auto h_kv = h/(d_head/d_kvhead);
                std::vector<float> attn_scores(d_tot); // 临时数据暂存（scores矩阵中的一行）
                // 计算Attention Scores
                for(size_t j=0;j<d_tot;++j){ // 遍历scores矩阵的每一列
                    attn_scores[j] = 0; 
                    for(size_t l=0;l<d_qk;++l){ // 计算Q@K.T
                        const auto q_idx = s*d_head*d_qk + h*d_qk + l;
                        // TODO Cache Hit优化,K应该变为[d_kvhead, d_tot, d_qk]
                        const auto k_idx = j*d_kvhead*d_qk + h_kv*d_qk + l; 
                        attn_scores[j] += TOF(q[q_idx]) * TOF(k[k_idx]);
                    }
                    attn_scores[j] *= scale;
                }
                // 计算Causal Softmax
                const auto causal_len = std::min(d_tot-d_seq+s+1, d_tot);
                float max_score = attn_scores[0];
                for(size_t j=1;j<causal_len;++j){
                    if(attn_scores[j] > max_score){
                        max_score = attn_scores[j];
                    }
                }
                float sum_exp = 0;
                for(size_t j=0;j<causal_len;++j){
                    attn_scores[j] = std::exp(attn_scores[j] - max_score);
                    sum_exp += attn_scores[j];
                }
                for(size_t j=0;j<causal_len;++j){
                    attn_scores[j] /= sum_exp;
                }

                // 计算scores @ V
                for(size_t j=0;j<d_v;++j){ // 遍历out的每一列
                    float res = 0;
                    for(size_t l=0;l<causal_len;++l){ // 计算scores @ V
                        // TODO Cache Hit优化,V应该变为[d_kvhead, d_v, d_tot]
                        const auto v_idx = l*d_kvhead*d_v + h_kv*d_v + j;
                        const auto a_idx = l;
                        res += attn_scores[a_idx] * TOF(v[v_idx]);
                    }
                    const auto out_idx = s*d_head*d_v + h*d_v + j;
                    attn_val[out_idx] = llaisys::utils::cast<T>(res);
                }
            }
        }
    }

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, 
    float scale, size_t d_seq, size_t d_head, size_t d_v, size_t d_qk, size_t d_kvhead, size_t d_tot,
    llaisysDataType_t type){
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return self_attention_(reinterpret_cast<float_t *>(attn_val),
                reinterpret_cast<const float_t *>(q),
                reinterpret_cast<const float_t *>(k),
                reinterpret_cast<const float_t *>(v),
                scale, d_seq, d_head, d_v, d_qk, d_kvhead, d_tot);
        case LLAISYS_DTYPE_BF16:
            return self_attention_(reinterpret_cast<bf16_t *>(attn_val),
                reinterpret_cast<const bf16_t *>(q),
                reinterpret_cast<const bf16_t *>(k),
                reinterpret_cast<const bf16_t *>(v),
                scale, d_seq, d_head, d_v, d_qk, d_kvhead, d_tot);
        case LLAISYS_DTYPE_F16:
            return self_attention_(reinterpret_cast<fp16_t *>(attn_val),
                reinterpret_cast<const fp16_t *>(q),
                reinterpret_cast<const fp16_t *>(k),
                reinterpret_cast<const fp16_t *>(v),
                scale, d_seq, d_head, d_v, d_qk, d_kvhead, d_tot);

        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}