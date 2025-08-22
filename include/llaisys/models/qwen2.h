#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer; // num_hidden_layers
        size_t hs; // hidden_size
        size_t nh; // num_attention_heads
        size_t nkvh; // num_key_value_heads
        size_t dh; // per_head_dim
        size_t di; // intermediate_size
        size_t maxseq; // max_position_embeddings
        size_t voc; // vocab_size
        float epsilon; // rms_norm_eps
        float theta; // rope_theta
        int64_t end_token; // eos_token_id
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model{
        llaisysDeviceType_t device;
        int* device_ids;
        size_t ndevice;
        const LlaisysQwen2Meta *meta;
        struct LlaisysQwen2Weights* weights;
        llaisysTensor_t* k_caches;
        llaisysTensor_t* v_caches;
        size_t kv_cached_row; // tot_seq_len
    };

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export void llaisysQwen2ModelAllocKVCache(struct LlaisysQwen2Model * model, size_t max_seq_len);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);
}
#endif // LLAISYS_MODELS_QWEN2_H
