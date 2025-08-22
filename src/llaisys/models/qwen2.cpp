#include "llaisys/models/qwen2.h"
#include "llaisys.h"
#include "llaisys/ops.h"
#include "llaisys/tensor.h"
#include "../llaisys_tensor.hpp"

#include "../../ops/ops.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#define DEBUG_ON 0

#if DEBUG_ON
#include <cstdio>
#include <cstdlib>
#define DEBUG(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)
#define DEBUG_TENSOR(X) X->debug()
#else
#define DEBUG(fmt, ...) 
#define DEBUG_TENSOR(X)
#endif

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice){
        auto model = new LlaisysQwen2Model;
        model->kv_cached_row = 0;
        model->k_caches = nullptr;
        model->v_caches = nullptr;

        model->meta = new LlaisysQwen2Meta(*meta);

        model->device = device;
        model->ndevice = ndevice;
        model->device_ids =  new int[ndevice];
        std::memcpy(model->device_ids, device_ids, sizeof(int) * ndevice);

        model->weights = new LlaisysQwen2Weights;

        auto alloc_weight = [&](llaisysTensor_t &ptr, std::vector<size_t> shape) {
            ptr = tensorCreate(shape.data(), shape.size(), meta->dtype, model->device, model->device_ids[0]);
        };

        auto alloc_layer_weights = [&](llaisysTensor_t *&ptr, std::vector<size_t> shape) {
            ptr = new llaisysTensor_t[meta->nlayer];
            for (size_t i = 0; i < meta->nlayer; ++i) {
                ptr[i] = tensorCreate(shape.data(), shape.size(), meta->dtype, model->device, model->device_ids[0]);
            }
        };

        // Input Embedding
        alloc_weight(model->weights->in_embed, {meta->voc, meta->hs});

        // Self-Attention
        alloc_layer_weights(model->weights->attn_norm_w, {meta->hs});
        alloc_layer_weights(model->weights->attn_q_w, {meta->hs, meta->nh * meta->dh});
        alloc_layer_weights(model->weights->attn_q_b, {meta->nh * meta->dh});
        alloc_layer_weights(model->weights->attn_k_w, {meta->nkvh * meta->dh, meta->hs});
        alloc_layer_weights(model->weights->attn_k_b, {meta->nkvh * meta->dh});
        alloc_layer_weights(model->weights->attn_v_w, {meta->nkvh * meta->dh, meta->hs});
        alloc_layer_weights(model->weights->attn_v_b, {meta->nkvh * meta->dh});
        alloc_layer_weights(model->weights->attn_o_w, {meta->nh * meta->dh, meta->hs});

        // MLP
        alloc_layer_weights(model->weights->mlp_norm_w, {meta->hs});
        alloc_layer_weights(model->weights->mlp_gate_w, {meta->di, meta->hs});
        alloc_layer_weights(model->weights->mlp_up_w,   {meta->di, meta->hs});
        alloc_layer_weights(model->weights->mlp_down_w, {meta->hs, meta->di});

        // Output Norm
        alloc_weight(model->weights->out_norm_w, {meta->hs});

        // Output Embedding
        alloc_weight(model->weights->out_embed, { meta->voc, meta->hs });


        return model;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model){
        if(!model) return;

        delete model->meta;

        // delete 会调用类的构析函数，类的构析函数会自动调用所有非指针成员变量的构析函数，但是不会处理指针成员变量
        delete [] model->weights->attn_norm_w;
        delete [] model->weights->attn_q_w;
        delete [] model->weights->attn_q_b;
        delete [] model->weights->attn_k_w;
        delete [] model->weights->attn_k_b;
        delete [] model->weights->attn_v_w;
        delete [] model->weights->attn_v_b;
        delete [] model->weights->attn_o_w;

        delete [] model->weights->mlp_norm_w;
        delete [] model->weights->mlp_gate_w;
        delete [] model->weights->mlp_up_w;
        delete [] model->weights->mlp_down_w;
        
        delete model->weights;

        if(model->k_caches)
            delete [] model->k_caches;
        if(model->v_caches)
            delete [] model->v_caches;

        delete [] model->device_ids;
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model){
        return model->weights;
    }

    void llaisysQwen2ModelAllocKVCache(struct LlaisysQwen2Model * model, size_t max_seq_len){
        if(!model) return;

        if(model->k_caches)
            delete [] model->k_caches;
        if(model->v_caches)
            delete [] model->v_caches;

        model->kv_cached_row = 0;

        auto alloc_kvcache = [&](llaisysTensor_t* &ptr, std::vector<size_t> shape){
            ptr = new llaisysTensor_t[model->meta->nlayer];
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                ptr[i] = tensorCreate(shape.data(), shape.size(), model->meta->dtype, model->device, model->device_ids[0]);
            }
        };

        alloc_kvcache(model->k_caches, {max_seq_len, model->meta->nkvh*model->meta->dh});
        alloc_kvcache(model->v_caches, {max_seq_len, model->meta->nkvh*model->meta->dh});
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken){
        // google::InstallFailureSignalHandler();

        if(!model || !token_ids || ntoken == 0 || !model->k_caches || !model->v_caches){
            return -1; // Invalid model or input
        }
        using llaisys::Tensor;

        const auto d_seq = ntoken;
        const auto d_cached = model->kv_cached_row;
        const auto dtype = model->meta->dtype;
        const auto device = model->device;
        const auto device_id = model->device_ids[0];

        const auto rms_eps = model->meta->epsilon;
        const auto rope_theta = model->meta->theta;

        const auto none = Tensor::create_none();

        auto inputs = Tensor::create({d_seq}, llaisysDataType_t::LLAISYS_DTYPE_I64, device, device_id);
        inputs->load(token_ids);

        auto pos = Tensor::create({d_seq}, llaisysDataType_t::LLAISYS_DTYPE_I64, device, device_id);
        for(size_t i=0;i<d_seq;++i){
            auto ptr = (int64_t*)(pos->data());
            ptr[i] = d_cached + i;
        }
        //
        // Input Embedding
        //
        DEBUG("Input Embedding");
        const auto embed_tokens_w = model->weights->in_embed->tensor;
        const auto d_emb = embed_tokens_w->shape()[1];
        auto hidden_states = Tensor::create({d_seq, d_emb}, dtype, device, device_id);
        llaisys::ops::embedding(hidden_states, inputs, embed_tokens_w);
        DEBUG_TENSOR(hidden_states);

        for(size_t layer=0;layer<model->meta->nlayer;++layer){
            DEBUG("Layer %ld", layer);
            DEBUG("Atten");
            //
            // 自注意力
            //
            const auto input_layernorm_w = model->weights->attn_norm_w[layer]->tensor;
            const auto q_proj_w = model->weights->attn_q_w[layer]->tensor;
            const auto q_proj_b = model->weights->attn_q_b[layer]->tensor;
            const auto k_proj_w = model->weights->attn_k_w[layer]->tensor;
            const auto k_proj_b = model->weights->attn_k_b[layer]->tensor;
            const auto v_proj_w = model->weights->attn_v_w[layer]->tensor;
            const auto v_proj_b = model->weights->attn_v_b[layer]->tensor;
            const auto o_proj_w = model->weights->attn_o_w[layer]->tensor;

            auto k_cache = model->k_caches[layer]->tensor;
            auto v_cache = model->v_caches[layer]->tensor;
            const auto kv_cached_row = model->kv_cached_row;
            
            const auto head_kv = model->meta->nkvh;
            const auto head_q = model->meta->nh;
            const auto d_qk = k_proj_w->shape()[0]/head_kv;
            const auto d_v = v_proj_w->shape()[0]/head_kv;

            // 输入归一化
            DEBUG("rms_norm");
            auto atten_norm = Tensor::create({d_seq, d_emb}, dtype, device, device_id);
            llaisys::ops::rms_norm(atten_norm, hidden_states, input_layernorm_w, rms_eps);
            DEBUG_TENSOR(hidden_states);
            
            // 投影Q
            DEBUG("Q");
            auto q = Tensor::create({d_seq, head_q*d_qk}, dtype, device, device_id);
            llaisys::ops::linear(q, atten_norm, q_proj_w, q_proj_b);
            q = q->view({d_seq, head_q, d_qk});
            // 位置编码
            llaisys::ops::rope(q, q, pos, rope_theta);
            DEBUG_TENSOR(q);

            // 只投影K新的部分
            DEBUG("K");
            auto k_new = k_cache->slice(0, kv_cached_row, kv_cached_row+d_seq);
            llaisys::ops::linear(k_new, atten_norm, k_proj_w, k_proj_b);
            k_new = k_new->view({d_seq, head_kv, d_qk});
            // 位置编码
            llaisys::ops::rope(k_new, k_new, pos, rope_theta);
            DEBUG_TENSOR(k_new);

            // 只投影V新的部分
            DEBUG("V");
            auto v_new = v_cache->slice(0, kv_cached_row, kv_cached_row+d_seq);
            llaisys::ops::linear(v_new, atten_norm, v_proj_w, v_proj_b);
            v_new = v_new->view({d_seq, head_kv, d_v});
            DEBUG_TENSOR(v_new);

            // 从KV Cache中获取完整的K和V
            auto k = k_cache->slice(0, 0, kv_cached_row+d_seq);
            k = k->view({kv_cached_row+d_seq, head_kv, d_qk});

            auto v = v_cache->slice(0, 0, kv_cached_row+d_seq);
            v = v->view({kv_cached_row+d_seq, head_kv, d_v});

            // 计算注意力分数
            DEBUG("scores");
            auto scores = Tensor::create({d_seq, head_q, d_v}, dtype, device, device_id);
            llaisys::ops::self_attention(scores, q, k, v, 1.f/std::sqrt(d_qk));
            scores = scores->view({d_seq, head_q*d_v});
            
            // 输出投影
            DEBUG("proj");
            auto atten_res = Tensor::create({d_seq, d_emb}, dtype, device, device_id);
            llaisys::ops::linear(atten_res, scores, o_proj_w, none);
            DEBUG_TENSOR(atten_res);

            // 残差连接
            DEBUG("residual1");
            llaisys::ops::add(hidden_states, hidden_states, atten_res);
            DEBUG_TENSOR(hidden_states);

            //
            // MLP
            //
            DEBUG("MLP");
            const auto mlp_norm_w = model->weights->mlp_norm_w[layer]->tensor;
            const auto mlp_gate_proj_w = model->weights->mlp_gate_w[layer]->tensor;
            const auto mlp_up_proj_w = model->weights->mlp_up_w[layer]->tensor;
            const auto mlp_down_proj_w = model->weights->mlp_down_w[layer]->tensor;

            const auto d_inter = model->meta->di; 
            // 归一化
            DEBUG("mlp_norm");
            auto mlp_norm = Tensor::create({d_seq, d_emb}, dtype, device, device_id);
            llaisys::ops::rms_norm(mlp_norm, hidden_states, mlp_norm_w, rms_eps);
            DEBUG_TENSOR(mlp_norm);

            DEBUG("gate");
            auto gate = Tensor::create({d_seq, d_inter}, dtype, device, device_id);
            llaisys::ops::linear(gate, mlp_norm, mlp_gate_proj_w, none);

            DEBUG("up");
            auto up = Tensor::create({d_seq, d_inter}, dtype, device, device_id);
            llaisys::ops::linear(up, mlp_norm, mlp_up_proj_w, none);

            // swiglu
            DEBUG("swiglu");
            auto swiglu_out = Tensor::create({d_seq, d_inter}, dtype, device, device_id);
            llaisys::ops::swiglu(swiglu_out, gate, up);

            DEBUG("mlp_proj");
            auto mlp_out = Tensor::create({d_seq, d_emb}, dtype, device, device_id);
            llaisys::ops::linear(mlp_out, swiglu_out, mlp_down_proj_w, none);
            DEBUG_TENSOR(mlp_out);
            //
            // 残差连接
            //
            DEBUG("residual2");
            llaisys::ops::add(hidden_states, hidden_states, mlp_out);
            DEBUG_TENSOR(hidden_states);
        }

        // 更新KV Cache
        model->kv_cached_row += d_seq;

        //
        // rms_norm
        //
        const auto norm_w = model->weights->out_norm_w->tensor;
        
        llaisys::ops::rms_norm(hidden_states, hidden_states, norm_w, rms_eps);

        //
        // LM Head
        //
        const auto lm_head_w = model->weights->out_embed->tensor;
        const auto d_vocab = lm_head_w->shape()[0];
        const auto logits = Tensor::create({d_seq, d_vocab}, dtype, device, device_id);
        
        llaisys::ops::linear(logits, hidden_states, lm_head_w, none);

        //
        // 
        //
        auto logits_last_row = logits->slice(0, logits->shape()[0]-1, logits->shape()[0])->view({logits->shape()[1]});
        auto max_idx = Tensor::create({1}, llaisysDataType_t::LLAISYS_DTYPE_I64, device, device_id);
        auto max_val = Tensor::create({1}, dtype, device, device_id);
        llaisys::ops::argmax(max_idx, max_val, logits_last_row);

        return *((int64_t*)max_idx->data()); 
    }
}