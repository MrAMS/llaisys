#include "llaisys/models/qwen2.h"
#include "llaisys.h"
#include "llaisys/ops.h"
#include "llaisys/tensor.h"
#include "../llaisys_tensor.hpp"
#include "../../paged_cache/paged_cache.hpp"

#include "../../ops/ops.hpp"
#include "../../sampler/sampler.hpp"

#include <cassert>
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
// #define DEBUG_TENSOR(X)
#else
#define DEBUG(fmt, ...) 
#define DEBUG_TENSOR(X)
#endif

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice){
        auto model = new LlaisysQwen2Model;

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

        model->scheduler = new llaisys::PagedCache::Scheduler(meta->max_running_seqs,
            meta->nlayer,
            meta->block_num, meta->block_size,
            meta->nkvh*meta->dh*llaisys::utils::dsize(meta->dtype),
            device, device_ids[0]);
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

        delete (llaisys::PagedCache::Scheduler*) model->scheduler;

        delete [] model->device_ids;
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model){
        return model->weights;
    }

    static llaisys::tensor_t run_model(struct LlaisysQwen2Model * model, llaisys::PagedCache::Sequence* seq, bool is_prefill){
        // google::InstallFailureSignalHandler();

        if(!model){
            return llaisys::Tensor::create_none(); // Invalid model or input
        }
        using llaisys::Tensor;

        const auto d_seq = is_prefill ? (seq->num_tokens()-seq->num_cached()) : 1;
        // const auto d_cached = model->kv_cached_row;
        const auto dtype = model->meta->dtype;
        const auto device = model->device;
        const auto device_id = model->device_ids[0];

        const auto rms_eps = model->meta->epsilon;
        const auto rope_theta = model->meta->theta;

        const auto none = Tensor::create_none();

        auto inputs = Tensor::create({d_seq}, llaisysDataType_t::LLAISYS_DTYPE_I64, device, device_id);
        inputs->load(seq->tokens_ptr(seq->num_tokens()-d_seq));

        auto pos = Tensor::create({d_seq}, llaisysDataType_t::LLAISYS_DTYPE_I64, device, device_id);
        for(size_t i=0;i<d_seq;++i){
            auto ptr = (int64_t*)(pos->data());
            // ptr[i] = d_cached + i;
            ptr[i] = seq->num_tokens()-(d_seq-i);
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
            
            const auto head_kv = model->meta->nkvh;
            const auto head_q = model->meta->nh;
            const auto d_qk = k_proj_w->shape()[0]/head_kv;
            const auto d_v = v_proj_w->shape()[0]/head_kv;

            const auto slot_mapping_k = seq->get_slot_mapping(layer, false);
            const auto slot_mapping_v = seq->get_slot_mapping(layer, true);

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
            std::vector<std::byte*> k_new;
            for(size_t i=slot_mapping_k.size()-d_seq;i<slot_mapping_k.size();++i) k_new.push_back(slot_mapping_k[i]);
            llaisys::ops::linear_paged(k_new.data(), atten_norm, k_proj_w, k_proj_b);
            // 位置编码
            llaisys::ops::rope_paged(k_new.data(), k_new.data(), pos, d_seq, head_kv, d_qk, hidden_states->dtype(), rope_theta);
            // DEBUG_TENSOR(k_new);

            // 只投影V新的部分
            DEBUG("V");
            std::vector<std::byte*> v_new;
            for(size_t i=slot_mapping_v.size()-d_seq;i<slot_mapping_v.size();++i) v_new.push_back(slot_mapping_v[i]);
            llaisys::ops::linear_paged(v_new.data(), atten_norm, v_proj_w, v_proj_b);
            // DEBUG_TENSOR(v_new);

            // 从KV Cache中获取完整的K和V
            std::vector<std::byte*> k;
            for(size_t i=0;i<slot_mapping_k.size();++i) k.push_back(slot_mapping_k[i]);
            std::vector<std::byte*> v;
            for(size_t i=0;i<slot_mapping_v.size();++i) v.push_back(slot_mapping_v[i]);

            // 计算注意力分数
            DEBUG("scores");
            auto scores = Tensor::create({d_seq, head_q, d_v}, dtype, device, device_id);
            llaisys::ops::self_attention_paged(scores, q, k.data(), v.data(), head_kv, k.size(), 1.f/std::sqrt(d_qk*1.f));
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
        return logits_last_row;

        // auto max_idx = Tensor::create({1}, llaisysDataType_t::LLAISYS_DTYPE_I64, device, device_id);
        // auto max_val = Tensor::create({1}, dtype, device, device_id);
        // llaisys::ops::argmax(max_idx, max_val, logits_last_row);

        // return *((int64_t*)max_idx->data()); 
    }

    void llaisysQwen2SchedulerAdd(struct LlaisysQwen2Model * model, uint64_t seq_id, int64_t * token_ids, size_t ntoken){
        auto scheduler = (llaisys::PagedCache::Scheduler*)(model->scheduler);
        scheduler->add(seq_id, std::vector<llaisys::PagedCache::token_t>(token_ids, token_ids+ntoken), model->meta->end_token);
    }

    bool llaisysQwen2SchedulerStep(struct LlaisysQwen2Model * model, uint64_t* nseq, uint64_t* seq_len, uint64_t* seq_ids, int64_t* token_ids){
        auto scheduler = (llaisys::PagedCache::Scheduler*)(model->scheduler);
        auto sampler = llaisys::sampler::SamplerArgmax();
        auto [seqs, is_prefill] = scheduler->schedule();
        *nseq = seqs.size();
        printf("nseq=%ld\n", *nseq);
        for(size_t i=0;i<seqs.size();++i){
            auto seq = seqs[i];
            llaisys::tensor_t logits= run_model(model, seq, is_prefill);
            const auto new_token = sampler.sample(logits);

            seq_len[i] = 1;
            seq_ids[i] = seq->get_id();
            *(token_ids++) = new_token;
            
            scheduler->postprocess(seq, new_token);
        }
        return scheduler->is_finished();
    }

}