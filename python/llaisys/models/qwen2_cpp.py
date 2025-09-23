from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, llaisysTensor_t
from ..libllaisys.qwen2_cpp import load_qwen2_cpp, LlaisysQwen2Meta, LlaisysQwen2Model, LlaisysQwen2Weights
from ..tensor import Tensor, DataType
from ..ops import Ops

from pathlib import Path
import safetensors

import numpy as np
import json
import dataclasses
import ctypes
import torch

load_qwen2_cpp(LIB_LLAISYS)

from huggingface_hub import snapshot_download

max_finished_seqs = 0
max_running_seqs = 2

block_num = 25
block_size = 4

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        self.device = device

        print("Reading config...", flush=True)
        with open(model_path / 'config.json', 'r') as config_file:
            config = json.load(config_file)
        self.vocab_size = config.get("vocab_size")
        self.hidden_size = config.get("hidden_size")
        self.num_layers = config.get('num_hidden_layers')
        self.intermediate_size = config.get("intermediate_size")
        self.max_position_embeddings = config.get("max_position_embeddings")
        self.eos_token_id = config.get('eos_token_id')
        self.num_atten_heads = config.get('num_attention_heads')
        self.num_key_value_heads = config.get('num_key_value_heads')
        self.rms_norm_eps = config.get('rms_norm_eps')
        self.rope_theta = config.get('rope_theta')
        self.per_head_dim = self.hidden_size // self.num_atten_heads
        self.per_kvhead_dim = self.per_head_dim
        self.dtype = DataType.from_config_json(config.get('torch_dtype'))
        if device == DeviceType.CPU:
            if self.dtype == DataType.BF16:
                # 在CPU上直接使用FP32权重以加速
                self.dtype = DataType.F32

        print("Model config loaded:", {
            "dtype":self.dtype,
            "nlayer":self.num_layers,
            "hs":self.hidden_size,
            "nh":self.num_atten_heads,
            "nkvh":self.num_key_value_heads,
            "dh":self.per_head_dim,
            "di":self.intermediate_size,
            "maxseq":self.max_position_embeddings,
            "voc":self.vocab_size,
            "epsilon":self.rms_norm_eps,
            "theta":self.rope_theta,
            "end_token":self.eos_token_id
        }, flush=True)

        meta = LlaisysQwen2Meta(
            dtype=self.dtype,
            nlayer=self.num_layers,
            hs=self.hidden_size,
            nh=self.num_atten_heads,
            nkvh=self.num_key_value_heads,
            dh=ctypes.c_size_t(self.per_head_dim),
            di=self.intermediate_size,
            maxseq=self.max_position_embeddings,
            voc=self.vocab_size,
            epsilon=self.rms_norm_eps,
            theta=self.rope_theta,
            end_token=self.eos_token_id,
            max_running_seqs=max_running_seqs,
            max_finished_seqs=max_finished_seqs,
            block_num=block_num,
            block_size=block_size,
        )

        device_ids = (ctypes.c_int * 1)(0)

        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            ctypes.c_int(device),
            device_ids,
            ctypes.c_int(1)
        )

        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)

        print("Loading weights...", flush=True)

        for file in sorted(model_path.glob("*.safetensors")):
            safetensors_data = safetensors.safe_open(file, framework="pt", device="cpu")

            def get_tensor(tensor_name):
                tensor = safetensors_data.get_tensor(tensor_name).contiguous()
                if self.dtype == DataType.F32:
                    tensor = tensor.to(torch.float32)
                return tensor

            def load_weight(cpp_name, weight_name):
                tensor = get_tensor(weight_name)
                ptr = ctypes.cast(getattr(weights.contents, cpp_name), llaisysTensor_t)
                LIB_LLAISYS.tensorLoad(ptr, tensor.data_ptr())

            def load_layer_weights(cpp_name, weight_name):
                ptr = ctypes.cast(getattr(weights.contents, cpp_name), ctypes.POINTER(llaisysTensor_t*self.num_layers)).contents

                for i in range(self.num_layers):
                    tensor_name = f"model.layers.{i}.{weight_name}"
                    tensor = get_tensor(tensor_name)
                    LIB_LLAISYS.tensorLoad(ptr[i], tensor.data_ptr())

            load_weight("in_embed", "model.embed_tokens.weight")

            load_layer_weights("attn_norm_w", "input_layernorm.weight")
            load_layer_weights("attn_q_w", "self_attn.q_proj.weight")
            load_layer_weights("attn_q_b", "self_attn.q_proj.bias")
            load_layer_weights("attn_k_w", "self_attn.k_proj.weight")
            load_layer_weights("attn_k_b", "self_attn.k_proj.bias")
            load_layer_weights("attn_v_w", "self_attn.v_proj.weight")
            load_layer_weights("attn_v_b", "self_attn.v_proj.bias")
            load_layer_weights("attn_o_w", "self_attn.o_proj.weight")

            load_layer_weights("mlp_norm_w", "post_attention_layernorm.weight")
            load_layer_weights("mlp_gate_w", "mlp.gate_proj.weight")
            load_layer_weights("mlp_up_w", "mlp.up_proj.weight")
            load_layer_weights("mlp_down_w", "mlp.down_proj.weight")

            load_weight("out_norm_w", "model.norm.weight")
            load_weight("out_embed", "lm_head.weight")



        print("Weights loaded.", flush=True)

    def __del__(self):
        LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
        self.model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if self.model is None:
            raise RuntimeError("Model is null")
        
        output = list(inputs)

        prefill_len = len(inputs)
        max_seq_len = prefill_len + max_new_tokens

        LIB_LLAISYS.llaisysQwen2SchedulerAdd(self.model, 
            ctypes.c_uint64(0),
            ctypes.cast((ctypes.c_int64 * prefill_len)(*inputs), ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_size_t(prefill_len),
            max_new_tokens
        )

        print("Generating...", flush=True)
        round=0
        while True:
            round += 1

            nseq = (ctypes.c_uint64 * max_running_seqs)()
            seq_len = (ctypes.c_uint64 * max_running_seqs)()
            seq_ids = (ctypes.c_uint64 * max_running_seqs)()
            token_ids = (ctypes.c_int64 * max_running_seqs)()


            finished = LIB_LLAISYS.llaisysQwen2SchedulerStep(
                self.model,
                nseq, seq_len, seq_ids, token_ids
            )

            if finished:
                break
            
            assert(seq_len[0] == 1)
            new_token = token_ids[0]

            print(f"Round {round}, token:", new_token, flush=True)

            output.append(new_token)

        
        return output
    
    def add_request(
        self,
        id: int,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        LIB_LLAISYS.llaisysQwen2SchedulerAdd(self.model, 
            ctypes.c_uint64(id),
            ctypes.cast((ctypes.c_int64 * len(inputs))(*inputs), ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_size_t(len(inputs)),
            max_new_tokens
        )


    def step(
        self
    ):
        nseq = (ctypes.c_uint64 * max_running_seqs)()
        seq_len = (ctypes.c_uint64 * max_running_seqs)()
        seq_ids = (ctypes.c_uint64 * max_running_seqs)()
        token_ids = (ctypes.c_int64 * max_running_seqs)()


        finished = LIB_LLAISYS.llaisysQwen2SchedulerStep(
            self.model,
            nseq, seq_len, seq_ids, token_ids
        )

        res = []
        pre = 0
        for i in range(nseq[0]):
            l = seq_len[i]
            res.append({
                'id': seq_ids[i],
                'tokens': list(token_ids[pre:pre+l])
            })
            pre += l
            
        return finished, res


