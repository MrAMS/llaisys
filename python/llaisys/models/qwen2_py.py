from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, llaisysDataType_t, llaisysTensor_t, llaisysDeviceType_t
from ..tensor import Tensor, DataType
from ..ops import Ops

from pathlib import Path
import safetensors

import numpy as np
import json
import dataclasses
import ctypes
import torch

from huggingface_hub import snapshot_download

_DEBUG = False

def debug(msg: str):
    if _DEBUG:
        print("[DEBUG]", msg, flush=True)

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        if model_path is not None and Path(model_path).exists():
            print(f"Using local model path: {model_path}", flush=True)
        else:
            print("Model path not provided or does not exist. Downloading from Huggingface...", flush=True)
            model_path = Path(snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"))
            print(f"Model downloaded to: {model_path}", flush=True)
        
        model_path = Path(model_path)
        self.device = device

        with open(model_path.joinpath('config.json'), 'r') as config_file:
            config = json.load(config_file)
            self.num_layers = config.get('num_hidden_layers')
            self.eos_token_id = config.get('eos_token_id')
            self.num_atten_heads = config.get('num_attention_heads')
            self.num_key_value_heads = config.get('num_key_value_heads')
            self.rms_norm_eps = config.get('rms_norm_eps')
            self.rope_theta = config.get('rope_theta')
            self.dtype = DataType.from_config_json(config.get('torch_dtype'))
            if device == DeviceType.CPU:
                if self.dtype == DataType.BF16:
                    # 在CPU上直接使用FP32权重加速
                    self.dtype = DataType.F32

        self.layers: Sequence[Qwen2Layer] = []
        for i in range(self.num_layers):
            self.layers.append(Qwen2Layer(i, self.num_atten_heads, self.num_key_value_heads, self.rms_norm_eps, self.rope_theta, device))

        print("Loading weights...", flush=True)

        for file in sorted(model_path.glob("*.safetensors")):
            # with open(file, "rb") as f:
            #     data_ = safetensors.deserialize(f.read()) # [("tensor_name", {"shape": [2, 3], "dtype": "F32", "data": b"\0\0.." }), (...)]
            # for tensor_ in data_:
            #     weight_name, weight_dict = tensor_
            #     shape = weight_dict["shape"]
            #     dtype = weight_dict["dtype"]
            #     raw_data = weight_dict["data"]
            with safetensors.safe_open(file, framework="pt", device='cpu') as f:
                for weight_name in f.keys():
                    tensor = f.get_tensor(weight_name).contiguous()
                    if self.dtype == DataType.F32:
                        tensor = tensor.to(torch.float32)
                    shape = tensor.shape
                    weight = Tensor(shape=shape, dtype=DataType.from_torch(tensor.dtype), device=device)
                    weight.load(tensor.data_ptr())
                    
                    if weight_name == "model.embed_tokens.weight":
                        self.embed_tokens_weight = weight
                    elif weight_name == "lm_head.weight":
                        self.lm_head_weight = weight
                    elif weight_name == "model.norm.weight":
                        self.norm_weight = weight
                    elif weight_name.startswith("model.layers."):
                        layer_idx = int(weight_name.split(".")[2])
                        if 0 <= layer_idx < self.num_layers:
                            self.layers[layer_idx].load_weight(weight_name, weight)
                    else:
                        raise RuntimeError(f"Unknown weight name: {weight_name}")
                
    def forward(self, tensor_in: Tensor, tensor_pos: Tensor) -> Tensor:
        dtype = self.embed_tokens_weight.dtype()

        d_seq = tensor_in.shape()[0]
        d_emb = self.embed_tokens_weight.shape()[1] # type: ignore
        hidden_states = Tensor(shape=(d_seq, d_emb), dtype=dtype, device=self.device)
        Ops.embedding(out=hidden_states, index=tensor_in, weight=self.embed_tokens_weight)

        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, tensor_pos)

        normalized = Tensor(shape=(d_seq, d_emb), dtype=dtype, device=self.device)
        Ops.rms_norm(out=normalized, inp=hidden_states, weight=self.norm_weight, eps=self.rms_norm_eps)

        d_vocab = self.lm_head_weight.shape()[0]
        logits = Tensor(shape=(d_seq, d_vocab), dtype=dtype, device=self.device)
        Ops.linear(out=logits, inp=normalized, weight=self.lm_head_weight)

        return logits

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        prefill_len = len(inputs)
        max_seq_len = prefill_len + max_new_tokens
        print("Allocate KV Cache...")
        # Allocate KV Cache
        for layer in self.layers:
            layer.reset_kvcache(max_seq_len)
        print("Prefilling...")
        # Prefill
        token = self._prefill(inputs)
        output = list(inputs)
        output.append(token)
        print("Prefill done, token:", token, flush=True)
        # Decode
        for pos in range(prefill_len, max_seq_len):
            token = self._decode(token, pos)
            print(f"Decode round {pos-prefill_len}, token:", token, flush=True)
            output.append(token)
            if token == self.eos_token_id:
                break

        return output
    
    def _prefill(self, inputs: Sequence[int]) -> int:
        # Prefill
        d_prefill = len(inputs)
        tensor_in = Tensor(shape=(d_prefill,), dtype=DataType.I64, device=self.device)
        tensor_in.load_np(np.array(inputs, dtype=np.int64))
        tensor_pos = Tensor(shape=(d_prefill,), dtype=DataType.I64, device=self.device)
        tensor_pos.load_np(np.arange(d_prefill, dtype=np.int64))

        logits = self.forward(tensor_in, tensor_pos)
        logits_last_row = logits.slice(0, -1).view(logits.shape()[1]) # type: ignore

        max_idx = Tensor(shape=(1,), dtype=DataType.I64, device=self.device)
        max_val = Tensor(shape=(1,), dtype=self.dtype, device=self.device)
        Ops.argmax(max_idx=max_idx, max_val=max_val, vals=logits_last_row)
        return max_idx.val() # type: ignore
    
    def _decode(self, token: int, pos: int) -> int:
        tensor_in = Tensor(shape=(1,), dtype=DataType.I64, device=self.device)
        tensor_in.load_np(np.array([token], dtype=np.int64))
        tensor_pos = Tensor(shape=(1,), dtype=DataType.I64, device=self.device)
        tensor_pos.load_np(np.array([pos], dtype=np.int64))

        logits = self.forward(tensor_in, tensor_pos)
        logits = logits.view(logits.shape()[1]) # type: ignore

        max_idx = Tensor(shape=(1,), dtype=DataType.I64, device=self.device)
        max_val = Tensor(shape=(1,), dtype=self.dtype, device=self.device)
        Ops.argmax(max_idx=max_idx, max_val=max_val, vals=logits)
        return max_idx.val() # type: ignore


@dataclasses.dataclass
class Qwen2Layer:
    layer_idx: int
    num_atten_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    device: DeviceType = DeviceType.CPU

    def load_weight(self, name: str, tensor: Tensor):
        # 注意力层权重
        if "self_attn.q_proj.weight" in name:
            # [d_q, d_emb]
            self.q_proj_weight = tensor
        elif "self_attn.q_proj.bias" in name:
            # [d_emb]
            self.q_proj_bias = tensor
        elif "self_attn.k_proj.weight" in name:
            # [d_k, d_emb]
            self.k_proj_weight = tensor
        elif "self_attn.k_proj.bias" in name:
            self.k_proj_bias = tensor
        elif "self_attn.v_proj.weight" in name:
            self.v_proj_weight = tensor
        elif "self_attn.v_proj.bias" in name:
            self.v_proj_bias = tensor
        elif "self_attn.o_proj.weight" in name:
            self.o_proj_weight = tensor
        # MLP层权重
        elif "mlp.gate_proj.weight" in name:
            self.gate_proj_weight = tensor
        elif "mlp.up_proj.weight" in name:
            self.up_proj_weight = tensor
        elif "mlp.down_proj.weight" in name:
            self.down_proj_weight = tensor
        # 归一化层权重
        elif "input_layernorm.weight" in name:
            self.input_layernorm_weight = tensor
        elif "post_attention_layernorm.weight" in name:
            self.post_atten_layernorm_weight = tensor
            
    def reset_kvcache(self, max_seq_len: int):
        self.k_cache = Tensor(shape=(max_seq_len, self.k_proj_weight.shape()[0]), dtype=self.k_proj_weight.dtype(), device=self.device)
        self.v_cache = Tensor(shape=(max_seq_len, self.v_proj_weight.shape()[0]), dtype=self.v_proj_weight.dtype(), device=self.device)
        self.kv_cached_rows = 0
    
    def forward(self, hidden_states: Tensor, tensor_pos: Tensor) -> Tensor:
        debug(f"forward layer {self.layer_idx}...")
        
        d_seq = hidden_states.shape()[0]
        d_emb = hidden_states.shape()[1] # type: ignore

        dtype = hidden_states.dtype()

        debug(f"input_layernorm...")
        # 输入归一化
        input_layernorm = Tensor(shape=(d_seq, d_emb), dtype=dtype, device=self.device)
        Ops.rms_norm(out=input_layernorm, inp=hidden_states, weight=self.input_layernorm_weight, eps=self.rms_norm_eps)
        
        # 自注意力
        atten_res = self._self_attention(input_layernorm, tensor_pos)

        # 残差连接
        debug(f"attn_residual...")
        attn_residual = Tensor(shape=(d_seq, d_emb), dtype=dtype, device=self.device)
        Ops.add(c=attn_residual, a=hidden_states, b=atten_res)

        # 归一化
        debug(f"post_atten_layernorm...")
        post_atten_layernorm = Tensor(shape=(d_seq, d_emb), dtype=dtype, device=self.device)
        Ops.rms_norm(out=post_atten_layernorm, inp=attn_residual, weight=self.post_atten_layernorm_weight, eps=self.rms_norm_eps)
        

        # MLP
        debug(f"mlp...")
        mlp_out = self._mlp(post_atten_layernorm)
        mlp_out.debug()
        

        # 残差连接
        debug(f"mlp_residual...")
        mlp_residual = Tensor(shape=(d_seq, d_emb), dtype=dtype, device=self.device)
        Ops.add(c=mlp_residual, a=attn_residual, b=mlp_out)
        mlp_residual.debug()
        exit()

        return mlp_residual
        
    
    def _self_attention(self, hidden_states: Tensor, tensor_pos: Tensor) -> Tensor:
        debug(f"_self_attention...")
        d_seq, d_emb = hidden_states.shape() # type: ignore
        dtype = hidden_states.dtype()
        head_kv = self.num_key_value_heads
        head_q = self.num_atten_heads

        d_qk = self.k_proj_weight.shape()[0] // head_kv
        d_v = self.v_proj_weight.shape()[0] // head_kv
        
        # 投影Q
        debug(f"Q...")
        q = Tensor(shape=(d_seq, head_q*d_qk), dtype=dtype, device=self.device)
        Ops.linear(out=q, inp=hidden_states, weight=self.q_proj_weight, bias=self.q_proj_bias)
        q = q.view(d_seq, head_q, d_qk)
        # 位置编码
        Ops.rope(out=q, inp=q, pos_ids=tensor_pos, theta=self.rope_theta)


        # 只投影K新的部分
        debug(f"K...")
        k_new = self.k_cache.slice(0, self.kv_cached_rows, self.kv_cached_rows+d_seq)
        Ops.linear(out=k_new, inp=hidden_states, weight=self.k_proj_weight, bias=self.k_proj_bias)
        k_new = k_new.view(d_seq, head_kv, d_qk)
        # 位置编码
        Ops.rope(out=k_new, inp=k_new, pos_ids=tensor_pos, theta=self.rope_theta)

        # 只投影V新的部分
        debug(f"V...")
        v_new = self.v_cache.slice(0, self.kv_cached_rows, self.kv_cached_rows+d_seq)
        Ops.linear(out=v_new, inp=hidden_states, weight=self.v_proj_weight, bias=self.v_proj_bias)
        v_new = v_new.view(d_seq, head_kv, d_v)

        # 更新KV Cache
        self.kv_cached_rows += d_seq

        # 从KV Cache中获取完整的K和V
        k = self.k_cache.slice(0, 0, self.kv_cached_rows)
        k = k.view(self.kv_cached_rows, head_kv, d_qk)
        v = self.v_cache.slice(0, 0, self.kv_cached_rows)
        v = v.view(self.kv_cached_rows, head_kv, d_v)

        # self-attention
        debug(f"scores...")
        scores = Tensor(shape=(d_seq, head_q, d_v), dtype=dtype, device=self.device)
        Ops.self_attention(attn_val=scores, q=q, k=k, v=v, scale=1.0 / np.sqrt(d_qk))

        scores = scores.view(d_seq, head_q*d_v)

        # 输出投影
        debug(f"out...")
        out = Tensor(shape=(d_seq, d_emb), dtype=dtype, device=self.device)
        Ops.linear(out=out, inp=scores, weight=self.o_proj_weight)

        return out



    def _mlp(self, hidden_states: Tensor) -> Tensor:
        d_seq, d_emb = hidden_states.shape() # type: ignore
        d_inter = self.gate_proj_weight.shape()[0] # type: ignore

        gate = Tensor(shape=(d_seq, d_inter), dtype=hidden_states.dtype(), device=self.device)
        Ops.linear(out=gate, inp=hidden_states, weight=self.gate_proj_weight)
        up = Tensor(shape=(d_seq, d_inter), dtype=hidden_states.dtype(), device=self.device)
        Ops.linear(out=up, inp=hidden_states, weight=self.up_proj_weight)

        swiglu_out = Tensor(shape=(d_seq, d_inter), dtype=hidden_states.dtype(), device=self.device)
        Ops.swiglu(out=swiglu_out, gate=gate, up=up)

        proj_out = Tensor(shape=(d_seq, d_emb), dtype=hidden_states.dtype(), device=self.device)
        Ops.linear(out=proj_out, inp=swiglu_out, weight=self.down_proj_weight)

        return proj_out
