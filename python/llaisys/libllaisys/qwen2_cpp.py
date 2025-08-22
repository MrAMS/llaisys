import ctypes
from . import DeviceType, llaisysDataType_t, llaisysTensor_t, llaisysDeviceType_t

class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", ctypes.c_size_t),
        ("hs", ctypes.c_size_t),
        ("nh", ctypes.c_size_t),
        ("nkvh", ctypes.c_size_t),
        ("dh", ctypes.c_size_t),
        ("di", ctypes.c_size_t),
        ("maxseq", ctypes.c_size_t),
        ("voc", ctypes.c_size_t),
        ("epsilon", ctypes.c_float),
        ("theta", ctypes.c_float),
        ("end_token", ctypes.c_int64)
    ]

class LlaisysQwen2Weights(ctypes.Structure):
    _fields_ = [
        ("in_embed",      llaisysTensor_t),
        ("out_embed",     llaisysTensor_t),
        ("out_norm_w",    llaisysTensor_t),

        ("attn_norm_w",   ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_w",      ctypes.POINTER(llaisysTensor_t)),
        ("attn_q_b",      ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_w",      ctypes.POINTER(llaisysTensor_t)),
        ("attn_k_b",      ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_w",      ctypes.POINTER(llaisysTensor_t)),
        ("attn_v_b",      ctypes.POINTER(llaisysTensor_t)),
        ("attn_o_w",      ctypes.POINTER(llaisysTensor_t)),

        ("mlp_norm_w",    ctypes.POINTER(llaisysTensor_t)),
        ("mlp_gate_w",    ctypes.POINTER(llaisysTensor_t)),
        ("mlp_up_w",      ctypes.POINTER(llaisysTensor_t)),
        ("mlp_down_w",    ctypes.POINTER(llaisysTensor_t)),
    ]

class LlaisysQwen2Model(ctypes.Structure):
    _fields_ = [
        ("meta", ctypes.POINTER(LlaisysQwen2Meta)),
        ("device", llaisysDeviceType_t),
        ("ndevice", ctypes.c_int),
        ("device_ids", ctypes.POINTER(ctypes.c_int)),
        ("weights", ctypes.POINTER(LlaisysQwen2Weights)),
        ("k_caches",    ctypes.POINTER(llaisysTensor_t)),
        ("v_caches",    ctypes.POINTER(llaisysTensor_t)),
        ("kv_cached_row", ctypes.POINTER(ctypes.c_size_t)),
    ]

def load_qwen2_cpp(lib):

    lib.llaisysQwen2ModelCreate.argtypes = [ctypes.POINTER(LlaisysQwen2Meta), llaisysDeviceType_t, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.llaisysQwen2ModelCreate.restype = ctypes.POINTER(LlaisysQwen2Model)

    lib.llaisysQwen2ModelDestroy.argtypes = [ctypes.POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [ctypes.POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelWeights.restype = ctypes.POINTER(LlaisysQwen2Weights)

    lib.llaisysQwen2ModelAllocKVCache.argtypes = [ctypes.POINTER(LlaisysQwen2Model), ctypes.c_size_t]
    lib.llaisysQwen2ModelAllocKVCache.restype = None

    # __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken);
    lib.llaisysQwen2ModelInfer.argtypes = [ctypes.POINTER(LlaisysQwen2Model), ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
    lib.llaisysQwen2ModelInfer.restype = ctypes.c_int64