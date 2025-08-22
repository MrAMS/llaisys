#include "op.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    const auto d_seq = attn_val->shape()[0];
    const auto d_head = attn_val->shape()[1];
    const auto d_v = attn_val->shape()[2];
    const auto d_qk = q->shape()[2];
    const auto d_kvhead = k->shape()[1];
    const auto d_tot = k->shape()[0];

    CHECK_ARGUMENT(q->ndim()==3, "SELF_ATTEN: q must be a 3D tensor");
    CHECK_ARGUMENT(k->ndim()==3, "SELF_ATTEN: k must be a 3D tensor");
    CHECK_ARGUMENT(v->ndim()==3, "SELF_ATTEN: v must be a 3D tensor");
    CHECK_ARGUMENT(attn_val->ndim()==3, "SELF_ATTEN: attn_val must be a 3D tensor");

    CHECK_ARGUMENT(q->shape()[0]==d_seq&&q->shape()[1]==d_head, "SELF_ATTEN: q tensor shape mismatch");
    CHECK_ARGUMENT(k->shape()[0]==d_tot&&k->shape()[1]==d_kvhead&&k->shape()[2]==d_qk, "SELF_ATTEN: k tensor shape mismatch");
    CHECK_ARGUMENT(v->shape()[0]==d_tot&&v->shape()[1]==d_kvhead&&v->shape()[2]==d_v, "SELF_ATTEN: v tensor shape mismatch");

    CHECK_ARGUMENT(attn_val->isContiguous(), "SELF_ATTEN: attn_val tensor must be contiguous");
    CHECK_ARGUMENT(q->isContiguous(), "SELF_ATTEN: q tensor must be contiguous");
    CHECK_ARGUMENT(k->isContiguous(), "SELF_ATTEN: k tensor must be contiguous");
    CHECK_ARGUMENT(v->isContiguous(), "SELF_ATTEN: v tensor must be contiguous");

    const auto device_type = attn_val->deviceType();
    llaisys::core::context().setDevice(device_type, attn_val->deviceId());

    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            scale,
            d_seq, d_head, d_v, d_qk, d_kvhead, d_tot,
            attn_val->dtype()
        );
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}
} // namespace llaisys::ops
