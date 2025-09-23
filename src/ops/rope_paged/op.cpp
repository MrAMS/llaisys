#include "op.hpp"

#include "cpu/rope_paged_cpu.hpp"

namespace llaisys::ops {
void rope_paged(std::byte** out_map, const std::byte* const* in_map, tensor_t pos_ids, size_t d_seq, size_t d_head, size_t d, llaisysDataType_t dtype, float theta) {
    // CHECK_SAME_DEVICE(out, in, pos_ids);
    // CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    // CHECK_SAME_SHAPE(out->shape(), in->shape());

    // CHECK_ARGUMENT(out->ndim()==3, "rope_paged: out must be a 3D tensor");
    // CHECK_ARGUMENT(in->ndim()==3, "rope_paged: in must be a 3D tensor");
    CHECK_ARGUMENT(pos_ids->ndim()==1, "rope_paged: pos_ids must be a 1D tensor");

    CHECK_ARGUMENT(pos_ids->dtype()==llaisysDataType_t::LLAISYS_DTYPE_I64, "rope_paged: pos_ids tensor dtype must be int64");
    // CHECK_ARGUMENT(out->shape()==in->shape(), "rope_paged: in and out tensor shape mismatch");
    // CHECK_ARGUMENT(out->shape()[0]==pos_ids->shape()[0], "rope_paged: pos_ids tensor shape mismatch");

    // CHECK_ARGUMENT(out->isContiguous(), "rope_paged: out tensor must be contiguous");
    // CHECK_ARGUMENT(in->isContiguous(), "rope_paged: in tensor must be contiguous");
    CHECK_ARGUMENT(pos_ids->isContiguous(), "rope_paged: pos_ids tensor must be contiguous");

    const auto device_type = pos_ids->deviceType();
    llaisys::core::context().setDevice(device_type, pos_ids->deviceId());

    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope_paged(out_map,
            in_map,
            pos_ids->data(),
            d_seq, d_head, d,
            theta,
            dtype
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
