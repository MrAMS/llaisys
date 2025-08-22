#include "op.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    CHECK_ARGUMENT(out->ndim()==3, "ROPE: out must be a 3D tensor");
    CHECK_ARGUMENT(in->ndim()==3, "ROPE: in must be a 3D tensor");
    CHECK_ARGUMENT(pos_ids->ndim()==1, "ROPE: pos_ids must be a 1D tensor");

    CHECK_ARGUMENT(pos_ids->dtype()==llaisysDataType_t::LLAISYS_DTYPE_I64, "ROPE: pos_ids tensor dtype must be int64");
    CHECK_ARGUMENT(out->shape()==in->shape(), "ROPE: in and out tensor shape mismatch");
    CHECK_ARGUMENT(out->shape()[0]==pos_ids->shape()[0], "ROPE: pos_ids tensor shape mismatch");

    CHECK_ARGUMENT(out->isContiguous(), "ROPE: out tensor must be contiguous");
    CHECK_ARGUMENT(in->isContiguous(), "ROPE: in tensor must be contiguous");
    CHECK_ARGUMENT(pos_ids->isContiguous(), "ROPE: pos_ids tensor must be contiguous");

    const auto device_type = out->deviceType();
    llaisys::core::context().setDevice(device_type, out->deviceId());

    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(),
            in->data(),
            pos_ids->data(),
            in->shape()[0],
            in->shape()[1],
            in->shape()[2],
            theta,
            in->dtype()
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
