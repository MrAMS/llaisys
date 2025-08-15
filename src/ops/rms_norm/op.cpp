#include "op.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    CHECK_ARGUMENT(in->ndim()==2, "RMS_NORM: in tensor must be 2D tensor for now");
    CHECK_ARGUMENT(weight->ndim()==1, "RMS_NORM: weight tensor must be 2D tensor for now");

    // weight:1*n, in:m*n, out:m*n 
    CHECK_ARGUMENT(in->shape()[1]==weight->shape()[0], "RMS_NORM: in and weight tensor shape mismatch");
    CHECK_ARGUMENT(out->shape()==in->shape(), "RMS_NORM: out tensor shape mismatch");

    CHECK_ARGUMENT(in->isContiguous(), "RMS_NORM: in tensor must be contiguous");
    CHECK_ARGUMENT(out->isContiguous(), "RMS_NORM: out tensor must be contiguous");
    CHECK_ARGUMENT(weight->isContiguous(), "RMS_NORM: weight tensor must be contiguous");

    const auto device_type = out->deviceType();
    llaisys::core::context().setDevice(device_type, out->deviceId());

    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(),
            in->data(),
            weight->data(),
            in->shape()[0],
            in->shape()[1],
            eps,
            out->dtype());
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
