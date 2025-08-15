#include "op.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if(bias) CHECK_SAME_DEVICE(out, bias);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if(bias) CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    
    CHECK_ARGUMENT(out->ndim()==2, "Linear: out must be a 2D tensor");
    CHECK_ARGUMENT(in->ndim()==2, "Linear: out must be a 2D tensor");
    CHECK_ARGUMENT(weight->ndim()==2, "Linear: out must be a 2D tensor");
    if(bias) CHECK_ARGUMENT(bias->ndim()==1, "Linear: out must be a 1D tensor");

    CHECK_ARGUMENT(out->shape()[0]==in->shape()[0] && out->shape()[1]==weight->shape()[0], "Linear: out tensor shape mismatched");
    if(bias) CHECK_ARGUMENT(out->shape()[1] == bias->shape()[0], "Linear: out tensor shape must match bias tensor");

    CHECK_ARGUMENT(out->isContiguous(), "Linear: out must be a contiguous tensor");
    CHECK_ARGUMENT(in->isContiguous(), "Linear: out must be a contiguous tensor");
    CHECK_ARGUMENT(weight->isContiguous(), "Linear: weight must be a contiguous tensor");
    if(bias) CHECK_ARGUMENT(bias->isContiguous(), "Linear: bias must be a contiguous tensor");

    const auto device_type = weight->deviceType();
    llaisys::core::context().setDevice(device_type, weight->deviceId());

    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(),
            in->data(),
            weight->data(),
            bias==nullptr?nullptr:bias->data(),
            in->shape()[0], in->shape()[1], weight->shape()[0],
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
