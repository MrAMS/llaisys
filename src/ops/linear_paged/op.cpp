#include "op.hpp"

#include "cpu/linear_paged_cpu.hpp"
#include <cstdio>

namespace llaisys::ops {
void linear_paged(std::byte** out_map, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(in, weight);
    if(!bias->isNone()) CHECK_SAME_DEVICE(in, bias);
    CHECK_SAME_DTYPE(in->dtype(), weight->dtype());
    // if(!bias->isNone()) CHECK_SAME_DTYPE(out->dtype(), bias->dtype());

    // CHECK_ARGUMENT(out->ndim()==2, "linear_paged: out must be a 2D tensor");
    CHECK_ARGUMENT(in->ndim()==2, "linear_paged: in must be a 2D tensor");
    CHECK_ARGUMENT(weight->ndim()==2, "linear_paged: weight must be a 2D tensor");
    if(!bias->isNone()) CHECK_ARGUMENT(bias->ndim()==1, "linear_paged: bias must be a 1D tensor");

    // 注意是in矩阵乘weight的转置
    // CHECK_ARGUMENT(out->shape()[0]==in->shape()[0] && out->shape()[1]==weight->shape()[0], "linear_paged: out tensor shape mismatched");
    CHECK_ARGUMENT(in->shape()[1]==weight->shape()[1], "linear_paged: in and weight tensor shape mismatch");
    // if(!bias->isNone()) CHECK_ARGUMENT(out->shape()[1] == bias->shape()[0], "linear_paged: out tensor shape must match bias tensor");

    // CHECK_ARGUMENT(out->isContiguous(), "linear_paged: out must be a contiguous tensor");
    CHECK_ARGUMENT(in->isContiguous(), "linear_paged: out must be a contiguous tensor");
    CHECK_ARGUMENT(weight->isContiguous(), "linear_paged: weight must be a contiguous tensor");
    if(!bias->isNone()) CHECK_ARGUMENT(bias->isContiguous(), "linear_paged: bias must be a contiguous tensor");

    const auto device_type = weight->deviceType();
    llaisys::core::context().setDevice(device_type, weight->deviceId());

    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear_paged(out_map,
            in->data(),
            weight->data(),
            !bias->isNone()?bias->data():nullptr,
            in->shape()[0], in->shape()[1], weight->shape()[0],
            in->dtype());
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
