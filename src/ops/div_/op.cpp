#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/div__cpu.hpp"

namespace llaisys::ops {
void div_(tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(a, b);
    CHECK_SAME_DTYPE(a->dtype(), b->dtype());
    CHECK_ARGUMENT(b->ndim()==1 && b->numel()==1, "DIV_: tensor b must be a scalar");
    ASSERT(a->isContiguous() && b->isContiguous(), "DIV_: all tensors must be contiguous.");

    llaisys::core::context().setDevice(a->deviceType(), a->deviceId());

    switch (a->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::div_(a->data(), b->data(), a->dtype(), a->numel());
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
