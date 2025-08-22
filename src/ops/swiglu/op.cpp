#include "op.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());

    CHECK_ARGUMENT(out->ndim()==2, "SWIGLU: out must be a 2D tensor");
    CHECK_ARGUMENT(gate->ndim()==2, "SWIGLU: gate must be a 2D tensor");
    CHECK_ARGUMENT(up->ndim()==2, "SWIGLU: up must be a 2D tensor");

    CHECK_ARGUMENT(out->isContiguous(), "SWIGLU: out tensor must be contiguous");
    CHECK_ARGUMENT(gate->isContiguous(), "SWIGLU: gate tensor must be contiguous");
    CHECK_ARGUMENT(up->isContiguous(), "SWIGLU: up tensor must be contiguous");

    const auto device_type = out->deviceType();
    llaisys::core::context().setDevice(device_type, out->deviceId());

    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(),
            up->data(),
            gate->data(),
            out->numel(),
            out->dtype()
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
