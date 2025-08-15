#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#include "llaisys.h"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(max_idx->dtype() == llaisysDataType_t::LLAISYS_DTYPE_I64, "Argmax: max_idx must be of type int64.");

    // TODO
    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax: vals must be a 1D tensor for now.");
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_idx->numel()==1, "Argmax: max_idx must be a 1D tensor for now.");
    CHECK_ARGUMENT(max_val->ndim() == 1 && max_val->numel()==1, "Argmax: max_val must be a 1D tensor for now.");
    CHECK_ARGUMENT(vals->isContiguous(), "Argmax: vals must be a contiguous tensor.");

    const auto device_type = vals->deviceType();
    llaisys::core::context().setDevice(device_type, vals->deviceId());

    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->numel(), vals->dtype());
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
