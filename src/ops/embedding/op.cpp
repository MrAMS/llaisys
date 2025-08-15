#include "op.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->dtype() == llaisysDataType_t::LLAISYS_DTYPE_I64, "Embedding: index must be of type int64.");
    CHECK_ARGUMENT(index->isContiguous(), "Embedding: index must be a contiguous tensor.");

    CHECK_ARGUMENT(index->ndim() == 1, "Embedding: index must be a 1D tensor for now.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight must be a 2D tensor for now.");
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding: out must be a 2D tensor for now.");

    const auto device_type = weight->deviceType();
    llaisys::core::context().setDevice(device_type, weight->deviceId());

    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), index->ndim(), weight->data(), weight->shape(), weight->strides(), weight->dtype());
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
