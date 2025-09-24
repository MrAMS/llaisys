#include "sampler.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/gumbelmax.hpp"

namespace llaisys::sampler {
SamplerGumbelmax::SamplerGumbelmax(float temperature):_temperature(temperature){

}
int64_t SamplerGumbelmax::sample(tensor_t logits){
    CHECK_ARGUMENT(logits->ndim()==1, "sampler.argmax: tensor logits must be a 1D tensor");
    ASSERT(logits->isContiguous(), "sampler.argmax: all tensors must be contiguous.");

    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());

    switch (logits->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return gumbelmax(logits->data(), logits->numel(), _temperature, logits->dtype());
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
