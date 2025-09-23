#pragma once

#include "../base/base.hpp"

namespace llaisys::sampler {
class SamplerArgmax : Sampler{
public:
    int64_t sample(tensor_t logits) override;
};

}
