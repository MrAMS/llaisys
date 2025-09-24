#pragma once

#include "../base/base.hpp"

namespace llaisys::sampler {
class SamplerGumbelmax : Sampler{
public:
    SamplerGumbelmax(float temperature);
    int64_t sample(tensor_t logits) override;
private:
    float _temperature;
};

}
