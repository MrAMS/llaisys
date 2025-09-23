#pragma once

#include <cstdint>
#include "../../tensor/tensor.hpp"

namespace llaisys::sampler {
class Sampler{
public:
    virtual int64_t sample(llaisys::tensor_t logits);
    virtual ~Sampler();
};
}