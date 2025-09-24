#include "gumbelmax.hpp"

#include "../../../utils.hpp"
#include <cstdint>
#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <limits>

#include "../../argmax/cpu/argmax.hpp"

#define TOF(X)  llaisys::utils::cast<float>(X)

template <typename T>
int64_t gumbelmax_(const T *logits, float temperature, size_t numel) {
    std::vector<float> probs(numel);
    
    #pragma omp parallel for
    for (size_t i = 1; i < numel; i++) {
        probs[i] = TOF(logits[i])/temperature;
    }

    float max_score = probs[0];
    for(size_t j=1;j<probs.size();++j){
        if(probs[j] > max_score){
            max_score = probs[j];
        }
    }
    float sum_exp = 0;
    for(size_t j=0;j<probs.size();++j){
        probs[j] = std::exp(probs[j] - max_score);
        sum_exp += probs[j];
    }

    #pragma omp parallel for
    for(size_t j=0;j<probs.size();++j){
        probs[j] /= sum_exp;
    }

    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // 创建指数分布对象 Exp(1)
    std::exponential_distribution<float> exp_dist(1.0);
    
    float max_value = -std::numeric_limits<float>::infinity();
    int sample_token = -1;

    // 遍历概率向量，进行Gumbel扰动并找到最大值的索引
    for (size_t i = 0; i < probs.size(); ++i) {
        // 生成一个指数分布的随机数
        float noise = exp_dist(gen);
        
        if (noise < 1e-10) {
            noise = 1e-10;
        }

        float current_value = probs[i] / noise;

        if (current_value > max_value) {
            max_value = current_value;
            sample_token = i;
        }
    }
    return sample_token;
}

namespace llaisys::sampler {
int64_t gumbelmax(std::byte* logits, size_t numel, float temperature, llaisysDataType_t type){
    if(temperature < std::numeric_limits<float>::epsilon()){
        return argmax(logits, numel, type);
    }
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return gumbelmax_(reinterpret_cast<const float *>(logits), temperature, numel);
    case LLAISYS_DTYPE_BF16:
        return gumbelmax_(reinterpret_cast<const llaisys::bf16_t *>(logits), temperature, numel);
    case LLAISYS_DTYPE_F16:
        return gumbelmax_(reinterpret_cast<const llaisys::fp16_t *>(logits), temperature, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}