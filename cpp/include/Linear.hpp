#ifndef LINEAR_HPP
#define LINEAR_HPP


#include <optional>
#include <random>
#include <string>


#include "Tensor.hpp"


class Linear {
public:
    Linear(size_t inFeatures, size_t outFeatures, bool useBias = true, uint32_t seed = 1337);

    // forward / backward
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& gradOutput);

    void zeroGrad();

    size_t inFeatures() const { return inFeatures_ ; }
    size_t outFeatures() const { return outFeatures_ ; }
    bool usesBias() const { return useBias_; }

    // params
    const Tensor& weight() const { return weight_; }
    const std::optional<Tensor>& bias() { return bias_; }

    //  gradient funcs
    const Tensor& gradWeight() const { return gradWeight_; }
    const std::optional<Tensor>& gradBias() const { return gradBias_; }

    // mutable access for optimizer
    Tensor& weightMutable() { return weight_; }
    std::optional<Tensor>& biasMutable() { return bias_; }

    Tensor& gradWeightMutable() { return gradWeight_; }
    std::optional<Tensor>& gradBiasMutable() { return gradBias_; }

private: 
    size_t inFeatures_;
    size_t outFeatures_;
    bool useBias_;

    Tensor weight_;                    
    std::optional<Tensor> bias_;

    Tensor gradWeight_;
    std::optional<Tensor> gradBias_;

    // cache for backward
    std::optional<Tensor> cachedInput_;
    std::vector<size_t> cachedInputShape_;

    std::mt19937 rng_;

    void initXavierUniform();
    static void requireAtLeast1D(const Tensor& t, const std::string& name);
    static size_t productOfLeadingDims(const std::vector<size_t>& shape);
};


#endif