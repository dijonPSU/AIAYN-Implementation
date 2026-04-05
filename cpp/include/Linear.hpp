#ifndef LINEAR_HPP
#define LINEAR_HPP

#include <memory>
#include <optional>
#include <random>
#include <string_view>

#include "Tensor.hpp"

class Linear {
public:
  Linear(size_t inFeatures, size_t outFeatures, bool useBias = true,
         uint32_t seed = 1337);

  // forward / backward
  Tensor forward(const Tensor &input);
  Tensor backward(const Tensor &gradOutput);
  void zeroGrad() noexcept;

  constexpr size_t inFeatures() const { return inFeatures_; }
  constexpr size_t outFeatures() const { return outFeatures_; }
  constexpr bool usesBias() const { return useBias_; }

  // params
  const Tensor &weight() const noexcept { return weight_; }
  const std::optional<Tensor> &bias() const noexcept { return bias_; }

  //  gradient accessors
  const Tensor &gradWeight() const noexcept { return gradWeight_; }
  const std::optional<Tensor> &gradBias() const noexcept { return gradBias_; }

  // mutable access for optimizer
  Tensor &weightMutable() noexcept { return weight_; }
  std::optional<Tensor> &biasMutable() noexcept { return bias_; }

  Tensor &gradWeightMutable() noexcept { return gradWeight_; }
  std::optional<Tensor> &gradBiasMutable() noexcept { return gradBias_; }

private:
  size_t inFeatures_;
  size_t outFeatures_;
  bool useBias_;

  Tensor weight_;
  std::optional<Tensor> bias_;

  Tensor gradWeight_;
  std::optional<Tensor> gradBias_;

  // cache for backward
  std::unique_ptr<Tensor> cachedInput_;

  void initXavierUniform(std::mt19937 &rng);
  static void requireAtLeast1D(const Tensor &t, std::string_view name);
  static size_t productOfLeadingDims(const std::vector<size_t> &shape);
};

#endif