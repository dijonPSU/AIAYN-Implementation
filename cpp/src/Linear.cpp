#include "Linear.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

void Linear::requireAtLeast1D(const Tensor &t, std::string_view name) {
  if (t.shape().empty()) {
    throw std::invalid_argument(std::string(name) + " must have a rank > 1");
  }
}

size_t Linear::productOfLeadingDims(const std::vector<size_t> &shape) {
  if (shape.size() == 1) {
    return 1;
  }

  size_t product = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) {
    product *= shape[i];
  }

  return product;
}

Linear::Linear(size_t inFeatures, size_t outFeatures, bool useBias,
               uint32_t seed)
    : inFeatures_(inFeatures), outFeatures_(outFeatures), useBias_(useBias),
      weight_({inFeatures, outFeatures}),
      gradWeight_({inFeatures, outFeatures}) {
  if (inFeatures_ == 0 || outFeatures_ == 0) {
    throw std::invalid_argument(
        "Linear inFeatures and outFeatures must be greater than 0");
  }

  if (useBias_) {
    bias_ = Tensor({outFeatures_});
    gradBias_ = Tensor({outFeatures_});
  }

  std::mt19937 rng(seed);
  initXavierUniform(rng);
  zeroGrad();
}

void Linear::initXavierUniform(std::mt19937 &rng) {
  static constexpr float XAVIER_SCALE = 6.0f;

  // glorot uniform
  const float fanIn = static_cast<float>(inFeatures_);
  const float fanOut = static_cast<float>(outFeatures_);
  const float limit = std::sqrt(XAVIER_SCALE / (fanIn + fanOut));

  std::uniform_real_distribution<float> dist(-limit, limit);

  for (size_t i = 0; i < inFeatures_; ++i) {
    for (size_t o = 0; o < outFeatures_; ++o) {
      weight_(i, o) = dist(rng);
    }
  }

  if (useBias_) {
    std::memset(bias_->data(), 0, outFeatures_ * sizeof(float));
  }
}

void Linear::zeroGrad() noexcept {
  std::memset(gradWeight_.data(), 0,
              inFeatures_ * outFeatures_ * sizeof(float));

  if (useBias_) {
    std::memset(gradBias_->data(), 0, outFeatures_ * sizeof(float));
  }
}

Tensor Linear::forward(const Tensor &input) {
  requireAtLeast1D(input, "Forward: input");

  const auto &inShape = input.shape();
  const size_t inDimension = inShape.back();

  // checks
  if (inDimension != inFeatures_) {
    throw std::runtime_error("Forward: Last dimension must equal inFeatures");
  }

  // flatten all leading dims into one batch dimension
  const size_t batch = productOfLeadingDims(inShape);

  // construct output
  std::vector<size_t> outShape = inShape;
  outShape.back() = outFeatures_;
  Tensor output(outShape);

  // read / write ptrs
  const float *inputPtr = input.data();
  const float *biasPtr = useBias_ ? bias_->data() : nullptr;
  const float *weightPtr = weight_.data();
  float *outputPtr = output.data();

  // matmul: output[b, o] = sum_i(input[b, i] * weight[i, o]) + bias[o]
  for (size_t b = 0; b < batch; ++b) {
    const size_t inputRowOffset = b * inFeatures_;
    const size_t outputRowOffset = b * outFeatures_;

    for (size_t o = 0; o < outFeatures_; ++o) {
      outputPtr[outputRowOffset + o] = biasPtr ? biasPtr[o] : 0.0f;
    }

    for (size_t i = 0; i < inFeatures_; ++i) {
      const float inputVal = inputPtr[inputRowOffset + i];
      const size_t weightRowOffset = i * outFeatures_;

      for (size_t o = 0; o < outFeatures_; ++o) {
        outputPtr[outputRowOffset + o] +=
            inputVal * weightPtr[weightRowOffset + o];
      }
    }
  }

  // cache flatten 2D input for backward
  Tensor flatInput({batch, inFeatures_});
  std::memcpy(flatInput.data(), input.data(),
              batch * inFeatures_ * sizeof(float));

  if (!cachedInput_) {
    cachedInput_ = std::make_unique<Tensor>(std::move(flatInput));
  } else {
    *cachedInput_ = std::move(flatInput);
  }

  return output;
}

Tensor Linear::backward(const Tensor &gradOutput) {
  requireAtLeast1D(gradOutput, "Backward: gradOutput");

  if (!cachedInput_) {
    throw std::runtime_error("Backward: Must call forward() before backward()");
  }

  const Tensor &input = *cachedInput_;
  const auto &inputShape = input.shape();
  const auto &gradOutputShape = gradOutput.shape();

  // cache Input is always flattened to 2D with shape [batch, inFeatures_]
  const size_t batch = inputShape[0];

  if (gradOutputShape.back() != outFeatures_) {
    throw std::invalid_argument(
        "Backward: last dimension must equal outFeatures");
  }

  const size_t gradOutputBatch = productOfLeadingDims(gradOutputShape);
  if (gradOutputBatch != batch) {
    throw std::invalid_argument(
        "Backward: cached input feature dimension mismatch");
  }

  const float *inputPtr = input.data();
  const float *gradOutputPtr = gradOutput.data();
  float *gradWeightPtr = gradWeight_.data();

  // gradWeight[i, o] += sum_b(input[b, i] * gradOutput[b, o])
  for (size_t b = 0; b < batch; ++b) {
    const size_t inputRowOffset = b * inFeatures_;
    const size_t gradOutputRowOffset = b * outFeatures_;

    for (size_t i = 0; i < inFeatures_; ++i) {
      const float inputVal = inputPtr[inputRowOffset + i];
      const size_t gradWeightRowOffset = i * outFeatures_;

      for (size_t o = 0; o < outFeatures_; ++o) {
        gradWeightPtr[gradWeightRowOffset + o] +=
            inputVal * gradOutputPtr[gradOutputRowOffset + o];
      }
    }
  }

  // gradBias[o] += sum_b(gradOutput[b, o])
  if (useBias_) {
    float *gradBiasPtr = gradBias_->data();

    for (size_t b = 0; b < batch; ++b) {
      const size_t gradOutputRowOffset = b * outFeatures_;

      for (size_t o = 0; o < outFeatures_; ++o) {
        gradBiasPtr[o] += gradOutputPtr[gradOutputRowOffset + o];
      }
    }
  }

  // gradInput[b, i] = sum_o(gradOutput[b, o] * weight[i, o])
  Tensor gradInput({batch, inFeatures_});
  const float *weightPtr = weight_.data();
  float *gradInputPtr = gradInput.data();

  for (size_t b = 0; b < batch; ++b) {
    const size_t gradInputRowOffset = b * inFeatures_;
    const size_t gradOutputRowOffset = b * outFeatures_;

    for (size_t i = 0; i < inFeatures_; ++i) {
      float sum = 0.0f;
      const size_t weightRowOffset = i * outFeatures_;

      for (size_t o = 0; o < outFeatures_; ++o) {
        sum += gradOutputPtr[gradOutputRowOffset + o] *
               weightPtr[weightRowOffset + o];
      }

      gradInputPtr[gradInputRowOffset + i] = sum;
    }
  }
  return gradInput;
}