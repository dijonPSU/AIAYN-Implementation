#include "Linear.hpp"

#include <cmath>
#include <stdexcept>


void Linear::requireAtLeast1D(const Tensor& t, const std::string& name) {
    if (t.shape().empty()) {
        throw std::invalid_argument(name + " must have a rank > 1");
    }
}

size_t Linear::productOfLeadingDims(const std::vector<size_t>& shape) {
    if (shape.size() == 1) {
        return 1;
    }

    size_t product = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        product *= shape[i];
    }

    return product;
}

Linear::Linear(size_t inFeatures, size_t outFeatures, bool useBias, uint32_t seed)
    : inFeatures_(inFeatures), outFeatures_(outFeatures), useBias_(useBias)
    , weight_({inFeatures, outFeatures}), gradWeight_({inFeatures, outFeatures})
    , rng_(seed)
{
    if (inFeatures_ == 0 || outFeatures_ == 0) {
        throw std::invalid_argument("Linear inFeatures and outFeatures must be greater than 0");
    }

    if (useBias_) {
        bias_ = Tensor({outFeatures_});
        gradBias_ = Tensor({outFeatures_});
    }

    initXavierUniform();
    zeroGrad();
}

void Linear::initXavierUniform() {
    static constexpr float XAVIER_SCALE = 6.0f;

    // glorot uniform
    const float fanIn = static_cast<float>(inFeatures_);
    const float fanOut = static_cast<float>(outFeatures_);
    const float limit = std::sqrt(XAVIER_SCALE / (fanIn + fanOut));

    std::uniform_real_distribution<float> dist(-limit, limit);

    for (size_t i = 0; i < inFeatures_; ++i) {
        for (size_t o = 0; o < outFeatures_; ++o) {
            weight_(i, o) = dist(rng_);
        }
    }

    if (useBias_) {
        for (size_t o = 0; o < outFeatures_; ++o) {
            (*bias_)({o}) = 0.0f;
        }
    }
}

void Linear::zeroGrad() {
    const float zeroFloat = 0.0f;

    for (size_t i = 0; i < inFeatures_; ++i) {
        for (size_t o = 0; o < outFeatures_; ++o) {
            gradWeight_(i, o) = zeroFloat;
        }
    }

    if (useBias_) {
        for (size_t o = 0; o < outFeatures_; ++o) {
            (*gradBias_)({o}) = zeroFloat;
        }
    }
}


Tensor Linear::forward(const Tensor& input) {
    requireAtLeast1D(input, "Forward: input");

    const std::vector<size_t>& inShape = input.shape();
    const size_t rank = inShape.size();
    const size_t inDimension = inShape.back();

    // checks
    if (inDimension != inFeatures_) {
        throw std::runtime_error("Forward: Last dimension must equal inFeatures");
    }

    // flatten all leading dims into one batch dimension
    const size_t batchFlat = productOfLeadingDims(inShape);

    // copy input into a 2D flat tensor
    Tensor flatInput({batchFlat, inFeatures_});

    const float* inputPtr = input.data();
    float* flatInputPtr = flatInput.data();

    const size_t rowSize = inFeatures_;
    for (size_t row = 0; row < batchFlat; ++row) {
        const size_t srcOffset = row * rowSize;
        const size_t dstOffset = row * rowSize;

        for (size_t i = 0; i < rowSize; ++i) {
            flatInputPtr[dstOffset + i] = inputPtr[srcOffset + i];
        }
    }

    // compute flat output
    Tensor flatOutput({batchFlat, outFeatures_});

    // matrix multiplication: flatOutput[row, o] * weight[i, o] + bias[o]
    for (size_t row = 0; row < batchFlat; ++row) {
        for (size_t o = 0; o < outFeatures_; ++o) {
            float sum = 0.0f;
            for (size_t i = 0; i < inFeatures_; ++i) {
                sum += flatInput(row, i) * weight_(i, o); // matrix is transposed

            }
            if (useBias_) {
                sum += (*bias_)({o});
            }
            flatOutput(row, o) = sum;
        }
    }

    // build output shape
    std::vector<size_t> outShape = inShape;
    outShape.back() = outFeatures_;
    Tensor output(outShape);

    // copy data to output storage
    const float* flatOutputPtr = flatOutput.data();
    float* outputPtr = output.data();

    const size_t outputRowSize = outFeatures_;
    for (size_t row = 0; row < batchFlat; ++row) {
        const size_t srcOffset = row * outputRowSize;
        const size_t dstOffset = row * outputRowSize;
        for (size_t o = 0; o < outputRowSize; ++o) {
            outputPtr[dstOffset + o] =  flatOutputPtr[srcOffset + o];
        }
    }

    // cache output for backward
    cachedInput_ = input;
    cachedInputShape_ = inShape;

    return output;
}