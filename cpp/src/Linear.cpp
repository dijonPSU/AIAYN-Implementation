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