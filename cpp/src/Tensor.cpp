#include "Tensor.hpp"
#include <numeric>
#include <functional>
#include <iostream>
#include <stdexcept>



Tensor::Tensor(const std::vector<size_t>& shape) : shape_(shape), size_(0) {
    if (shape_.empty()) {
        throw std::invalid_argument("Shape cannot be empty");
    }

    const size_t start_multiplier = 1;
    size_ = std::accumulate(shape_.begin(), shape_.end(), start_multiplier, std::multiplies<size_t>());
    data_.assign(size_, 0.0f);
}

// compute flat index
size_t Tensor::computeIndex(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Dimension mismatch");
    }

    size_t index = 0, stride = 1;

    // start from the last dimension to first
    for (size_t i = shape_.size(); i-- > 0;) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of range");
        }

        index += indices[i] * stride;
        stride *= shape_[i];
    }

    return index;
}

// returns mutable reference to element at the given indices
float& Tensor::operator()(const std::vector<size_t>& indices) {
    size_t idx = computeIndex(indices);
    return data_[idx];
}

// returns element value at given indices
const float& Tensor::operator()(const std::vector<size_t>& indices) const {
    size_t idx = computeIndex(indices);
    return data_[idx];
}

// returns result of element-wise addition with another tensor of the same shape
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch in operator+");
    }

    Tensor result(shape_);
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }

    return result;
}

// returns result of element-wise subtraction with another tensor of the same shape
Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shape mismatch in operator-");
    }

    Tensor result(shape_);
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }

    return result;
}

// returns scaled tensor
Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = scalar * data_[i];
    }

    return result;
}

// prints the tensor shape and all data vlues
void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) std::cout << ", ";
    }
    std::cout << "], size=" << size_ << ")\n";

    std::cout << "data=[";
    for (size_t i = 0; i < size_; ++i) {
        std::cout << data_[i];
        if (i + 1 < size_) std::cout << ", ";
    }
    std::cout << "]\n";
}

// returns mutable size of tensor
float& Tensor::operator()(size_t i, size_t j) {
    int required_size = 2;

    if (shape_.size() != required_size) {
        throw std::invalid_argument("Operator()(i, j) requires 2D tensor");
    }

    if (i >= shape_[0] || j>= shape_[1]) {
        throw std::out_of_range("Index out of range");
    }

    return data_[i * shape_[1] + j];
}

// returns inmutable size of tensor
const float Tensor::operator()(size_t i, size_t j) const {
    int required_size = 2;

    if (shape_.size() != required_size) {
        throw std::invalid_argument("Operator()(i, j) requires 2D tensor");
    }


    if (i >= shape_[0] || j>= shape_[1]) {
        throw std::out_of_range("Index out of range");
    }

    return data_[i * shape_[1] + j];
}