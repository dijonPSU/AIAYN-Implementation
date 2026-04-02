#include "Tensor.hpp"
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
  if (shape_.empty()) {
    throw std::invalid_argument("Shape cannot be empty");
  }
  constexpr float initial_value = 0.0f;
  constexpr size_t start_multiplier = 1;
  const size_t size =
      std::accumulate(shape_.begin(), shape_.end(), start_multiplier,
                      std::multiplies<size_t>());
  data_.assign(size, initial_value);
}

// compute flat index
size_t Tensor::computeIndex(const std::vector<size_t> &indices) const {
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
float &Tensor::operator()(const std::vector<size_t> &indices) {
  const size_t idx = computeIndex(indices);
  return data_[idx];
}

// returns element value at given indices
const float &Tensor::operator()(const std::vector<size_t> &indices) const {
  const size_t idx = computeIndex(indices);
  return data_[idx];
}

// returns result of element-wise addition with another tensor of the same shape
Tensor Tensor::operator+(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shape mismatch in operator+");
  }

  Tensor result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = data_[i] + other.data_[i];
  }

  return result;
}

// returns result of element-wise subtraction with another tensor of the same
// shape
Tensor Tensor::operator-(const Tensor &other) const {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shape mismatch in operator-");
  }

  Tensor result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = data_[i] - other.data_[i];
  }

  return result;
}

// returns scaled tensor
Tensor Tensor::operator*(float scalar) const {
  Tensor result(shape_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = scalar * data_[i];
  }

  return result;
}

// prints the tensor shape and all data values
void Tensor::print() const {
  std::cout << "Tensor(shape=[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    std::cout << shape_[i];
    if (i + 1 < shape_.size())
      std::cout << ", ";
  }
  std::cout << "], size=" << data_.size() << ")\n";

  std::cout << "data=[";
  for (size_t i = 0; i < data_.size(); ++i) {
    std::cout << data_[i];
    if (i + 1 < data_.size())
      std::cout << ", ";
  }
  std::cout << "]\n";
}

// 2D accessor
float &Tensor::operator()(size_t i, size_t j) {
  constexpr size_t required_size = 2;

  if (shape_.size() != required_size) {
    throw std::invalid_argument("Operator()(i, j) requires 2D tensor");
  }

  if (i >= shape_[0] || j >= shape_[1]) {
    throw std::out_of_range("Index out of range");
  }

  return data_[i * shape_[1] + j];
}

// 2D accessor (const version)
const float &Tensor::operator()(size_t i, size_t j) const {
  constexpr size_t required_size = 2;

  if (shape_.size() != required_size) {
    throw std::invalid_argument("Operator()(i, j) requires 2D tensor");
  }

  if (i >= shape_[0] || j >= shape_[1]) {
    throw std::out_of_range("Index out of range");
  }

  return data_[i * shape_[1] + j];
}