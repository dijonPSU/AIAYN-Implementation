#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstddef>
#include <vector>

class Tensor {
public:
  // constructors
  Tensor() = default;
  explicit Tensor(std::vector<size_t> shape);

  // accessors
  const std::vector<size_t> &shape() const noexcept { return shape_; }
  size_t size() const noexcept { return data_.size(); }
  float *data() noexcept { return data_.data(); }
  const float *data() const noexcept { return data_.data(); }

  // element accessors
  float &operator()(size_t i, size_t j);
  const float &operator()(size_t i, size_t j) const;
  float &operator()(const std::vector<size_t> &indices);
  const float &operator()(const std::vector<size_t> &indices) const;

  // operators
  Tensor operator+(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator*(float scalar) const;

  // output
  void print() const;

private:
  std::vector<float> data_;
  std::vector<size_t> shape_;

  size_t computeIndex(const std::vector<size_t> &indices) const;
};

#endif