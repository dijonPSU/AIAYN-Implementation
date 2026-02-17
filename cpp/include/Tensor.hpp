#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cstddef>


class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const std::vector<size_t>& shape);

    float& operator()(const std::vector<size_t>& indices);
    const float& operator()(const std::vector<size_t>& indices) const;

    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }

    float* data() {return data_.data(); }
    const float* data() const {return data_.data(); }

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(float scalar) const;

    float& operator()(size_t i, size_t j);
    const float operator()(size_t i, size_t j) const;

    void print() const;

private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
    size_t size_ = 0;

    size_t computeIndex(const std::vector<size_t>& indices) const;
};

#endif