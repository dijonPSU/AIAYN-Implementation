#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <iostream>
#include <stdexcept>


class Tensor {
private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
    size_t size_;

    size_t computeIndex(const std::vector<size_t>& indicies);
public:
    Tensor() : size_(0) {};
    Tensor(const std::vector<size_t>& shape);

    float& Tensor::operator()(const std::vector<size_t>& indicies);
    float Tensor::operator()(const std::vector<size_t>& indices) const;

    std::vector<size_t> shape() const { return shape_; };
    size_t size() const { return size_; };

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(float scalar) const;

    void print() const;
};

#endif