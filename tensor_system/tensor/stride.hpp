#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <shape.hpp>

// Stores the memory step for each dimension of a tensor.
class Stride{
public:
    // Calculates row-major (C-style) strides from the tensor shape.
    explicit Stride (const Shape& shape){
        size_t n=shape.ndim();
        strides_.resize(n);

        if (n==0) return ; // scalar

        strides_[n-1]=1;
        for (size_t i=n-1;i-->0;){
            strides_[i]=strides_[i + 1] * shape[i + 1];
        }
    }

    // Creates a Stride object from caller-provided stride values.
    explicit Stride(std::vector<size_t> explicit_strides)
        : strides_(std::move(explicit_strides)) {}
    
    // Returns the number of dimensions represented by these strides.
    size_t ndim() const {
        return strides_.size();
    }

    // Returns the stride for one dimension, with bounds checking.
    size_t operator[](size_t i) const {
        if (i >= strides_.size()) {
            throw std::out_of_range("Stride index out of range");
        }
        return strides_[i];
    }

    // Converts multidimensional indices into a flat storage offset.
    size_t offset(const std::vector<size_t>& indices) const {
        if (indices.size() != strides_.size()) {
            throw std::invalid_argument("Number of indices must match number of dimensions");
        }
        size_t off = 0;
        for (size_t i = 0; i < strides_.size(); ++i) {
            off += indices[i] * strides_[i];
        }
        return off;
    }

    // Checks whether these strides match the shape's standard contiguous layout.
    bool is_contiguous(const Shape& shape) const {
        Stride expected(shape);
        if (expected.ndim() != ndim()) return false;
        for (size_t i = 0; i < ndim(); ++i) {
            if (expected[i] != strides_[i]) return false;
        }
        return true;
    }
private:
    std::vector<size_t> strides_;
};