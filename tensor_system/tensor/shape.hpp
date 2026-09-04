#pragma once
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <numeric>


class Shape{
public:
    Shape(std::initializer_list<int> dims){
        dims_.reserve(dims.size());
        for (int d : dims){
            if (d<0){
                throw std::invalid_argument("The dimention cannot be negative");
            }
            dims_.push_back(static_cast<size_t>(d));
        }
        numel_=compute_numel();
    }

    // needed because Tensor builds shapes at runtime,
    // not from a literal list, when permuting dims for transpose/reshape
    explicit Shape(std::vector<size_t> dims) : dims_(std::move(dims)) {
        numel_ = compute_numel();
    }

    size_t ndim() const{
        return dims_.size();
    }

    size_t operator[](size_t i) const{     //when we call them using the square braket we get the size eg: in Shape shape{2,3,4} and in shape[0] gets 2
        if (i>=dims_.size()){
            throw std::out_of_range("Shape index out of range");
        }
        return dims_[i];
    }

    size_t numel() const{
        return numel_;  // cached at construction — safe because Shape is immutable
    }

private:
    size_t compute_numel() const{      //calculating the size eg:{2,3,4} gives 2*3*4= 24
        return std::accumulate(dims_.begin(),dims_.end(),
                static_cast<size_t>(1),
                std::multiplies<size_t>());
    }
    std::vector<size_t> dims_;
    size_t numel_;
};