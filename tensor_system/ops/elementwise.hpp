#pragma once
#include "tensor.hpp"

namespace ops {
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor sub(const Tensor& a, const Tensor& b);
    Tensor mul(const Tensor& a, const Tensor& b);
    Tensor div(const Tensor& a, const Tensor& b);

    Tensor neg(const Tensor& a);
    Tensor sqrt(const Tensor& a);
    Tensor exp(const Tensor& a);
    Tensor log(const Tensor& a);
}