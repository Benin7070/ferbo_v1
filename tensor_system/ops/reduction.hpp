#pragma once
#include <tensor.hpp>

namespace ops{
    
Tensor sum(const Tensor& a);                 // full reduction -> scalar
Tensor sum(const Tensor& a, size_t axis);    // reduce along one axis

Tensor mean(const Tensor& a);
Tensor mean(const Tensor& a, size_t axis);

Tensor max(const Tensor& a);
Tensor max(const Tensor& a, size_t axis);

Tensor min(const Tensor& a);
Tensor min(const Tensor& a, size_t axis);

}