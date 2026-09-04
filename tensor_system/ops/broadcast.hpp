#pragma once
#include <shape.hpp>
#include <stride.hpp>


//define the rules for broadcasting tensor shapes during operations such as addition or multiplication.

namespace ops{
    Shape broadcast_shapes(const Shape& a,const Shape& b);
    Stride broadcast_strides(const Shape& shape, const Stride& stride, const Shape& target_shape);
}