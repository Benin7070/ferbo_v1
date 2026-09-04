#include "broadcast.hpp"
#include <stdexcept>
#include <algorithm>

namespace ops {

Shape broadcast_shapes(const Shape& a, const Shape& b) {
    size_t ndim = std::max(a.ndim(), b.ndim());
    std::vector<size_t> out_dims(ndim);

    // Walk from the rightmost dim of each shape (i counts distance from the end).
    for (size_t i = 0; i < ndim; ++i) {
        size_t a_dim = (i < a.ndim()) ? a[a.ndim() - 1 - i] : 1;   // missing dim = implicit 1
        size_t b_dim = (i < b.ndim()) ? b[b.ndim() - 1 - i] : 1;

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::invalid_argument("broadcast: shapes not compatible");
        }
        out_dims[ndim - 1 - i] = std::max(a_dim, b_dim);
    }
    return Shape(out_dims);
}

Stride broadcast_strides(const Shape& shape, const Stride& stride, const Shape& target_shape) {
    size_t target_ndim = target_shape.ndim();
    std::vector<size_t> out_strides(target_ndim, 0);   // default 0 covers "dim doesn't exist" case

    for (size_t i = 0; i < shape.ndim(); ++i) {
        size_t target_idx = target_ndim - 1 - i;
        size_t shape_idx  = shape.ndim() - 1 - i;

        if (shape[shape_idx] == 1 && target_shape[target_idx] != 1) {
            out_strides[target_idx] = 0;              // real broadcast: freeze this axis
        } else {
            out_strides[target_idx] = stride[shape_idx];  // real dim: keep its actual stride
        }
    }
    return Stride(out_strides);
}

}