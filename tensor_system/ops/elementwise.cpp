#include "elementwise.hpp"
#include "broadcast.hpp"
#include <cmath>
#include <stdexcept>
#include <functional>

namespace ops{

namespace{
    std::vector<size_t> unravel_index(size_t flat, const Shape& shape) {
        std::vector<size_t> indices(shape.ndim());
        for (size_t i = shape.ndim(); i-- > 0; ) {
            indices[i] = flat % shape[i];
            flat /= shape[i];
        }
        return indices;
    }

    void check_fp32(const Tensor& t, const char* op_name) {
        if (t.dtype().value() != DType::FP32) {
            throw std::runtime_error(std::string(op_name) + ": only FP32 supported for now");
        }
    }

    template <typename Fn>
    Tensor binary_op(const Tensor& a, const Tensor& b, Fn fn, const char* op_name) {
        check_fp32(a, op_name);
        check_fp32(b, op_name);

        Shape out_shape = ops::broadcast_shapes(a.shape(), b.shape());
        Stride a_bstride = ops::broadcast_strides(a.shape(), a.stride(), out_shape);
        Stride b_bstride = ops::broadcast_strides(b.shape(), b.stride(), out_shape);

        Tensor out(out_shape, a.dtype(), a.device());
        float* out_ptr = static_cast<float*>(out.data());
        const float* a_base = static_cast<const float*>(a.data());
        const float* b_base = static_cast<const float*>(b.data());

        for (size_t flat = 0; flat < out.numel(); ++flat) {
            auto idx = unravel_index(flat, out_shape);
            float a_val = a_base[a.base_offset() + a_bstride.offset(idx)];
            float b_val = b_base[b.base_offset() + b_bstride.offset(idx)];
            out_ptr[flat] = fn(a_val, b_val);
        }
        return out;
    }

    template <typename Fn>
    Tensor unary_op(const Tensor& a, Fn fn, const char* op_name) {
        check_fp32(a, op_name);

        Tensor out(a.shape(), a.dtype(), a.device());
        float* out_ptr = static_cast<float*>(out.data());
        const float* a_base = static_cast<const float*>(a.data());

        size_t n = out.numel();
        for (size_t flat = 0; flat < n; ++flat) {
            auto idx = unravel_index(flat, out.shape());
            out_ptr[flat] = fn(a_base[a.offset_of(idx)]);
        }
        return out;
    }

} //anonymous namespace

Tensor add(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x + y; }, "add");
}
Tensor sub(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x - y; }, "sub");
}
Tensor mul(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x * y; }, "mul");
}
Tensor div(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, [](float x, float y) { return x / y; }, "div");
}

Tensor neg(const Tensor& a) {
    return unary_op(a, [](float x) { return -x; }, "neg");
}
Tensor sqrt(const Tensor& a) {
    return unary_op(a, [](float x) { return std::sqrt(x); }, "sqrt");
}
Tensor exp(const Tensor& a) {
    return unary_op(a, [](float x) { return std::exp(x); }, "exp");
}
Tensor log(const Tensor& a) {
    return unary_op(a, [](float x) { return std::log(x); }, "log");
}

} //ops namespace