#include "elementwise.hpp"
#include "broadcast.hpp"
#include "dispatch.hpp"
#include <cmath>
#include <stdexcept>

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

    Tensor binary_op(const Tensor& a, const Tensor& b, backend::BinaryFn kernel,
                     const char* op_name) {
        check_fp32(a, op_name);
        check_fp32(b, op_name);

        Shape out_shape = ops::broadcast_shapes(a.shape(), b.shape());
        Stride a_bstride = ops::broadcast_strides(a.shape(), a.stride(), out_shape);
        Stride b_bstride = ops::broadcast_strides(b.shape(), b.stride(), out_shape);

        Tensor out(out_shape, a.dtype(), a.device());
        float* out_ptr = static_cast<float*>(out.data());
        const float* a_base = static_cast<const float*>(a.data());
        const float* b_base = static_cast<const float*>(b.data());
        std::vector<float> a_values(out.numel());
        std::vector<float> b_values(out.numel());

        for (size_t flat = 0; flat < out.numel(); ++flat) {
            auto idx = unravel_index(flat, out_shape);
            a_values[flat] = a_base[a_bstride.offset(idx)];
            b_values[flat] = b_base[b_bstride.offset(idx)];
        }
        kernel(a.device().type(), a_values.data(), b_values.data(), out_ptr, out.numel());
        return out;
    }

    Tensor unary_op(const Tensor& a, backend::UnaryFn kernel, const char* op_name) {
        check_fp32(a, op_name);

        Tensor out(a.shape(), a.dtype(), a.device());
        float* out_ptr = static_cast<float*>(out.data());
        const float* a_base = static_cast<const float*>(a.data());
        std::vector<float> a_values(out.numel());

        for (size_t flat = 0; flat < out.numel(); ++flat) {
            auto idx = unravel_index(flat, out.shape());
            a_values[flat] = a_base[a.offset_of(idx)];
        }
        kernel(a.device().type(), a_values.data(), out_ptr, out.numel());
        return out;
    }

} //anonymous namespace

Tensor add(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, backend::dispatch_add, "add");
}
Tensor sub(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, backend::dispatch_sub, "sub");
}
Tensor mul(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, backend::dispatch_mul, "mul");
}
Tensor div(const Tensor& a, const Tensor& b) {
    return binary_op(a, b, backend::dispatch_div, "div");
}

Tensor neg(const Tensor& a) {
    return unary_op(a, backend::dispatch_neg, "neg");
}
Tensor sqrt(const Tensor& a) {
    return unary_op(a, backend::dispatch_sqrt, "sqrt");
}
Tensor exp(const Tensor& a) {
    return unary_op(a, backend::dispatch_exp, "exp");
}
Tensor log(const Tensor& a) {
    return unary_op(a, backend::dispatch_log, "log");
}

} //ops namespace