#include "elementwise.hpp"
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

    void check_same_shape(const Tensor& a, const Tensor& b, const char* op_name) {
        if (a.ndim() != b.ndim()) {
            throw std::invalid_argument(std::string(op_name) + ": dimension count mismatch");
        }
        for (size_t i = 0; i < a.ndim(); ++i) {
            if (a.shape()[i] != b.shape()[i]) {
                throw std::invalid_argument(std::string(op_name) + ": shape mismatch");
            }
        }
    }

    void check_fp32(const Tensor& t, const char* op_name) {
        if (t.dtype().value() != DType::FP32) {
            throw std::runtime_error(std::string(op_name) + ": only FP32 supported for now");
        }
    }

    template <typename Fn>
    Tensor binary_op(const Tensor& a, const Tensor& b, Fn fn, const char* op_name) {
        check_same_shape(a, b, op_name);
        check_fp32(a, op_name);
        check_fp32(b, op_name);

        Tensor out(a.shape(), a.dtype(), a.device());   // fresh, contiguous — never shares storage
        float* out_ptr = static_cast<float*>(out.data());
        const float* a_base = static_cast<const float*>(a.data());
        const float* b_base = static_cast<const float*>(b.data());

        size_t n = out.numel();
        for (size_t flat = 0; flat < n; ++flat) {
            auto idx = unravel_index(flat, out.shape());
            float a_val = a_base[a.offset_of(idx)];
            float b_val = b_base[b.offset_of(idx)];
            out_ptr[flat] = fn(a_val, b_val);   // writing flat is safe: out is guaranteed contiguous
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