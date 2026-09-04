#include <reduction.hpp>
#include <stdexcept>
#include <limits>

namespace ops{
namespace{

void check_fp32(const Tensor& t, const char* op_name) {
    if (t.dtype().value() != DType::FP32) {
        throw std::runtime_error(std::string(op_name) + ": only FP32 supported for now");
    }
}

std::vector<size_t> unravel_index(size_t flat, const Shape& shape) {
    std::vector<size_t> indices(shape.ndim());
    for (size_t i = shape.ndim(); i-- > 0; ) {
        indices[i] = flat % shape[i];
        flat /= shape[i];
    }
    return indices;
}

std::vector<size_t> insert_axis(const std::vector<size_t>& out_idx, size_t axis, size_t axis_val) {
    std::vector<size_t> full_idx;
    full_idx.reserve(out_idx.size() + 1);
    for (size_t i = 0; i < out_idx.size(); ++i) {
        if (i == axis) full_idx.push_back(axis_val);
        full_idx.push_back(out_idx[i]);
    }
    if (axis == out_idx.size()) full_idx.push_back(axis_val);  // reducing the last axis
    return full_idx;
}

template <typename Fn>
Tensor reduce_axis(const Tensor& a, size_t axis, float init, Fn combine, const char* op_name) {
    check_fp32(a, op_name);
    if (axis >= a.ndim()) throw std::out_of_range(std::string(op_name) + ": axis out of range");

    std::vector<size_t> out_dims;
    for (size_t i = 0; i < a.ndim(); ++i) {
        if (i != axis) out_dims.push_back(a.shape()[i]);
    }
    Shape out_shape(out_dims);
    Tensor out(out_shape, a.dtype(), a.device());

    float* out_ptr = static_cast<float*>(out.data());
    const float* a_base = static_cast<const float*>(a.data());
    size_t axis_len = a.shape()[axis];

    for (size_t flat = 0; flat < out.numel(); ++flat) {
        auto out_idx = unravel_index(flat, out_shape);
        float acc = init;
        for (size_t k = 0; k < axis_len; ++k) {
            auto full_idx = insert_axis(out_idx, axis, k);
            acc = combine(acc, a_base[a.offset_of(full_idx)]);
        }
        out_ptr[flat] = acc;
    }
    return out;
}
template <typename Fn>
Tensor reduce_all(const Tensor& a, float init, Fn combine, const char* op_name) {
    check_fp32(a, op_name);
    Tensor out(Shape({}), a.dtype(), a.device());   // scalar output
    const float* a_base = static_cast<const float*>(a.data());

    float acc = init;
    for (size_t flat = 0; flat < a.numel(); ++flat) {
        auto idx = unravel_index(flat, a.shape());
        acc = combine(acc, a_base[a.offset_of(idx)]);
    }
    static_cast<float*>(out.data())[0] = acc;
    return out;
}

}  // anonymous namespace

Tensor sum(const Tensor& a, size_t axis) {
    return reduce_axis(a, axis, 0.0f, [](float acc, float x) { return acc + x; }, "sum");
}
Tensor sum(const Tensor& a) {
    return reduce_all(a, 0.0f, [](float acc, float x) { return acc + x; }, "sum");
}

Tensor mean(const Tensor& a, size_t axis) {
    Tensor s = sum(a, axis);
    float* ptr = static_cast<float*>(s.data());
    float count = static_cast<float>(a.shape()[axis]);
    for (size_t i = 0; i < s.numel(); ++i) ptr[i] /= count;   // s is fresh+contiguous, flat write is safe
    return s;
}
Tensor mean(const Tensor& a) {
    Tensor s = sum(a);
    static_cast<float*>(s.data())[0] /= static_cast<float>(a.numel());
    return s;
}
Tensor max(const Tensor& a, size_t axis) {
    float ninf = -std::numeric_limits<float>::infinity();
    return reduce_axis(a, axis, ninf, [](float acc, float x) { return acc > x ? acc : x; }, "max");
}
Tensor max(const Tensor& a) {
    float ninf = -std::numeric_limits<float>::infinity();
    return reduce_all(a, ninf, [](float acc, float x) { return acc > x ? acc : x; }, "max");
}

Tensor min(const Tensor& a, size_t axis) {
    float pinf = std::numeric_limits<float>::infinity();
    return reduce_axis(a, axis, pinf, [](float acc, float x) { return acc < x ? acc : x; }, "min");
}
Tensor min(const Tensor& a) {
    float pinf = std::numeric_limits<float>::infinity();
    return reduce_all(a, pinf, [](float acc, float x) { return acc < x ? acc : x; }, "min");
}

} //namespace ops