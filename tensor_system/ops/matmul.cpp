#include "matmul.hpp"
#include "dispatch.hpp"
#include <stdexcept>

namespace ops {
namespace {

void check_fp32(const Tensor& t, const char* op_name) {
    if (t.dtype().value() != DType::FP32) {
        throw std::runtime_error(std::string(op_name) + ": only FP32 supported for now");
    }
}

}  // anonymous namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
    check_fp32(a, "matmul");
    check_fp32(b, "matmul");

    // Only plain 2D matmul for now — batched matmul (needed for multi-head
    // attention, where you matmul per-head across a batch dim) is real
    // future scope, flagging it rather than building it now.
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul: only 2D tensors supported for now");
    }

    size_t M = a.shape()[0];
    size_t K = a.shape()[1];
    size_t K2 = b.shape()[0];
    size_t N = b.shape()[1];

    if (K != K2) {
        throw std::invalid_argument("matmul: inner dimensions do not match (" +
            std::to_string(K) + " vs " + std::to_string(K2) + ")");
    }

    Tensor out(Shape(std::vector<size_t>{M, N}), a.dtype(), a.device());

    if (a.is_contiguous() && b.is_contiguous()) {
        // Fast path: hand straight to the backend kernel as flat buffers.
        backend::dispatch_matmul(a.device().type(),
                                  static_cast<const float*>(a.data()),
                                  static_cast<const float*>(b.data()),
                                  static_cast<float*>(out.data()),
                                  M, K, N);
        return out;
    }

    // General path: stride-aware, handles transposed/non-contiguous inputs.
    // Same triple loop, but reading through offset_of() instead of flat
    // pointer arithmetic.
    float* out_ptr = static_cast<float*>(out.data());
    const float* a_base = static_cast<const float*>(a.data());
    const float* b_base = static_cast<const float*>(b.data());

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += a_base[a.offset_of({i, k})] * b_base[b.offset_of({k, j})];
            }
            out_ptr[i * N + j] = acc;   // out is guaranteed fresh + contiguous — flat write is safe
        }
    }
    return out;
}

}