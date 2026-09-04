#include <cpu_kernels.hpp>
#include <cmath>

namespace backend {

void cpu_add(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
void cpu_sub(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}
void cpu_mul(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}
void cpu_div(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
}
void cpu_neg(const float* a, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = -a[i];
}
void cpu_sqrt(const float* a, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = std::sqrt(a[i]);
}
void cpu_exp(const float* a, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = std::exp(a[i]);
}
void cpu_log(const float* a, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = std::log(a[i]);
}

void cpu_matmul(const float* a, const float* b, float* out,
                 size_t M, size_t K, size_t N) {
    // a is M×K row-major, b is K×N row-major, out is M×N row-major.
    // Naive triple loop — correctness first, per the spec. No tiling,
    // no blocking, no vectorization yet.
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                acc += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = acc;
        }
    }
}

}