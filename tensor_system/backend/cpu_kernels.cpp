#include <cpu_kernels.hpp>
#include <algorithm>
#include <cmath>

namespace backend {

void cpu_add(const float* a, const float* b, float* out, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}
void cpu_sub(const float* a, const float* b, float* out, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}
void cpu_mul(const float* a, const float* b, float* out, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}
void cpu_div(const float* a, const float* b, float* out, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
}
void cpu_neg(const float* a, float* out, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) out[i] = -a[i];
}
void cpu_sqrt(const float* a, float* out, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) out[i] = std::sqrt(a[i]);
}
void cpu_exp(const float* a, float* out, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) out[i] = std::exp(a[i]);
}
void cpu_log(const float* a, float* out, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) out[i] = std::log(a[i]);
}

void cpu_matmul(const float* a, const float* b, float* out,
                 size_t M, size_t K, size_t N) {
    // a is M×K row-major, b is K×N row-major, out is M×N row-major.
    // Blocked i-k-j traversal improves cache reuse without changing the API.
    constexpr size_t block = 64;
    std::fill(out, out + M * N, 0.0f);
    #pragma omp parallel for schedule(static)
    for (size_t i0 = 0; i0 < M; i0 += block) {
        for (size_t k0 = 0; k0 < K; k0 += block) {
            for (size_t j0 = 0; j0 < N; j0 += block) {
                size_t i_end = std::min(i0 + block, M);
                size_t k_end = std::min(k0 + block, K);
                size_t j_end = std::min(j0 + block, N);

                for (size_t i = i0; i < i_end; ++i) {
                    for (size_t k = k0; k < k_end; ++k) {
                        const float value = a[i * K + k];
                        for (size_t j = j0; j < j_end; ++j) {
                            out[i * N + j] += value * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

}