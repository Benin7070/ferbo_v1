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

}