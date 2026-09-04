#pragma once
#include <cstddef>

namespace backend {

void cpu_add(const float* a, const float* b, float* out, size_t n);
void cpu_sub(const float* a, const float* b, float* out, size_t n);
void cpu_mul(const float* a, const float* b, float* out, size_t n);
void cpu_div(const float* a, const float* b, float* out, size_t n);

void cpu_neg(const float* a, float* out, size_t n);
void cpu_sqrt(const float* a, float* out, size_t n);
void cpu_exp(const float* a, float* out, size_t n);
void cpu_log(const float* a, float* out, size_t n);

void cpu_matmul(const float* a, const float* b, float* out,
                 size_t M, size_t K, size_t N);
}