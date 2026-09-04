#pragma once
#include <cstddef>

namespace backend {

void blas_matmul(const float* a, const float* b, float* out,
                 size_t M, size_t K, size_t N);

}