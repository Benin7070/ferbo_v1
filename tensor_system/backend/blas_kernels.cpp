#include "blas_kernels.hpp"
#include <cblas.h>

namespace backend {

void blas_matmul(const float* a, const float* b, float* out,
				 size_t M, size_t K, size_t N) {
	cblas_sgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasNoTrans,
		static_cast<int>(M),
		static_cast<int>(N),
		static_cast<int>(K),
		1.0f,
		a,
		static_cast<int>(K),
		b,
		static_cast<int>(N),
		0.0f,
		out,
		static_cast<int>(N));
}

}
