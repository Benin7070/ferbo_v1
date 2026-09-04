// backend/dispatch.cpp
#include "dispatch.hpp"
#include "cpu_kernels.hpp"
#include <stdexcept>
#include <string>

namespace backend {

namespace {
[[noreturn]] void unsupported(const char* op, DeviceType d) {
    // Today this only ever fires for CUDA/NPU — the whole point of the
    // dispatcher is that this is the ONLY place that needs to know that.
    throw std::runtime_error(std::string(op) + ": device not yet supported");
}
}

void dispatch_add(DeviceType d, const float* a, const float* b, float* out, size_t n) {
    if (d == DeviceType::CPU) { cpu_add(a, b, out, n); return; }
    unsupported("add", d);
}
void dispatch_sub(DeviceType d, const float* a, const float* b, float* out, size_t n) {
    if (d == DeviceType::CPU) { cpu_sub(a, b, out, n); return; }
    unsupported("sub", d);
}
void dispatch_mul(DeviceType d, const float* a, const float* b, float* out, size_t n) {
    if (d == DeviceType::CPU) { cpu_mul(a, b, out, n); return; }
    unsupported("mul", d);
}
void dispatch_div(DeviceType d, const float* a, const float* b, float* out, size_t n) {
    if (d == DeviceType::CPU) { cpu_div(a, b, out, n); return; }
    unsupported("div", d);
}
void dispatch_neg(DeviceType d, const float* a, float* out, size_t n) {
    if (d == DeviceType::CPU) { cpu_neg(a, out, n); return; }
    unsupported("neg", d);
}
void dispatch_sqrt(DeviceType d, const float* a, float* out, size_t n) {
    if (d == DeviceType::CPU) { cpu_sqrt(a, out, n); return; }
    unsupported("sqrt", d);
}
void dispatch_exp(DeviceType d, const float* a, float* out, size_t n) {
    if (d == DeviceType::CPU) { cpu_exp(a, out, n); return; }
    unsupported("exp", d);
}
void dispatch_log(DeviceType d, const float* a, float* out, size_t n) {
    if (d == DeviceType::CPU) { cpu_log(a, out, n); return; }
    unsupported("log", d);
}

void dispatch_matmul(DeviceType d, const float* a, const float* b, float* out,
                      size_t M, size_t K, size_t N) {
    if (d == DeviceType::CPU) { cpu_matmul(a, b, out, M, K, N); return; }
    unsupported("matmul", d);
}


}