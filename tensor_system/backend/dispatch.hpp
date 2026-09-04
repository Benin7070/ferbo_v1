#pragma once
#include <cstddef>
#include "device.hpp"

namespace backend {

using BinaryFn = void(*)(DeviceType, const float*, const float*, float*, size_t);
using UnaryFn  = void(*)(DeviceType, const float*, float*, size_t);

void dispatch_add(DeviceType device, const float* a, const float* b, float* out, size_t n);
void dispatch_sub(DeviceType device, const float* a, const float* b, float* out, size_t n);
void dispatch_mul(DeviceType device, const float* a, const float* b, float* out, size_t n);
void dispatch_div(DeviceType device, const float* a, const float* b, float* out, size_t n);

void dispatch_neg(DeviceType device, const float* a, float* out, size_t n);
void dispatch_sqrt(DeviceType device, const float* a, float* out, size_t n);
void dispatch_exp(DeviceType device, const float* a, float* out, size_t n);
void dispatch_log(DeviceType device, const float* a, float* out, size_t n);

void dispatch_matmul(DeviceType device, const float* a, const float* b, float* out,
                      size_t M, size_t K, size_t N);
                      
}