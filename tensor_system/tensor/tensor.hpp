#pragma once
#include <memory>
#include <vector>
#include <random>
#include <cstring>
#include <stdexcept>

#include "storage.hpp"
#include "shape.hpp"
#include "stride.hpp"
#include "dtype.hpp"
#include "device.hpp"

class Tensor{
public:
    // Primary constructor: allocates fresh Storage for a brand-new tensor.
    Tensor(Shape shape, DTypeInfo dtype = DTypeInfo(DType::FP32), Device device = Device(DeviceType::CPU))
        : shape_(std::move(shape)),
          stride_(shape_),             // derive default contiguous strides from shape
          dtype_(dtype),
          device_(device),
          offset_(0)
    
    {
        size_t bytes = shape_.numel() * dtype_.size();
        storage_ = std::make_shared<Storage>(bytes, device_);
    }

     // --- Creation ---
    static Tensor zeros(Shape shape, DTypeInfo dtype = DTypeInfo(DType::FP32), Device device = Device(DeviceType::CPU)) {
        Tensor t(shape, dtype, device);
        std::memset(t.storage_->data(), 0, t.storage_->size());
        return t;
    }

    static Tensor full(Shape shape, float value, DTypeInfo dtype = DTypeInfo(DType::FP32), Device device = Device(DeviceType::CPU)) {
        if (dtype.value() != DType::FP32) throw std::runtime_error("full() only supports FP32 for now");
        Tensor t(shape, dtype, device);
        float* ptr = static_cast<float*>(t.storage_->data());
        for (size_t i = 0; i < t.numel(); ++i) ptr[i] = value;
        return t;
    }

    static Tensor ones(Shape shape, DTypeInfo dtype = DTypeInfo(DType::FP32), Device device = Device(DeviceType::CPU)) {
        return full(shape, 1.0f, dtype, device);
    }

    static Tensor randn(Shape shape, DTypeInfo dtype = DTypeInfo(DType::FP32), Device device = Device(DeviceType::CPU)) {
        if (dtype.value() != DType::FP32) throw std::runtime_error("randn() only supports FP32 for now");
        Tensor t(shape, dtype, device);
        static std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float* ptr = static_cast<float*>(t.storage_->data());
        for (size_t i = 0; i < t.numel(); ++i) ptr[i] = dist(gen);
        return t;
    }

    // --- Inspection ---
    const Shape& shape() const { return shape_; }
    const Stride& stride() const { return stride_; }
    DTypeInfo dtype() const { return dtype_; }
    Device device() const { return device_; }
    size_t numel() const { return shape_.numel(); }
    size_t ndim() const { return shape_.ndim(); }

    // --- Access ---
    void* data() const {
        char* base = static_cast<char*>(storage_->data());
        return base + offset_ * dtype_.size();
    }
    // Returns a reference, not a value — so t.at({0,1}) = 5.0f works as a setter too.
    float& at(const std::vector<size_t>& indices) {
        if (dtype_.value() != DType::FP32) throw std::runtime_error("at() only supports FP32 for now");
        size_t elem_offset = offset_ + stride_.offset(indices);
        return static_cast<float*>(storage_->data())[elem_offset];
    }

    // --- Manipulation ---
    Tensor transpose(size_t dim0, size_t dim1) const {
        if (dim0 >= ndim() || dim1 >= ndim()) throw std::out_of_range("transpose: dim out of range");

        std::vector<size_t> new_dims(ndim()), new_strides(ndim());
        for (size_t i = 0; i < ndim(); ++i) {
            new_dims[i] = shape_[i];
            new_strides[i] = stride_[i];
        }
        std::swap(new_dims[dim0], new_dims[dim1]);
        std::swap(new_strides[dim0], new_strides[dim1]);

        return Tensor(storage_, Shape(new_dims), Stride(new_strides), dtype_, device_, offset_);
    }
    Tensor reshape(const std::vector<size_t>& new_dims) const {
        Shape new_shape(new_dims);
        if (new_shape.numel() != numel()) throw std::invalid_argument("reshape: element count mismatch");
        if (!stride_.is_contiguous(shape_)) throw std::runtime_error("reshape: not contiguous, call .contiguous() first");

        Stride new_stride(new_shape);   // fresh contiguous strides for the new shape
        return Tensor(storage_, new_shape, new_stride, dtype_, device_, offset_);
    }

    Tensor view(const std::vector<size_t>& new_dims) const {
        return reshape(new_dims);   // same no-copy contract; kept as a separate name deliberately
    }
private:
    // Internal constructor: builds a Tensor that SHARES existing storage.
    // Never allocates — this is what view/reshape/transpose actually call.
    Tensor(std::shared_ptr<Storage> storage, Shape shape, Stride stride,
           DTypeInfo dtype, Device device, size_t offset)
        : storage_(std::move(storage)), shape_(std::move(shape)), stride_(std::move(stride)),
          dtype_(dtype), device_(device), offset_(offset) {}

    std::shared_ptr<Storage> storage_;
    Shape shape_;
    Stride stride_;
    DTypeInfo dtype_;
    Device device_;
    size_t offset_;
};