#pragma once
#include <cstddef>
#include <string>

// Supported hardware backends where tensor data can be stored or processed.
enum class DeviceType{
    CPU,
    CUDA,
    NPU
};

class Device{
public:
    // Creates a device; index identifies a specific device for CUDA or NPU.
    Device(DeviceType type, int index = 0) : type_(type), index_(index) {}

    // Returns the hardware backend type.
    DeviceType type() const { return type_; }

    // Returns the device index, such as GPU 0 or NPU 1.
    int index() const { return index_; }

    // Devices are equal when both their backend and index match.
    bool operator==(const Device& other) const {
        return type_ == other.type_ && index_ == other.index_;
    }

    // Returns true when the two devices are different.
    bool operator!=(const Device& other) const {
        return !(*this == other);
    }

    // Converts the device to a readable form, such as "cpu" or "cuda:0".
    std::string to_string() const {
        switch (type_) {
            case DeviceType::CPU:  return "cpu";
            case DeviceType::CUDA: return "cuda:" + std::to_string(index_);
            case DeviceType::NPU:  return "npu:" + std::to_string(index_);
        }
        return "unknown";
    }

private:
    DeviceType type_;
    int index_;
};