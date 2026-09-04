#pragma once
#include <cstddef>
#include <stdexcept>
#include <device.hpp>


class Storage{
public:
    Storage(size_t size_in_bytes, Device device);
    ~Storage();

    Storage(const Storage&)= delete;   //this disables the copy constructor
    Storage& operator=(const Storage&) = delete;  //this disables the copy assignment operator 

    Storage(Storage&& other) noexcept; //transfering the ownership
    Storage& operator=(Storage&& other) noexcept;   //move assignment operator

    void* data() const { return data_; }
    size_t size() const { return size_; }
    Device device() const {return device_;}

private:
    void allocate();
    void release();

    void* data_;
    size_t size_;
    Device device_;

};
