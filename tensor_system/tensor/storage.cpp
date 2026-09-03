#include <storage.hpp>
#include <cstdlib> //lib for the malloc and free
#include <new>

Storage::Storage(size_t size_in_bytes , Device device)       //creating an storage object and storing the data into it
    : data_(nullptr), size_(size_in_bytes), device_(device){
        allocate();
    }

Storage::~Storage(){
    release();
}


void Storage::allocate(){
    switch (device_){
        case Device::CPU:
            data_=std::malloc(size_);
            if (size_>0 && data_ == nullptr){
                throw std::bad_alloc();   //allocation failed so throwing and bad allocation error
            }
            break;
    }
}

void Storage::release(){
    switch (device_)
    {
    case Device::CPU:
        std::free(data_);
        break;
    }
    data_=nullptr;   //to remove the dangling pointer
}

Storage::Storage(Storage&& other) noexcept    //handling the transfering of ownership using the rvalue reference
    : data_(other.data_),size_(other.size_),device_(other.device_){
        other.data_=nullptr;
        other.size_=0;
}

Storage& Storage::operator=(Storage&& other) noexcept{  //handling the move operator assignment
    if (this != &other){
        release();
        data_=other.data_;
        size_=other.size_;
        device_=other.device_;
        other.data_=nullptr;
        other.size_=0;
    }
    return *this;
}