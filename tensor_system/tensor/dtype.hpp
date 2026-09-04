#pragma once
#include <cstddef>

enum class DType{
    FP32,
    FP16,
    BF16,
    INT32,
    INT64,
    INT8,
};

inline size_t dtype_size(DType dt){   //return the size of the data type
    switch(dt){
        case DType::FP32: return 4;
        case DType::FP16: return 2;
        case DType::BF16: return 2;
        case DType::INT32: return 4;
        case DType::INT64: return 8;
        case DType::INT8: return 1;
    }
    return 0;
}

inline bool dtype_is_floating_point(DType dt){  //to check the datatype as the floating point datatype or not
    switch (dt)
    {
    case DType::FP32:
    case DType::FP16:
    case DType::BF16:
        return true;
    case DType::INT64:
    case DType::INT32:
    case DType::INT8:
        return false;
    }
    return false;
}


class DTypeInfo{  //wrapper for above function to get the data about the datatype
public:
    DTypeInfo(DType dt) : dt_(dt) {}  //constructor

    size_t size() const { return dtype_size(dt_); }
    bool is_floating_point() const { return dtype_is_floating_point(dt_); }
    DType value() const { return dt_; }

private:
    DType dt_;
};