//
// Created by 27823 on 2024/9/28.
//

#include "types.h"  // 包含 TensorRT 数据类型定义的头文件

// 返回给定 TensorRT 数据类型的大小（以字节为单位）
size_t getDataTypeSize(nvinfer1::DataType dataType) {
    switch (dataType) {
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kFLOAT:
            return 4U;   // 32位整数和浮点数占用 4 字节
        case nvinfer1::DataType::kHALF:
            return 2U;   // 半精度浮点数占用 2 字节
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kFP8:
            return 1U;   // 布尔、无符号8位、带符号8位整数和FP8占用 1 字节
    }
    return 0;  // 未知数据类型返回0
}

// 计算多维张量的总体积
int64_t calculateVolume(const nvinfer1::Dims& dims) {
    int64_t volume = 1;
    // 遍历每个维度，计算体积
    for (int i = 0; i < dims.nbDims; ++i) {
        volume *= static_cast<int64_t>(dims.d[i]);  // 将每个维度大小相乘
    }
    return volume;  // 返回整体体积
}

// 将数字 n 向上取整到指定对齐大小
int64_t roundUp(int64_t n, int64_t align) {
    return (n + align - 1) / align * align;  // 计算并返回对齐后的结果
}
