#include "macro.h"  // 包含一些可能的宏定义，例如错误检查宏
#include "tensor.h" // 包含 Tensor 类的声明

// Tensor 类的实现

// 用于在主机（CPU）上重新分配内存
void Tensor::reallocHost(int64_t bytes) {
    // 如果当前分配容量小于所需字节数，则重新分配内存
    if (hostCap < bytes) {
        // 释放当前的主机内存（假设 CUDA 是一个用于检查 CUDA API 结果的宏）
        CUDA(cudaFreeHost(hostPtr));
        // 分配新的主机内存
        CUDA(cudaMallocHost(&hostPtr, bytes));
        // 更新当前容量
        hostCap = bytes;
    }
    // 更新当前使用的字节数
    hostBytes = bytes;
}

// 用于在设备（GPU）上重新分配内存
void Tensor::reallocDevice(int64_t bytes) {
    // 如果当前设备内存容量小于所需字节数，则重新分配内存
    if (deviceCap < bytes) {
        // 释放当前设备内存
        CUDA(cudaFree(devicePtr));
        // 分配新的设备内存
        CUDA(cudaMalloc(&devicePtr, bytes));
        // 更新设备内存容量
        deviceCap = bytes;
    }
    // 更新当前使用的设备内存字节数
    deviceBytes = bytes;
}

// Tensor 类析构函数，负责释放在主机和设备上分配的内存
Tensor::~Tensor() {
    // 如果主机指针不为空，释放主机内存
    if (hostPtr != nullptr) {
        CUDA(cudaFreeHost(hostPtr));
//        hostBytes = 0;
    }
    // 如果设备指针不为空，释放设备内存
    if (devicePtr != nullptr) {
        CUDA(cudaFree(devicePtr));
//        deviceBytes = 0;
    }
}

// 获取指定大小的主机内存指针
void* Tensor::host(int64_t size) {
    // 调用 reallocHost 以确保分配足够的内存
    reallocHost(size);
    return hostPtr;
}

// 获取指定大小的设备内存指针
void* Tensor::device(int64_t size) {
    // 调用 reallocDevice 以确保分配足够的内存
    reallocDevice(size);
    return devicePtr;
}
