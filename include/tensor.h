//
// Created by 27823 on 2024/9/29.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <string>

#include "types.h"

#pragma once

/**
 * @brief 一个简单的张量类，用于管理主机和设备内存。
 *
 * 该类提供了在主机和设备之间分配和管理内存的功能，支持自动调整内存大小。
 */
class Tensor {
public:
    // 默认构造函数
    explicit Tensor() = default;

    // 析构函数，负责释放分配的内存
    ~Tensor();

    // 返回指向主机内存的指针
    void* host() { return hostPtr; }

    /**
     * @brief 返回指向主机内存的指针，并根据需要分配内存
     *
     * @param bytes 要求的内存大小（字节）
     * @return 指向主机内存的指针
     */
    void* host(int64_t bytes);

    // 返回指向设备内存的指针
    void* device() { return devicePtr; }

    /**
     * @brief 返回指向设备内存的指针，并根据需要分配内存
     *
     * @param bytes 要求的内存大小（字节）
     * @return 指向设备内存的指针
     */
    void* device(int64_t bytes);

    /**
     * @brief 重新分配主机内存来适应指定大小
     *
     * @param bytes 新的内存大小（字节）
     */
    void reallocHost(int64_t bytes);

    /**
     * @brief 重新分配设备内存来适应指定大小
     *
     * @param bytes 新的内存大小（字节）
     */
    void reallocDevice(int64_t bytes);

private:
    void* hostPtr = nullptr;      /**< 指向主机内存的指针 */
    void* devicePtr = nullptr;    /**< 指向设备内存的指针 */
    int64_t hostBytes = 0;        /**< 主机内存的大小（字节） */
    int64_t deviceBytes = 0;      /**< 设备内存的大小（字节） */
    int64_t hostCap = 0;          /**< 主机内存的容量（字节） */
    int64_t deviceCap = 0;        /**< 设备内存的容量（字节） */
};

#endif // CUDA_RUN_TENSOR_H
