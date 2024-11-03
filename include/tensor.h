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


class Tensor {
public:
    explicit Tensor() = default;
    ~Tensor();

    void* host() {return hostPtr;}
    void* host(int64_t bytes);
    void* device() {return devicePtr;}
    void* device(int64_t bytes);
    void reallocHost(int64_t bytes);
    void reallocDevice(int64_t bytes);
private:
    void*   hostPtr     = nullptr; /**< Pointer to host memory */
    void*   devicePtr   = nullptr; /**< Pointer to device memory */
    int64_t hostBytes   = 0;       /**< Size of host memory in bytes */
    int64_t deviceBytes = 0;       /**< Size of device memory in bytes */
    int64_t hostCap     = 0;       /**< Capacity of host memory in bytes */
    int64_t deviceCap   = 0;       /**< Capacity of device memory in bytes */

};


#endif //CUDA_RUN_TENSOR_H
