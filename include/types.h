#ifndef CUDA_RUN_TYPES_H
#define CUDA_RUN_TYPES_H
#pragma once

#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>

constexpr int defaultAlignment = 32; // 默认对齐大小

/**
 * @brief 获取指定数据类型的字节大小。
 *
 * @param dataType 数据类型。
 * @return size_t 数据类型所占的字节大小。
 */
size_t getDataTypeSize(nvinfer1::DataType dataType);

/**
 * @brief 根据张量的维度计算其体积（总元素个数）。
 *
 * @param dims 张量的维度。
 * @return int64_t 张量的总元素个数。
 */
uint64_t calculateVolume(const nvinfer1::Dims& dims);

/**
 * @brief 将数字 n 向上取整到 align 的最接近倍数。
 *
 * @param n 要取整的数字。
 * @param align 对齐值（默认是 defaultAlignment）。
 * @return int64_t 取整后的值。
 */
[[maybe_unused]] uint64_t roundUp(uint64_t n, uint64_t align = defaultAlignment);



#endif //CUDA_RUN_TYPES_H
