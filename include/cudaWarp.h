#ifndef CUDAWARP_H
#define CUDAWARP_H
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

/**
 * @brief 表示用于仿射变换的 2x3 转换矩阵的结构体。
 */
extern "C" struct TransformMatrix {
    float3 matrix[2];   // 用于仿射变换的 2x3 矩阵。
    int    lastWidth;   // 上次处理的源图像的宽度。
    int    lastHeight;  // 上次处理的源图像的高度。

    /**
     * @brief 根据源图像和目标图像尺寸的变化更新变换矩阵。
     *
     * @param fromWidth 源图像的宽度。
     * @param fromHeight 源图像的高度。
     * @param toWidth 目标图像的宽度。
     * @param toHeight 目标图像的高度。
     */
    void update(int fromWidth, int fromHeight, int toWidth, int toHeight);

    /**
     * @brief 使用变换矩阵转换一个点。
     *
     * @param x 点的 X 坐标。
     * @param y 点的 Y 坐标。
     * @param[out] ox 变换后的 X 坐标。
     * @param[out] oy 变换后的 Y 坐标。
     */
    void transform(float x, float y, float* ox, float* oy) const;
};

/**
 * @brief 使用 CUDA 应用仿射变换。
 *
 * @param input 指向输入图像数据的指针。
 * @param inputWidth 输入图像的宽度。
 * @param inputHeight 输入图像的高度。
 * @param output 指向输出图像数据的指针。
 * @param outputWidth 输出图像的宽度。
 * @param outputHeight 输出图像的高度。
 * @param matrix 仿射变换矩阵。
 * @param stream 用于异步执行的 CUDA 流（可选）。
 */
extern "C" void cudaWarpAffine(
        uint8_t* input, uint32_t inputWidth, uint32_t inputHeight,
        float* output, uint32_t outputWidth, uint32_t outputHeight, float3 matrix[2], cudaStream_t stream);

#endif //CUDAWARP_H
