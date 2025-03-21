﻿#ifndef CUDAWARP_H
#define CUDAWARP_H
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

/**
 * @brief 表示用于仿射变换的 2x3 转换矩阵的结构体。
 */
extern "C" struct TransformMatrix {
    float3 matrix[2];   // 用于仿射变换的 2x3 矩阵。

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
 * @brief 表示用于仿射变换的 2x3 转换矩阵的结构体。
 */
extern "C" struct WarpAffine {
    dim3 BlocksPerGrid{};   // 用于仿射变换的 2x3 矩阵。
    dim3 threadsPerBlock{};   // 用于仿射变换的 2x3 矩阵。

    uint8_t * inputPtr;
    uint32_t fromWidth, fromHeight;
    float * outputPtr;
    uint32_t toWidth, toHeight;

    uint32_t Channel;

    TransformMatrix* transforms{nullptr};
    cudaStream_t stream{nullptr};

    bool BGR{false};

    WarpAffine(void *input, uint32_t imageWidth, uint32_t imageHeight,
               void *output, uint32_t outputWidth, uint32_t outputHeight,
               uint32_t Channel, bool BGR,
               cudaStream_t stream);


    void updateImageOutSize(uint32_t imageWidth, uint32_t imageHeight, uint32_t imageChannel = 3);

    void updateImageInputPtr(void *input);

    void updateBGR(bool bgr);

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
    void cudaWarpAffine();
    void cudaCutImg();
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
        float* output, uint32_t outputWidth, uint32_t outputHeight,
        float3 matrix[2], cudaStream_t stream);

#endif //CUDAWARP_H
