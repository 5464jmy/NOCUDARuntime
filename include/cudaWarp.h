//
// Created by 27823 on 2024/9/29.
//

#ifndef CUDA_RUN_CUDAWARP_H
#define CUDA_RUN_CUDAWARP_H
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

/**
* @brief Struct representing a 2x3 transformation matrix for affine warp.
*/
extern "C" struct TransformMatrix {
    float3 matrix[2];   // The 2x3 transformation matrix for affine warp.
    int    lastWidth;   // Width of the last processed source image.
    int    lastHeight;  // Height of the last processed source image.

    /**
     * @brief Updates the warp matrix based on the change in source and target image dimensions.
     *
     * @param fromWidth Width of the source image.
     * @param fromHeight Height of the source image.
     * @param toWidth Width of the target image.
     * @param toHeight Height of the target image.
     */
    void update(int fromWidth, int fromHeight, int toWidth, int toHeight);

    /**
     * @brief Transforms a point using the warp matrix.
     *
     * @param x X-coordinate of the point.
     * @param y Y-coordinate of the point.
     * @param[out] ox Transformed X-coordinate.
     * @param[out] oy Transformed Y-coordinate.
     */
    void transform(float x, float y, float* ox, float* oy) const;
};

/**
* @brief Applies an affine warp transformation using CUDA.
*
* @param input Pointer to the input image data.
* @param inputWidth Width of the input image.
* @param inputHeight Height of the input image.
* @param output Pointer to the output image data.
* @param outputWidth Width of the output image.
* @param outputHeight Height of the output image.
* @param matrix Affine transformation matrix.
* @param stream CUDA stream for asynchronous execution (optional).
*/
extern "C" void cudaWarpAffine(
        uint8_t* input, uint32_t inputWidth, uint32_t inputHeight,
        float* output, uint32_t outputWidth, uint32_t outputHeight, float3 matrix[2], cudaStream_t stream);

#endif //CUDA_RUN_CUDAWARP_H