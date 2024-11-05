#pragma once

#include <cuda_runtime.h>
#include <iostream>

// 定义用于动态库导出的宏，根据平台条件进行定义。
#ifdef ENABLE_DEPLOY_BUILDING_DLL
#if defined(_WIN32)
#define DEPLOY_DECL __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define DEPLOY_DECL __attribute__((visibility("default")))
#else
#define DEPLOY_DECL
#endif
#else
#define DEPLOY_DECL
#endif

/**
 * @brief 检查 CUDA API 调用的返回值，并在发生错误时输出错误信息。
 *
 * @param code CUDA API 的返回值。
 * @param file 调用发生的文件名。
 * @param line 调用发生的行号。
 * @return 如果没有错误发生则返回 true，发生错误返回 false。
 */
inline bool cudaError(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error:\n";
        std::cerr << "    File:       " << file << "\n";
        std::cerr << "    Line:       " << line << "\n";
        std::cerr << "    Error code: " << code << "\n";
        std::cerr << "    Error text: " << cudaGetErrorString(code) << "\n";
        return false;
    }
    return true;
}

/**
 * @brief 简化 CUDA 错误检查的宏。
 *
 * 该宏封装了 `cudaError` 函数，提供了一种简洁的方法来检查 CUDA API 调用。
 * 它评估给定的 CUDA API 调用 `code`，如果返回错误，`cudaError` 函数会被调用以输出错误信息。
 *
 * @param code 要执行和检查错误的 CUDA API 调用。
 */
#define CUDA(code) cudaError((code), __FILE__, __LINE__)
