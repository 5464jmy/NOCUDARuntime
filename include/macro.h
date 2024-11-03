//
// Created by 27823 on 2024/9/29.
//

#pragma once

#include <cuda_runtime.h>

#include <iostream>

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
* @brief Macro for simplified CUDA error checking.
*
* This macro wraps the `cudaError` function, allowing easy and concise checking
* of CUDA API calls. It evaluates the given CUDA API call `code`, and if it returns
* an error, the `cudaError` function is called to print error information.
*
* @param code The CUDA API call to be executed and checked for errors.
*/
#define CUDA(code) cudaError((code), __FILE__, __LINE__)

