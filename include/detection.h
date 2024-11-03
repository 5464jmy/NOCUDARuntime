#ifndef RUNTIME_H
#define RUNTIME_H

#ifdef _WIN32
#ifdef DLL_EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif
#else
#define API
#endif

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>

#include <windows.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cudaWarp.h"
#include "tensor.h"
#include "types.h"
#include "core.h"

class API Runtime {
public:
    Runtime(std::string shmName, int inputWidth, std::string enginePath);
    void createSharedMemory();
    void pointSharedMemory();
    void setupTensors();
    void createGraph();

    void predict();

    ~Runtime();

    std::string shm_name{};
    std::string engine_path{};
    int height{};
    int width{};
    int imageWidth{};

    nvinfer1::Dims32 input_dims{};
    nvinfer1::Dims32 output_dims{};

    Tensor output_Tensor{};
private:
    void* host_ptr{nullptr};
    HANDLE hMapFile{};

    std::shared_ptr<EngineContext> engineCtx{};

    cudaGraphExec_t inferGraphExec{};
    cudaStream_t inferStream{nullptr};
    cudaGraph_t inferGraph{};

    Tensor input_Tensor{};
    int64_t input_bytes{};

    int64_t output_bytes{};

    std::shared_ptr<Tensor> imageTensor{};
    int imageSize{};

    TransformMatrix transforms{};
};

#endif // RUNTIME_H
