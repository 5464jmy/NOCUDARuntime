#ifndef RUNTIME_H
#define RUNTIME_H

// Windows 平台下的 DLL 导出和导入宏定义
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
#include <opencv2//opencv.hpp>
#include <device_launch_parameters.h>

#include "cudaWarp.h"
#include "tensor.h"
#include "types.h"
#include "core.h"
using namespace std;
// 使用 API 宏确保在共享库中导出类
class API Runtime {
public:

    // 构造函数，接受共享内存名称、输入宽度和引擎路径
    Runtime(std::string& shmName,
            const vector<int>& shapes,
            std::string& enginePath,
            bool ultralytics);
    // 构造函数，接受共享内存名称、输入宽度和引擎路径
    Runtime(void* image_ptr,
            const vector<int>& shapes,
            std::string& enginePath,
            bool ultralytics);

    void InitCUDA();

    void setImagePtr(void* image_ptr);
    void setShmName(std::string& shmName);
    std::string getShmName();
    void setEnginePath(std::string& shmName);
    std::string getEnginePath();
    void setShapes(const vector<int>& shapes);
    vector<int> getShapes();

    // 创建共享内存
    void createSharedMemory();

    // 指向共享内存
    void pointSharedMemory();

    // 设置张量
    void setupTensors();

    // 创建 CUDA 图
    void createGraph();

    // 进行预测
    void predict();

    // 析构函数
    ~Runtime();

    // 公有成员变量
    std::string shm_name{};
    std::string engine_path{};
    bool ultralytics{};
    int height{};
    int width{};
    int imageWidth{};
    int imageHeight{};
    int imageChannels{};

    // 输入和输出张量的维度
    nvinfer1::Dims32 input_dims{};
    nvinfer1::Dims32 output_dims{};

    // 输出张量
    Tensor output_Tensor{};
    void* host_ptr{nullptr};         // 主机内存指针

private:
    HANDLE hMapFile{};               // 文件映射句柄，用于共享内存

    std::shared_ptr<EngineContext> engineCtx{};  // 引擎上下文指针

    cudaGraphExec_t inferGraphExec{}; // CUDA 图执行实例
    cudaStream_t inferStream{nullptr}; // CUDA 流
    cudaGraph_t inferGraph{};          // CUDA 图

    Tensor input_Tensor{};             // 输入张量
    int64_t input_bytes{};             // 输入张量的字节数

    int64_t output_bytes{};            // 输出张量的字节数

    std::shared_ptr<Tensor> imageTensor{}; // 图像张量的智能指针
    int imageSize{};                      // 图像大小

    TransformMatrix transforms{};         // 用于图像变换的矩阵
};

#endif // RUNTIME_H
