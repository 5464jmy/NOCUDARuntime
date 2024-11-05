#ifndef RUNTIME_H
#define RUNTIME_H

// Windows平台下的DLL导出和导入宏定义
#ifdef _WIN32
#ifdef DLL_EXPORT
#define API __declspec(dllexport)  // 导出符号
#else
#define API __declspec(dllimport)  // 导入符号
#endif
#else
#define API  // 如果不是在Windows平台上，API定义为空
#endif

// 引入了所需的库头文件
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>

// CUDA和TensorRT相关库
#include <windows.h>
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <opencv2//opencv.hpp>
#include <device_launch_parameters.h>

// 自定义头文件
#include "cudaWarp.h"
#include "tensor.h"
#include "types.h"
#include "core.h"

using namespace std;

// 使用API宏确保在共享库中导出类
class API RuntimeWithGraph {
public:
    // 构造函数，接受共享内存名称、输入宽度和引擎路径
    RuntimeWithGraph(string& shmName, const vector<int>& shapes, string& enginePath, bool ultralytics);

    // 构造函数，接受图像指针、输入宽度和引擎路径
    RuntimeWithGraph(void* image_ptr, const vector<int>& shapes, string& enginePath, bool ultralytics);

    // 成员函数的声明
    void setImagePtr(void* image_ptr);
    void setShmName(string& shmName);
    string getShmName();
    void setEnginePath(string& shmName, bool ultralytics_in);
    string getEnginePath();
    void setShapes(const vector<int>& shapes);
    vector<int> getShapes();

    void InitTensors();  // 初始化张量

    void createSharedMemory();  // 创建共享内存

    void pointSharedMemory();  // 指向共享内存

    void createGraph();  // 创建CUDA图

    void predict();  // 执行预测

    ~RuntimeWithGraph();  // 析构函数

    // 公有成员变量
    string shm_name{};
    string engine_path{};
    bool ultralytics{};
    int height{};
    int width{};
    int imageWidth{};
    int imageHeight{};
    int imageChannels{};

    nvinfer1::Dims32 input_dims{};  // 输入张量维度
    nvinfer1::Dims32 output_dims{}; // 输出张量维度

    Tensor output_Tensor{};  // 输出张量
    void* host_ptr{nullptr}; // 主机内存指针

private:
    HANDLE hMapFile{};  // 文件映射句柄，用于共享内存

    std::shared_ptr<EngineContext> engineCtx{};  // 引擎上下文指针

    cudaGraphExec_t inferGraphExec{};  // CUDA图执行实例
    cudaStream_t inferStream{nullptr}; // CUDA流
    cudaGraph_t inferGraph{};  // CUDA图

    Tensor input_Tensor{};  // 输入张量
    int64_t input_bytes{};  // 输入张量的字节数
    int64_t output_bytes{}; // 输出张量的字节数

    std::shared_ptr<Tensor> imageTensor{}; // 图像张量的智能指针
    int64_t imageSize{};  // 图像大小

    TransformMatrix transforms{};  // 用于图像变换的矩阵
};

#endif // RUNTIME_H