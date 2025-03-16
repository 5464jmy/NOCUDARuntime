#ifndef RUNTIME_H
#define RUNTIME_H

//// Windows 平台下的 DLL 导出和导入宏定义
//#ifdef EXPORT_DLL  // 如果在 DLL 项目中定义
//    #define API __declspec(dllexport)
//#else  // 如果在使用 DLL 的项目中定义
//    #define API __declspec(dllimport)
//#endif


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
#include <device_launch_parameters.h>

#include <opencv2//opencv.hpp>
// 自定义头文件
#include "cudaWarp.h"
#include "tensor.h"
#include "types.h"
#include "core.h"

using namespace std;

vector<char> loadFile(const string& filePath, bool ultralytics);
void pointSharedMemory(string& shm_name, uint64_t& imageSize, void** host_ptr, HANDLE& hMapFile);

class API Base{
public:
    Base(string& enginePath, bool ultralytics = false, bool dynamic = false, uint32_t batch=1);

    void Init();

    void InitTensor();  // 初始化张量

    void predict(float * image);  // 执行预测

    virtual ~Base();  // 析构函数

    virtual void setEnginePath(string &engine_path, bool ultralytics_in);

    string enginePath{};
    bool ultralytics{};
    int height{};
    int width{};

    bool  dynamic{false};
    uint32_t batch{1};

    nvinfer1::Dims32 inputDims{};
    nvinfer1::Dims32 outputDims{};

    Tensor inputTensor{};
    Tensor outputTensor{};
protected:
    std::shared_ptr<EngineContext> engineCtx{};
    cudaStream_t inferStream{nullptr};

    uint64_t input_bytes{};
    uint64_t output_bytes{};
    uint64_t imageSize{0};  // 图像大小
};

class API BaseWithWarpS : public Base{
public:
    BaseWithWarpS(string& enginePath, bool BGR= false, bool ultralytics = false, bool dynamic= false, uint32_t batch=1);;

    void predict(uint8_t* image, const nvinfer1::Dims3& dims);  // 执行预测
    virtual ~BaseWithWarpS() override;  // 析构函数
protected:
    uint64_t imageSize{0};  // 图像大小

    std::shared_ptr<Tensor> imageTensor{}; // 图像张量的智能指针
    WarpAffine* warpAffine{nullptr};
};

class API BaseWithWarpT : public Base{
public:
    BaseWithWarpT(const vector<int>& shapes, string& enginePath, bool BGR = false, bool ultralytics = false, bool dynamic= false, uint32_t batch=1);;

    bool BGR{false};
    int imageWidth{0};
    int imageHeight{0};
    int imageChannels{0};
    void predict(void* image);  // 执行预测
    virtual ~BaseWithWarpT();  // 析构函数
protected:

    std::shared_ptr<Tensor> imageTensor{}; // 图像张量的智能指针
    WarpAffine* warpAffine{nullptr};
};

class API Runtime : public BaseWithWarpT{

public:
    Runtime(const vector<int>& shapes, string& enginePath, bool BGR= false,
            bool ultralytics= false, bool dynamic=false, uint32_t batch=1);
    Runtime(string& shmName, const vector<int>& shapes, string& enginePath, bool BGR= false,
            bool ultralytics= false, bool dynamic=false, uint32_t batch=1);
    Runtime(void* image_ptr, const vector<int>& shapes, string& enginePath, bool BGR= false,
            bool ultralytics= false, bool dynamic=false, uint32_t batch=1);

    void predict();  // 执行预测
    virtual ~Runtime() ;  // 析构函数

    virtual void setImagePtr(void* image_ptr);
    virtual void setShmName(string& shmName);
    virtual string getShmName();
    virtual void setShapes(const vector<int> &shapes);
    virtual vector<int> getShapes();
    // 公有成员变量
    string shm_name;
    void* host_ptr{nullptr}; // 主机内存指针
protected:
    HANDLE hMapFile{};  // 文件映射句柄，用于共享内存
};

// 使用API宏确保在共享库中导出类
class API RuntimeCG : public Runtime {
public:
    RuntimeCG(const vector<int>& shapes, string& enginePath, bool BGR=false,
              bool ultralytics=false, bool dynamic=false, uint32_t batch=1);
    RuntimeCG(string& shmName, const vector<int>& shapes, string& enginePath, bool BGR=false,
              bool ultralytics=false, bool dynamic=false, uint32_t batch=1);
    RuntimeCG(void* imagePtr, const vector<int>& shapes, string& enginePath, bool BGR= false,
              bool ultralytics=false, bool dynamic=false, uint32_t batch=1);

    void createGraph();  // 创建CUDA图
    void predict() ;  // 执行预测
    virtual ~RuntimeCG() ;  // 析构函数

    // 成员函数的声明
    void setImagePtr(void* image_ptr) override;
    void setShmName(string& shmName) override;
    void setEnginePath(string &engine_path, bool ultralytics_in) override;
    void setShapes(const vector<int> &shapes) override;
private:
    // CUDA图执行实例
    cudaGraph_t inferGraph{};  // CUDA图
    cudaGraphExec_t inferGraphExec{};
};
#endif // RUNTIME_H


//// 使用API宏确保在共享库中导出类
//class API RuntimeCG : public Runtime {
//public:
////    RuntimeCG(string& shmName, const vector<int>& shapes, string& enginePath):
////            RuntimeCG(shmName, shapes, enginePath, false, false, 1){};
////    RuntimeCG(string& shmName, const vector<int>& shapes, string& enginePath, bool ultralytics):
////            RuntimeCG(shmName, shapes, enginePath, ultralytics, false, 1){};
////    RuntimeCG(string& shmName, const vector<int>& shapes, string& enginePath, bool dynamic, uint32_t batch):
////            RuntimeCG(shmName, shapes, enginePath, false, dynamic, batch){};
//    RuntimeCG(string& shmName, const vector<int>& shapes, string& enginePath, bool BGR= false,
//              bool ultralytics= false, bool dynamic=false, uint32_t batch=1);
//
////    RuntimeCG(void* image_ptr, const vector<int>& shapes, string& enginePath):
////            RuntimeCG(image_ptr, shapes, enginePath, false, false, 1){};
////    RuntimeCG(void* image_ptr, const vector<int>& shapes, string& enginePath, bool ultralytics):
////            RuntimeCG(image_ptr, shapes, enginePath, ultralytics, false, 1){};
////    RuntimeCG(void* image_ptr, const vector<int>& shapes, string& enginePath, bool dynamic, uint32_t batch):
////            RuntimeCG(image_ptr, shapes, enginePath, false, dynamic, batch){};
//    RuntimeCG(void* image_ptr, const vector<int>& shapes, string& enginePath, bool BGR= false,
//              bool ultralytics=false, bool dynamic=false, uint32_t batch=1);
//
//    void createGraph();  // 创建CUDA图
//    void predict() override ;  // 执行预测
//    ~RuntimeCG() override;  // 析构函数
//
//    // 成员函数的声明
//    void setImagePtr(void* image_ptr);
//    void setShmName(string& shmName);
//    string getShmName();
//    void setEnginePath(string &enginePath, bool ultralytics_in);
//    string getEnginePath();
//    void setShapes(const vector<int> &shapes);
//    vector<int> getShapes();
//private:
//    // CUDA图执行实例
//    cudaGraph_t inferGraph{};  // CUDA图
//    cudaGraphExec_t inferGraphExec{};
//};
//#endif // RUNTIME_H