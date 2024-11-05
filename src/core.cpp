#include <iostream>
#include "core.h"

/**
 * @brief TrtLogger 用于处理和输出 TensorRT 日志消息。
 */
void TrtLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    // 过滤掉低于当前设置级别的日志
    if (severity > mSeverity) return;

    // 根据严重级别输出不同的消息前缀
    switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
    }
    // 输出日志消息
    std::cerr << msg << '\n';
}

/**
 * @brief EngineContext 类用于管理 TensorRT 推理引擎的上下文。
 */
void EngineContext::destroy() {
    // 重置上下文、引擎和运行时指针以销毁资源
    mContext.reset();
    mEngine.reset();
    mRuntime.reset();
}

/**
 * @brief 构建推理引擎上下文。
 *
 * @param data 反序列化 CUDA 引擎的数据指针
 * @param size 数据的大小
 * @return 如果构建成功则返回 true，否则返回 false。
 */
bool EngineContext::construct(const void* data, size_t size) {
    // 首先销毁当前的引擎
    destroy();

    if (data == nullptr || size == 0) return false;

    // 创建 Runtime 对象，带有自定义删除器
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(mLogger), [](nvinfer1::IRuntime* ptr) {
                if (ptr != nullptr) delete ptr;
            });
    if (mRuntime == nullptr) return false;

    // 反序列化 CUDA 引擎，带有自定义删除器
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(data, size),
            [](nvinfer1::ICudaEngine* ptr) {
                if (ptr != nullptr) delete ptr;
            });
    if (mEngine == nullptr) return false;

    // 创建执行上下文，带有自定义删除器
    mContext = std::shared_ptr<nvinfer1::IExecutionContext>(
            mEngine->createExecutionContext(), [](nvinfer1::IExecutionContext* ptr) {
                if (ptr != nullptr) delete ptr;
            });
    return mContext != nullptr;
}
