#ifndef CUDA_RUN_CORE_H
#define CUDA_RUN_CORE_H
#pragma once

#include <NvInferPlugin.h>
#include <memory>

/**
 * @brief TensorRT 的自定义日志记录器，用于处理日志消息。
 */
class TrtLogger : public nvinfer1::ILogger {
private:
    nvinfer1::ILogger::Severity mSeverity; /**< 用于过滤日志消息的严重性级别。 */

public:
    /**
     * @brief TrtLogger 的构造函数。
     * @param severity 要记录的消息的最低严重性级别。
     */
    explicit TrtLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
            : mSeverity(severity) {}

    /**
     * @brief 使用给定的严重性记录一条消息。
     * @param severity 日志消息的严重性。
     * @param msg 要记录的消息。
     */
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};

/**
 * @brief 管理 TensorRT 引擎和执行上下文。
 */
class EngineContext {
private:
    TrtLogger mLogger{ nvinfer1::ILogger::Severity::kERROR }; /**< 用于处理 TensorRT 消息的日志记录器实例。 */

    /**
     * @brief 清理与引擎上下文相关的资源。
     */
    void destroy();

public:
    std::shared_ptr<nvinfer1::IExecutionContext> mContext = nullptr; /**< TensorRT 引擎的执行上下文。 */
    std::shared_ptr<nvinfer1::ICudaEngine>       mEngine = nullptr; /**< TensorRT 引擎实例。 */
    std::shared_ptr<nvinfer1::IRuntime>          mRuntime = nullptr; /**< TensorRT 运行环境。 */

    /**
     * @brief 默认构造函数初始化 TensorRT 插件。
     */
    EngineContext() {
        initLibNvInferPlugins(&mLogger, ""); /**< 使用自定义日志记录器初始化 TensorRT 插件。 */
    }

    // 允许复制构造和赋值。
    EngineContext(const EngineContext&)            = default;
    EngineContext& operator=(const EngineContext&) = default;

    // 删除移动构造函数和移动赋值运算符。
    EngineContext(EngineContext&&)                 = delete;
    EngineContext& operator=(EngineContext&&)      = delete;

    /**
     * @brief 析构函数销毁 EngineContext 对象并释放相关资源。
     */
    ~EngineContext() {
        destroy(); /**< 调用 destroy 函数释放资源。 */
    }

    /**
     * @brief 从序列化的引擎数据构造 TensorRT 执行上下文。
     * @param data 指向序列化引擎数据的指针。
     * @param size 序列化引擎数据的大小，单位为字节。
     * @return 如果构造成功则返回 true，否则返回 false。
     */
    bool construct(const void* data, size_t size);
};

#endif // CUDA_RUN_CORE_H
