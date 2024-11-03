//
// Created by 27823 on 2024/9/28.
//

#ifndef CUDA_RUN_CORE_H
#define CUDA_RUN_CORE_H
#pragma once

#include <NvInferPlugin.h>

#include <memory>


class TrtLogger : public nvinfer1::ILogger {
private:
    nvinfer1::ILogger::Severity mSeverity; /**< Severity level for logging. */

public:

    explicit TrtLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) : mSeverity(severity) {}
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};

/**
 * @brief Manages the TensorRT engine and execution context.
 */
class EngineContext {
private:
    TrtLogger mLogger{ nvinfer1::ILogger::Severity::kERROR }; /**< Logger for handling TensorRT messages. */
    void destroy();

public:
    std::shared_ptr<nvinfer1::IExecutionContext> mContext = nullptr; /**< Execution context for TensorRT engine. */
    std::shared_ptr<nvinfer1::ICudaEngine>       mEngine = nullptr; /**< TensorRT engine. */
    std::shared_ptr<nvinfer1::IRuntime>          mRuntime = nullptr; /**< TensorRT runtime. */


    EngineContext() {
        initLibNvInferPlugins(&mLogger, ""); /**< Initializes TensorRT plugins with custom logger. */
    }
    EngineContext(const EngineContext&)            = default;
    EngineContext(EngineContext&&)                 = delete;
    EngineContext& operator=(const EngineContext&) = default;
    EngineContext& operator=(EngineContext&&)      = delete;
    ~EngineContext() {
        destroy(); /**< Destroys the EngineContext object and releases associated resources. */
    }
    bool construct(const void* data, size_t size);
};

#endif //CUDA_RUN_CORE_H
