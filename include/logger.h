#ifndef LOGGER_H
#define LOGGER_H

#include <NvInfer.h>
#include <iostream>

// 自定义 TensorRT Logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 屏蔽 kINFO 级别以下的信息
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

#endif // LOGGER_H
