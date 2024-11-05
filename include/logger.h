#ifndef LOGGER_H
#define LOGGER_H

#include <NvInfer.h>
#include <iostream>

// 自定义 TensorRT logger
class logger : public nvinfer1::ILogger {
public:
    /**
     * @brief 重写日志记录函数，根据消息的严重性输出日志信息。
     *
     * @param severity 日志消息的严重性等级。
     * @param msg 要记录的日志消息。
     */
    void log(Severity severity, const char* msg) noexcept override {
        // 屏蔽 kINFO 级别以下的信息
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

#endif // LOGGER_H
