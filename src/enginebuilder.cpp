#include "enginebuilder.h"
#include "logger.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <stdexcept>

void buildEngine(const std::string& onnxFilePath,
                 const std::string& engineFilePath,
                 int multiplier = 1,
                 int exponent = 22,
                 bool half = false,
                 bool ultralytics = false) {
    // 创建日志记录器
    logger logger;

    // 创建 TensorRT Builder
    auto builder = nvinfer1::createInferBuilder(logger);
    if (!builder) throw std::runtime_error("Failed to create TensorRT builder");

    // 创建网络定义
    auto network = builder->createNetworkV2(1U << (unsigned int)nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    if (!network) {
        builder->destroy();
        throw std::runtime_error("Failed to create network definition");
    }

    // 创建配置
    auto config = builder->createBuilderConfig();
    if (!config) {
        network->destroy();
        builder->destroy();
        throw std::runtime_error("Failed to create builder config");
    }

    // 创建 ONNX 解析器
    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser) {
        config->destroy();
        network->destroy();
        builder->destroy();
        throw std::runtime_error("Failed to create ONNX parser");
    }

    // 解析 ONNX 文件
    if (!parser->parseFromFile(onnxFilePath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        parser->destroy();
        config->destroy();
        network->destroy();
        builder->destroy();
        throw std::runtime_error("Failed to parse ONNX file");
    }

    // 设置内存限制
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, (1U << exponent) * multiplier);

    // 检查是否支持 FP16，并根据需求启用
    if (builder->platformHasFastFp16() && half) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // 构建序列化网络
    auto plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        parser->destroy();
        config->destroy();
        network->destroy();
        builder->destroy();
        throw std::runtime_error("Failed to build serialized network");
    }

    // 打开文件流写入引擎文件
    std::ofstream file(engineFilePath, std::ios::binary);
    if (!file) {
        plan->destroy();
        parser->destroy();
        config->destroy();
        network->destroy();
        builder->destroy();
        throw std::runtime_error("Failed to open engine file for writing");
    }

    // 如果 ultralytics 模式启用，写入元数据
    if (ultralytics) {
        nlohmann::json metadata = {
                {"author", "Ultralytics"},
                {"license", "AGPL-3.0 https://ultralytics.com/license"},
                {"stride", 32},
                {"task", "pose"},
                {"batch", 1},
                {"imgsz", {640, 640}},
                {"names", {{"0", "CT"}, {"1", "T"}}},
                {"kpt_shape", {1, 3}}
        };

        std::string metadataStr = metadata.dump(4);  // 格式化 JSON
        int32_t metaSize = static_cast<int32_t>(metadataStr.size());// 确保 metaSize 是 4 字节有符号整数
        file.write(reinterpret_cast<const char*>(&metaSize), sizeof(metaSize));  // 写入元数据大小
        file.write(metadataStr.data(), metaSize);  // 写入元数据内容
    }

    // 写入序列化引擎数据
    file.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    // 释放所有资源
    plan->destroy();
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}
