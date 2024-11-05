#include "EngineBuilder.h"
#include "Logger.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <stdexcept>

void buildEngine(std::string& onnxFilePath, std::string& engineFilePath, bool half) {
    // 使用 nlohmann::json 创建 JSON 对象
    nlohmann::json metadata;
    metadata["author"] = "Ultralytics";
    metadata["license"] = "AGPL-3.0 https://ultralytics.com/license";
    metadata["stride"] = 32;
    metadata["task"] = "pose";
    metadata["batch"] = 1;
    metadata["imgsz"] = {640, 640};
    metadata["names"] = {{"0", "CT"}, {"1", "T"}};
    metadata["kpt_shape"] = {1, 3};

    // TensorRT Logger
    Logger logger;

    // TensorRT Builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << (unsigned int) nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto config = builder->createBuilderConfig();
    auto parser = nvonnxparser::createParser(*network, logger);

    // 解析 ONNX 模型
    if (!parser->parseFromFile(onnxFilePath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        throw std::runtime_error("Failed to parse ONNX file");
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
    if (builder->platformHasFastFp16() && half) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // 序列化网络
    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        throw std::runtime_error("Failed to build serialized network");
    }

    // 写入引擎文件
    std::ofstream file(engineFilePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file for writing");
    }

    // 写入元数据
    std::string metadataStr = metadata.dump(4);  // 格式化输出为 JSON 字符串
    size_t metaSize = metadataStr.size();
    file.write(reinterpret_cast<const char*>(&metaSize), sizeof(metaSize));
    file.write(metadataStr.c_str(), metaSize);
    file.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    // 清理资源
    plan->destroy();
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}
