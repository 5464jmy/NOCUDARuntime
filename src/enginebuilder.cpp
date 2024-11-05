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
                 bool ultralytics = false){
    // 创建 TensorRT 的日志记录器以跟踪输出信息
    logger logger;

    // 创建 TensorRT Builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << (unsigned int) nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto config = builder->createBuilderConfig();
    auto parser = nvonnxparser::createParser(*network, logger);

    // 解析 ONNX 模型
    if (!parser->parseFromFile(onnxFilePath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        throw std::runtime_error("Failed to parse ONNX file");  // 如果解析失败，抛出异常
    }

    // 设置内存池限制，以1GB为最大工作空间
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, (1U << exponent) * multiplier);

    // 如果硬件支持 FP16 并且用户希望使用半精度，则启用 FP16 模式
    if (builder->platformHasFastFp16() && half) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // 序列化网络到 IHostMemory
    nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
    if (!plan) {
        throw std::runtime_error("Failed to build serialized network");  // 如果序列化失败，抛出异常
    }

    // 打开文件流以二进制方式写入引擎数据
    std::ofstream file(engineFilePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file for writing");  // 如果文件打开失败，抛出异常
    }

    if (ultralytics){
        // 使用 nlohmann::json 创建 JSON 对象
        // 创建一个 JSON 对象以存储模型的元数据
        nlohmann::json metadata;
        metadata["author"] = "Ultralytics";
        metadata["license"] = "AGPL-3.0 https://ultralytics.com/license";
        metadata["stride"] = 32;
        metadata["task"] = "pose";
        metadata["batch"] = 1;
        metadata["imgsz"] = {640, 640};
        metadata["names"] = {{"0", "CT"}, {"1", "T"}};
        metadata["kpt_shape"] = {1, 3};
        // 写入元数据到文件
        std::string metadataStr = metadata.dump(4);  // 将 JSON 元数据格式化为字符串
        size_t metaSize = metadataStr.size();
        file.write(reinterpret_cast<const char*>(&metaSize), sizeof(metaSize));  // 写入元数据大小
        file.write(metadataStr.c_str(), metaSize);  // 写入元数据内容
    }

    // 写入序列化的引擎数据
    file.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    // 清理资源，释放所有动态分配的对象
    plan->destroy();
    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
}
