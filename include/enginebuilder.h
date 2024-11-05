#ifndef ENGINE_BUILDER_H
#define ENGINE_BUILDER_H

#ifdef _WIN32  // Windows 平台特定的导出设置
#ifdef DLL_EXPORT
#define API __declspec(dllexport)  /**< 如果定义了 DLL_EXPORT，则导出函数。 */
#else
#define API __declspec(dllimport)  /**< 如果未定义 DLL_EXPORT，则导入函数。 */
#endif
#else
#define API  /**< 非 Windows 平台不需要特殊标记。 */
#endif

#include <string>
#include <nlohmann/json.hpp>  // 包含 nlohmann/json.hpp 头文件以使用 JSON 功能

/**
 * @brief 构建 TensorRT 引擎。
 *
 * @param onnxFilePath 指定输入的 ONNX 模型文件路径。
 * @param engineFilePath 指定输出的 TensorRT 引擎文件路径。
 * @param half 指定是否使用半精度（FP16）模式。默认为 false。
 */
extern API void buildEngine(const std::string& onnxFilePath,
                            const std::string& engineFilePath,
                            int multiplier, int exponent, bool half, bool ultralytics);

#endif // ENGINE_BUILDER_H
