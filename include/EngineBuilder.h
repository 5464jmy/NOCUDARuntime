#ifndef ENGINE_BUILDER_H
#define ENGINE_BUILDER_H
#ifdef _WIN32
#ifdef DLL_EXPORT
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif
#else
#define API
#endif
#include <string>
#include <nlohmann/json.hpp>  // 使用 nlohmann::json 需要包含头文件

// 声明外部符号 buildEngine 函数
extern API void buildEngine(std::string& onnxFilePath, std::string& engineFilePath, bool half = false);

#endif // ENGINE_BUILDER_H
