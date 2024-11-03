//
// Created by 27823 on 2024/9/29.
//#include "detection.h"
//
//int main(){
////    Runtime detect{};
////    detect.Runtime("image", 320, R"(F:\Pyqt_project\AimbobyUI_3\weights\best copy 4.engine)");
//    Runtime detect("image", 320, R"(F:\Pyqt_project\AimbobyUI_3\weights\best copy 4.engine)");
//    detect.predict();
//
//    return 0;
//}
//
//#include <iostream>
//#include <cstring>
//#include <Windows.h>
//#include <cuda_runtime.h>
//#include <vector>
////#include <opencv2/opencv.hpp> // 包含OpenCV的头文件
//
//int main() {
//    const char* shmName = "image";
//
//    // 检查共享内存是否成功打开
//    HANDLE hMapFile = OpenFileMappingA(FILE_MAP_READ, FALSE, shmName);
//    if (hMapFile == nullptr) {
//        std::cerr << "Could not open file mapping object: " << GetLastError() << std::endl;
//        return 1;
//    }
//
//    // 挂载共享内存
//    void* pBuf = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
//    if (pBuf == nullptr) {
//        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
//        CloseHandle(hMapFile);
//        return 1;
//    }
//
//    // CUDA部分：从共享内存复制数据到CPU内存
//    const size_t dataSize = 320 * 320 * 3; // 数据大小
////    while (TRUE){
////        // 使用 vector 自动管理内存（此处在CPU上处理）
////        std::vector<uint8_t> image(dataSize);
////        std::memcpy(image.data(), pBuf, dataSize * sizeof(uint8_t));
////
////        // 创建 OpenCV 矩阵并展示图片
////        cv::Mat output_image(320, 320, CV_8UC3, image.data()); //使用 8 位无符号整型
////        cv::imshow("input_image", output_image);
////        // 检查是否按下了键，按 'q' 键退出循环
////        if (cv::waitKey(1) == 'q') {
////            break;
////        }
////    }
//
//
//    // 清理和释放资源
//    UnmapViewOfFile(pBuf);
//    CloseHandle(hMapFile);
//
//    return 0;
//}
#include "EngineBuilder.h"
#include <iostream>

int main() {
    std::string onnxFilePath = R"(D:\Base\ultralytics-modifiy\runs\pose\train6\weights\best.onnx)";
    std::string engineFilePath = R"(D:\Base\ultralytics-modifiy\runs\pose\train6\weights\best.engine)";

    try {
        buildEngine(onnxFilePath, engineFilePath, true);
        std::cout << "Engine built and written to file successfully." << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}
