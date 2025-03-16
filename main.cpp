
//#include "runtime.h"
//int main(){
//    std::string shmName = "image";
//    std::vector<int> shapes = {320, 320, 3};
//    std::string enginePath = R"(E:\Pyqt_project\AimbobyUI_5\weights\best1.engine)";
//
//    cv::Mat image1 = cv::imread(R"(E:\Pyqt_project\AimbobyUI_5\app\resource\images\7.jpg)");
//    cv::resize(image1, image1, cv::Size(320,320));
//
//    RuntimeWithGraph detect(image1.data, shapes, enginePath, false);
//    auto* output_ptr = static_cast<float*>(detect.output_Tensor.host());  // 获取指针
//    detect.predict();
//
////    enginePath = R"(E:\Pyqt_project\AimbobyUI_4\weights\best.engine)";
////    detect.setEnginePath(enginePath, true);
////    detect.predict();
////
////    cv::Mat image2 = cv::imread(R"(E:\Pyqt_project\AimbobyUI_4\app\resource\images\7.jpg)");
////    cv::resize(image2, image2, cv::Size(320,320));
////    detect.setImagePtr(image2.data);
////    detect.predict();
//
//
////    cv::Mat image = cv::imread(R"(E:\Pyqt_project\AimbobyUI_4\app\resource\images\8.jpg)");
////    cv::Mat image1 = cv::Mat(320,320,CV_8UC3, detect.host_ptr);
////    cv::resize(image, image1, cv::Size(320,320));
////
////    detect.predict();
//
//    return 0;
//}
////
////#include <iostream>
////#include <cstring>
////#include <Windows.h>
////#include <cuda_runtime.h>
////#include <vector>
//////#include <opencv2/opencv.hpp> // 包含OpenCV的头文件
////
////int main() {
////    const char* shmName = "image";
////
////    // 检查共享内存是否成功打开
////    HANDLE hMapFile = OpenFileMappingA(FILE_MAP_READ, FALSE, shmName);
////    if (hMapFile == nullptr) {
////        std::cerr << "Could not open file mapping object: " << GetLastError() << std::endl;
////        return 1;
////    }
////
////    // 挂载共享内存
////    void* pBuf = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
////    if (pBuf == nullptr) {
////        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
////        CloseHandle(hMapFile);
////        return 1;
////    }
////
////    // CUDA部分：从共享内存复制数据到CPU内存
////    const size_t dataSize = 320 * 320 * 3; // 数据大小
//////    while (TRUE){
//////        // 使用 vector 自动管理内存（此处在CPU上处理）
//////        std::vector<uint8_t> image(dataSize);
//////        std::memcpy(image.data(), pBuf, dataSize * sizeof(uint8_t));
//////
//////        // 创建 OpenCV 矩阵并展示图片
//////        cv::Mat output_image(320, 320, CV_8UC3, image.data()); //使用 8 位无符号整型
//////        cv::imshow("input_image", output_image);
//////        // 检查是否按下了键，按 'q' 键退出循环
//////        if (cv::waitKey(1) == 'q') {
//////            break;
//////        }
//////    }
////
////
////    // 清理和释放资源
////    UnmapViewOfFile(pBuf);
////    CloseHandle(hMapFile);
////
////    return 0;
////}
//
//
//#include "enginebuilder.h"
//#include <iostream>
//
//int main() {
//    std::string onnxFilePath = R"(D:\Base\D-FINE\runs\train\best_stg1.onnx)";
//    std::string engineFilePath = R"(best.engine)";
//
//    try {
//        buildEngine(onnxFilePath, engineFilePath, 1, 22, false, true);
//        std::cout << "Engine built and written to file successfully." << std::endl;
//    } catch (const std::exception& ex) {
//        std::cerr << "Error: " << ex.what() << std::endl;
//    }
//
//    return 0;
//}
#include "runtime.h"


int main() {
    cudaGraph_t inferGraph{};  // CUDA图
    cudaGraphExec_t inferGraphExec{};  // CUDA图执行实例
    cudaStream_t inferStream{nullptr};
    cudaStreamCreate(&inferStream);  // 创建CUDA流
    cv::Mat image = cv::imread(R"(D:\Base\cutImg\2.jpg)");
    void* imagePtr = nullptr;    /**< 指向设备内存的指针 */
    cudaMalloc(&imagePtr, image.cols * image.rows * image.channels());
    void* hostPtr = nullptr;      /**< 指向主机内存的指针 */
    void* devicePtr = nullptr;    /**< 指向设备内存的指针 */
    cudaMallocHost(&hostPtr, 1 * 8 * 640 * 640 * image.channels());
    cudaMalloc(&devicePtr, 1 * 8 * 640 * 640 * image.channels());


    WarpAffine warpAffine = WarpAffine(imagePtr, image.cols, image.rows,
                                       devicePtr, 640, 640, inferStream);

    auto re = cudaMemcpyAsync(imagePtr, image.data, image.cols * image.rows * image.channels(), cudaMemcpyHostToDevice, inferStream);
    warpAffine.cudaCutImg();
    re =cudaMemcpyAsync(hostPtr, devicePtr, 1 * 8 * 640 * 640 * image.channels(), cudaMemcpyDeviceToHost, inferStream);
    cv::Mat image2(640, 640 * 1, CV_8UC3, static_cast<uint8_t*>(hostPtr) + 7 * 640 * 640 * image.channels());
    cv::imshow("image", image2);
    cv::waitKey(0);
    return 0;
}
