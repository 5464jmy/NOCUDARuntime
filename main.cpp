
//#include "runtime.h"
//int main(){
//    std::string shmName = "image";
//    std::vector<int> shapes = {320, 320, 3};
//    std::string enginePath = R"(E:\Pyqt_project\AimbobyUI_5\weights\best1.engine)";
//
//    cv::Mat image1 = cv::imread(R"(E:\Pyqt_project\AimbobyUI_5\app\resource\images\7.jpg)");
//    cv::resize(image1, image1, cv::Size(320,320));
//
//    RuntimeCG detect(image1.data, shapes, enginePath, false);
//    auto* output_ptr = static_cast<float*>(detect.outputTensor.host());  // 获取指针
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
//    std::string onnxFilePath = R"(C:\Users\5464jmy\Desktop\AimBot\yolo\weights\cs2\yolov11n-480.onnx)";
//    std::string engineFilePath = R"(C:\Users\5464jmy\Desktop\AimBot\yolo\weights\cs2\yolov11n-480.engine)";
//
//    try {
//        buildEngine(onnxFilePath, engineFilePath);
//        std::cout << "Engine built and written to file successfully." << std::endl;
//    } catch (const std::exception& ex) {
//        std::cerr << "Error: " << ex.what() << std::endl;
//    }
//
//    return 0;
//}


#include "Runtime.h"


int main() {
//    cudaGraph_t inferGraph{};  // CUDA图
//    cudaGraphExec_t inferGraphExec{};  // CUDA图执行实例
//    cudaStream_t inferStream{nullptr};
//    cudaStreamCreate(&inferStream);  // 创建CUDA流
//    void* imagePtr = nullptr;    /**< 指向设备内存的指针 */
//    cudaMalloc(&imagePtr, image.cols * image.rows * image.channels());
//    void* hostPtr = nullptr;      /**< 指向主机内存的指针 */
//    void* devicePtr = nullptr;    /**< 指向设备内存的指针 */
//    cudaMallocHost(&hostPtr, 1 * 8 * 640 * 640 * image.channels() * 4);
//    cudaMalloc(&devicePtr, 1 * 8 * 640 * 640 * image.channels() * 4);
//
//
//    WarpAffine warpAffine = WarpAffine(imagePtr, image.cols, image.rows,
//                                       devicePtr, 640, 640, inferStream);
//    cudaStreamBeginCapture(inferStream, cudaStreamCaptureModeGlobal);
//    auto re = cudaMemcpyAsync(imagePtr, image.data, image.cols * image.rows * image.channels(), cudaMemcpyHostToDevice, inferStream);
//    warpAffine.cudaCutImg();
//    re =cudaMemcpyAsync(hostPtr, devicePtr, 1 * 8 * 640 * 640 * image.channels() * 4, cudaMemcpyDeviceToHost, inferStream);
//    cudaStreamEndCapture(inferStream, &inferGraph);
//    cudaGraphInstantiate(&inferGraphExec, inferGraph, nullptr, nullptr, 0);
//
//
//    cudaGraphLaunch(inferGraphExec, inferStream);
//    cudaStreamSynchronize(inferStream);
//
//    cv::Mat image2(640, 640, CV_32FC3, (static_cast<float *>(hostPtr)) +  7 * 640 * 640 * 3);
//    cv::imshow("image", image2);
//    cv::waitKey(0);


    cv::Mat image = cv::imread(R"(E:\CS2\AimbobyUI_5\app\resource\images\7.jpg)");
    // 缩放图像到 640x640
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(640, 640));

    // 将图像转换为浮点型并归一化到 [0, 1]
    cv::Mat floatImage;
    resizedImage.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    // 将 [640, 640, 3] 转换为 [3, 640, 640]
    std::vector<cv::Mat> channels(3);
    cv::split(floatImage, channels); // 分离通道

    // 将通道数据重新排列为 [3, 640, 640]
    cv::Mat chwImage(3, 640 * 640, CV_32F);
    for (int i = 0; i < 3; ++i) {
        // 将每个通道展平并复制到 chwImage 的对应行
        channels[i].reshape(1, 1).copyTo(chwImage.row(i));
    }

    // 如果需要，可以将 chwImage 转换为 3x640x640 的连续内存
    chwImage = chwImage.reshape(3, 640);

    std::string enginePath = R"(E:\CS2\AimbobyUI_5\weights\best1.engine)";
    Base detect(enginePath);
    detect.predict((float *)chwImage.data);
    int i = 0;
    do{
        detect.predict((float *)chwImage.data);
        i = i + 1;
    } while (i <= 10);


//    string shn_name = "image";
//    cv::resize(image, image, cv::Size(320, 320));
////    cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
//    std::string enginePath = R"(E:\CS2\AimbobyUI_5\weights\best1.engine)";
//    std::vector<int> shapes = {image.cols, image.rows, 3};
//
////    Base detect(enginePath);
////    nvinfer1::Dims3 dims{image.cols, image.rows, 4};
////    detect.predict(image.data, dims);
//
//    Runtime detect(image.data, shapes, enginePath, true);
//    detect.setImagePtr(image.data);
//    detect.predict();
////
//    int i = 0;
////    do{
////        cudaGraphLaunch(detect.inferGraphExec, detect.inferStream);
////        cudaStreamSynchronize(detect.inferStream);
////        detect.predict();
////        i = i + 1;
////    } while (i <= 10);
//
//    do{
//        detect.predict();
//        i = i + 1;
//    } while (i <= 10);
//    do{
//        detect.predict(image.data, dims);
//        i = i + 1;
//    } while (i <= 10);

//    string shn_name = "image";
//    cv::Mat image = cv::imread(R"(D:\github\NOCUDARuntime1\2.jpg)");
////    cv::resize(image, image, cv::Size(320, 320));
////    cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
//    std::string enginePath = R"(D:\github\NOCUDARuntime1\best1.engine)";
//    std::vector<int> shapes = {image.cols, image.rows, 3};
//
//    RuntimeCG detect(image.data, shapes, enginePath, true, 8);
////    nvinfer1::Dims dims{3, image.cols, image.rows, 4};
//    nvinfer1::Dims3 dims{image.cols, image.rows, 3};
//    detect.predict();
//    int i = 0;
//    do{
//        detect.predict();
//        i = i + 1;
//    } while (i <= 10);




//    cudaGraphLaunch(detect.inferGraphExec, detect.inferStream);
//    cudaStreamSynchronize(detect.inferStream);
//
//    for (int i= 0 ;i<10;i++) {
//        cudaGraphLaunch(detect.inferGraphExec, detect.inferStream);
//        cudaStreamSynchronize(detect.inferStream);
//
//        auto re =cudaMemcpyAsync(detect.inputTensor.host(), detect.inputTensor.device(),
//                                 8 * 640 * 640 * 3 * 4, cudaMemcpyDeviceToHost, detect.inferStream);
//        cv::Mat image2(640, 640, CV_32FC3, (static_cast<float *>(detect.inputTensor.host())) +  7 * 640 * 640 * 3);
//        cv::imshow("image", image2);
//        cv::waitKey(0);
//    }

//    std::string enginePath = R"(D:\github\NOCUDARuntime1\best.engine)";
//    std::vector<int> shapes = {image.cols, image.rows, 3};
//    Runtime detect(image.data, shapes, enginePath, false);
//    for (int i= 0 ;i<10;i++) {
//        detect.predict();
//    }


    return 0;
}
