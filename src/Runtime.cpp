#include "Runtime.h"

// 连接到共享内存
void pointSharedMemory(string& shm_name, uint64_t& imageSize, void** host_ptr, HANDLE& hMapFile) {
    // 打开命名文件映射对象
    hMapFile = OpenFileMapping(
            FILE_MAP_ALL_ACCESS, // 可读写
            FALSE,               // 不继承
            shm_name.c_str());   // 名称

    if (hMapFile == nullptr) {
        cerr << "Could not open file mapping object" << endl;
        return;
    }

    // 映射这段共享内存以获得一个可访问的指针
    *host_ptr = MapViewOfFile(
            hMapFile,            // 映射对象
            FILE_MAP_ALL_ACCESS, // 读写权限
            0,                   // 文件偏移高
            0,                   // 文件偏移低
            imageSize);          // 映射大小

    if (host_ptr == nullptr) {
        cerr << "Could not map view of file: " << GetLastError() << endl;
        CloseHandle(hMapFile);
        hMapFile = nullptr;
    }
}
// 创建CUDA图形执行计划
void RuntimeCG::createGraph() {
    cudaStreamBeginCapture(inferStream, cudaStreamCaptureModeGlobal);
    cudaMemcpyAsync(imageTensor->device(), host_ptr, imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice, inferStream);
    warpAffine->cudaWarpAffine();
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw runtime_error("Failed to enqueueV3 during graph creation");
    }
    cudaMemcpyAsync(outputTensor.host(), outputTensor.device(), output_bytes, cudaMemcpyDeviceToHost, inferStream);
//    cudaMemcpyAsync(inputTensor.host(), inputTensor.device(), input_bytes, cudaMemcpyDeviceToHost, inferStream);
    cudaStreamEndCapture(inferStream, &inferGraph);
    cudaGraphInstantiate(&inferGraphExec, inferGraph, nullptr, nullptr, 0);
}
//Runtime::Runtime(string& shmName, const vector<int>& shapes, string& enginePath,
//                 bool BGR, bool ultralytics, bool dynamic, uint32_t batch):
//        Base(enginePath, ultralytics, dynamic, batch){
//     // 初始化参数
//    imageWidth = shapes[0];
//    imageHeight = shapes[1];
//    imageChannels = shapes[2];
//    imageSize = imageWidth * imageHeight * imageChannels;
//
//    shm_name = shmName;
//    pointSharedMemory(shm_name, imageSize, &host_ptr, hMapFile);
////    Init();
//
////    imageTensor->host(imageSize * sizeof(uint8_t));
//    imageTensor->device(imageSize * sizeof(uint8_t));
//    warpAffine = new WarpAffine(imageTensor->device(), imageWidth, imageHeight,
//                                inputTensor.device(), width, height,
//                                imageChannels, BGR,
//                                inferStream);
//}


//Runtime::Runtime(void* image_ptr, const vector<int>& shapes, string& enginePath,
//                 bool BGR, bool ultralytics, bool dynamic, uint32_t batch):
//        Base(enginePath, ultralytics, dynamic, batch){
//    // 初始化参数
//    this->dynamic = dynamic;
//    this->batch = batch;
//    imageWidth = shapes[0];
//    imageHeight = shapes[1];
//    imageChannels = shapes[2];
//    imageSize = imageWidth * imageHeight * imageChannels;
//    host_ptr = image_ptr;
////    Init();
//
////    imageTensor->host(imageSize * sizeof(uint8_t));
//    imageTensor->device(imageSize * sizeof(uint8_t));
//    warpAffine = new WarpAffine(imageTensor->device(), imageWidth, imageHeight,
//                                inputTensor.device(), width, height,
//                                imageChannels, BGR,
//                                inferStream);
//}

//Runtime:: ~Runtime(){
//    // 解除和关闭共享内存
//    if (hMapFile != nullptr) {
//        UnmapViewOfFile(host_ptr);
//        CloseHandle(hMapFile);
//        host_ptr = nullptr;
//        hMapFile = nullptr;
//    }
//    // 释放其他资源
//    engineCtx.reset(); // 重置智能指针
//    imageTensor.reset();
//}
//void Runtime::predict(){
//    // 将图像数据从主机复制到设备
//    cudaMemcpyAsync(imageTensor->device(), host_ptr, imageSize, cudaMemcpyHostToDevice, inferStream);
//    // 执行图像扭曲变换
//    warpAffine->cudaWarpAffine();
////    warpAffine->cudaCutImg();
//    // 将推理操作入队
//    if (!engineCtx->mContext->enqueueV3(inferStream)) {
//        throw runtime_error("Failed to enqueueV3 during graph creation");
//    } // 将输出数据从设备复制到主机
//    cudaMemcpyAsync(outputTensor.host(), outputTensor.device(), output_bytes, cudaMemcpyDeviceToHost, inferStream);
//    cudaStreamSynchronize(inferStream);
//
////    cudaMemcpyAsync(inputTensor.host(), inputTensor.device(),
////                        input_bytes, cudaMemcpyDeviceToHost, inferStream);
////    cv::Mat image2(height, width, CV_32FC3, (static_cast<float *>(inputTensor.host())));
////    cv::imshow("image", image2);
////    cv::waitKey(0);
//}


//string Base::getEnginePath() {
//    return enginePath;
//}
void Base::setEnginePath(string &engine_path, bool ultralytics_in){
    // 释放现有资源
    engineCtx.reset(); // 重置引擎上下文指针
    ultralytics = ultralytics_in;
    engine_path = engine_path;  // 更新引擎路径
    InitTensor();  // 重新初始化张量
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw runtime_error("Failed to enqueueV3 before graph creation");
    }
    cudaStreamSynchronize(inferStream);
}

void RuntimeCG::setEnginePath(string &engine_path, bool ultralytics_in) {
    // 释放现有资源
    if (inferGraphExec != nullptr) {
        cudaGraphExecDestroy(inferGraphExec);
        inferGraphExec = nullptr;
    }
    if (inferGraph != nullptr) {
        cudaGraphDestroy(inferGraph);
        inferGraph = nullptr;
    }
    Runtime::setEnginePath(engine_path, ultralytics_in);
    createGraph();  // 创建CUDA图
}
// 获取共享内存名称
string Runtime::getShmName() {
    return shm_name;
}

void Runtime::setShmName(string &shmName) {
    // 释放现有资源
    if (hMapFile != nullptr) {
        UnmapViewOfFile(host_ptr);
        CloseHandle(hMapFile);
        host_ptr = nullptr;
        hMapFile = nullptr;
    }
    shm_name = shmName;
    pointSharedMemory(shm_name, imageSize, &host_ptr, hMapFile);
}

void RuntimeCG::setShmName(string &shmName) {
    if (inferGraphExec != nullptr) {
        cudaGraphExecDestroy(inferGraphExec);
        inferGraphExec = nullptr;
    }
    if (inferGraph != nullptr) {
        cudaGraphDestroy(inferGraph);
        inferGraph = nullptr;
    }
    Runtime::setShmName(shmName);
    createGraph();        // 创建CUDA图
}

void Runtime::setImagePtr(void* image_ptr){
    if (hMapFile != nullptr) {
        UnmapViewOfFile(host_ptr);
        CloseHandle(hMapFile);
        hMapFile = nullptr;
    }
    shm_name = "";
    host_ptr = image_ptr;
}

vector<int> Runtime::getShapes() {
    return vector<int>{imageWidth, imageHeight, imageChannels};
}

void RuntimeCG::setImagePtr(void* image_ptr) {
    if (inferGraphExec != nullptr) {
        cudaGraphExecDestroy(inferGraphExec);
        inferGraphExec = nullptr;
    }
    if (inferGraph != nullptr) {
        cudaGraphDestroy(inferGraph);
        inferGraph = nullptr;
    }
    Runtime::setImagePtr(image_ptr);
    createGraph();
}

void Runtime::setShapes(const vector<int> &shapes) {
    imageWidth = shapes[0];
    imageHeight = shapes[1];
    imageChannels = shapes[2];
    imageSize = imageWidth * imageHeight * imageChannels;
    imageTensor->device(imageSize * sizeof(uint8_t));
    warpAffine->transforms->update(imageWidth, imageHeight, width, height);
}
void RuntimeCG::setShapes(const vector<int> &shapes) {
    Runtime::setShapes(shapes);
    createGraph();
}


