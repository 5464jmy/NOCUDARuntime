#include "runtime.h"

// 从文件加载数据到字符向量
std::vector<char> loadFile(const std::string& filePath, bool ultralytics) {
    std::ifstream file(filePath, std::ios::binary);  // 以二进制方式打开文件
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filePath);
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (ultralytics){
        // 读取序列化引擎的大小
        int size = 4;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));

        // 计算总偏移
        size += sizeof(size);

        // 调整文件大小并设置文件位置
        fileSize -= size;
        file.seekg(size, std::ios::beg);
    }


    // 如果文件为空或大小为负，返回空向量
    if (fileSize <= 0) {
        return {};
    }

    // 将文件内容读取到向量中
    std::vector<char> fileContent(static_cast<std::size_t>(fileSize));
    file.read(fileContent.data(), fileSize);

    // 检查读取错误
    if (!file) {
        throw std::runtime_error("Error reading file: " + filePath);
    }

    return fileContent;
}
// 带有共享内存名的构造函数
Runtime::Runtime(std::string& shmName,
                 const vector<int>& shapes,
                 std::string& enginePath,
                 bool ultralytics) {
    this->ultralytics = ultralytics;
    imageWidth = shapes[0];
    imageHeight = shapes[1];
    imageChannels = shapes[2];
    imageSize = imageWidth * imageHeight * imageChannels;
    engine_path = enginePath;
    // 初始化图像张量
    imageTensor = std::make_shared<Tensor>();
    imageTensor->host(imageSize * sizeof(uint8_t));
    imageTensor->device(imageSize * sizeof(uint8_t));

    shm_name = shmName;
//    createSharedMemory();
    pointSharedMemory();   // 指向共享内存
    InitCUDA();            // 初始化CUDA
}

// 带有图像指针的构造函数
Runtime::Runtime(void* image_ptr,
                 const vector<int>& shapes,
                 std::string& enginePath,
                 bool ultralytics) {

    this->ultralytics = ultralytics;
    imageWidth = shapes[0];
    imageHeight = shapes[1];
    imageChannels = shapes[2];
    imageSize = imageWidth * imageHeight * imageChannels;
    imageTensor = std::make_shared<Tensor>();
    imageTensor->device(imageSize * sizeof(uint8_t));

    engine_path = enginePath;
    host_ptr = image_ptr;
    InitCUDA();
}

// 初始化CUDA资源
void Runtime::InitCUDA() {
    cudaStreamCreate(&inferStream);  // 创建CUDA流

    setupTensors();              // 设置张量

    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw std::runtime_error("Failed to enqueueV3 before graph creation");
    }
    cudaStreamSynchronize(inferStream);

    transforms.update(imageWidth, imageHeight, width, height);

    createGraph();               // 创建CUDA图
}
// 设置运行时的张量
void Runtime::setupTensors() {
    auto data = loadFile(engine_path, ultralytics);
    engineCtx = std::make_shared<EngineContext>();
    if (!engineCtx->construct(data.data(), data.size())) {
        throw std::runtime_error("Failed to construct engine context.");
    }
    int tensorNum = engineCtx->mEngine->getNbIOTensors();
    for (int i = 0; i < tensorNum; i++) {
        const char* name = engineCtx->mEngine->getIOTensorName(i);
        auto dtype = engineCtx->mEngine->getTensorDataType(name);
        auto typesz = getDataTypeSize(dtype);

        // 计算张量大小（字节）
        if ((engineCtx->mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)) {
            input_dims = engineCtx->mEngine->getTensorShape(name);
            height = input_dims.d[2];
            width = input_dims.d[3];
            input_bytes = calculateVolume(input_dims) * typesz;
            input_Tensor.host(input_bytes);
            engineCtx->mContext->setTensorAddress(name, input_Tensor.device(input_bytes));
        } else {
            output_dims = engineCtx->mEngine->getTensorShape(name);
            output_bytes = calculateVolume(output_dims) * typesz;
            engineCtx->mContext->setTensorAddress(name, output_Tensor.device(output_bytes));
            output_Tensor.host(output_bytes);
        }
    }
}

// 创建图形执行计划
void Runtime::createGraph() {
    cudaStreamBeginCapture(inferStream, cudaStreamCaptureModeGlobal);
    cudaMemcpyAsync(imageTensor->device(), host_ptr, imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice, inferStream);
    cudaWarpAffine(static_cast<uint8_t*>(imageTensor->device()), imageWidth, imageWidth,
                   static_cast<float*>(input_Tensor.device()), width, height,
                   transforms.matrix, inferStream);
    // 将推理操作入队
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw std::runtime_error("Failed to enqueueV3 during graph creation");
    }
    cudaMemcpyAsync(output_Tensor.host(), output_Tensor.device(), output_bytes, cudaMemcpyDeviceToHost, inferStream);
    cudaStreamEndCapture(inferStream, &inferGraph);
    cudaGraphInstantiate(&inferGraphExec, inferGraph, nullptr, nullptr, 0);
}

// 创建共享内存
void Runtime::createSharedMemory() {
    // 创建共享内存的映射文件
    hMapFile = CreateFileMappingA(
            INVALID_HANDLE_VALUE, // 使用系统分页文件
            nullptr,              // 默认安全性
            PAGE_READWRITE,       // 读写权限
            0,                    // 高位文件大小
            imageSize,            // 低位文件大小
            shm_name.c_str());    // 名字

    if (hMapFile == nullptr) {
        std::cerr << "Could not create file mapping object" << std::endl;
        return;
    }
}

// 连接到共享内存
void Runtime::pointSharedMemory() {
    // 打开命名文件映射对象
    hMapFile = OpenFileMapping(
            FILE_MAP_ALL_ACCESS, // 可读写
            FALSE,               // 不继承
            shm_name.c_str());   // 名称

    if (hMapFile == nullptr) {
        std::cerr << "Could not open file mapping object" << std::endl;
        return;
    }

    // 映射视图
    host_ptr = MapViewOfFile(
            hMapFile,            // 映射对象
            FILE_MAP_ALL_ACCESS, // 读写
            0,                   // 文件偏移高
            0,                   // 文件偏移低
            imageSize);          // 映射大小

    if (host_ptr == nullptr) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        hMapFile = nullptr;
    }
}

// 析构函数
Runtime::~Runtime() {
    if (inferGraphExec != nullptr) {
        cudaGraphExecDestroy(inferGraphExec);
        inferGraphExec = nullptr;
    }

    // 销毁CUDA图
    if (inferGraph != nullptr) {
        cudaGraphDestroy(inferGraph);
        inferGraph = nullptr;
    }

    // 销毁CUDA流
    if (inferStream != nullptr) {
        cudaStreamDestroy(inferStream);
        inferStream = nullptr;
    }

    if (hMapFile != nullptr){
        UnmapViewOfFile(host_ptr);
        CloseHandle(hMapFile);
        host_ptr = nullptr;
        hMapFile = nullptr;
    }

    // 释放其他资源
    engineCtx.reset();
    imageTensor.reset();
}

// 推理方法
void Runtime::predict() {
//    cudaMemcpyAsync(imageTensor->device(), host_ptr, imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice, inferStream);
//    cudaMemcpyAsync(imageTensor->host(), imageTensor->device(), imageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost, inferStream);
//    cv::Mat mat(imageWidth, imageHeight, CV_8UC3, imageTensor->host());
//    cv::imshow("IMAGE", mat);
    cudaGraphLaunch(inferGraphExec, inferStream);
    cudaStreamSynchronize(inferStream);
//    cudaMemcpyAsync(input_Tensor.host(), input_Tensor.device(), input_bytes, cudaMemcpyDeviceToHost, inferStream);
//    cv::Mat mat1(640, 640, CV_32FC3, input_Tensor.host());
//    cv::imshow("IMAGE1", mat1);
//    cv::waitKey(0);
}



// 获取共享内存名称
std::string Runtime::getShmName() {
    return shm_name;
}

// 设置共享内存名称
void Runtime::setShmName(std::string &shmName) {
    // 释放现有资源
    if (hMapFile != nullptr){
        UnmapViewOfFile(host_ptr);
        CloseHandle(hMapFile);
        host_ptr = nullptr;
        hMapFile = nullptr;
    }
    if (inferGraphExec != nullptr) {
        cudaGraphExecDestroy(inferGraphExec);
        inferGraphExec = nullptr;
    }
    if (inferGraph != nullptr) {
        cudaGraphDestroy(inferGraph);
        inferGraph = nullptr;
    }
    if (inferStream != nullptr) {
        cudaStreamDestroy(inferStream);
        inferStream = nullptr;
    }
    shm_name = shmName;
    pointSharedMemory();    // 指向新的共享内存
    InitCUDA();             // 重新初始化CUDA
}

// 设置图像指针
void Runtime::setImagePtr(void* image_ptr){
    // 释放现有资源
    if (hMapFile != nullptr){
        UnmapViewOfFile(host_ptr);
        CloseHandle(hMapFile);
        host_ptr = nullptr;
        hMapFile = nullptr;

    }
    if (inferGraphExec != nullptr) {
        cudaGraphExecDestroy(inferGraphExec);
        inferGraphExec = nullptr;
    }
    if (inferGraph != nullptr) {
        cudaGraphDestroy(inferGraph);
        inferGraph = nullptr;
    }
    if (inferStream != nullptr) {
        cudaStreamDestroy(inferStream);
        inferStream = nullptr;
    }
    shm_name = nullptr;
    host_ptr = image_ptr;
    InitCUDA();  // 重新初始化CUDA
}

// 设置引擎路径
void Runtime::setEnginePath(string &enginePath) {
    // 释放现有资源
    if (inferGraphExec != nullptr) {
        cudaGraphExecDestroy(inferGraphExec);
        inferGraphExec = nullptr;
    }
    if (inferGraph != nullptr) {
        cudaGraphDestroy(inferGraph);
        inferGraph = nullptr;
    }
    if (inferStream != nullptr) {
        cudaStreamDestroy(inferStream);
        inferStream = nullptr;
    }
    engineCtx.reset();
    imageTensor.reset();
    engine_path = enginePath; // 更新引擎路径
    InitCUDA();  // 重新初始化CUDA
}

// 获取引擎路径
std::string Runtime::getEnginePath() {
    return engine_path;
}

void Runtime::setShapes(const vector<int> &shapes) {
    imageTensor.reset();

    imageWidth = shapes[0];
    imageHeight = shapes[1];
    imageChannels = shapes[2];
    imageSize = imageWidth * imageHeight * imageChannels;

    imageTensor = std::make_shared<Tensor>();
    imageTensor->device(imageSize * sizeof(uint8_t));
}

vector<int> Runtime::getShapes() {
    return vector<int>{imageWidth, imageHeight, imageChannels};
}
