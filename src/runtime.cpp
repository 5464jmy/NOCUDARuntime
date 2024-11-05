#include "runtime.h"

// 从文件加载数据到字符向量
vector<char> loadFile(const string& filePath, bool ultralytics) {
    ifstream file(filePath, ios::binary);  // 以二进制方式打开文件
    if (!file.is_open()) {
        throw runtime_error("Error opening file: " + filePath);
    }

    // 获取文件大小
    file.seekg(0, ios::end);
    streampos fileSize = file.tellg();
    file.seekg(0, ios::beg);

    if (ultralytics){
        // 如果使用ultralytics模型，读取偏移序列化引擎大小
        int size = 4;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));

        // 计算总偏移并调整文件大小
        size += sizeof(size);
        fileSize -= size;
        file.seekg(size, ios::beg);
    }

    // 如果文件为空或大小为负，返回空向量
    if (fileSize <= 0) {
        return {};
    }

    // 将文件内容读取到向量中
    vector<char> fileContent(static_cast<size_t>(fileSize));
    file.read(fileContent.data(), fileSize);

    // 检查读取错误
    if (!file) {
        throw runtime_error("Error reading file: " + filePath);
    }

    return fileContent;
}

// 带有共享内存名的构造函数
RuntimeWithGraph::RuntimeWithGraph(string& shmName, const vector<int>& shapes, string& enginePath, bool ultralytics) {
    // 初始化参数
    this->ultralytics = ultralytics;
    imageWidth = shapes[0];
    imageHeight = shapes[1];
    imageChannels = shapes[2];
    imageSize = imageWidth * imageHeight * imageChannels;
    engine_path = enginePath;

    // 初始化图像张量
    imageTensor = make_shared<Tensor>();
    imageTensor->host(imageSize * sizeof(uint8_t));
    imageTensor->device(imageSize * sizeof(uint8_t));

    shm_name = shmName;
//    createSharedMemory();
    pointSharedMemory(); // 指向共享内存
    InitTensors(); // 初始化张量
    transforms.update(imageWidth, imageHeight, width, height);
    cudaStreamCreate(&inferStream);  // 创建CUDA流
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw runtime_error("Failed to enqueueV3 before graph creation");
    }
    createGraph(); // 创建CUDA图
}

// 带有图像指针的构造函数
RuntimeWithGraph::RuntimeWithGraph(void* image_ptr, const vector<int>& shapes, string& enginePath, bool ultralytics) {
    // 初始化参数
    this->ultralytics = ultralytics;
    imageWidth = shapes[0];
    imageHeight = shapes[1];
    imageChannels = shapes[2];
    imageSize = imageWidth * imageHeight * imageChannels;

    // 初始化图像张量
    imageTensor = make_shared<Tensor>();
    imageTensor->device(imageSize * sizeof(uint8_t));

    engine_path = enginePath;
    host_ptr = image_ptr;

    InitTensors(); // 初始化张量
    transforms.update(imageWidth, imageHeight, width, height);
    cudaStreamCreate(&inferStream);  // 创建CUDA流
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw runtime_error("Failed to enqueueV3 before graph creation");
    }
    cudaStreamSynchronize(inferStream);
    createGraph(); // 创建CUDA图
}

// 设置运行时的张量
void RuntimeWithGraph::InitTensors() {
    // 加载引擎文件到字符向量
    auto data = loadFile(engine_path, ultralytics);
    engineCtx = make_shared<EngineContext>();

    // 构建引擎上下文
    if (!engineCtx->construct(data.data(), data.size())) {
        throw runtime_error("Failed to construct engine context.");
    }

    // 获取引擎中的张量信息
    int tensorNum = engineCtx->mEngine->getNbIOTensors();
    for (int i = 0; i < tensorNum; i++) {
        const char* name = engineCtx->mEngine->getIOTensorName(i);
        auto dtype = engineCtx->mEngine->getTensorDataType(name);
        auto typesz = getDataTypeSize(dtype);

        // 如果是输入张量，获取输入张量的维度
        if ((engineCtx->mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)) {
            input_dims = engineCtx->mEngine->getTensorShape(name);
            height = input_dims.d[2];
            width = input_dims.d[3];
            input_bytes = calculateVolume(input_dims) * typesz;
            engineCtx->mContext->setTensorAddress(name, input_Tensor.device(input_bytes));
        } else {
            // 如果是输出张量，获取输出张量的维度
            output_dims = engineCtx->mEngine->getTensorShape(name);
            output_bytes = calculateVolume(output_dims) * typesz;
            engineCtx->mContext->setTensorAddress(name, output_Tensor.device(output_bytes));
            output_Tensor.host(output_bytes);
        }
    }
}

// 创建CUDA图形执行计划
void RuntimeWithGraph::createGraph() {
    // 开始捕获CUDA流中的操作
    cudaStreamBeginCapture(inferStream, cudaStreamCaptureModeGlobal);

    // 将图像数据从主机复制到设备
    cudaMemcpyAsync(imageTensor->device(), host_ptr, imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice, inferStream);

    // 执行图像扭曲变换
    cudaWarpAffine(static_cast<uint8_t*>(imageTensor->device()), imageWidth, imageWidth,
                   static_cast<float*>(input_Tensor.device()), width, height,
                   transforms.matrix, inferStream);

    // 将推理操作入队
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw runtime_error("Failed to enqueueV3 during graph creation");
    }
    // 将输出数据从设备复制到主机
    cudaMemcpyAsync(output_Tensor.host(), output_Tensor.device(), output_bytes, cudaMemcpyDeviceToHost, inferStream);

    // 结束捕获并创建CUDA图
    cudaStreamEndCapture(inferStream, &inferGraph);
    cudaGraphInstantiate(&inferGraphExec, inferGraph, nullptr, nullptr, 0);
}

// 创建共享内存
void RuntimeWithGraph::createSharedMemory() {
    // 创建共享内存映射文件
    hMapFile = CreateFileMappingA(
            INVALID_HANDLE_VALUE, // 使用系统分页文件
            nullptr,              // 默认安全性
            PAGE_READWRITE,       // 读写权限
            0,                    // 高位文件大小
            imageSize,            // 低位文件大小
            shm_name.c_str());    // 共享内存名称

    if (hMapFile == nullptr) {
        cerr << "Could not create file mapping object" << endl;
        return;
    }
}

// 连接到共享内存
void RuntimeWithGraph::pointSharedMemory() {
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
    host_ptr = MapViewOfFile(
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

// 析构函数
RuntimeWithGraph::~RuntimeWithGraph() {
    // 销毁图执行实例
    if (inferGraphExec != nullptr) {
        cudaGraphExecDestroy(inferGraphExec);
        inferGraphExec = nullptr;
    }

    // 销毁图
    if (inferGraph != nullptr) {
        cudaGraphDestroy(inferGraph);
        inferGraph = nullptr;
    }

    // 销毁流
    if (inferStream != nullptr) {
        cudaStreamDestroy(inferStream);
        inferStream = nullptr;
    }

    // 解除和关闭共享内存
    if (hMapFile != nullptr) {
        UnmapViewOfFile(host_ptr);
        CloseHandle(hMapFile);
        host_ptr = nullptr;
        hMapFile = nullptr;
    }

    // 释放其他资源
    engineCtx.reset(); // 重置智能指针
    imageTensor.reset();
}

// 推理方法
void RuntimeWithGraph::predict() {
    // 执行推理
    cudaGraphLaunch(inferGraphExec, inferStream);
    cudaStreamSynchronize(inferStream);
}

// 获取共享内存名称
string RuntimeWithGraph::getShmName() {
    return shm_name;
}

// 设置共享内存名称
void RuntimeWithGraph::setShmName(string &shmName) {
    // 释放现有资源
    if (hMapFile != nullptr) {
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
    shm_name = shmName;
    pointSharedMemory();  // 指向新的共享内存
    createGraph();        // 创建CUDA图
}

// 设置图像指针
void RuntimeWithGraph::setImagePtr(void* image_ptr) {
    // 释放现有资源
    if (hMapFile != nullptr) {
        UnmapViewOfFile(host_ptr);
        CloseHandle(hMapFile);
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
    shm_name = "";
    host_ptr = image_ptr;
    createGraph();  // 创建CUDA图
}

// 设置引擎路径
void RuntimeWithGraph::setEnginePath(string &enginePath, bool ultralytics_in) {
    // 释放现有资源
    if (inferGraphExec != nullptr) {
        cudaGraphExecDestroy(inferGraphExec);
        inferGraphExec = nullptr;
    }
    if (inferGraph != nullptr) {
        cudaGraphDestroy(inferGraph);
        inferGraph = nullptr;
    }
    engineCtx.reset(); // 重置引擎上下文指针

    ultralytics = ultralytics_in;
    engine_path = enginePath;  // 更新引擎路径

    InitTensors();  // 重新初始化张量
    transforms.update(imageWidth, imageHeight, width, height);
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw runtime_error("Failed to enqueueV3 before graph creation");
    }
    cudaStreamSynchronize(inferStream);

    createGraph();  // 创建CUDA图
}

// 获取引擎路径
string RuntimeWithGraph::getEnginePath() {
    return engine_path;
}

// 设置形状参数
void RuntimeWithGraph::setShapes(const vector<int> &shapes) {
    imageTensor.reset();  // 重置图像张量

    imageWidth = shapes[0];
    imageHeight = shapes[1];
    imageChannels = shapes[2];
    imageSize = imageWidth * imageHeight * imageChannels;

    imageTensor = make_shared<Tensor>();
    imageTensor->device(imageSize * sizeof(uint8_t));
    transforms.update(imageWidth, imageHeight, width, height);
    createGraph();  // 创建CUDA图
}

// 获取形状参数
vector<int> RuntimeWithGraph::getShapes() {
    return vector<int>{imageWidth, imageHeight, imageChannels};
}