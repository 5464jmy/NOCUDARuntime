#include "Runtime.h"

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

// 设置运行时的张量
void Base::InitTensor() {
    // 加载引擎文件到字符向量
    auto data = loadFile(enginePath, ultralytics);
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
            inputDims = engineCtx->mEngine->getTensorShape(name);
            if ((inputDims.d[0] == -1) & dynamic) {
                inputDims.d[0] = batch;
            }
            height = inputDims.d[2];
            width = inputDims.d[3];
            imageSize = calculateVolume(inputDims);
            input_bytes = calculateVolume(inputDims) * typesz;
//            inputTensor.host(input_bytes);
            engineCtx->mContext->setTensorAddress(name, inputTensor.device(input_bytes));
        } else {
            // 如果是输出张量，获取输出张量的维度
            outputDims = engineCtx->mEngine->getTensorShape(name);
            if ((outputDims.d[0] == -1) & dynamic) {
                outputDims.d[0] = batch;
            }
            output_bytes = calculateVolume(outputDims) * typesz;
            engineCtx->mContext->setTensorAddress(name, outputTensor.device(output_bytes));
            outputTensor.host(output_bytes);
        }
    }
    if (dynamic) {
        engineCtx->mContext->setOptimizationProfileAsync(0, inferStream);
        engineCtx->mContext->setInputShape("images", inputDims);
    }
}


void Base::predict(float* image){
    std::cerr << 1 << std::endl;
    cudaMemcpyAsync(inputTensor.device(), image, input_bytes, cudaMemcpyHostToDevice, inferStream);
    // 将推理操作入队
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw runtime_error("Failed to enqueueV3 during graph creation");
    } // 将输出数据从设备复制到主机
    cudaMemcpyAsync(outputTensor.host(), outputTensor.device(), output_bytes, cudaMemcpyDeviceToHost, inferStream);
//    cudaMemcpyAsync(inputTensor.host(), inputTensor.device(), input_bytes, cudaMemcpyDeviceToHost, inferStream);
    cudaStreamSynchronize(inferStream);


//    cv::Mat image2(height, width, CV_32FC3, (static_cast<float *>(inputTensor.host())));
//    cv::imshow("image", image2);
//    cv::waitKey(0);
}


void BaseWithWarpT::predict(void* image) {
    cudaMemcpyAsync(imageTensor->device(), image, imageSize, cudaMemcpyHostToDevice, inferStream);
    warpAffine->cudaWarpAffine();
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw runtime_error("Failed to enqueueV3 during graph creation");
    }
    cudaMemcpyAsync(outputTensor.host(), outputTensor.device(), output_bytes, cudaMemcpyDeviceToHost, inferStream);
    cudaStreamSynchronize(inferStream);
}

void BaseWithWarpS::predict(uint8_t* image, const nvinfer1::Dims3& dims) {
    imageSize = calculateVolume(dims);
    imageTensor->device(imageSize);
    warpAffine->updateImageInputPtr(imageTensor->device());
    warpAffine->updateImageOutSize(dims.d[1], dims.d[0]);

    cudaMemcpyAsync(imageTensor->device(), image, imageSize, cudaMemcpyHostToDevice, inferStream);
    warpAffine->cudaWarpAffine();
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw runtime_error("Failed to enqueueV3 during graph creation");
    }
    cudaMemcpyAsync(outputTensor.host(), outputTensor.device(), output_bytes, cudaMemcpyDeviceToHost, inferStream);
    cudaStreamSynchronize(inferStream);
}

void Runtime::predict(){
    cudaMemcpyAsync(imageTensor->device(), host_ptr, imageSize, cudaMemcpyHostToDevice, inferStream);
    warpAffine->cudaWarpAffine();
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw runtime_error("Failed to enqueueV3 during graph creation");
    }
    cudaMemcpyAsync(outputTensor.host(), outputTensor.device(), output_bytes, cudaMemcpyDeviceToHost, inferStream);
    cudaStreamSynchronize(inferStream);
}

// 推理方法
void RuntimeCG::predict() {
    cudaGraphLaunch(inferGraphExec, inferStream);
    cudaStreamSynchronize(inferStream);
}
void Base::Init(){
    cudaStreamCreate(&inferStream);
    InitTensor();
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw runtime_error("Failed to enqueueV3 before graph creation");
    }
    cudaStreamSynchronize(inferStream);
}

Base::Base(std::string &enginePath, bool ultralytics, bool dynamic, uint32_t batch) {
    this->dynamic = dynamic;
    this->batch = batch;
    this->enginePath = enginePath;
    this->ultralytics = ultralytics;
    Init();
}

BaseWithWarpS::BaseWithWarpS(string& enginePath, bool BGR, bool ultralytics, bool dynamic, uint32_t batch):
        Base(enginePath, ultralytics, dynamic, batch) {
    imageTensor = make_shared<Tensor>();
    imageTensor->device(imageSize);
    warpAffine = new WarpAffine(imageTensor->device(), 0, 0,
                                inputTensor.device(), width, height,
                                0, BGR,
                                inferStream);
}

BaseWithWarpT::BaseWithWarpT(const vector<int>& shapes, string& enginePath, bool BGR, bool ultralytics, bool dynamic, uint32_t batch):
        Base(enginePath, ultralytics, dynamic, batch) {
    this->BGR = BGR;
    imageWidth = shapes[0];
    imageHeight = shapes[1];
    imageChannels = shapes[2];
    imageSize = imageWidth * imageHeight * imageChannels;
    imageTensor = make_shared<Tensor>();
    imageTensor->device(imageSize);
//    std::cerr << BGR << std::endl;
    warpAffine = new WarpAffine(imageTensor->device(), imageWidth, imageHeight,
                                inputTensor.device(), width, height,
                                imageChannels, BGR,
                                inferStream);
}
Runtime::Runtime(const vector<int>& shapes, string& enginePath, bool BGR, bool ultralytics, bool dynamic, uint32_t batch):
        BaseWithWarpT(shapes, enginePath, BGR, ultralytics, dynamic, batch){
}
Runtime::Runtime(void* image_ptr, const vector<int>& shapes, string& enginePath, bool BGR, bool ultralytics, bool dynamic, uint32_t batch):
        BaseWithWarpT(shapes, enginePath, BGR, ultralytics, dynamic, batch){
    host_ptr = image_ptr;
}
Runtime::Runtime(string& shmName, const vector<int>& shapes, string& enginePath, bool BGR, bool ultralytics, bool dynamic, uint32_t batch):
        BaseWithWarpT(shapes, enginePath, BGR, ultralytics, dynamic, batch){
    shm_name = shmName;
    pointSharedMemory(shm_name, imageSize, &host_ptr, hMapFile);
}
RuntimeCG::RuntimeCG(const vector<int>& shapes, string& enginePath, bool BGR, bool ultralytics, bool dynamic, uint32_t batch):
        Runtime(shapes, enginePath, BGR, ultralytics, dynamic, batch){
}
RuntimeCG::RuntimeCG(string& shmName, const vector<int>& shapes, string& enginePath, bool BGR, bool ultralytics, bool dynamic, uint32_t batch):
        Runtime(shmName, shapes, enginePath, BGR, ultralytics, dynamic, batch){
    createGraph();
}

RuntimeCG::RuntimeCG(void* imagePtr, const vector<int>& shapes, string& enginePath, bool BGR, bool ultralytics, bool dynamic, uint32_t batch):
        Runtime(imagePtr, shapes, enginePath, BGR, ultralytics, dynamic, batch){
    createGraph();
}

Base::~Base(){
    engineCtx.reset();
}

BaseWithWarpS::~BaseWithWarpS(){
    imageTensor.reset();
}

BaseWithWarpT::~BaseWithWarpT(){
    imageTensor.reset();
}

Runtime:: ~Runtime(){
    if (hMapFile != nullptr) {
        UnmapViewOfFile(host_ptr);
        CloseHandle(hMapFile);
        host_ptr = nullptr;
        hMapFile = nullptr;
    }
}

RuntimeCG::~RuntimeCG() {
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
}










//void Base::predict(uint8_t* image, const nvinfer1::Dims3& dims){
//    // 将图像数据从主机复制到设备
//    uint64_t imageByte = calculateVolume(dims);
//    imageTensor->device(imageByte);
//    warpAffine = new WarpAffine(imageTensor->device(), dims.d[1], dims.d[0],
//                                inputTensor.device(), width, height,
//                                imageChannels, false,
//                                inferStream);
//
//    cudaMemcpyAsync(imageTensor->device(), image, imageByte, cudaMemcpyHostToDevice, inferStream);
//    // 执行图像扭曲变换
//    warpAffine->cudaWarpAffine();
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

//void Base::predict(float* image, const nvinfer1::Dims& dims){
//    // 将图像数据从主机复制到设备
//    uint32_t imageByte = calculateVolume(dims) * 4;
//    imageTensor->device(imageByte);
//    cudaMemcpyAsync(imageTensor->device(), image, imageSize, cudaMemcpyHostToDevice, inferStream);
//
//    // 执行图像扭曲变换
//    warpAffine->cudaWarpAffine();
////    warpAffine->cudaCutImg();
//
//    // 将推理操作入队
//    if (!engineCtx->mContext->enqueueV3(inferStream)) {
//        throw runtime_error("Failed to enqueueV3 during graph creation");
//    } // 将输出数据从设备复制到主机
//    cudaMemcpyAsync(outputTensor.host(), outputTensor.device(), output_bytes, cudaMemcpyDeviceToHost, inferStream);
//    cudaStreamSynchronize(inferStream);
//
////    r = cudaMemcpyAsync(imageTensor->host(), imageTensor->device(), imageSize * sizeof(uint8_t), cudaMemcpyDeviceToHost, inferStream);
////    cv::Mat image2(imageHeight, imageWidth, CV_8UC3, imageTensor->host());
////    cv::imshow("image", image2);
////    cv::waitKey(0);
//
////    cudaMemcpyAsync(inputTensor.host(), inputTensor.device(),
////                        input_bytes, cudaMemcpyDeviceToHost, inferStream);
////    cv::Mat image2(height, width, CV_32FC3, (static_cast<float *>(inputTensor.host())));
////    cv::imshow("image", image2);
////    cv::waitKey(0);
//}