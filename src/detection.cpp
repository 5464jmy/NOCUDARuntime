#include "detection.h"

#include <utility>

// Function to load a file into a vector of char
std::vector<char> loadFile(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);  // Open file in binary mode
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filePath);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the size of the serialized engine
    int size = 4;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    // Calculate total offset
    size += sizeof(size); // Account for the size of the integer itself

    // Adjust fileSize and set file position
    fileSize -= size;
    file.seekg(size, std::ios::beg);

    // If file is empty or size is negative, return an empty vector
    if (fileSize <= 0) {
        return {};
    }

    // Read file content into vector
    std::vector<char> fileContent(static_cast<std::size_t>(fileSize));
    file.read(fileContent.data(), fileSize);

    // Check for read errors
    if (!file) {
        throw std::runtime_error("Error reading file: " + filePath);
    }

    return fileContent;
}

void Runtime::setupTensors() {
    auto data = loadFile(engine_path);
    engineCtx = std::make_shared<EngineContext>();
    if (!engineCtx->construct(data.data(), data.size())) {
        throw std::runtime_error("Failed to construct engine context.");
    }
    int tensorNum = engineCtx->mEngine->getNbIOTensors();
    for (int i = 0; i < tensorNum; i++) {
        const char* name   = engineCtx->mEngine->getIOTensorName(i);
        auto        dtype  = engineCtx->mEngine->getTensorDataType(name);
        auto      typesz = getDataTypeSize(dtype);

        // Calculate the tensor size in bytes
        if ((engineCtx->mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)) {
            input_dims = engineCtx->mEngine->getTensorShape(name);
            height = input_dims.d[2];
            width  = input_dims.d[3];
            input_bytes = calculateVolume(input_dims) * typesz;
            engineCtx->mContext->setTensorAddress(name, input_Tensor.device(input_bytes));
        }else{
            output_dims = engineCtx->mEngine->getTensorShape(name);
            output_bytes = calculateVolume(output_dims) * typesz;
            engineCtx->mContext->setTensorAddress(name, output_Tensor.device(output_bytes));
            output_Tensor.host(output_bytes);
        }
    }
}

void Runtime::createGraph() {
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw std::runtime_error("Failed to enqueueV3 before graph creation");
    }
    cudaStreamSynchronize(inferStream);

    cudaStreamBeginCapture(inferStream, cudaStreamCaptureModeGlobal);
    cudaMemcpyAsync(imageTensor->device(), host_ptr, imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice, inferStream);
    cudaWarpAffine(static_cast<uint8_t*>(imageTensor->device()), imageWidth, imageWidth,
                   static_cast<float*>(input_Tensor.device()), width, height,
                   transforms.matrix, inferStream);
    //  Enqueue the inference operation
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw std::runtime_error("Failed to enqueueV3 during graph creation");
    }
    cudaMemcpyAsync(output_Tensor.host(), output_Tensor.device(), output_bytes, cudaMemcpyDeviceToHost, inferStream);
    cudaStreamEndCapture(inferStream, &inferGraph);
    cudaGraphInstantiate(&inferGraphExec, inferGraph, nullptr, nullptr, 0);
}



void Runtime::createSharedMemory() {
    // ??????????
    hMapFile = CreateFileMappingA(
            INVALID_HANDLE_VALUE, // ?????????
            nullptr,                 // ?????????
            PAGE_READWRITE,       // ?????д
            0,                    // ???????С????λ??
            imageSize,    // ???????С????λ??
            shm_name.c_str());   // ??????????????// ????

    if (hMapFile == nullptr) {
        std::cerr << "Could not create file mapping object" << std::endl;
        return ;
    }
}

void Runtime::pointSharedMemory() {
//     2. ???????????
    hMapFile = OpenFileMapping(
            FILE_MAP_ALL_ACCESS,   // ??д???
            FALSE,                 // ????о??
            shm_name.c_str());             // ???????????

    if (hMapFile == nullptr) {
        std::cerr << "Could not open file mapping object" << std::endl;
        return ;
    }

    // ??乲????????????????
    host_ptr = MapViewOfFile(
            hMapFile,            // ????????????
            FILE_MAP_ALL_ACCESS, // ?????д???????
            0,                   // ??????????λ
            0,                   // ??????????λ
            imageSize);  // ??????????

    if (host_ptr == nullptr) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return ;
    }
}

Runtime::~Runtime() {
    if (inferGraphExec != nullptr) {
        cudaGraphExecDestroy(inferGraphExec);
        inferGraphExec = nullptr;
    }

    // Release CUDA graph
    if (inferGraph != nullptr) {
        cudaGraphDestroy(inferGraph);
        inferGraph = nullptr;
    }

    // Release CUDA stream
    if (inferStream != nullptr) {
        cudaStreamDestroy(inferStream);
        inferStream = nullptr;
    }

    // Release other resources
    engineCtx.reset();
    imageTensor.reset();
    UnmapViewOfFile(host_ptr);
    CloseHandle(hMapFile);
}

void Runtime::predict() {
    cudaGraphLaunch(inferGraphExec, inferStream);
    cudaStreamSynchronize(inferStream);
}
Runtime::Runtime(std::string shmName, int inputWidth, std::string enginePath):
shm_name(std::move(shmName)), imageWidth(inputWidth), engine_path(std::move(enginePath)) {
    imageSize = inputWidth * inputWidth * 3;


    setupTensors();
    createSharedMemory();
    pointSharedMemory();


    imageTensor = std::make_shared<Tensor>();
    imageTensor->device(imageSize * sizeof(uint8_t));
    transforms.update(imageWidth, imageWidth, width, height);
    cudaStreamCreate(&inferStream);
    createGraph();
}


