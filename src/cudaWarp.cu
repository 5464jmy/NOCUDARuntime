#include <algorithm>

#include "cudaWarp.h"


inline __device__ __host__ uint32_t iDivUp(uint32_t a, uint32_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

extern "C" __global__ void gpuBilinearWarpAffine3BGR(uint8_t* input, int inputWidth, int inputHeight,
                                                     float* output, int outputWidth, int outputHeight,
                                                     float3 m0, float3 m1) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    // 提前退出无效线程
    if (x >= outputWidth || y >= outputHeight) return;

    // 计算映射的输入坐标
    float inputX = m0.x * x + m0.y * y + m0.z;
    float inputY = m1.x * x + m1.y * y + m1.z;

    // 初始化输出值
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;

    // 判断是否在输入图像范围内
    if (inputX > -1 && inputX < inputWidth && inputY > -1 && inputY < inputHeight) {
        int lowX = __float2int_rd(inputX);
        int lowY = __float2int_rd(inputY);

        // 计算边界和插值权重
        float lx = inputX - lowX;
        float ly = inputY - lowY;
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;

        // 边界检查 (lowX 和 highX 是连续的)
        lowX = max(0, min(lowX, inputWidth - 1));
        lowY = max(0, min(lowY, inputHeight - 1));
        int highX = min(lowX + 1, inputWidth - 1);
        int highY = min(lowY + 1, inputHeight - 1);

        // 计算像素地址
        int lineSize = inputWidth * 3;
        uint8_t* v1 = input + lowY * lineSize + lowX * 3;
        uint8_t* v2 = input + lowY * lineSize + highX * 3;
        uint8_t* v3 = input + highY * lineSize + lowX * 3;
        uint8_t* v4 = input + highY * lineSize + highX * 3;

        // 直接进行插值计算
        float w1 = hx * hy, w2 = lx * hy, w3 = hx * ly, w4 = lx * ly;
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

//    int index = (y * toWidth + x) * 3;
//    outputPtr[index + 0] = c0 * 0.00392156862f;  // 归一化到 [0, 1]
//    outputPtr[index + 1] = c1 * 0.00392156862f;
//    outputPtr[index + 2] = c2 * 0.00392156862f;

    // 归一化并存储到输出
    int index = y * outputWidth + x;
    output[index]                  = c0 * 0.00392156862f;  // 归一化到 [0, 1]
    output[index + outputWidth * outputHeight] = c1 * 0.00392156862f;
    output[index + 2 * outputWidth * outputHeight] = c2 * 0.00392156862f;
}
extern "C" __global__ void gpuBilinearWarpAffine3RGB(uint8_t* input, int inputWidth, int inputHeight,
                                                     float* output, int outputWidth, int outputHeight,
                                                     float3 m0, float3 m1) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    // 提前退出无效线程
    if (x >= outputWidth || y >= outputHeight) return;

    // 计算映射的输入坐标
    float inputX = m0.x * x + m0.y * y + m0.z;
    float inputY = m1.x * x + m1.y * y + m1.z;

    // 初始化输出值
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;

    // 判断是否在输入图像范围内
    if (inputX > -1 && inputX < inputWidth && inputY > -1 && inputY < inputHeight) {
        int lowX = __float2int_rd(inputX);
        int lowY = __float2int_rd(inputY);

        // 计算边界和插值权重
        float lx = inputX - lowX;
        float ly = inputY - lowY;
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;

        // 边界检查 (lowX 和 highX 是连续的)
        lowX = max(0, min(lowX, inputWidth - 1));
        lowY = max(0, min(lowY, inputHeight - 1));
        int highX = min(lowX + 1, inputWidth - 1);
        int highY = min(lowY + 1, inputHeight - 1);

        // 计算像素地址
        int lineSize = inputWidth * 3;
        uint8_t* v1 = input + lowY * lineSize + lowX * 3;
        uint8_t* v2 = input + lowY * lineSize + highX * 3;
        uint8_t* v3 = input + highY * lineSize + lowX * 3;
        uint8_t* v4 = input + highY * lineSize + highX * 3;

        // 直接进行插值计算
        float w1 = hx * hy, w2 = lx * hy, w3 = hx * ly, w4 = lx * ly;
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

//    int index = (y * toWidth + x) * 3;
//    outputPtr[index + 0] = c0 * 0.00392156862f;  // 归一化到 [0, 1]
//    outputPtr[index + 1] = c1 * 0.00392156862f;
//    outputPtr[index + 2] = c2 * 0.00392156862f;

    // 归一化并存储到输出
    int index = y * outputWidth + x;
    output[index]                  = c2 * 0.00392156862f;  // 归一化到 [0, 1]
    output[index + outputWidth * outputHeight] = c1 * 0.00392156862f;
    output[index + 2 * outputWidth * outputHeight] = c0 * 0.00392156862f;
}

extern "C" __global__ void gpuBilinearWarpAffine4(uint8_t* input, int inputWidth, int inputHeight,
                                                  float* output, int outputWidth, int outputHeight,
                                                  float3 m0, float3 m1) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    // 提前退出无效线程
    if (x >= outputWidth || y >= outputHeight) return;

    // 计算映射的输入坐标
    float inputX = m0.x * x + m0.y * y + m0.z;
    float inputY = m1.x * x + m1.y * y + m1.z;

    // 初始化输出值
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;

    // 判断是否在输入图像范围内
    if (inputX > -1 && inputX < inputWidth && inputY > -1 && inputY < inputHeight) {
        int lowX = __float2int_rd(inputX);
        int lowY = __float2int_rd(inputY);

        // 计算边界和插值权重
        float lx = inputX - lowX;
        float ly = inputY - lowY;
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;

        // 边界检查 (lowX 和 highX 是连续的)
        lowX = max(0, min(lowX, inputWidth - 1));
        lowY = max(0, min(lowY, inputHeight - 1));
        int highX = min(lowX + 1, inputWidth - 1);
        int highY = min(lowY + 1, inputHeight - 1);

        // 计算像素地址
        int lineSize = inputWidth * 4;
        uint8_t* v1 = input + lowY * lineSize + lowX * 4;
        uint8_t* v2 = input + lowY * lineSize + highX * 4;
        uint8_t* v3 = input + highY * lineSize + lowX * 4;
        uint8_t* v4 = input + highY * lineSize + highX * 4;

        // 直接进行插值计算
        float w1 = hx * hy, w2 = lx * hy, w3 = hx * ly, w4 = lx * ly;
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

//    int index = (y * toWidth + x) * 3;
//    outputPtr[index + 0] = c0 * 0.00392156862f;  // 归一化到 [0, 1]
//    outputPtr[index + 1] = c1 * 0.00392156862f;
//    outputPtr[index + 2] = c2 * 0.00392156862f;

    // 归一化并存储到输出
    int index = y * outputWidth + x;
    output[index]                  = c0 * 0.00392156862f;  // 归一化到 [0, 1]
    output[index + outputWidth * outputHeight] = c1 * 0.00392156862f;
    output[index + 2 * outputWidth * outputHeight] = c2 * 0.00392156862f;
}
extern "C" __global__ void cutImg(uint8_t* input, uint32_t inputWidth, uint32_t inputHeight,
                                  float * output, uint32_t outputWidth, uint32_t outputHeight) {

    uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;
    uint32_t z = blockDim.z * blockIdx.z + threadIdx.z;

    uint32_t col = z % 4;
    uint32_t row = z / 4;

    uint32_t a1, b1;
    if ((inputWidth - col * outputWidth) < outputWidth) {
        a1 = inputWidth - outputWidth + x;
    }
    else {
        a1 = col * outputWidth + x;
    }

    if ((inputHeight - row * outputHeight) < outputHeight) {
        b1 = inputHeight - outputHeight + y;
    }
    else {
        b1 = row * outputHeight + y;
    }
    uint32_t idx = (b1 * inputWidth + a1) * 3;
    uint32_t index = (y * outputWidth + x + z * outputWidth * outputHeight) * 3;
//    uint32_t area = toWidth * toHeight;
//    outputPtr[index + 0 * area] = inputPtr[idx + 2] / 255.0f;
//    outputPtr[index + 1 * area] = inputPtr[idx + 1] / 255.0f;
//    outputPtr[index + 2 * area] = inputPtr[idx] / 255.0f;
    output[index + 2] = input[idx + 2] / 255.0f;
    output[index + 1] = input[idx + 1] / 255.0f;
    output[index + 0] = input[idx] / 255.0f;
//    uint32_t index = (y * toWidth + x + z * toWidth * toHeight) * 3;
//    uint32_t index = (x * toWidth + y + z * toWidth * toHeight);
//    uint32_t index = (threadIdx.x + threadIdx.y * BlocksPerGrid.x + threadIdx.z * BlocksPerGrid.x * BlocksPerGrid.y) +
//            (blockIdx.x + blockIdx.y * threadsPerBlock.x + blockIdx.z * threadsPerBlock.x * threadsPerBlock.y) *
//            (BlocksPerGrid.x * BlocksPerGrid.y * BlocksPerGrid.z);
//    outputPtr[index + 0] = 255;
//    outputPtr[index + 1] = 255;
//    outputPtr[index + 2] = 255;

}

extern "C" void TransformMatrix::update(int fromWidth, int fromHeight, int toWidth, int toHeight) {

    float scale  = std::min(static_cast<float>(toWidth) / fromWidth, static_cast<float>(toHeight) / fromHeight);
    float offset = 0.5f * scale - 0.5f;

    float scaleFromWidth  = -0.5f * scale * fromWidth;
    float scaleFromHeight = -0.5f * scale * fromHeight;
    float halfToWidth     = 0.5f * toWidth;
    float halfToHeight    = 0.5f * toHeight;

    float invD = (scale != 0.0) ? 1.0f / (scale * scale) : 0.0f;
    float A    = scale * invD;

    matrix[0] = make_float3(A, 0.0, -A * (scaleFromWidth + halfToWidth + offset));
    matrix[1] = make_float3(0.0, A, -A * (scaleFromHeight + halfToHeight + offset));
}

extern "C" void TransformMatrix::transform(float x, float y, float* ox, float* oy) const {
    *ox = matrix[0].x * x + matrix[0].y * y + matrix[0].z;
    *oy = matrix[1].x * x + matrix[1].y * y + matrix[1].z;
}

extern "C" void cudaWarpAffine(uint8_t* input, uint32_t inputWidth, uint32_t inputHeight,
                               float* output, uint32_t outputWidth, uint32_t outputHeight,
                               float3 matrix[2], cudaStream_t stream) {
    // launch kernel
    const dim3 threadsPerBlock(32, 32);
    const dim3 BlocksPerGrid(iDivUp(outputWidth, threadsPerBlock.x), iDivUp(outputHeight, threadsPerBlock.y));
    gpuBilinearWarpAffine3BGR<<<BlocksPerGrid, threadsPerBlock, 0, stream>>>(
            input, inputWidth, inputHeight,
            output, outputWidth, outputHeight,
            matrix[0], matrix[1]);
}

extern "C" WarpAffine::WarpAffine(void * input, uint32_t imageWidth, uint32_t imageHeight,
                                  void * output, uint32_t outputWidth, uint32_t outputHeight,
                                  uint32_t Channel, bool BGR,
                                  cudaStream_t stream) {

    this->fromWidth = imageWidth;
    this->fromHeight = imageHeight;
    this->toWidth = outputWidth;
    this->toHeight = outputHeight;
    this->Channel = Channel;
    this->BGR = BGR;

    this->inputPtr = static_cast<uint8_t*>(input);
    this->outputPtr = static_cast<float *>(output);
    this->stream = stream;

    threadsPerBlock = dim3(32, 32, 1);
    BlocksPerGrid = dim3(iDivUp(toWidth, threadsPerBlock.x), iDivUp(toHeight, threadsPerBlock.y), 1);

    transforms = new TransformMatrix();
    transforms->update(imageWidth, imageHeight, toWidth, toHeight);
}

extern "C" void WarpAffine::cudaWarpAffine() {
    if (Channel == 3){
        if (BGR) {
            gpuBilinearWarpAffine3BGR<<<BlocksPerGrid, threadsPerBlock, 0, stream>>>(
                    inputPtr, fromWidth, fromHeight,
                    outputPtr, toWidth, toHeight,
                    transforms->matrix[0], transforms->matrix[1]);
        }else{
            gpuBilinearWarpAffine3RGB<<<BlocksPerGrid, threadsPerBlock, 0, stream>>>(
                    inputPtr, fromWidth, fromHeight,
                    outputPtr, toWidth, toHeight,
                    transforms->matrix[0], transforms->matrix[1]);
        }
    }else{
        gpuBilinearWarpAffine4<<<BlocksPerGrid, threadsPerBlock, 0, stream>>>(
                inputPtr, fromWidth, fromHeight,
                outputPtr, toWidth, toHeight,
                transforms->matrix[0], transforms->matrix[1]);
    }
}
extern "C" void WarpAffine::cudaCutImg() {
    cutImg <<<BlocksPerGrid, threadsPerBlock, 0, stream>>> (
            inputPtr, fromWidth, fromHeight,
            outputPtr, toWidth, toHeight
    );
}

extern "C" void WarpAffine::updateImageInputPtr(void *input) {
    this->inputPtr = static_cast<uint8_t*>(input);
}

extern "C" void WarpAffine::updateImageOutSize(uint32_t imageWidth, uint32_t imageHeight, uint32_t imageChannel){
    Channel = imageChannel;
    fromHeight = imageHeight;
    fromWidth =imageWidth;
    transforms->update(imageWidth, imageHeight, toWidth, toHeight);
}
extern "C" void WarpAffine::updateBGR(bool bgr) {
    BGR = bgr;
}