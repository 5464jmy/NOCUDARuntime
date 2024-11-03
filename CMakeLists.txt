cmake_minimum_required(VERSION 3.12)
project(CUDARuntime LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_ARCHITECTURES "52;60;61;70;75;80")

# 为 MSVC 编译器设置 UTF-8 编码支持（仅针对 C 和 CXX 编译器）
if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
endif()

# 查找自定义 Python 可执行文件的位置
set(PYTHON_EXECUTABLE "D:/element/anaconda3/envs/CS2-1/python.exe")
set(PYBIND11_PYTHON_VERSION 3.10)

# Set custom CUDA path
set(CUDA_TOOLKIT_ROOT_DIR "D:/element/cuda/gpu")
find_package(CUDA REQUIRED)

# TensorRT
set(TENSORRT_INCLUDE_DIR "D:/element/TensorRT-8.6.1.6/include")
set(TENSORRT_LIBRARY_DIR "D:/element/TensorRT-8.6.1.6/lib")
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})

# OpenCV
set(OpenCV_DIR "D:/element/opencv/build")
find_package(OpenCV REQUIRED)

# Include directories
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
include_directories(${INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} "D:/element/cuda/samples/common/inc")

# Find source files
file(GLOB_RECURSE SOURCES
        "${SRC_DIR}/*.cpp"
        "${SRC_DIR}/*.cu"
        "D:/element/cuda/samples/common/src/*.cpp"
)

# Libraries
set(LIBS ${CUDA_LIBRARIES} ${OpenCV_LIBS}
        d3d12 dxgi d3dcompiler dxguid nvinfer nvonnxparser nvinfer_plugin
)

# Shared and Static libraries
add_library(detect_dll SHARED ${SOURCES})
add_library(detect_dllStatic STATIC ${SOURCES})

# Link CUDA
target_link_libraries(detect_dll PRIVATE ${LIBS})

# Pybind11 module
add_subdirectory(pybind11)
pybind11_add_module(detect python/bindings.cpp)
target_link_libraries(detect PRIVATE detect_dll ${LIBS})

# Executable
add_executable(MyExecutable ${SOURCES})

# Link libraries
target_link_libraries(MyExecutable PRIVATE ${LIBS})

# Apply CUDA architectures to targets
set_property(TARGET detect_dll PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
set_property(TARGET detect_dllStatic PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
set_property(TARGET detect PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
set_property(TARGET MyExecutable PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
