#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "EngineBuilder.h"
#include "detection.h"
//#include "person.h"


namespace py = pybind11;

// 辅助函数，将 Dims32 转换为 std::vector<int>
std::vector<int> dims_to_vector(const nvinfer1::Dims32& dims) {
    return std::vector<int>(dims.d, dims.d + dims.nbDims);
}

// 将 Python list 转换为 Dims32
nvinfer1::Dims32 vector_to_dims(const std::vector<int>& vec) {
    nvinfer1::Dims32 dims{};
    dims.nbDims = static_cast<int>(vec.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        dims.d[i] = vec[i];
    }
    return dims;
}

PYBIND11_MODULE(detect, m) {
    py::class_<Runtime>(m, "Runtime", py::buffer_protocol())
            .def_readwrite("shm_name", &Runtime::shm_name, "Shared memory name")
            .def_readwrite("engine_path", &Runtime::engine_path, "Path to the engine file")
            .def_readonly("height", &Runtime::height, "Height of the input image")
            .def_readonly("width", &Runtime::width, "Width of the input image")
            .def_readonly("imageWidth", &Runtime::imageWidth, "Width of the image")

                    // 自定义 getter 和 setter 来处理 Dims32 类型
            .def_property("input_dims",
                          [](const Runtime& self) { return dims_to_vector(self.input_dims); },
                          [](Runtime& self, const std::vector<int>& dims) {
                              self.input_dims = vector_to_dims(dims);
                          })

            .def_property("output_dims",
                          [](const Runtime& self) { return dims_to_vector(self.output_dims); },
                          [](Runtime& self, const std::vector<int>& dims) {
                              self.output_dims = vector_to_dims(dims);
                          })

            .def(py::init<std::string, int, std::string>(),
                 py::arg("shmName"), py::arg("inputWidth"), py::arg("enginePath"),
                 "Initialize Runtime with shared memory name, input width, and engine path")
            .def("createSharedMemory", &Runtime::createSharedMemory, "Create shared memory segment")
            .def("pointSharedMemory", &Runtime::pointSharedMemory, "Point to shared memory segment")
            .def("setupTensors", &Runtime::setupTensors, "Setup tensors for inference")
            .def("createGraph", &Runtime::createGraph, "Create the computational graph")
            .def("predict", &Runtime::predict, "Execute prediction on the input data")

                    // 使用 def_buffer 来暴露 output_Tensor 内存
            .def_buffer([](Runtime& self) -> py::buffer_info {
                auto* ptr = static_cast<float*>(self.output_Tensor.host());  // 获取指针
                if (!ptr) {
                    throw std::runtime_error("Output tensor host memory is null.");
                }

                // 获取输出的维度
                std::vector<size_t> shape(self.output_dims.d,
                                          self.output_dims.d + self.output_dims.nbDims);

                // 每个维度的步长
                std::vector<size_t> strides;
                size_t stride = sizeof(float);  // 每个元素的大小
                for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
                    strides.insert(strides.begin(), stride);
                    stride *= *it;
                }

                // 使用大括号初始化 buffer_info
                return py::buffer_info{
                        ptr,                             // 指向内存的指针
                        sizeof(float),                   // 元素的大小
                        py::format_descriptor<float>::format(),  // 数据类型
                        self.output_dims.nbDims,                    // 维度数量
                        shape,                           // 维度大小
                        strides                          // 步长
                };
            })
            ;
//    py::class_<Person>(m, "Person")
//            .def(py::init<const std::string&, int>(),
//                 py::arg("name"), py::arg("age"),
//                 "Initialize Person with name and age")
//            .def("introduce", &Person::introduce, "Introduce the person");

    m.def("buildEngine", &buildEngine,
          py::arg("onnxFilePath"), py::arg("engineFilePath"), py::arg("half") = false,
          "Build a TensorRT engine from an ONNX file");
}

