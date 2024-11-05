#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "enginebuilder.h"
#include "runtime.h"


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

PYBIND11_MODULE(NOCUDARuntime, m) {
    py::class_<RuntimeWithGraph>(m, "RuntimeWithGraph", py::buffer_protocol())
            .def(py::init<std::string&, std::vector<int>&, std::string&, bool>(),
                 py::arg("shared memory name"),
                 py::arg("image [imageWidth, imageHeight]"),
                 py::arg("engine path"),
                 py::arg("ultralytics") = false,
                 "Initialize Runtime with shared memory name, input width, and engine path")
            .def(py::init([](const py::array_t<uint8_t>& array,
                    const std::vector<int>& shapes,
                    std::string& enginePath,
                    bool ultralytics) {
                     // 获取 numpy 数组的缓冲区信息
                     py::buffer_info buf = array.request();
                     // 检查数据维度 (3D array checker)
                     if (buf.ndim != 3) {
                         throw std::runtime_error("输入数组必须是三维的");
                     }
                     // 调用 Runtime 构造函数
                     return new RuntimeWithGraph(buf.ptr, shapes, enginePath, ultralytics);
                 }),
                 py::arg("array"),
                 py::arg("shapes"),
                 py::arg("enginePath"),
                 py::arg("ultralytics") = true,
                 "Initialize Runtime with array, shapes, and engine path")

            .def_property("shapes", &RuntimeWithGraph::getShapes, &RuntimeWithGraph::setShapes)
            .def_property("shm_name", &RuntimeWithGraph::getShmName, &RuntimeWithGraph::setShmName)
            .def_property_readonly("engine_path", &RuntimeWithGraph::getEnginePath, "")
                     // 定义只读属性
            .def_property_readonly("input_dims",
                                   [](const RuntimeWithGraph& self) { return dims_to_vector(self.input_dims); },
                                   "Get input dimensions as a vector of ints")

            .def_property_readonly("output_dims",
                                   [](const RuntimeWithGraph& self) { return dims_to_vector(self.output_dims); },
                                   "Get output dimensions as a vector of ints")

            .def("predict", &RuntimeWithGraph::predict, "Execute prediction on the input data")
            .def("setImage", [](RuntimeWithGraph &self, const py::array_t<uint8_t> &array) {
                // 获取 numpy 数组的缓冲区信息
                py::buffer_info buf = array.request();
                // 将数据指针传递给 Runtime 的 setImagePtr 方法
                self.setImagePtr(buf.ptr);
            }, "Set image using numpy array")
            .def("setEnginePath", &RuntimeWithGraph::setEnginePath, py::arg("enginePath"), py::arg("ultralytics") = false)
            // 使用 def_buffer 来暴露 output_Tensor 内存
            .def_buffer([](RuntimeWithGraph& self) -> py::buffer_info {
                auto* output_ptr = static_cast<float*>(self.output_Tensor.host());  // 获取指针
                if (!output_ptr) {
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
                        output_ptr,                      // 指向内存的指针
                        sizeof(float),                   // 元素的大小
                        py::format_descriptor<float>::format(),  // 数据类型
                        self.output_dims.nbDims,                    // 维度数量
                        shape,                           // 维度大小
                        strides                          // 步长
                };
            })

            .def_readonly("ultralytics", &RuntimeWithGraph::ultralytics, "engine style")
            ;

    m.def("buildEngine", &buildEngine,
          py::arg("onnxFilePath"),
          py::arg("engineFilePath"),
          py::arg("exponent") = 1,
          py::arg("exponent") = 22,
          py::arg("half") = false,
          py::arg("ultralytics") = false,
          "Build a TensorRT engine from an ONNX file");
}

