#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "enginebuilder.h"
#include "Runtime.h"
#include "tensor.h"


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
    py::class_<Base>(m, "Base", py::buffer_protocol())
            .def(py::init<std::string&, bool, bool, uint32_t>(),
                 py::arg("engine path"),
                 py::arg("ultralytics") = false,
                 py::arg("dynamic") = false,
                 py::arg("batch") = 1,
                 "Initialize Base with shared memory name, inputPtr width, and engine path")

            .def_property_readonly("input_dims", [](const Base& self) { return dims_to_vector(self.inputDims); },
                                   "Get inputPtr dimensions as a vector of ints")

            .def_property_readonly("output_dims", [](const Base& self) { return dims_to_vector(self.outputDims); },
                                   "Get outputPtr dimensions as a vector of ints")
            .def_readonly("ultralytics", &Base::ultralytics, "engine style")
            .def_readonly("engine", &Base::enginePath, "engine style")
            .def("set_engine", &Base::setEnginePath, py::arg("enginePath"), py::arg("ultralytics") = false)
            .def("predict", [](Base& self, const py::array_t<float>& array){
                py::buffer_info buf = array.request();
                self.predict((float *)(buf.ptr));
                }, "Execute prediction on the inputPtr data")
            // 使用 def_buffer 来暴露 outputTensor 内存
            .def_buffer([](Base& self) -> py::buffer_info {
                auto* output_ptr = (float*)(self.outputTensor.host());  // 获取指针
                if (!output_ptr) {
                    throw std::runtime_error("Output tensor host memory is null.");
                }

                // 获取输出的维度
                std::vector<size_t> shape(self.outputDims.d, self.outputDims.d + self.outputDims.nbDims);
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
                        self.outputDims.nbDims,                    // 维度数量
                        shape,                           // 维度大小
                        strides                          // 步长
                };
            });

    py::class_<BaseWithWarpS, Base>(m, "BaseWithWarpS", py::buffer_protocol())
            .def(py::init<std::string&, bool, bool, bool, uint32_t>(),
                 py::arg("engine path"),
                 py::arg("BGR") = false,
                 py::arg("ultralytics") = false,
                 py::arg("dynamic") = false,
                 py::arg("batch") = 1,
                 "Initialize Base with shared memory name, inputPtr width, and engine path")

            .def("predict", [](BaseWithWarpS& self, const py::array_t<uint8_t>& array){
                auto shape = array.shape();
                nvinfer1::Dims3 dims3 = nvinfer1::Dims3(shape[1], shape[0], shape[2]);
                py::buffer_info buf = array.request();
                self.predict(static_cast<uint8_t*>(buf.ptr), dims3);
                }, "Execute prediction on the inputPtr data");

    py::class_<BaseWithWarpT, Base>(m, "BaseWithWarpT", py::buffer_protocol())
            .def(py::init<std::vector<int>&, std::string&, bool, bool, bool, uint32_t>(),
                 py::arg("image [imageWidth, imageHeight, imageChannels]"),
                 py::arg("engine path"),
                 py::arg("BGR") = false,
                 py::arg("ultralytics") = false,
                 py::arg("dynamic") = false,
                 py::arg("batch") = 1,
                 "Initialize Base with shared memory name, inputPtr width, and engine path")

//            .def_property("bgr", &BaseWithWarpT::getBGR, &BaseWithWarpT::setBGR)
            .def_readonly("bgr", &BaseWithWarpT::BGR)
            .def("predict", [](BaseWithWarpT& self, const py::array_t<uint8_t>& array){
//                py::buffer_info buf = array.request();
                self.predict(array.request().ptr);
            }, "Execute prediction on the inputPtr data");

    py::class_<Runtime, BaseWithWarpT>(m, "Runtime", py::buffer_protocol())
            .def(py::init<std::vector<int>&, std::string&, bool, bool, bool, uint32_t>(),
                 py::arg("image [imageWidth, imageHeight, imageChannels]"),
                 py::arg("engine path"),
                 py::arg("BGR") = false,
                 py::arg("ultralytics") = false,
                 py::arg("dynamic") = false,
                 py::arg("batch") = 1,
                 "Initialize Runtime with shared memory name, inputPtr width, and engine path")
            .def(py::init<std::string&, std::vector<int>&, std::string&, bool, bool, bool, uint32_t>(),
                 py::arg("shared memory name"),
                 py::arg("image [imageWidth, imageHeight, imageChannels]"),
                 py::arg("engine path"),
                 py::arg("BGR") = false,
                 py::arg("ultralytics") = false,
                 py::arg("dynamic") = false,
                 py::arg("batch") = 1,
                 "Initialize Runtime with shared memory name, inputPtr width, and engine path")
            .def(py::init([](py::array_t<uint8_t>& array, std::vector<int>& shapes,
                             std::string& enginePath, bool BGR, bool ultralytics, bool dynamic,
                             uint32_t batch) {
                     py::buffer_info buf = array.request();
                     if (buf.ndim != 3) {
                         throw std::runtime_error("输入数组必须是三维的");
                     }
                     return new Runtime(buf.ptr, shapes, enginePath, BGR, ultralytics, dynamic, batch);
                 }),
                 py::arg("Array"),
                 py::arg("image [imageWidth, imageHeight, imageChannels]"),
                 py::arg("engine path"),
                 py::arg("BGR") = false,
                 py::arg("ultralytics") = false,
                 py::arg("dynamic") = false,
                 py::arg("batch") = 1,
                 "Initialize Runtime with array, shapes, and engine path")

            .def("set_image", [](Runtime &self, const py::array_t<uint8_t> &array) {
                py::buffer_info buf = array.request();
                self.setImagePtr(buf.ptr);
            }, "Set image using numpy array")
            .def_property("shm_name", &Runtime::getShmName, &Runtime::setShmName)
            .def_property("image_shape", &Runtime::getShapes, &Runtime::setShapes)
            .def("set_engine", &Runtime::setEnginePath, py::arg("enginePath"), py::arg("ultralytics") = false)

            .def("predict", &Runtime::predict, "Execute prediction on the inputPtr data");

    py::class_<RuntimeCG, Runtime>(m, "RuntimeCG", py::buffer_protocol())
            .def(py::init<std::vector<int>&, std::string&, bool, bool, bool, uint32_t>(),
                 py::arg("image [imageWidth, imageHeight, imageChannels]"),
                 py::arg("engine path"),
                 py::arg("BGR") = false,
                 py::arg("ultralytics") = false,
                 py::arg("dynamic") = false,
                 py::arg("batch") = 1,
                 "Initialize Runtime with shared memory name, inputPtr width, and engine path")
            .def(py::init<std::string&, std::vector<int>&, std::string&, bool, bool, bool, uint32_t>(),
                 py::arg("shared memory name"),
                 py::arg("image [imageWidth, imageHeight, imageChannels]"),
                 py::arg("engine path"),
                 py::arg("BGR") = false,
                 py::arg("ultralytics") = false,
                 py::arg("dynamic") = false,
                 py::arg("batch") = 1,
                 "Initialize Runtime with shared memory name, inputPtr width, and engine path")
            .def(py::init([](const py::array_t<uint8_t>& array, const std::vector<int>& shapes,
                             std::string& enginePath, bool BGR, bool ultralytics, bool dynamic,
                             uint32_t batch) {
                     py::buffer_info buf = array.request();
                     if (buf.ndim != 3) {
                         throw std::runtime_error("输入数组必须是三维的");
                     }
                     return new RuntimeCG(buf.ptr, shapes, enginePath, BGR, ultralytics, dynamic, batch);
                 }),
                 py::arg("array"),
                 py::arg("image [imageWidth, imageHeight, imageChannels]"),
                 py::arg("engine path"),
                 py::arg("BGR") = false,
                 py::arg("ultralytics") = false,
                 py::arg("dynamic") = false,
                 py::arg("batch") = 1,
                 "Initialize Runtime with array, shapes, and engine path")

            .def("set_image", [](RuntimeCG &self, const py::array_t<uint8_t> &array) {
                py::buffer_info buf = array.request();
                self.setImagePtr(buf.ptr);
            }, "Set image using numpy array")
            .def_property("shm_name", &RuntimeCG::getShmName, &RuntimeCG::setShmName)
            .def_property("shapes", &RuntimeCG::getShapes, &RuntimeCG::setShapes)
            .def("set_engine", &RuntimeCG::setEnginePath, py::arg("enginePath"), py::arg("ultralytics") = false)
            .def("predict", &RuntimeCG::predict, "Execute prediction on the inputPtr data");

    m.def("buildEngine", &buildEngine,
          py::arg("onnxFilePath"),
          py::arg("engineFilePath"),
          py::arg("multiplier") = 1,
          py::arg("exponent") = 22,
          py::arg("half") = false,
          py::arg("dynamic") = false,
          py::arg("ultralytics") = false,
          "Build a TensorRT engine from an ONNX file");
}

