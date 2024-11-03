# 不使用CUDA软件对YOLO进行TensorRT推理

其中含有对该功能的pyd打包

## 环境

* Windows11
* CLion
* MSCV

## 推理技术

* TensorRT
* CUDA Graphsc

## 接口

* buildEngine  arg：onnxFilePath，engineFilePath， half(bool)
* Runtime arg: shmName(共享空间名)， inputWidth(输入文件宽度，正方形图片)， enginePath
* ```
  detect = Runtime("image", 640, enginePath)
  ```

## 输出

* 原有TensorRT的输出格式
* 怎么启动检测 将图片信息传递进共享空间后
* ```
  detect.predict()
  ```
* 怎么获取输出 对实例化进行Numpy
* ```
  output_host = np.array(detect, copy=False) #存储结果的buffer
  ```

  Numpy一次后永久有效可重复从output_host获取结果

  ```
  output_host = np.array(detect, copy=False) #存储结果的buffer
  detect.predict()
  preds = output_host[conf_mask]
  detect.predict()
  preds = output_host[conf_mask]
  ```

## 注意

* engine格式

  本项目完全模拟ultralytics方法 **ultralytics下转成的模型前缀有4 + 196个字节长度为文件描述信息**

  可以使用buildEngine接口对onnx转化 其格式也是ultralytics转化最大空间为1U<<30
* 后续会做全部优化

## 图片操作提示

* 图片无需预操作调整，内部使用算子进行自适应调整成模型需要的大小 该项目图片获取地址为共享空间

## 离线需要的DLL文件

* cublas64_11.dll
* cublasLt64_11.dll
* cudart64_110.dll
* cudnn64_8.dll
* nvinfer_plugin.dll
* nvinfer.dll
* nvonnxparser.dll
* 编译出的DLL文件
