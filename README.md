# 不使用CUDA软件对YOLO进行TensorRT推理

## 环境

* Windows11
* CLion
* MSCV
* 根据实际情况调整cmake中库路径

## 推理技术

* TensorRT
* CUDA Graphsc

## 注意

* 该模块类方法均会调用核函数对图片进行**双线性缩放**成模型需要的尺寸
* 在传进图片前均需要先预设好传入方法的图像大小，而后可自动将传进去的图片缩放成模型需要的尺寸，且缩放后图片**位于中心**。
* 在类进行初始化时需要传进一个初始图片大小，后续有调整图片大小可通过类方法重新设置传入方法的图像大小
* engine格式
  本项目完全模拟ultralytics方法 **ultralytics下转成的模型前缀有4 + 196个字节长度为文件描述信息**
  可以使用buildEngine接口对onnx转化 其格式也是ultralytics转化
  最大空间为1U<<30
* 后续会做全部优化

## Python接口

### buildEngine

#### 参数

* onnxPath：提供的onnx路径
* enginePath：生成engine路径
* multiplier，exponent：
  ```
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, (1U << exponent) * multiplier);
  ```
* half：半精度 默认false
* ultralytics：ultralytics下转成的模型前缀有4 + 196个字节长度为文件描述信息 默认false

### RuntimeWithGraph

#### 初始化参数

1. ``RuntimeWithGraph(string& shmName, const vector<int>& shapes, string& enginePath, bool ultralytics)``;

   * shmName：共享空间名(“image”)
   * shapes：图片大小 ([320, 320, 3])
   * enginePath：engine路径
   * ultralytics: 读取文件格式 默认false
     ```
     detect = Runtime("image", [640,640,3], "xxx.engine", True)
     ```
2. ``RuntimeWithGraph(void* image_ptr, const vector<int>& shapes, string& enginePath, bool ultralytics)``;

   * nadarray：numpy数组或cv2读取的图像
   * shapes：图片大小 ([320, 320, 3])
   * enginePath：engine路径
   * ultralytics: 读取文件格式 默认false
     ```
     image= cv2.imread("xxx.jpg") 
     detect = Runtime(image, [640,640,3], enginePath, True)
     ```

#### Python类方法

1. 启动检测
   ```
   detect.predict()
   ```
2. 修改共享内存名
   ```
   detect.shm_name = "'xxx'"
   ```
3. 修改获取图像的地址(图片要是传进去的shapes大小)
   ```
   image = cv2.imread("xxx.jpg")  
   detect.setImage(image)
   ```
4. 更换engine
   ```
   detect.setEnginePath("xxx.engine", True)
   ```
5. 修改输入的图片尺寸
   ```
   detect.shapes = [320, 320, 3]
   ```

#### 可查询属性

* shapes：传进方法的图片大小
* input_dims：模型输入大小
* output_dims：模型输出大小
* engine_path：engine路径
* shm_name：共享空间名
* ultralytics：engine格式

#### 输出

* 原有TensorRT的输出格式
* 怎么获取输出 对实例化进行Numpy ``output = np.array(detect, copy=False) #存储结果的buffer``
* Numpy一次后永久有效可重复从output获取结果，不需要重复np.array(detect, copy=False)(除非更换engine）

```
  output = np.array(detect, copy=False) #存储结果的buffer
  detect.predict()
  detect.predict()
```

* 更换engine后需重新对实例化进行Numpy

```
  output = np.array(shot, copy=False)
  detect.setEnginePath("xxx.engine", True)
  output = np.array(shot, copy=False)
```

#### 使用建议

    * 批量一个尺寸的图片或者截屏 录屏检测
    * 需要将图片村放到统一的地址才能读取

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
