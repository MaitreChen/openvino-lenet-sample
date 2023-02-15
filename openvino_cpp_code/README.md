## OpenVINO C++ API 推理部署

该文件夹中包含了两个infer.cpp，含义如下：

* infer_mnist1：采用OpenVINO 2021 的API开发；
* infer_mnist2：采用OpenVINO 2022 的API开发；

由于OpenVINO 2022采用了全新的API，所以这里提供了旧版本程序以作对比。

### Attention

* matU8ToBlob.cpp为OpenVINO 2021提供的源文件，在OpenVINO 2022已移除。其主要作用在于将OpenCV读取到的**mat**数据对象转换为OpenVINO推理需要的**Blob**对象，同时也是为了将**NHWC**格式转换为**NCHW**！

- 在OpenVINO 2022中，可以参考matU8ToBlob的实现方式将图片数据复制到Tensor中。