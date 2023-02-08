# openvino-lenet-sample

## Dependence

- Win 10
- Python 3.8
- PyTorch 1.10.0
- Visual Studio 2019
- OpenVINO 2022.1 Runtime
- OpenVINO 2022.1 Dev

---

PyTorch安装教程详见：[Windows下深度学习环境搭建（PyTorch）](https://zhuanlan.zhihu.com/p/538386791)

OpenVINO 2022 Runtime安装详见文章**第三部分**：[VS+OpenCV+OpenVINO2022详细配置](https://zhuanlan.zhihu.com/p/603685184)

OpenVINO 2022 Dev安装详见文章第3部分：[OpenVINO2022 运行分类Sample](https://zhuanlan.zhihu.com/p/603740365)

---

## Quick Start

### 模型训练

```python
python train.py
```

### 模型导出

```python
python export_onnx.py -m model/best.ckpt
```

### ONNX推理

```python
python inference_onnx.py -m model/best.onnx -i img.jpg
```

### 模型优化

```python
mo --input_model model/best.onnx --output_dir model
```

### OpenVINO Python推理

```python
python inference_openvino.py -m model/best.xml -i img.jpg -d CPU
```

推理设备可指定：[CPUGPU, MYRIAD]

其中，MYRIAD是NSC2的视觉处理器，需要连接NSC2才可成功执行！

### OpenVINO C++ 推理

源码见openvino_cpp_code。

---

详见：基于OpenVINO2022 C++ API 的模型部署 - 阿斌有话说的文章 - 知乎 https://zhuanlan.zhihu.com/p/604351639