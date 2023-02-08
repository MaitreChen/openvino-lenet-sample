# openvino-lenet-sample

## Dependence

- Win 10
- Python 3.8
- PyTorch 1.10.0
- Visual Studio 2019
- OpenVINO 2022.1 Runtime
- OpenVINO 2022.1 Dev

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