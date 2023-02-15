<div align="center">
   <a href="https://img.shields.io/badge/Nickname-阿斌~-blue"><img src="https://img.shields.io/badge/Nickname-阿斌~-blue.svg"></a>
   <a href="https://img.shields.io/badge/Hello-Buddy~-red"><img src="https://img.shields.io/badge/Hello-Buddy~-red.svg"></a>
   <a href="https://img.shields.io/badge/Enjoy-Yourself-brightgreen"><img src="https://img.shields.io/badge/Enjoy-Yourself-brightgreen.svg"></a>
</div>

# Introduction

该仓库包含了深度学习应用的基本开发流程，包括模型训练、模型部署，旨在帮助小伙伴建立系统的认知！

在模型部署阶段，主要使用OpenVINO Inference API开发推理程序。

接下来的工作为继续完善POT工具的使用，进一步优化模型，使其以更少的存储资源与推理时延部署在边缘端。

# Dependence

- Win 10
- Python 3.8
- PyTorch 1.10.0
- Visual Studio 2019
- OpenVINO 2022.3 Runtime
- OpenVINO 2022.3 Dev

PyTorch安装教程详见：[Windows下深度学习环境搭建（PyTorch）](https://zhuanlan.zhihu.com/p/538386791)

OpenVINO 2022 Runtime安装详见文章**第三部分**：[VS+OpenCV+OpenVINO2022详细配置](https://zhuanlan.zhihu.com/p/603685184)

OpenVINO 2022 Dev安装详见文章**第三部分**：[OpenVINO2022 运行分类Sample](https://zhuanlan.zhihu.com/p/603740365)

# Quick Start

### 模型训练

```bash
python train.py
```

### PyTorch 模型推理

```bash
python inference_torch.py -m model/best.ckpt -i img.jpg -d cpu
```

### 模型导出

```bash
python export_onnx.py -m model/best.ckpt
```

### ONNX推理

```bash
python inference_onnx.py -m model/best.onnx -i img.jpg
```

### 模型优化

```bash
mo --input_model model/best.onnx --output_dir model
```

### OpenVINO Python 推理

```bash
python inference_openvino.py --model model/best.xml --img img.jpg --mode sync --device CPU
```

OpenVINO模型推理时，可指定同步推理或异步推理：[sync、async]

推理设备可指定：[CPU，GPU，MYRIAD]

其中，MYRIAD是NSC2的视觉处理器，需要连接NSC2才可成功执行！

### OpenVINO C++ 推理

源码见openvino_cpp_code文件夹。

---

详见：[基于OpenVINO2022 C++ API 的模型部署](https://zhuanlan.zhihu.com/p/604351639)
