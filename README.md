<div align="center">
   <a href="https://img.shields.io/badge/Nickname-阿斌~-blue"><img src="https://img.shields.io/badge/Nickname-阿斌~-blue.svg"></a>
   <a href="https://img.shields.io/badge/Hello-Buddy~-red"><img src="https://img.shields.io/badge/Hello-Buddy~-red.svg"></a>
   <a href="https://img.shields.io/badge/Enjoy-Yourself-brightgreen"><img src="https://img.shields.io/badge/Enjoy-Yourself-brightgreen.svg"></a>
</div>

# 📣Introduction

该仓库包含了深度学习应用的基本开发流程，包括模型训练与模型部署，旨在帮助小伙伴建立系统的认知！💖

在模型部署阶段，我们希望能够以更小的模型、更少的计算代价运行在资源受限的设备上，如树莓派、神经计算棒。尽管我们采用的LeNet-5模型大小仅几百K，部署在边缘设备毫无压力，但本项目同时也在于帮助我们熟悉各类基本的模型优化方法，其中包括模型剪枝、模型量化与知识蒸馏等。

特别的，在模型部署阶段，本项目主要使用OpenVINO Inference API开发推理程序。

----

🚩 **New Updates**

- ✅ April 14, 2023. Add pruning based l1-norm.





# 💊Dependence

- Win 10
- Python 3.8
- PyTorch 1.10.0
- Visual Studio 2019
- OpenVINO 2022.3 Runtime
- OpenVINO 2022.3 Dev

PyTorch安装教程详见：[Windows下深度学习环境搭建（PyTorch）](https://zhuanlan.zhihu.com/p/538386791)

OpenVINO 2022 Runtime安装详见文章**第三部分**：[VS+OpenCV+OpenVINO2022详细配置](https://zhuanlan.zhihu.com/p/603685184)

OpenVINO 2022 Dev安装详见文章**第三部分**：[OpenVINO2022 运行分类Sample](https://zhuanlan.zhihu.com/p/603740365)



# 🧨Usage

```bash
python main.py [OPTIONS]
```

### Options

| Option                        | Description                           |
| ----------------------------- | ------------------------------------- |
| `-h, --help`                  | show this help message and exit       |
| `--batch-size BATCH_SIZE`     | batch size for training               |
| `--epoch EPOCH`               | number of epochs for training         |
| `--optim-policy OPTIM_POLICY` | optimizer for training. [sgd          |
| `--lr LR`                     | learning rate                         |
| `--use-gpu`                   | turn on flag to use GPU               |
| `--prune`                     | turn on flag to prune                 |
| `--output-dir OUTPUT_DIR`     | checkpoints of pruned model           |
| `--ratio RATIO`               | pruning scale. (default: 0.5)         |
| `--retrain-mode RETRAIN_MODE` | [train from scratch:0 \| fine-tune:1] |
| `--p-epoch P_EPOCH`           | number of epochs for retraining       |
| `--p-lr P_LR`                 | learning rate for retraining          |
| `--visualize VISUALIZE`       | select to visualize                   |





# ✨Quick Start

### 模型训练

```bash
python main.py
```

### 模型剪枝

指定prune开启剪枝模式，默认剪枝比例为0.5。

```bash
python main.py --prune
```

若要修改剪枝比例，指定ratio参数即可，范围在 [0-1]之间。

```bash
python main.py --prune --ratio 0.6
```

模型剪枝后再训练可以使用`fine-tune`或`train-from-scratch`，默认为fine-tune。

若要修改再训练方式，指定retrain-mode参数即可，参数对应情况如下：

* train from scratch：0
* fine-tune：1

----

train-from-scratch

```bash
python main.py --prune --ratio 0.6 --retrain-mode 0
```

-----

### PyTorch 模型推理

```bash
python inference_torch.py -m model_data/best_sparse.ckpt -i img.jpg -d cpu
```

### 模型导出

```bash
python export_onnx.py -m model_data/best_sparse.ckpt
```

### ONNX推理

```bash
python inference_onnx.py -m model_data/best.onnx -i img.jpg
```

### 模型优化

```bash
mo --input_model model_data/best.onnx --output_dir model_data
```

### OpenVINO Python 推理

```bash
python inference_openvino.py --model_data model_data/best.xml --img img.jpg --mode sync --device CPU
```

OpenVINO模型推理时，可指定同步推理或异步推理：[sync、async]

推理设备可指定：[CPU，GPU，MYRIAD]

其中，MYRIAD是NSC2的视觉处理器，需要连接NSC2才可成功执行！

### OpenVINO C++ 推理

源码见openvino_cpp_code文件夹。

---

详见：[基于OpenVINO2022 C++ API 的模型部署](https://zhuanlan.zhihu.com/p/604351639)
