<div align="center">
   <a href="https://img.shields.io/badge/Nickname-阿斌~-blue"><img src="https://img.shields.io/badge/Nickname-阿斌~-blue.svg"></a>
   <a href="https://img.shields.io/badge/Hello-Buddy~-red"><img src="https://img.shields.io/badge/Hello-Buddy~-red.svg"></a>
   <a href="https://img.shields.io/badge/Enjoy-Yourself-brightgreen"><img src="https://img.shields.io/badge/Enjoy-Yourself-brightgreen.svg"></a>
</div>

# 📣Introduction

欢迎来到这个仓库！这里是一个充满神奇的角落，我们将共同探索模型训练和部署的奇妙之旅！💖

在模型部署的舞台上，我们追求的是小而美，让模型在树莓派和神经计算棒等计算资源匮乏设备上飞起！虽然我们的LeNet-5模型大小只有几百K，能够轻松部署在边缘设备上，但这个项目不仅仅是为了让你的模型轻装上阵，更是为了让你了解模型优化的魔法，包括**模型剪枝、模型量化和知识蒸馏**等招数！🎉

值得一提的是，在模型部署阶段，我们用的是**OpenVINO Inference API**。这不仅让我们的推理程序变得迅捷高效，也让整个项目更加魔法般的有趣！🚀

踏上深度学习探索之旅，让我们一起为代码世界注入更多奇妙的魔力吧！✨

----

🚩 **New Updates**

- ✅ April 14, 2023. Add pruning based l1-norm.





# 💊Dependence

- Win 10
- Python 3.8
- PyTorch 1.10.0
- Visual Studio 2019
- OpenVINO 2022.3.0 Runtime
- OpenVINO 2022.3.0 Dev

PyTorch安装教程详见：[Windows下深度学习环境搭建（PyTorch）](https://zhuanlan.zhihu.com/p/538386791)

OpenVINO 2022 Runtime安装详见文章**第三部分**：[VS+OpenCV+OpenVINO2022详细配置](https://zhuanlan.zhihu.com/p/603685184)

OpenVINO 2022 Dev安装详见文章**第三部分**：[OpenVINO2022 运行分类Sample](https://zhuanlan.zhihu.com/p/603740365)

----

OpenVINO 安装完成后，在命令行测试可用性：

```bash
python -c "from openvino import Core; print(Core().available_devices)"
```





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

----

由于模型剪枝后精度会下降，因此需要再训练以恢复精度，甚至超过原先的精度。

🚩再训练方式包括两种：微调（fine-tune）和 从头训练（train-from-scratch）。

默认采用`fine-tune`，若要修改再训练方式，指定`retrain-mode`参数即可，参数对应情况如下：

* train from scratch：0
* fine-tune：1

 即若采用`train-from-scratch`的策略，最终会得到 **model_data/best_pruned.ckpt**：

```bash
python main.py --prune --ratio 0.6 --retrain-mode 0
```

----

🤔对`fine-tune`和`train-from-scratch`的说明：

常规的剪枝流程通常是：训练 - 剪枝 - 微调，直到2019年的一篇文章[《Rethinking the Value of Network Pruning》](https://arxiv.org/abs/1810.05270)，文章通过对当前最先进的结构化剪枝算法做了实验，发现：**剪枝模型fine-tune后的性能仅与使用随机初始化权重的模型相当或更差**。换句话说，**剪枝后的网络架构本身要比网络权重更重要**。

因此，剪枝流程可以调整为：训练 - 剪枝 - 从头训练。

### PyTorch 模型推理

```bash
python inference_torch.py -m model_data/best.ckpt -i img.jpg -d cpu
```

若要对剪枝模型推理，需要初始化剪枝后的网络结构，这样才可以将 `model` 与我们再训练得到的 `weight` **(\best_pruned.ckpt)** 相匹配。 

若采用了上述的剪枝比例 `ratio 0.6` 进行剪枝，可以看到在 console 输出下列剪枝后的通道信息：

```bash
Conv2d    In shape: 1, Out shape 3.
Conv2d    In shape: 3, Out shape 8.
Linear    In shape: 200, Out shape 60.
Linear    In shape: 60, Out shape 42.
Linear    In shape: 42, Out shape 10.
```

于是，我们可以根据 `out shape` 重构网络结构，修改**inference_torch**中的**load model and params**部分即可：

```python
from src.net import LeNet

net = LeNet(cfg=[3, 8, 60, 42, 10])
```

### 模型导出

```bash
python export_onnx.py -m model_data/best.ckpt
```

【注】剪枝模型导出ONNX格式同上~

### ONNX Runtime推理

```bash
python inference_onnx.py -m model_data/best.onnx -i img.jpg
```

### 模型优化

```bash
mo --input_model model_data/best.onnx --output_dir model_data
```

### OpenVINO Python 推理

```bash
python inference_openvino.py --model model_data/best.xml --img img.jpg --mode sync --device CPU
```

OpenVINO模型推理时，可指定**同步推理或异步推理**：[sync、async]

推理设备可指定：[CPU，GPU，MYRIAD]

其中，MYRIAD是NSC2的视觉处理器，需要连接NSC2才可成功执行！

### OpenVINO C++ 推理

源码见openvino_cpp_code文件夹。

详见：[基于OpenVINO2022 C++ API 的模型部署](https://zhuanlan.zhihu.com/p/604351639)
