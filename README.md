# 🌾 Rice-Categorize: 基于 GoogLeNet 的大米品种识别系统

> 本项目利用深度学习技术，通过自定义的 **GoogLeNet (Inception v1)** 模型对大米图像进行多分类，实现对 Arborio、Basmati 等品种的精准识别。

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📑 项目架构与文件说明

项目代码结构清晰，涵盖了从数据处理到模型推理的全流程：

* **`model.py`**: 核心算法文件。实现了自定义的 `Inception` 模块及完整的 `GoogLeNet` 网络，并包含 Kaiming 初始化权重逻辑。
* **`data_partitioning.py`**: 数据预处理工具。负责将原始数据集按 **9:1** 的比例自动划分为训练集与测试集。
* **`mean_std.py`**: 像素特征分析脚本。计算全数据集的均值 (Mean) 与标准差 (Std)，用于 `transforms.Normalize` 预处理。
* **`train_model.py`**: 模型训练引擎。包含数据加载、Adam 优化器配置及损失函数计算，并实时可视化训练过程。
* **`test_model.py`**: 推理评估脚本。加载 `best_model.pth` 进行单张图像预测，并支持结果的反归一化可视化展示。

---

## 🚀 算法核心细节

### 1. 模型实现
模型采用并行卷积结构（Inception），能同时提取不同尺度的特征：
* **输入尺寸**: 224 x 224 (RGB)
* **分类输出**: 5 类大米品种
* **特征提取**: 通过 1x1, 3x3, 5x5 卷积核并行运算

### 2. 标准化参数 (Normalization)
根据 `mean_std.py` 计算，项目采用了特定的归一化参数以提升收敛速度：
* **Mean**: `[0.0420662, 0.04281093, 0.04413987]`
* **Std**: `[0.03315472, 0.03433457, 0.03628447]`

---

## 🛠 使用步骤

### 第一步：准备数据
将原始图像放入 `Rice_Image_Dataset` 目录，然后运行划分脚本：
```bash
python data_partitioning.py
```

### 第二步：模型训练
启动训练程序，系统会自动保存验证集表现最优的权重文件：
```bash
python train_model.py
```

### 第三步：模型推理
将待识别的图片（如 `OIP.jfif`）放入根目录，运行预测脚本查看结果：
```bash
python test_model.py
```

---

## ⚠️ 注意点
> * **路径配置**: 在 `train_model.py` 中保存权重时，请确保代码中的路径在你的电脑上存在，或修改为相对路径 `./best_model.pth`。
> * **显卡支持**: 代码默认优先使用 **CUDA** 进行加速，若无 GPU 环境则自动切换至 CPU。

---

## 📦 环境依赖
* `torch >= 1.7.0`
* `torchvision`
* `pandas`
* `matplotlib`
* `Pillow`
