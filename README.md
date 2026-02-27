Rice-Categorize: 基于 GoogLeNet 的大米品种识别系统

本项目是一个基于 Deep Learning 的图像分类任务，旨在利用 GoogLeNet (Inception v1) 网络模型对不同品种的大米图像进行自动化识别与分类。

项目架构

项目包含完整的数据预处理、模型构建、训练及推理流程：

model.py: 核心算法文件。实现了自定义的 Inception 模块以及完整的 GoogLeNet 网络结构，并包含 Kaiming 初始化权重逻辑。

data_partitioning.py: 数据集划分工具。自动将原始数据集按 9:1 的比例划分为训练集（train）和测试集（test）。

mean_std.py: 像素特征分析。遍历整个数据集，计算所有图像像素的均值（Mean）和标准差（Std），用于后续的标准化预处理。

train_model.py: 模型训练引擎。包含数据加载、Adam 优化器配置、交叉熵损失函数计算，并实时可视化训练/验证的准确率与 Loss。

test_model.py: 推理与评估脚本。加载训练好的 best_model.pth，对单张大米图片进行品种预测，并利用 Matplotlib 进行反归一化可视化展示。

算法细节