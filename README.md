# 食物图片分类项目

本项目是一个基于深度学习的图像分类系统，旨在识别11个不同类别中的食物。项目采用迁移学习方法，利用了预训练的ResNet50V2模型，以实现高精度的分类效果。

## 项目简介

项目目标是构建一个稳健的食物图像分类器。整个工作流程遵循标准的机器学习项目步骤，包括数据预处理、模型构建、训练和评估。

- **实现方法**: 迁移学习 (Transfer Learning)
- **基础模型**: ResNet50V2 (在ImageNet上预训练)
- **开发框架**: TensorFlow / Keras
- **最终评估准确率**: **约 81.9%**

## 数据集说明

本项目使用 **Food-11** 数据集。该数据集包含11个食物类别，已被划分为三个部分：

- `training/`: 用于训练模型的图片，共 9,866 张。
- `validation/`: 用于在训练过程中验证模型的图片，共 3,430 张。
- `evaluation/`: 用于最终评估模型性能的图片，共 3,347 张。

**注意**: 由于数据集体积较大，并未上传至本仓库。您可以从以下来源下载：
*   **Kaggle官方来源**: [Food-11 Image Dataset](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)

下载后，请解压并将`training`、`validation`和`evaluation`三个文件夹放置在项目根目录下。

## 文件结构

```
.
├── training/               # 训练图片 (需自行下载)
├── validation/             # 验证图片 (需自行下载)
├── evaluation/             # 评估图片 (需自行下载)
├── food_classification.py    # 用于训练和保存模型的主脚本
├── evaluate.py             # 用于加载和评估已保存模型的脚本
├── plot.py                 # 用于生成训练历史图表的脚本
├── food_classifier_model.keras # 已训练好的Keras模型文件
├── training_history.pkl    # 已保存的训练历史数据
├── requirements.txt        # 项目依赖的Python库
└── README.md               # 本说明文件
```

## 如何运行

1.  **克隆仓库**
    ```bash
    git clone https://github.com/shiqipzy/Machine_Learning.git
    cd Machine_Learning
    ```

2.  **环境配置**
    建议使用Python虚拟环境以避免依赖冲突。
    ```bash
    # 创建虚拟环境 (推荐)
    python -m venv venv

    # 激活虚拟环境
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    # source venv/bin/activate

    # 安装所需依赖
    pip install -r requirements.txt
    ```

3.  **下载并准备数据**
    - 从 [Kaggle](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset) 下载数据集。
    - 解压并将 `training`, `validation`, `evaluation` 文件夹移动到项目根目录。

4.  **运行脚本**
    - **从头开始训练模型** (这将花费较长时间，并覆盖已有的模型文件):
        ```bash
        python food_classification.py
        ```
    - **评估已提供的模型** (需要 `food_classifier_model.keras` 文件存在):
        ```bash
        python evaluate.py
        ```
    - **生成训练历史图表** (需要 `training_history.pkl` 文件存在):
        ```bash
        python plot.py
        ```