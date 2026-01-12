# DINOv2 NumPy 图像检索项目

A pure NumPy implementation of DINOv2 (ViT-Base/14) for image retrieval, with no PyTorch/TensorFlow dependencies. This project completes feature extraction alignment and image retrieval tasks as required.

## 项目概述

本项目基于纯 NumPy 实现 DINOv2 视觉Transformer模型，核心目标是：
1. 实现与官方 PyTorch 版本高度对齐的特征提取（余弦相似度 ≥ 0.99）
2. 构建10,000+图像的图库索引
3. 支持图像检索功能，返回Top-10相似图像

### 核心特性
- **无框架依赖**：纯 NumPy + SciPy 实现，无需 PyTorch/TensorFlow
- **高效对齐**：特征提取与官方DINOv2对齐，数值误差控制在极小范围
- **完整流程**：包含数据预处理、模型推理、索引构建、图像检索全链路
- **轻量化**：代码简洁易懂，无冗余逻辑，适合学习与部署

## 项目结构

```
├── build_index.py        # 构建图库索引（提取10000+图像特征）
├── debug.py              # 调试脚本（验证特征与官方对齐）
├── dinov2_numpy.py       # 核心模型（纯NumPy实现DINOv2）
├── preprocess_image.py   # 图像预处理（缩放、归一化等）
├── search.py             # 检索脚本（输入图像返回Top-10相似结果）
├── data.csv              # 图库图像URL列表（需自行准备）
├── vit-dinov2-base.npz   # DINOv2权重文件（NumPy格式，需自行转换）
├── demo_data/            # 测试数据目录
│   ├── cat.jpg           # 测试图像1
│   ├── dog.jpg           # 测试图像2
│   └── cat_dog_feature.npy  # 官方参考特征（用于对齐验证）
├── gallery_features.npy  # 生成的图库特征（build_index.py输出）
├── metadata.json         # 图库图像元数据（build_index.py输出）
└── README.md             # 项目说明文档
```

## 环境依赖

```bash
# 安装核心依赖（Python 3.8+）
pip install numpy scipy pillow pandas requests
```

## 快速开始

### 1. 准备文件
#### （1）权重文件
- 下载官方DINOv2-Base PyTorch权重：[facebook/dinov2-base](https://huggingface.co/facebook/dinov2-base)
- 转换为NumPy格式（提供转换脚本如下）：

```python
# convert_pytorch_to_numpy.py
import torch
import numpy as np
from dinov2 import DinoV2

def main():
    # 加载PyTorch权重
    model = DinoV2.from_pretrained("facebook/dinov2-base", device_map="cpu")
    state_dict = model.state_dict()
    
    # 转换为NumPy格式
    np_weights = {k: v.cpu().numpy() for k, v in state_dict.items()}
    np.savez("vit-dinov2-base.npz", **np_weights)
    print("权重转换完成！生成 vit-dinov2-base.npz")

if __name__ == "__main__":
    main()
```

运行转换脚本：
```bash
pip install git+https://github.com/facebookresearch/dinov2.git
python convert_pytorch_to_numpy.py
```

#### （2）图库数据
- 准备 `data.csv` 文件，包含两列：`image_url`（图像URL）、`caption`（图像描述）
- 示例格式：
  ```csv
  image_url,caption
  https://example.com/image1.jpg,a cat sitting on sofa
  https://example.com/image2.jpg,a dog running in park
  ```

### 2. 调试与验证（任务一）
验证模型特征提取是否与官方对齐：
```bash
# 运行调试脚本
python debug.py
```

**预期输出**：
```
余弦相似度: 0.999234
✅ 调试通过：特征对齐完美！
```

- 若余弦相似度 ≥ 0.99，说明模型对齐成功
- 若未找到参考特征文件，会输出特征预览值

### 3. 构建图库索引（任务二）
提取10,000+图库图像特征并构建索引：
```bash
# 运行索引构建脚本（约5-10分钟，取决于网络速度）
python build_index.py
```

**输出文件**：
- `gallery_features.npy`：图库所有图像的特征矩阵（shape: [N, 768]）
- `metadata.json`：图库图像的元数据（URL和描述）

### 4. 图像检索（任务二）
输入查询图像，返回Top-10相似结果：
```bash
# 运行检索脚本（替换为你的查询图像路径）
python search.py
```

**预期输出**：
```
🔍 搜索结果 Top 10:
相似度: 0.9876 | 描述: a cat sitting on sofa
相似度: 0.9754 | 描述: a cat lying on bed
相似度: 0.9632 | 描述: a cat playing with ball
...
```

## 核心功能说明

### 1. 图像预处理（preprocess_image.py）
- `resize_short_side`：缩放图像短边至224，同时确保宽高均为14的倍数（DINOv2 Patch Size要求）
- 支持输入：图像路径或PIL图像对象
- 标准化：使用ImageNet统计量（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）

### 2. 模型实现（dinov2_numpy.py）
纯NumPy实现DINOv2核心模块：
- **Embeddings**：Patch嵌入 + CLS Token + 位置编码（双三次插值适配不同尺寸）
- **MultiHeadAttention**：多头注意力机制（支持12头，维度768）
- **TransformerBlock**：Pre-LN结构 + LayerScale + MLP
- **GELU**：使用官方精确公式（0.5 * x * (1 + erf(x / sqrt(2)))）

### 3. 索引构建（build_index.py）
- 多线程下载图像（20线程加速）
- 批量提取图像特征并归一化（余弦相似度可直接通过点积计算）
- 自动跳过下载失败或处理异常的图像

### 4. 图像检索（search.py）
- 支持余弦相似度计算（高效点积实现）
- 快速排序返回Top-10相似图像
- 输出相似度分数和图像描述

## 注意事项
1. 权重文件要求：`vit-dinov2-base.npz` 必须包含完整的DINOv2-Base权重（键名需与模型预期一致）
2. 图像尺寸要求：预处理后图像宽高需为14的倍数，否则会报错
3. 网络要求：构建索引时需保证网络能访问 `data.csv` 中的图像URL
4. 性能优化：若图库图像超过10万张，建议使用FAISS优化检索速度

## 常见问题
### Q1: 运行debug.py提示KeyError（权重键名不匹配）
A1: 检查权重文件键名格式，官方DINOv2权重键名如 `encoder.layer.0.attention.query.weight`，若权重键名带 `model.` 前缀，需在模型代码中添加前缀适配。

### Q2: 构建索引时下载图像失败
A2: 检查URL有效性，或增加超时时间（修改 `process_row` 函数中的 `timeout=5` 为 `timeout=10`）。

### Q3: 检索相似度偏低
A3: 确保查询图像预处理流程与图库一致，或重新生成参考特征文件。

## 许可证
本项目基于MIT许可证开源，仅供学习和研究使用。使用时请遵守DINOv2官方许可证要求。

## 致谢
- 基于Facebook Research的DINOv2模型：[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- 参考PyTorch官方实现与NumPy数值计算最佳实践
