import numpy as np
import json
import os
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side


def perform_search(query_img_path):
    print("⏳ 正在加载索引库和模型...")
    # 1. 加载模型
    weights = np.load("vit-dinov2-base.npz")
    model = Dinov2Numpy(weights)

    # 2. 加载你刚刚跑完的 100% 成果
    gallery = np.load("gallery_features.npy")
    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"🔎 正在检索图片: {query_img_path}")
    # 3. 提取查询图特征并归一化
    query_tensor = resize_short_side(query_img_path)
    query_feat = model(query_tensor)[0]
    query_feat = query_feat / np.linalg.norm(query_feat)

    # 4. 计算余弦相似度 (矩阵乘法)
    similarities = gallery @ query_feat

    # 5. 取得分最高的 Top 10
    top_k = 10
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print("\n🏆 检索结果 Top 10:")
    print("-" * 60)
    for i in top_indices:
        score = similarities[i]
        info = metadata[i]
        print(f"[相似度: {score:.4f}]")
        print(f"描述: {info['caption']}")
        print(f"链接: {info['url']}\n")


if __name__ == "__main__":
    # 请确保这个路径下有一张图片用于测试
    test_image = "./demo_data/cat.jpg"

    if os.path.exists(test_image):
        perform_search(test_image)
    else:
        print(f"❌ 错误：找不到测试图 {test_image}，请指定一个存在的图片路径。")