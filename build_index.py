import pandas as pd
import numpy as np
import requests
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side
from PIL import Image

# ================= 配置区 =================
LIMIT = 10000  # 任务二最低标准
MAX_WORKERS = 20  # 并行下载线程数
DATA_PATH = "data.csv"  # 原始数据
WEIGHTS_PATH = "vit-dinov2-base.npz"


# ==========================================

def process_row(row, model):
    """单条数据处理逻辑"""
    url = row['image_url']
    caption = row['caption']
    try:
        # 1. 下载图片
        resp = requests.get(url, timeout=5, stream=True)
        if resp.status_code != 200:
            return None

        # 2. 预处理
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img_tensor = resize_short_side(img)  # 调用你写的 224 + 14倍数对齐函数

        # 3. 提取特征
        feat = model(img_tensor)[0]

        # 4. 归一化 (方便后续直接用点积算相似度)
        norm_feat = feat / (np.linalg.norm(feat) + 1e-6)

        return {
            "feature": norm_feat,
            "metadata": {"url": url, "caption": caption}
        }
    except Exception:
        # 忽略下载失败或格式错误的图片
        return None


def main():
    # 0. 环境检查
    if not os.path.exists(WEIGHTS_PATH):
        print(f"❌ 错误: 找不到权重文件 {WEIGHTS_PATH}")
        return

    print("🚀 正在初始化 DINOv2 模型...")
    weights = np.load(WEIGHTS_PATH)
    model = Dinov2Numpy(weights)

    print(f"📖 正在读取数据并准备处理前 {LIMIT} 条...")
    df = pd.read_csv(DATA_PATH).head(LIMIT)

    all_features = []
    all_metadata = []
    count = 0
    success_count = 0

    print(f"⚡ 开始并行构建索引 (线程数: {MAX_WORKERS})...")

    # 使用 ThreadPoolExecutor 并实时打印进度
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(process_row, row, model): row for _, row in df.iterrows()}

        for future in as_completed(futures):
            count += 1
            result = future.result()

            if result:
                all_features.append(result["feature"])
                all_metadata.append(result["metadata"])
                success_count += 1

            if count % 100 == 0:
                print(f"⏳ 进度: {count}/{LIMIT} | 成功提取: {success_count}")

    # ================= 保存逻辑 (核心修改) =================
    print("\n💾 正在将索引写入磁盘...")

    if len(all_features) > 0:
        # 获取当前绝对路径，防止文件“失踪”
        current_dir = os.path.dirname(os.path.abspath(__file__))
        feat_path = os.path.join(current_dir, "gallery_features.npy")
        meta_path = os.path.join(current_dir, "metadata.json")

        # 执行保存
        np.save(feat_path, np.array(all_features))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

        print(f"✅ 构建完成！")
        print(f"📦 最终有效索引数量: {success_count}")
        print(f"📍 特征库路径: {feat_path}")
        print(f"📍 元数据路径: {meta_path}")
    else:
        print("❌ 严重错误: 没有成功提取到任何特征，请检查网络连接或 data.csv 中的图片链接。")


if __name__ == "__main__":
    main()