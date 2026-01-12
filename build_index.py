import pandas as pd
import numpy as np
import requests, io, json
from concurrent.futures import ThreadPoolExecutor
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side
from PIL import Image

# 初始化
model = Dinov2Numpy(np.load("vit-dinov2-base.npz"))
LIMIT = 10000  # 满足任务要求的最低标准


def process_row(row):
    try:
        resp = requests.get(row['image_url'], timeout=5)
        img = Image.open(io.BytesIO(resp.content))
        feat = model(resize_short_side(img))[0]
        # 预先归一化，搜索时直接点积即为余弦相似度
        return feat / np.linalg.norm(feat), {"url": row['image_url'], "caption": row['caption']}
    except:
        return None


def main():
    df = pd.read_csv("data.csv").head(LIMIT)
    print(f"🚀 正在构建索引 (10,000张)...")

    feats, metas = [], []
    count = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        # 使用 as_completed 或直接循环 map 来打印进度
        for result in executor.map(process_row, [row for _, row in df.iterrows()]):
            count += 1
            if result:
                feats.append(result[0])
                metas.append(result[1])

            # 每 100 张打印一次进度
            if count % 100 == 0:
                print(f"⌛ 已处理: {count}/10000 ({(count / 10000) * 100:.1f}%)")


if __name__ == "__main__":
    main()