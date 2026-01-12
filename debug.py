import numpy as np
from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

# 1. 加载模型
weights = np.load("vit-dinov2-base.npz")
vit = Dinov2Numpy(weights)

# 2. 提取特征
cat_pixels = center_crop("./demo_data/cat.jpg")
cat_feat = vit(cat_pixels)[0]

# 3. 对比参考值
try:
    ref_features = np.load("./demo_data/cat_dog_feature.npy")
    cat_ref = ref_features[0]

    # 余弦相似度计算
    cos_sim = np.dot(cat_feat, cat_ref) / (np.linalg.norm(cat_feat) * np.linalg.norm(cat_ref))
    print(f"余弦相似度: {cos_sim:.6f}")
    if cos_sim > 0.999:
        print("✅ 调试通过：特征对齐完美！")
    else:
        print("⚠️ 注意：特征对齐仍有偏差。")
except FileNotFoundError:
    print("未找到参考特征文件，仅输出前 5 位特征值：", cat_feat[:5])