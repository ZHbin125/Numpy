import numpy as np
from PIL import Image


def resize_short_side(img_or_path, target_size=224):
    """任务二核心：缩放短边并对齐 14 倍数"""
    if isinstance(img_or_path, str):
        image = Image.open(img_or_path).convert("RGB")
    else:
        image = img_or_path.convert("RGB")

    w, h = image.size
    # 1. 计算比例，缩放短边到 224
    if w < h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))

    # 2. 强制对齐到 14 的倍数 (DINOv2 要求)
    new_w = (new_w // 14) * 14
    new_h = (new_h // 14) * 14

    image = image.resize((new_w, new_h), resample=Image.BICUBIC)

    # 标准化
    arr = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)