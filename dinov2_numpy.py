import numpy as np
from scipy.ndimage import zoom


def gelu(x):
    # DINOv2 官方使用精确版 GELU
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


class Embeddings:
    def __init__(self, weights):
        self.hidden_size = 768
        self.patch_size = 14
        self.cls_token = weights["embeddings.cls_token"]
        self.position_embeddings = weights["embeddings.position_embeddings"]
        self.patch_embed_w = weights["embeddings.patch_embeddings.projection.weight"].reshape(768, -1).T
        self.patch_embed_b = weights["embeddings.patch_embeddings.projection.bias"].reshape(1, 768)

    def interpolate_pos_encoding(self, height, width):
        """将 37x37 的位置编码插值到目标网格大小 """
        cls_token_pos = self.position_embeddings[:, :1, :]
        patch_pos = self.position_embeddings[:, 1:, :]

        old_grid_size = int(np.sqrt(patch_pos.shape[1]))
        patch_pos = patch_pos.reshape(1, old_grid_size, old_grid_size, self.hidden_size)

        new_h, new_w = height // self.patch_size, width // self.patch_size
        if new_h == old_grid_size and new_w == old_grid_size:
            return self.position_embeddings

        zoom_factors = (1, new_h / old_grid_size, new_w / old_grid_size, 1)
        interpolated_pos = zoom(patch_pos, zoom_factors, order=3)  # order=3 为双三次插值
        return np.concatenate([cls_token_pos, interpolated_pos.reshape(1, -1, self.hidden_size)], axis=1)

    def __call__(self, pixel_values):
        B, C, H, W = pixel_values.shape
        # 使用矩阵变换代替循环以提高效率
        x = pixel_values.reshape(B, C, H // 14, 14, W // 14, 14).transpose(0, 2, 4, 1, 3, 5).reshape(B, -1, C * 14 * 14)
        embeddings = x @ self.patch_embed_w + self.patch_embed_b

        cls_token = np.tile(self.cls_token, (B, 1, 1))
        embeddings = np.concatenate([cls_token, embeddings], axis=1)
        return embeddings + self.interpolate_pos_encoding(H, W)


class MultiHeadAttention:
    def __init__(self, config, prefix, weights):
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_size'] // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = Linear(weights[f"{prefix}.attention.query.weight"], weights[f"{prefix}.attention.query.bias"])
        self.k_proj = Linear(weights[f"{prefix}.attention.key.weight"], weights[f"{prefix}.attention.key.bias"])
        self.v_proj = Linear(weights[f"{prefix}.attention.value.weight"], weights[f"{prefix}.attention.value.bias"])
        self.out_proj = Linear(weights[f"{prefix}.output.dense.weight"], weights[f"{prefix}.output.dense.bias"])

    def __call__(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        attn = softmax((q @ k.transpose(0, 1, 3, 2)) * self.scale)
        context = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.out_proj(context)


# 基础模块定义
class Linear:
    def __init__(self, w, b): self.w, self.b = w, b

    def __call__(self, x): return x @ self.w.T + self.b


class LayerNorm:
    def __init__(self, w, b, eps=1e-6): self.w, self.b, self.eps = w, b, eps

    def __call__(self, x):
        u = x.mean(-1, keepdims=True)
        s = x.var(-1, keepdims=True)
        return self.w * (x - u) / np.sqrt(s + self.eps) + self.b


class TransformerBlock:
    def __init__(self, config, idx, weights):
        p = f"encoder.layer.{idx}"
        self.norm1 = LayerNorm(weights[f"{p}.norm1.weight"], weights[f"{p}.norm1.bias"])
        self.attn = MultiHeadAttention(config, f"{p}.attention", weights)
        self.ls1 = weights[f"{p}.layer_scale1.lambda1"]
        self.norm2 = LayerNorm(weights[f"{p}.norm2.weight"], weights[f"{p}.norm2.bias"])
        self.mlp = MLP(f"{p}", weights)
        self.ls2 = weights[f"{p}.layer_scale2.lambda1"]

    def __call__(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class MLP:
    def __init__(self, p, weights):
        self.fc1 = Linear(weights[f"{p}.mlp.fc1.weight"], weights[f"{p}.mlp.fc1.bias"])
        self.fc2 = Linear(weights[f"{p}.mlp.fc2.weight"], weights[f"{p}.mlp.fc2.bias"])

    def __call__(self, x): return self.fc2(gelu(self.fc1(x)))


class Dinov2Numpy:
    def __init__(self, weights):
        self.config = {"hidden_size": 768, "num_heads": 12, "num_layers": 12}
        self.embeddings = Embeddings(weights)
        self.blocks = [TransformerBlock(self.config, i, weights) for i in range(12)]
        self.norm = LayerNorm(weights["layernorm.weight"], weights["layernorm.bias"])

    def __call__(self, x):
        x = self.embeddings(x)
        for blk in self.blocks: x = blk(x)
        return self.norm(x)[:, 0]