import torch
import torch.nn as nn


def trunc_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -----------------------------
# Patch Embedding (EEG专用)
# -----------------------------
class PatchEmbed1D(nn.Module):
    """
    将输入的 EEG 2D 数据 (B, 1, 26, 496)
    沿时间维 (width=496) 切成 patch，保留所有通道维度 (26)
    输出为 (B, num_patches, embed_dim)
    """
    def __init__(self, in_chans=1, num_channels=26, signal_len=496, patch_size=16, embed_dim=128):
        super().__init__()
        assert signal_len % patch_size == 0, "采样点数量必须能被 patch_size 整除"
        self.num_patches = signal_len // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(num_channels, patch_size), stride=(num_channels, patch_size))

    def forward(self, x):
        # x: (B, 1, 26, 496)
        x = self.proj(x)            # (B, embed_dim, 1, num_patches)
        x = x.squeeze(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# -----------------------------
# Transformer Block
# -----------------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -----------------------------
# Transformer 分类模型
# -----------------------------
class EEGTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes=5,
        num_channels=26,
        signal_len=496,
        patch_size=16,
        embed_dim=128,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed1D(1, num_channels, signal_len, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token + 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer encoder
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        cls_output = x[:, 0]
        return self.head(cls_output)

if __name__ == "__main__":
    x = torch.randn(16, 1, 26, 496)
    model = EEGTransformerClassifier(
        num_classes=10,
        num_channels=26,
        signal_len=496,
        patch_size=16,
        embed_dim=128,
        depth=6,
        num_heads=4
    )
    y = model(x)
    print("输出维度:", y.shape)
