import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------
# U-net Model Definition
# ----------------------

# Full UNet1D with Full Channel + Temporal Attention
class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1D, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.bottleneck = DoubleConv(1024, 1024)

        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)

        self.outc = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.bottleneck(x5)

        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

# Attention Block: Channel + Temporal
class AttentionBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AttentionBlock1D, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Temporal Attention
        self.conv_temporal = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel-wise
        ca = self.fc(self.avg_pool(x))
        x = x * ca
        # Temporal-wise
        ta = self.conv_temporal(x)
        x = x * ta
        return x

# DoubleConv with Attention
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.attn = AttentionBlock1D(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        return x

# Downsampling block
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)

# Upsampling block
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if size mismatch due to rounding
        diff = x2.size(-1) - x1.size(-1)
        if diff > 0:
            x1 = F.pad(x1, (0, diff))
        elif diff < 0:
            x2 = F.pad(x2, (0, -diff))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

