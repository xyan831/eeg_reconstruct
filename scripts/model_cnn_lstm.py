import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden=128, lstm_layers=1):
        super(CNN_LSTM, self).__init__()
        # -----------------
        # CNN特征提取部分
        # -----------------
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))  # 保留“时间序列”方向

        # -----------------
        # LSTM时序建模部分
        # -----------------
        # 假设 CNN 提取的最后特征是 (B, 64, 1, T)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # -----------------
        # 分类层
        # -----------------
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)  # 双向LSTM输出×2

    def forward(self, x):
        # x: (B, 1, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # (B, 64, 1, T)
        
        # 调整维度以匹配LSTM输入: (B, T, 64)
        x = x.squeeze(2).permute(0, 2, 1)

        # LSTM输出
        lstm_out, _ = self.lstm(x)  # (B, T, 2*hidden)
        last_out = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # 分类
        out = self.fc(last_out)  # (B, num_classes)
        return out
