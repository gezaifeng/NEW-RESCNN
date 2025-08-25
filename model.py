import torch
import torch.nn as nn
import torch.nn.functional as F
class ResBlock(nn.Module):
    """带 BatchNorm 和 Dropout 的残差块"""
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(0.2)

        # 匹配维度的投影
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += identity
        return F.relu(out)

class ResCNN(nn.Module):
    def __init__(self, out_dim: int = 76):
        super(ResCNN, self).__init__()

        # ===== 特征提取主干（保持你原来的结构不变）=====
        self.block1 = ResBlock(3, 32)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.block2 = ResBlock(32, 64)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.block3 = ResBlock(64, 128)

        self.flatten_dim = 128 * 1 * 1

        self.fc1     = nn.Linear(self.flatten_dim, 256)
        self.bn_fc1  = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.3)

        self.fc2     = nn.Linear(256, 128)
        self.bn_fc2  = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.3)

        # ===== 主体光谱头：输出 76 维细节（峰形等）=====
        self.fc3 = nn.Linear(128, out_dim)

        # ===== 基线校正头（新增）=====
        # 作用：输出低维系数，表示全谱的“平滑基线（偏移+缓慢弯曲）”
        # 好处：把“整体抬升/弯曲”等低频趋势单独建模，主干更专注峰形细节
        self.base_head = nn.Linear(128, 3)   # [b0, b1, b2] 3 个系数

        # ===== 多项式基（新增）=====
        # 在规范化波长轴 [0,1] 上构造 1、λ、λ^2，后续与 [b0,b1,b2] 相乘生成基线
        lam  = torch.linspace(0.0, 1.0, out_dim)          # 76 个点
        poly = torch.stack([torch.ones_like(lam), lam, lam**2], dim=0)  # 形状 (3,76)
        self.register_buffer("poly", poly)  # 作为 buffer，参与前向传播，且随模型保存/加载

    def forward(self, x):
        # ===== 主干特征提取（保持不变）=====
        x = self.pool1(self.block1(x))   # (B,32,2,3)
        x = self.pool2(self.block2(x))   # (B,64,1,1)
        x = self.block3(x)               # (B,128,1,1)

        x = x.view(x.size(0), -1)        # (B,128)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))

        # ===== 分头输出 =====
        spec_core = self.fc3(x)          # (B,76) 主体光谱：负责峰形等高频细节
        base_coef = self.base_head(x)    # (B,3)  基线系数：[b0,b1,b2]

        # ===== 基线展开并相加（核心新增逻辑）=====
        # base = [b0,b1,b2] @ [1; λ; λ^2]  →  (B,3)@(3,76) = (B,76)
        base = base_coef @ self.poly      # (B,76)

        y = spec_core + base              # (B,76) 最终预测 = 细节 + 平滑基线

        # ❌ 移除 softplus：否则会把应为负的小值抬正，导致尾段/基线系统性偏高
        # return F.softplus(y)
        return y
