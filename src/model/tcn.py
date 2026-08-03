"""VideoPose3D 风格因果膨胀 TCN。
输入: (B, T, C_in), 输出: (B, num_joints, 3)
因果: padding 只在时间左侧 (CausalConv1d), 输出取最后一帧。
"""
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1D 卷积, 左侧 padding, 右侧裁剪, 保证因果性。"""

    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.pad)

    def forward(self, x):
        x = self.conv(x)
        if self.pad > 0:
            x = x[..., :x.shape[-1] - self.pad]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.residual = (in_ch == out_ch)
        if not self.residual:
            self.res_conv = CausalConv1d(in_ch, out_ch, 1, 1)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        y = self.dropout(y)
        if self.residual:
            return self.relu(y + x)
        return self.relu(y + self.res_conv(x))


class TemporalConvNet(nn.Module):
    def __init__(self, num_input_channels=34, num_joints=17, receptive_field=81,
                 causal=True, num_layers=5, channels=1024, kernel_size=3,
                 dropout=0.25):
        super().__init__()
        self.causal = causal
        self.num_joints = num_joints

        # 计算膨胀率: 使总感受野 >= receptive_field
        dilations = []
        rf = 1
        d = 1
        while rf < receptive_field and len(dilations) < num_layers:
            dilations.append(d)
            rf += d * (kernel_size - 1)
            d *= 2
        assert len(dilations) > 0, "num_layers 不足以覆盖 receptive_field"
        self.receptive_field = rf

        # 输入投影
        self.in_conv = CausalConv1d(num_input_channels, channels, kernel_size, 1)
        self.in_bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        # 残差块
        blocks = []
        for dil in dilations:
            blocks.append(ResidualBlock(channels, channels, kernel_size, dil, dropout))
        self.blocks = nn.Sequential(*blocks)

        # 输出
        self.out_conv = nn.Conv1d(channels, num_joints * 3, 1)
        self.out_bn = nn.BatchNorm1d(num_joints * 3)

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)              # (B, C, T)
        y = self.relu(self.in_bn(self.in_conv(x)))
        y = self.blocks(y)
        y = self.out_bn(self.out_conv(y))  # (B, 51, T)
        y = y[..., -1]                     # 只取最后一帧 (因果)
        return y.reshape(-1, self.num_joints, 3)
