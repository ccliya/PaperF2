import torch
import torch.nn as nn
import torch.nn.functional as F

class KernelPredictor(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.out_channels = in_channels

        hidden_dim = in_channels // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels * kernel_size * kernel_size, 3, padding=1)
        )

    def forward(self, style_feat):
        B, C, H, W = style_feat.shape
        kernels = self.encoder(style_feat)  # [B, C * K*K, H, W]
        kernels = kernels.view(B, C, self.kernel_size * self.kernel_size, H, W)  # [B, C, K*K, H, W]
        return kernels

class AdaConv2d(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x, dynamic_kernel):
        B, C, H, W = x.shape
        K = self.kernel_size

        # unfold → [B, C*K*K, H*W]
        x_unfold = F.unfold(x, kernel_size=K, padding=self.padding)  # [B, C*K*K, H*W]
        x_unfold = x_unfold.view(B, C, K * K, H, W)  # [B, C, K*K, H, W]

        # element-wise mul + sum over K*K
        out = (x_unfold * dynamic_kernel).sum(dim=2)  # [B, C, H, W]
        return out

class AdaConvModule(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.kernel_predictor = KernelPredictor(in_channels, kernel_size)
        self.adaptive_conv = AdaConv2d(in_channels, kernel_size)

    def forward(self, content_feat, style_feat):
        # 输入: content_feat, style_feat → 输出结构感知风格融合特征
        kernels = self.kernel_predictor(style_feat)
        out = self.adaptive_conv(content_feat, kernels)
        return out
