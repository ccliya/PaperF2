import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.color as Kcolor
from cube_attention import AdaConvModule


# —— AdaIN 对齐函数 —— #
def adain(x, y, eps=1e-5):
    b, c, h, w = x.size()
    # x 统计
    x_mean = x.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    x_std  = x.view(b, c, -1).std(dim=2).view(b, c, 1, 1) + eps
    # y 统计
    y_mean = y.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    y_std  = y.view(b, c, -1).std(dim=2).view(b, c, 1, 1) + eps
    return (x - x_mean) / x_std * y_std + y_mean

class ELA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw      = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw      = nn.Conv2d(channels, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.dw(x); y = self.pw(y)
        return x * self.sigmoid(y)

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv  = nn.Conv2d(64,64,3,padding=1,bias=False)
        self.drop  = nn.Dropout2d(0.2)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(64,61,3,padding=1,bias=False)
    def forward(self, x):
        fmap, inp = x
        y = self.relu(self.drop(self.conv(self.relu(self.drop(self.conv(fmap))))))
        y = self.relu(self.conv1(y))
        return torch.cat([y, inp], dim=1), inp

# —— DEConv 插件 —— #
class Conv2d_cd(nn.Module):
    def __init__(self,in_ch,out_ch,theta=1.0):
        super().__init__()
        self.conv  = nn.Conv2d(in_ch,out_ch,3,padding=1,bias=True)
        self.theta = theta
    def get_weight(self):
        w = self.conv.weight; o,i,k1,k2 = w.shape
        flat = w.view(i,o,-1)
        cd   = flat.clone(); cd[:,:,4] -= flat.sum(dim=2)
        cd   = cd.view(i,o,k1,k2)
        return cd, self.conv.bias

class Conv2d_ad(nn.Module):
    def __init__(self,in_ch,out_ch,theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,3,padding=1,bias=True)
        self.theta= theta
    def get_weight(self):
        w = self.conv.weight; o,i,k1,k2 = w.shape
        flat = w.view(i,o,-1)
        idx  = [3,0,1,6,4,2,7,8,5]
        ad   = flat - self.theta*flat[:,:,idx]
        ad   = ad.view(i,o,k1,k2)
        return ad, self.conv.bias

class Conv2d_rd(nn.Module):
    def __init__(self,in_ch,out_ch,theta=1.0):
        super().__init__()
        self.conv  = nn.Conv2d(in_ch,out_ch,3,padding=2,bias=True)
        self.theta = theta
    def forward(self,x):
        if abs(self.theta)<1e-8:
            return self.conv(x)
        w = self.conv.weight; o,i,k1,k2 = w.shape
        flat = w.view(i,o,-1)
        rd   = x.new_zeros(i,o,25)
        rd[:,:, [0,2,4,10,14,20,22,24]] = flat[:,:,1:]
        rd[:,:, [6,7,8,11,13,16,17,18]] = -flat[:,:,1:]*self.theta
        rd[:,:,12] = flat[:,:,0]*(1-self.theta)
        rd   = rd.view(i,o,5,5)
        return F.conv2d(x, rd, bias=self.conv.bias,
                        padding=self.conv.padding,
                        stride=self.conv.stride,
                        groups=self.conv.groups)

class Conv2d_hd(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch,out_ch,3,padding=1,bias=True)
    def get_weight(self):
        w = self.conv.weight; o,i,k = w.shape
        hd= w.new_zeros(i,o,9)
        hd[:,:, [0,3,6]] = w[:,:,0:1]
        hd[:,:, [2,5,8]] = -w[:,:,0:1]
        hd = hd.view(i,o,3,3)
        return hd, self.conv.bias

class Conv2d_vd(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch,out_ch,3,padding=1,bias=True)
    def get_weight(self):
        w = self.conv.weight; o,i,k = w.shape
        vd= w.new_zeros(i,o,9)
        vd[:,:,0:3] = w[:,:,0:1]
        vd[:,:,6:9] = -w[:,:,0:1]
        vd = vd.view(i,o,3,3)
        return vd, self.conv.bias

class DEConv(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.cd  = Conv2d_cd(dim,dim)
        self.hd  = Conv2d_hd(dim,dim)
        self.vd  = Conv2d_vd(dim,dim)
        self.ad  = Conv2d_ad(dim,dim)
        self.rd  = Conv2d_rd(dim,dim)
        self.std = nn.Conv2d(dim,dim,3,padding=1,bias=True)
    def forward(self,x):
        w1,b1 = self.cd.get_weight()
        w2,b2 = self.hd.get_weight()
        w3,b3 = self.vd.get_weight()
        w4,b4 = self.ad.get_weight()
        y5    = self.rd(x)
        w6,b6 = self.std.weight, self.std.bias
        w_sum = w1+w2+w3+w4+w6; b_sum = b1+b2+b3+b4+b6
        out   = F.conv2d(x,w_sum,bias=b_sum,padding=1)
        return out + y5


class CrossAttentionFuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_a = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv_b = nn.Conv2d(channels, channels, 1, bias=False)

        self.attn_a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=True),
            nn.Sigmoid()
        )

        self.attn_b = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=True),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_a, feat_b):
        """
        feat_a: ol0
        feat_b: o0
        返回融合增强后的两个分支
        """
        # 通道注意力 (基于对方输入生成门控因子)
        gate_a = self.attn_b(feat_b)  # B -> A 的引导
        gate_b = self.attn_a(feat_a)  # A -> B 的引导

        # 特征调制 + 残差连接
        out_a = feat_a + self.gamma * self.conv_a(feat_a * gate_a)
        out_b = feat_b + self.gamma * self.conv_b(feat_b * gate_b)

        return out_a, out_b
    
class UWnet(nn.Module):
    def __init__(self, num_layers=3, base_ch=64):
        super().__init__()
        # ——— 共享编码器部分 ——— #
        self.rgb_in  = nn.Conv2d(3, base_ch, 3, padding=1, bias=False)
        self.relu    = nn.ReLU(inplace=True)
        self.pool    = nn.MaxPool2d(2)
        self.reduce  = nn.Conv2d(base_ch*2, base_ch, 1, bias=False)

        # ConvBlocks
        self.blocks  = nn.Sequential(*[ConvBlock() for _ in range(num_layers)])
        self.ela     = ELA(base_ch)
        self.deconv  = DEConv(base_ch)

        # 浅层融合
        self.shallow_fuse = nn.Conv2d(base_ch*2, base_ch, 1, bias=False)
        self.shallow_norm = nn.InstanceNorm2d(base_ch, affine=True)
        # 中层融合
        self.mid_fuse     = nn.Conv2d(base_ch*2, base_ch, 1, bias=False)
        self.mid_norm     = nn.InstanceNorm2d(base_ch, affine=True)
        # 深层融合
        self.deep_fuse    = nn.Conv2d(base_ch*2, base_ch, 1, bias=False)
        self.deep_norm    = nn.InstanceNorm2d(base_ch, affine=True)

        # 输出层
        self.output       = nn.Conv2d(base_ch, 3, 3, padding=1, bias=False)

        self.fusion = AdaConvModule(base_ch)
    
    def forward(self, x):
        B, _, H, W = x.shape
        # —— RGB 路径浅层特征 —— #
        f   = self.relu(self.rgb_in(x))      # [B,64,H,W]
        p   = self.pool(f)                   # [B,64,⌊H/2⌋,⌊W/2⌋]
        # —— **改动：按原始 f 的 H×W 做插值** —— #
        u   = F.interpolate(p, size=(H, W),
                            mode='bilinear', align_corners=False)
        c   = torch.cat([u, f], dim=1)       # [B,128,H,W]
        o0  = self.relu(self.reduce(c))      # [B,64,H,W]

        # —— Lab 路径浅层特征 —— #
        lab = Kcolor.rgb_to_lab(x)
        fl  = self.relu(self.rgb_in(lab))
        pl  = self.pool(fl)
        ul  = F.interpolate(pl, size=(H, W),
                            mode='bilinear', align_corners=False)
        cl  = torch.cat([ul, fl], dim=1)
        ol0 = self.relu(self.reduce(cl))

        # —— 浅层融合 + AdaIN 对齐 —— #
        ash = self.fusion(ol0, o0)
        sh  = self.shallow_fuse(torch.cat([ash, ol0], dim=1))
        sh  = self.shallow_norm(sh)

        # —— 中层特征 —— #
        om, _ = self.blocks((sh, x))         # [B,64,H,W]
        om    = self.ela(om)
        om    = self.deconv(om)

        # —— Lab 中层 —— #
        olm, _ = self.blocks((ol0, lab))
        olm    = self.ela(olm)
        olm    = self.deconv(olm)

        # —— 中层融合 + AdaIN —— #
        am  = self.fusion(olm, om)
        mid = self.mid_fuse(torch.cat([am, olm], dim=1))
        mid = self.mid_norm(mid)

        # —— 深层特征 —— #
        od, _ = self.blocks((mid, x))
        od    = self.ela(od)
        od    = self.deconv(od)

        # —— Lab 深层 —— #
        old, _ = self.blocks((olm, lab))
        old    = self.ela(old)
        old    = self.deconv(old)

        # —— 深层融合 + AdaIN —— #
        ad  = self.fusion(old, od)
        dp  = self.deep_fuse(torch.cat([ad, old], dim=1))
        dp  = self.deep_norm(dp)

        return self.output(dp)

