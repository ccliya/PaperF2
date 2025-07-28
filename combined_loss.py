from VGG_loss import *
from torchvision import models

class combinedloss(nn.Module):
    def __init__(self, config):
        super(combinedloss, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        print("VGG model is loaded")
        self.vggloss = VGG_loss(vgg, config)
        for param in self.vggloss.parameters():
            param.requires_grad = False
        self.mseloss = nn.MSELoss().to(config.device)
        self.l1loss = nn.L1Loss().to(config.device)

    def forward(self, out, label):
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        mse_loss = self.mseloss(out, label)
        vgg_loss = self.l1loss(inp_vgg, label_vgg)
        total_loss = mse_loss + vgg_loss
        return total_loss, mse_loss, vgg_loss

from pytorch_msssim import ssim
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss2(nn.Module):
    def __init__(self, config):
        super(CombinedLoss2, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        print("VGG model is loaded")
        self.vggloss = VGG_loss(vgg, config)
        for param in self.vggloss.parameters():
            param.requires_grad = False

        self.mseloss = nn.MSELoss().to(config.device)
        self.l1loss = nn.L1Loss().to(config.device)
        self.device = config.device

    def ssim_loss(self, out, label):
        return 1 - ssim(out, label, data_range=1, size_average=True)

    def tv_loss(self, x):
        tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return tv_h + tv_w

    def color_loss(self, out, label):
        mean_out = out.mean(dim=[2, 3])
        mean_label = label.mean(dim=[2, 3])
        return F.l1_loss(mean_out, mean_label)

    def forward(self, out, label):
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)

        mse_loss = self.mseloss(out, label)
        vgg_loss = self.l1loss(inp_vgg, label_vgg)
        ssim_loss = self.ssim_loss(out, label)
        tv_loss = self.tv_loss(out)
        color_loss = self.color_loss(out, label)
        total_loss = (
            1.0 * mse_loss +
            0.2 * vgg_loss +
            0.5 * ssim_loss +
            0.05 * tv_loss +
            0.1 * color_loss 
        )

        return total_loss, mse_loss, vgg_loss
