import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIToMemoryEncoder(nn.Module):
    """
    Input:
        roi_feat: [B, C, H, W], e.g. [4, 256, 7, 7]
    Output:
        memory_token: [B, 1, C], e.g. [4, 1, 256]
    """
    def __init__(self, dim=256, cfg=None,reduction=4,use_spatial_gate=False):
        super().__init__()
        self.cfg = cfg
        self.use_spatial_gate = use_spatial_gate
        hidden = max(dim // reduction, 32)

        self.local_fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # channel reweighting
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # spatial reweighting (not pooling)
        if self.use_spatial_gate:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

        # output projection in dense feature space
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, roi_feat):
        """
        roi_feat: [B, C, H, W]
        return:   [B, 1, C]
        """
        # local enhancement
        x = self.local_fuse(roi_feat) + roi_feat   # residual
        # channel gating
        ch_gate = self.channel_gate(x)             # [B, C, 1, 1]
        x = x * ch_gate
        # spatial gating
        if self.use_spatial_gate:
            sp_gate = self.spatial_gate(x)         # [B, 1, H, W]
            x = x * sp_gate
        # dense projection
        x = self.out_proj(x) + x                   # residual

        return x


def build_memory_encoder(encoder):
    num_channels_enc = encoder.num_channels
    memory_encoder = ROIToMemoryEncoder(dim=num_channels_enc)
    return memory_encoder
