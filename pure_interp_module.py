# -*- coding: utf-8 -*-
"""Anchor-Interpolated Convolution with a compact learnable kernel."""
import torch
from torch import nn
import torch.nn.functional as F


class PureInterpolationConv2d(nn.Module):
    """Interpolate a learnable 3x3 kernel to a dense odd-sized kernel."""

    def __init__(
        self,
        in_channels,
        out_channels,
        target_size,
        stride=1,
        bias=False,
    ):
        super().__init__()
        if target_size < 3 or target_size % 2 == 0:
            raise ValueError("target_size must be an odd integer >= 3")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 3
        self.expanded_size = target_size
        self.stride = stride
        self.padding = target_size // 2

        self.odd_kernels = nn.Parameter(
            torch.empty(out_channels, in_channels, 3, 3)
        )
        nn.init.kaiming_normal_(
            self.odd_kernels, mode="fan_out", nonlinearity="relu"
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_channels)) if bias else None
        )

    def expanded_kernel(self):
        if self.expanded_size == 3:
            return self.odd_kernels
        return F.interpolate(
            self.odd_kernels,
            size=(self.expanded_size, self.expanded_size),
            mode="bilinear",
            align_corners=True,
        )

    def forward(self, x):
        return F.conv2d(
            x,
            self.expanded_kernel(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
        )
