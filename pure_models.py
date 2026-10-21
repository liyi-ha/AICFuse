# -*- coding: utf-8 -*-
"""AICFuse backbone with anchor-interpolated 5x5, 7x7 and 9x9 kernels."""
import torch
from torch import nn

from pure_interp_module import PureInterpolationConv2d


CHANNEL = 64
VARIANT_TO_SIZE = {
    "PureInterp5": 5,
    "PureInterp7": 7,
    "PureInterp9": 9,
}
VARIANTS = tuple(VARIANT_TO_SIZE)
EXPECTED_PARAMS = {variant: 260_546 for variant in VARIANTS}


def block(variant, in_channels, out_channels, activation):
    layers = [
        PureInterpolationConv2d(
            in_channels,
            out_channels,
            target_size=VARIANT_TO_SIZE[variant],
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    ]
    if activation == "silu":
        layers.append(nn.SiLU(inplace=True))
    elif activation == "tanh":
        layers.append(nn.Tanh())
    elif activation == "sigmoid":
        layers.append(nn.Sigmoid())
    else:
        raise ValueError(activation)
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.cov1 = block(variant, 1, CHANNEL, "silu")
        self.cov2 = block(variant, CHANNEL, CHANNEL, "silu")
        self.cov3 = block(variant, CHANNEL, CHANNEL, "tanh")
        self.cov4 = block(variant, CHANNEL, CHANNEL, "tanh")

    def forward(self, x):
        f1 = self.cov1(x)
        f2 = self.cov2(f1)
        fb = self.cov3(f2)
        fd = self.cov4(f2)
        return f1, f2, fb, fd


class Decoder(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.cov5 = block(variant, CHANNEL * 2, CHANNEL, "silu")
        self.cov6 = block(variant, CHANNEL * 2, CHANNEL, "silu")
        self.cov7 = block(variant, CHANNEL * 2, 1, "sigmoid")

    def forward(self, f1, f2, fb, fd):
        o1 = self.cov5(torch.cat([fb, fd], dim=1))
        o2 = self.cov6(torch.cat([o1, f2], dim=1))
        return self.cov7(torch.cat([o2, f1], dim=1))


def build(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown pure-interpolation variant: {variant}")
    return Encoder(variant), Decoder(variant)


def nparams(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def self_check():
    for variant, target_size in VARIANT_TO_SIZE.items():
        encoder, decoder = build(variant)
        x = torch.randn(2, 1, 31, 35)
        with torch.no_grad():
            features = encoder(x)
            output = decoder(*features)
        params = nparams(encoder) + nparams(decoder)
        assert output.shape == x.shape
        assert params == EXPECTED_PARAMS[variant]
        for module in list(encoder.modules()) + list(decoder.modules()):
            if isinstance(module, PureInterpolationConv2d):
                assert module.expanded_kernel().shape[-2:] == (
                    target_size, target_size
                )
                assert not hasattr(module, "gate_logit")
        print(
            f"{variant:12s} 3x3 -> {target_size}x{target_size} "
            f"Params={params:,} Shape={tuple(x.shape)}->{tuple(output.shape)}"
        )
    print("[PASS] pure interpolation 5/7/9 models are internally consistent.")


if __name__ == "__main__":
    self_check()
