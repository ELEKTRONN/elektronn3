# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""Neural network layers"""

from typing import Optional, Tuple

import torch
from torch import nn


class Identity(nn.Module):
    def forward(self, x):
        return x


class GatherExcite(nn.Module):
    """Gather-Excite module (https://arxiv.org/abs/1810.12348),

    a generalization of the Squeeze-and-Excitation module
    (https://arxiv.org/abs/1709.01507).

    Args:
        channels: Number of input channels (= number of output channels)
        extent: extent factor that determines how much the gather operator
            output is smaller than its input. The special value ``extent=0``
            activates global gathering (so the gathered information has no
            spatial extent).
        param_gather: If ``True``, the gather operator is parametrized
            according to https://arxiv.org/abs/1810.12348.
        param_excite: If ``True``, the excitation operator is parametrized
            according to https://arxiv.org/abs/1810.12348 (also equivalent to
            the original excitation operator proposed in
            https://arxiv.org/abs/1709.01507).
        reduction:  Channel reduction rate of the parametrized excitation
            operator.
        spatial_shape: Spatial shape of the module input. This needs to be
            specified if ``param_gather=0 and extent=0`` (parametrized global
            gathering).
    """
    def __init__(
            self,
            channels: int,
            extent: int = 0,
            param_gather: bool = False,
            param_excite: bool = True,
            reduction: int = 16,
            spatial_shape: Optional[Tuple[int, ...]] = None
    ):
        super().__init__()
        if extent == 1:
            raise NotImplementedError('extent == 1 doesn\'t make sense.')
        if param_gather:
            if extent == 0:  # Global parametrized gather operator
                if spatial_shape is None:
                    raise ValueError(
                        'With param_gather=True, extent=0, you will need to specify spatial_shape.')
                self.gather = nn.Sequential(
                    nn.Conv3d(channels, channels, spatial_shape),
                    nn.BatchNorm3d(channels),
                    nn.ReLU()
                )
            else:
                # This will make the model much larger with growing extent!
                # TODO: This is ugly and I'm not sure if it should even be supported
                assert extent in [2, 4, 8, 16]
                num_convs = int(torch.log2(torch.tensor(extent, dtype=torch.float32)))
                self.gather = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv3d(channels, channels, 3, stride=2, padding=1),
                        nn.BatchNorm3d(channels),
                        nn.ReLU()
                    ) for _ in range(num_convs)
                ])
        else:
            if extent == 0:
                self.gather = nn.AdaptiveAvgPool3d(1)  # Global average pooling
            else:
                self.gather = nn.AvgPool3d(extent)
        if param_excite:
            self.excite = nn.Sequential(
                nn.Conv3d(channels, channels // reduction, 1),
                nn.ReLU(),
                nn.Conv3d(channels // reduction, channels, 1)
            )
        else:
            self.excite = Identity()

        if extent == 0:
            self.interpolate = Identity()  # Use broadcasting instead of interpolation
        else:
            self.interpolate = torch.nn.functional.interpolate

    def forward(self, x):
        y = self.gather(x)
        y = self.excite(y)
        y = torch.sigmoid(self.interpolate(y, x.shape[2:]))
        return x * y


class AdaptiveConv3d(nn.Module):
    """Equivalent to ``torch.nn.Conv3d`` except that if
    ``kernel_size[0] == 1``, ``torch.nn.Conv2d`` is used internally in
    order to improve computation speed.

    This is a workaround for https://github.com/pytorch/pytorch/issues/7740.

    Current limitations:
    - Expects ``kernel_size`` to be passed as a keyword arg, not positional."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        ks = kwargs['kernel_size']
        if isinstance(ks, tuple) and ks[0] == 1:
            kwargs['kernel_size'] = ks[1:]
            kwargs['stride'] = kwargs.get('stride', (0, 1, 1))[1:]
            kwargs['padding'] = kwargs.get('padding', (0, 0, 0))[1:]
            kwargs['dilation'] = kwargs.get('dilation', (1, 1, 1))[1:]
            self.conv = nn.Conv2d(*args, **kwargs)
            self.forward = self.forward2d
        else:
            self.conv = nn.Conv3d(*args, **kwargs)
            self.forward = self.forward3d

    def forward2d(self, x):
        n, c, d, h, w = x.shape
        transp = x.transpose(1, 2)  # -> (N, D, C, H, W)
        view2d = transp.reshape(n * d, c, h, w)  # -> (N * D, C, H, W)
        out2dtransp = self.conv(view2d)
        h, w = out2dtransp.shape[-2:]  # H and W can be changed due to convolution
        c = self.conv.out_channels
        out3dtransp = out2dtransp.reshape(n, d, c, h, w)  # -> (N, D, C, H, W)
        out3d = out3dtransp.transpose(1, 2)  # -> (N, C, D, H, W)

        return out3d

    def forward3d(self, x):
        return self.conv(x)

    def forward(self, x): raise NotImplementedError()  # Chosen by __init__()


class AdaptiveConvTranspose3d(nn.Module):
    """Equivalent to ``torch.nn.ConvTranspose3d`` except that if
    ``kernel_size[0] == 1``, ``torch.nn.ConvTranspose2d`` is used internally in
    order to improve computation speed.

    This is a workaround for https://github.com/pytorch/pytorch/issues/7740.

    Current limitations:
    - Expects ``kernel_size`` to be passed as a keyword arg, not positional."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        ks = kwargs['kernel_size']
        if isinstance(ks, tuple) and ks[0] == 1:
            kwargs['kernel_size'] = ks[1:]
            kwargs['stride'] = kwargs.get('stride', (0, 1, 1))[1:]
            kwargs['padding'] = kwargs.get('padding', (0, 0, 0))[1:]
            kwargs['dilation'] = kwargs.get('dilation', (1, 1, 1))[1:]
            self.conv = nn.ConvTranspose2d(*args, **kwargs)
            self.forward = self.forward2d
        else:
            self.conv = nn.ConvTranspose3d(*args, **kwargs)
            self.forward = self.forward3d

    def forward2d(self, x):
        n, c, d, h, w = x.shape
        transp = x.transpose(1, 2)  # -> (N, D, C, H, W)
        view2d = transp.reshape(n * d, c, h, w)  # -> (N * D, C, H, W)
        out2dtransp = self.conv(view2d)
        h, w = out2dtransp.shape[-2:]  # H and W can be changed due to convolution
        c = self.conv.out_channels
        out3dtransp = out2dtransp.reshape(n, d, c, h, w)  # -> (N, D, C, H, W)
        out3d = out3dtransp.transpose(1, 2)  # -> (N, C, D, H, W)

        return out3d

    def forward3d(self, x):
        return self.conv(x)

    def forward(self, x): raise NotImplementedError()  # Chosen by __init__()


class ResizeConv(nn.Module):
    """Upsamples by 2x and applies a convolution.

    This is meant as a replacement for transposed convolution to avoid
    checkerboard artifacts. See

    - https://distill.pub/2016/deconv-checkerboard/
    - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, planar=False, dim=3, adaptive=False,
                 upsampling_mode='nearest'):
        super().__init__()
        # Avoiding cyclical import hell by importing here
        from elektronn3.modules import layer_helpers
        self.upsampling_mode = upsampling_mode
        self.scale_factor = 2
        if dim == 3 and planar:  # Only interpolate (H, W) dims, leave D as is
            self.scale_factor = layer_helpers.planar_kernel(self.scale_factor)
        self.dim = dim
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode=self.upsampling_mode)
        # TODO: Investigate if 3x3 or 1x1 conv makes more sense here and choose default accordingly
        # Preliminary notes:
        # - conv3 increases global parameter count by ~10%, compared to conv1 and is slower overall
        # - conv1 is the simplest way of aligning feature dimensions
        # - conv1 may be enough because in all common models later layers will apply conv3
        #   eventually, which could learn to perform the same task...
        #   But not exactly the same thing, because this layer operates on
        #   higher-dimensional features, which subsequent layers can't access
        #   (at least in U-Net out_channels == in_channels // 2).
        # --> Needs empirical evaluation
        if kernel_size == 3:
            self.conv = layer_helpers.conv3(
                in_channels, out_channels, padding=1,
                planar=planar, dim=dim, adaptive=adaptive
            )
        elif kernel_size == 1:
            self.conv = layer_helpers.conv1(in_channels, out_channels, dim=dim)
        else:
            raise ValueError(f'kernel_size={kernel_size} is not supported. Choose 1 or 3.')

    def forward(self, x):
        return self.conv(self.upsample(x))
