import copy
from typing import Optional, Tuple

import torch
from torch import nn


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


def get_conv(dim=3, adaptive=False):
    if dim == 3:
        return AdaptiveConv3d if adaptive else nn.Conv3d
    elif dim == 2:
        return nn.Conv2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_convtranspose(dim=3, adaptive=False):
    if dim == 3:
        return AdaptiveConvTranspose3d if adaptive else nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_maxpool(dim=3):
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d
    else:
        raise ValueError('dim has to be 2 or 3')


def get_batchnorm(dim=3):
    if dim == 3:
        return nn.BatchNorm3d
    elif dim == 2:
        return nn.BatchNorm2d
    else:
        raise ValueError('dim has to be 2 or 3')


def planar_kernel(x):
    if isinstance(x, int):
        return (1, x, x)
    else:
        return x


def planar_pad(x):
    if isinstance(x, int):
        return (0, x, x)
    else:
        return x


def _conv3(in_channels, out_channels, kernel_size=3, stride=1,
           padding=1, bias=True, planar=False, dim=3, adaptive=False):
    if planar:
        stride = planar_kernel(stride)
        padding = planar_pad(padding)
        kernel_size = planar_kernel(kernel_size)
    return get_conv(dim, adaptive)(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias
    )


def _upconv2(in_channels, out_channels, mode='transpose', planar=False, dim=3, adaptive=False):
    kernel_size = 2
    stride = 2
    scale_factor = 2
    if planar:
        kernel_size = planar_kernel(kernel_size)
        stride = planar_kernel(stride)
        scale_factor = planar_kernel(scale_factor)
    if mode == 'transpose':
        return get_convtranspose(dim, adaptive)(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
    else:
        # out_channels is always going to be the same
        # as in_channels
        mode = 'trilinear' if dim == 3 else 'bilinear'
        return nn.Sequential(
            nn.Upsample(mode=mode, scale_factor=scale_factor),
            _conv1(in_channels, out_channels, dim=dim)
        )


def _conv1(in_channels, out_channels, dim=3):
    return get_conv(dim)(
        in_channels,
        out_channels,
        kernel_size=1,
    )


class LinearActivation(nn.Module):
    def forward(self, x):
        return x


def get_activation(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky':
            return nn.LeakyReLU(negative_slope=0.1)
        elif activation == 'prelu':
            return nn.PReLU(num_parameters=1)
        elif activation == 'rrelu':
            return nn.RReLU()
        elif activation == 'lin':
            return LinearActivation()
    else:
        # Deep copy is necessary in case of paremtrized activations
        return copy.deepcopy(activation)
