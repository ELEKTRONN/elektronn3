# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""Small helper functions for abstracting over 2D and 3D networks"""

import copy

from torch import nn

from elektronn3.modules import AdaptiveConv3d, AdaptiveConvTranspose3d, Identity


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


def conv3(in_channels, out_channels, kernel_size=3, stride=1,
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


def upconv2(in_channels, out_channels, mode='transpose', planar=False, dim=3, adaptive=False):
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
    elif mode == 'resize':
        """
        # TODO: needs refinement to work with arbitrary kernel size, stride and padding etc.
        https://distill.pub/2016/deconv-checkerboard/
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/190
        """
        assert dim == 2
        return nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
           nn.ReflectionPad2d(1),
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1))
    else:
        # out_channels is always going to be the same
        # as in_channels
        mode = 'trilinear' if dim == 3 else 'bilinear'
        return nn.Sequential(
            nn.Upsample(mode=mode, scale_factor=scale_factor),
            conv1(in_channels, out_channels, dim=dim)
        )


def conv1(in_channels, out_channels, dim=3):
    return get_conv(dim)(
        in_channels,
        out_channels,
        kernel_size=1,
    )


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
            return Identity()
    else:
        # Deep copy is necessary in case of paremtrized activations
        return copy.deepcopy(activation)
