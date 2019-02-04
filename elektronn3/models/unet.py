# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Author: Martin Drawitsch

"""
This is a modified version of the U-Net CNN architecture for biomedical
image segmentation. U-Net was originally published in
https://arxiv.org/abs/1505.04597 by Ronneberger et al.

A pure-3D variant of U-Net has been proposed by Çiçek et al.
in https://arxiv.org/abs/1606.06650, but the below implementation
is based on the original U-Net paper, with several improvements.

This code is based on https://github.com/jaxony/unet-pytorch
(c) 2017 Jackson Huang, released under MIT License,
which implements (2D) U-Net with user-defined network depth
and a few other improvements of the original architecture.

Major differences of this version from Huang's code:
- Operates on 3D image data (5D tensors) instead of 2D data
- Uses 3D convolution, 3D pooling etc. by default
- planar_blocks architecture parameter for mixed 2D/3D convnets
  (see UNet class docstring for details)
- Improved tests (see the bottom of the file)
- Cleaned up parameter/variable names and formatting, changed default params
- Updated for PyTorch 0.4.0 and Python 3.6 (earlier versions unsupported)
- (Optional DEBUG mode for optional printing of debug information)
- Extended documentation
"""

__all__ = ['UNet']

import copy
import itertools

from typing import Sequence, Union, Tuple

import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.checkpoint import checkpoint


DEBUG = False
if DEBUG:
    _print = print
else:
    def _print(*args, **kwargs):
        pass


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


def _get_activation(activation):
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


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, planar=False, activation='relu',
                 batch_norm=False, dim=3, conv_mode='same', adaptive=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.batch_norm = batch_norm
        padding = 1 if 'same' in conv_mode else 0

        self.conv1 = _conv3(
            self.in_channels, self.out_channels, planar=planar, dim=dim, padding=padding,
            adaptive=adaptive
        )
        self.conv2 = _conv3(
            self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding,
            adaptive=adaptive
        )

        if self.pooling:
            kernel_size = 2
            if planar:
                kernel_size = planar_kernel(kernel_size)
            self.pool = get_maxpool(dim)(kernel_size=kernel_size)

        self.act1 = _get_activation(activation)
        self.act2 = _get_activation(activation)

        if self.batch_norm:
            self.bn = get_batchnorm(dim)(self.out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)
        if self.batch_norm:
            y = self.bn(y)
        y = self.act2(y)
        before_pool = y
        if self.pooling:
            y = self.pool(y)
        return y, before_pool


@torch.jit.script
def autocrop(from_down: torch.Tensor, from_up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if from_down.shape[2:] != from_up.shape[2:]:
        # If VALID convolutions are used (not SAME), we need to center-crop to
        #  make features combinable.
        ds = from_down.shape[2:]
        us = from_up.shape[2:]
        assert ds[0] >= us[0]
        assert ds[1] >= us[1]
        if from_down.dim() == 4:
            from_down = from_down[
                :,
                :,
                ((ds[0] - us[0]) // 2):((ds[0] + us[0]) // 2),
                ((ds[1] - us[1]) // 2):((ds[1] + us[1]) // 2)
            ]
        elif from_down.dim() == 5:
            assert ds[2] >= us[2]
            from_down = from_down[
                :,
                :,
                ((ds[0] - us[0]) // 2):((ds[0] + us[0]) // 2),
                ((ds[1] - us[1]) // 2):((ds[1] + us[1]) // 2),
                ((ds[2] - us[2]) // 2):((ds[2] + us[2]) // 2),
            ]
    return from_down, from_up


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose', planar=False,
                 activation='relu', batch_norm=False, dim=3, conv_mode='same', adaptive=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.batch_norm = batch_norm
        padding = 1 if 'same' in conv_mode else 0

        self.upconv = _upconv2(self.in_channels, self.out_channels,
            mode=self.up_mode, planar=planar, dim=dim, adaptive=adaptive
        )

        if self.merge_mode == 'concat':
            self.conv1 = _conv3(
                2*self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding,
                adaptive=adaptive
            )
        else:
            # num of input channels to conv2 is same
            self.conv1 = _conv3(
                self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding,
                adaptive=adaptive
            )
        self.conv2 = _conv3(
            self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding,
            adaptive=adaptive
        )

        self.act0 = _get_activation(activation)
        self.act1 = _get_activation(activation)
        self.act2 = _get_activation(activation)

        if self.batch_norm:
            self.bn = get_batchnorm(dim)(self.out_channels)

    def forward(self, enc, dec):
        """ Forward pass
        Arguments:
            enc: Tensor from the encoder pathway
            dec: Tensor from the decoder pathway (to be upconv'd)
        """
        updec = self.upconv(dec)
        crenc, upcdec = autocrop(enc, updec)
        if self.up_mode == 'transpose':
            # Only for transposed convolution.
            # (In case of bilinear upsampling we omit activation)
            updec = self.act0(updec)
        if self.merge_mode == 'concat':
            mrg = torch.cat((updec, crenc), 1)
        else:
            mrg = updec + crenc
        y = self.conv1(mrg)
        y = self.act1(y)
        y = self.conv2(y)
        if self.batch_norm:
            y = self.bn(y)
        y = self.act2(y)
        return y


# TODO: Suppress known TracerWarnings?
# TODO: Pre-calculate output sizes when using valid convolutions
class UNet(nn.Module):
    """Modified version of U-Net, adapted for 3D biomedical image segmentation

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding, expansive pathway)
    about an input tensor is merged with information representing the
    localization of details (from the encoding, compressive pathway).

    - Original paper: https://arxiv.org/abs/1505.04597
    - Base implementation: https://github.com/jaxony/unet-pytorch


    Modifications to the original paper (@jaxony):
    (1) Padding is used in size-3-convolutions to prevent loss
        of border pixels.
    (2) Merging outputs does not require cropping due to (1).
    (3) Residual connections can be used by specifying
        UNet(merge_mode='add').
    (4) If non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose').

    Additional modifications (@mdraw):
    (5) Operates on 3D image data (5D tensors) instead of 2D data
    (6) Uses 3D convolution, 3D pooling etc. by default
    (7) Each network block pair (the two corresponding submodules in the
        encoder and decoder pathways) can be configured to either work
        in 3D or 2D mode (3D/2D convolution, pooling etc.)
        with the `planar_blocks` parameter.
        This is helpful for dealing with data anisotropy (commonly the
        depth axis has lower resolution in SBEM data sets, so it is not
        as important for convolution/pooling) and can reduce the complexity of
        models (parameter counts, speed, memory usage etc.).
        Note: If planar blocks are used, the input patch size should be
        adapted by reducing depth and increasing height and width of inputs.
    (8) Configurable activation function.
    (9) Optional batch normalization

    Args:
        in_channels: Number of input channels
            (e.g. 1 for single-grayscale inputs, 3 for RGB images)
            Default: 1
        out_channels: Number of output channels (number of classes).
            Default: 2
        n_blocks: Number of downsampling/convolution blocks (max-pooling)
            in the encoder pathway. The decoder (upsampling/upconvolution)
            pathway will consist of `n_blocks - 1` blocks.
            Increasing `n_blocks` has two major effects:
            - The network will be deeper
              (n + 1 -> 4 additional convolution layers)
            - Since each block causes one additional downsampling, more
              contextual information will be available for the network,
              enhancing the effective visual receptive field.
              (n + 1 -> receptive field is approximately doubled in each
                  dimension, except in planar blocks, in which it is only
                  doubled in the H and W image dimensions)
            **Important note**: Always make sure that the spatial shape of
            your input is divisible by the number of blocks, because
            else, concatenating downsampled features will fail.
        start_filts: Number of filters for the first convolution layer.
            Note: The filter counts of the later layers depend on the
            choice of `merge_mode`.
        up_mode: Upsampling method in the decoder pathway.
            Choices:
            - 'transpose' (default): Use transposed convolution
              ("Upconvolution")
            - 'upsample': Use nearest neighbour upsampling.
            For a detailed empirical evaluation of this option (in 2D U-Net),
            see https://ai.intel.com/biomedical-image-segmentation-u-net/
        merge_mode: How the features from the encoder pathway should
            be combined with the decoder features.
            Choices:
            - 'concat' (default): Concatenate feature maps along the
              `C` axis, doubling the number of filters each block.
            - 'add': Directly add feature maps (like in ResNets).
              The number of filters thus stays constant in each block.
            Note: According to https://arxiv.org/abs/1701.03056, feature
            concatenation ('concat') generally leads to better model
            accuracy than 'add' in typical medical image segmentation
            tasks.
        planar_blocks: Each number i in this sequence leads to the i-th
            block being a "planar" block. This means that all image
            operations performed in the i-th block in the encoder pathway
            and its corresponding decoder counterpart disregard the depth
            (`D`) axis and only operate in 2D (`H`, `W`).
            This is helpful for dealing with data anisotropy (commonly the
            depth axis has lower resolution in SBEM data sets, so it is
            not as important for convolution/pooling) and can reduce the
            complexity of models (parameter counts, speed, memory usage
            etc.).
            Note: If planar blocks are used, the input patch size should
            be adapted by reducing depth and increasing height and
            width of inputs.
        activation: Name of the non-linear activation function that should be
            applied after each network layer.
            Choices (see https://arxiv.org/abs/1505.00853 for details):
            - 'relu' (default)
            - 'leaky': Leaky ReLU (slope 0.1)
            - 'prelu': Parametrized ReLU. Best for training accuracy, but
                tends to increase overfitting.
            - 'rrelu': Can improve generalization at the cost of training
                accuracy.
            - Or you can pass an nn.Module instance directly, e.g.
              ``activation=torch.nn.ReLU()``
        batch_norm: If batch normalization should be applied at the end of
            each block. Note that BN is applied after the activated conv
            layers, not before the activation. This scheme differs from the
            original batch normalization paper and the BN scheme of 3D U-Net,
            but it delivers better results this way
            (see https://redd.it/67gonq).
        dim: Spatial dimensionality of the network. Choices:
            - 3 (default): 3D mode. Every block fully works in 3D unless
              it is excluded by the ``planar_blocks`` setting.
              The network expects and operates on 5D input tensors
              (N, C, D, H, W).
            - 2: Every block and every operation works in 2D, expecting
              4D input tensors (N, C, H, W).
        conv_mode: Padding mode of convolutions. Choices:
            - 'same' (default): Use SAME-convolutions in every layer:
              zero-padding inputs so that all convolutions preserve spatial
              shapes and don't produce an offset at the boundaries.
            - 'valid': Use VALID-convolutions in every layer: no padding is
              used, so every convolution layer reduces spatial shape by 2 in
              each dimension. Intermediate feature maps of the encoder pathway
              are automatically cropped to compatible shapes so they can be
              merged with decoder features.
              Advantages:
              - Less resource consumption than SAME because feature maps
                have reduced sizes especially in deeper layers.
              - No "fake" data (that is, the zeros from the SAME-padding)
                is fed into the network. The output regions that are influenced
                by zero-padding naturally have worse quality, so they should
                be removed in post-processing if possible (see
                ``overlap_shape`` in `py:mod:`elektronn3.inference`).
                Using VALID convolutions prevents the unnecessary computation
                of these regions that need to be cut away anyways for
                high-quality tiled inference.
              - Avoids the issues described in https://arxiv.org/abs/1811.11718.
              - Since the network will not receive zero-padded inputs, it is
                not required to learn a robustness against artificial zeros
                being in the border regions of inputs. This should reduce the
                complexity of the learning task and allow the network to
                specialize better on understanding the actual, unaltered
                inputs (effectively requiring less parameters to fit).
              Disadvantages:
              - Using this mode poses some additional constraints on input
                sizes and requires you to center-crop your targets,
                so it's harder to use in practice than the 'same' mode.
              - In some cases it might be preferable to get low-quality
                outputs at image borders as opposed to getting no outputs at
                the borders. Most notably this is the case if you do training
                and inference not on small patches, but on complete images in
                a single step.
            - 'valid_padback': Same as valid, but the final output of the
              network is zero-padded to the same spatial shape as the input
              tensor. This mode is only intended for testing purposes and
              shouldn't be used normally.
        adaptive: If ``True``, use custom convolution/transposed
            convolution layers for improved performance in planar blocks.
            This is an experimental feature and it is not guaranteed to give
            the same results as the native PyTorch convolution/transposed
            convolution implementations.
        checkpointing: If ``True``, use gradient checkpointing to reduce memory
            consumption while training. This makes the backward pass a bit
            slower, but the memory savings can be huge (usually around
            20% - 50%, depending on hyperparameters).
            Checkpoints are made after each network *block*.
            See https://pytorch.org/docs/master/checkpoint.html and
            https://arxiv.org/abs/1604.06174 for more details.
    """

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 2,
            n_blocks: int = 3,
            start_filts: int = 64,
            up_mode: str = 'transpose',
            merge_mode: str = 'concat',
            planar_blocks: Sequence = (),
            activation: Union[str, nn.Module] = 'relu',
            batch_norm: bool = True,
            dim: int = 3,
            conv_mode: str = 'same',
            adaptive: bool = False,
            checkpointing: bool = False
    ):
        super().__init__()

        if n_blocks < 1:
            raise ValueError('n_blocks must be > 1.')

        if dim not in {2, 3}:
            raise ValueError('dim has to be 2 or 3')
        if dim == 2 and planar_blocks != ():
            raise ValueError(
                'If dim=2, you can\'t use planar_blocks since everything will '
                'be planar (2-dimensional) anyways.\n'
                'Either set dim=3 or set planar_blocks=().'
            )

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "n_blocks channels (by half).")

        if len(planar_blocks) > n_blocks:
            raise ValueError('planar_blocks can\'t be longer than n_blocks.')
        if planar_blocks and (max(planar_blocks) >= n_blocks or min(planar_blocks) < 0):
            raise ValueError(
                'planar_blocks has invalid value range. All values have to be'
                'block indices, meaning integers between 0 and (n_blocks - 1).'
            )

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.n_blocks = n_blocks
        self.batch_norm = batch_norm
        self.conv_mode = conv_mode
        self.activation = activation
        self.dim = dim
        self.adaptive = adaptive
        self.checkpointing = checkpointing

        self.down_convs = []
        self.up_convs = []

        # Indices of blocks that should operate in 2D instead of 3D mode,
        # to save resources
        self.planar_blocks = planar_blocks

        # create the encoder pathway and add to a list
        for i in range(n_blocks):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < n_blocks - 1 else False
            planar = i in self.planar_blocks
            _print(f'D{i}: planar = {planar}')

            down_conv = DownConv(
                ins,
                outs,
                pooling=pooling,
                planar=planar,
                activation=activation,
                batch_norm=batch_norm,
                dim=dim,
                conv_mode=conv_mode,
                adaptive=adaptive
            )
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires n_blocks-1 blocks
        for i in range(n_blocks - 1):
            ins = outs
            outs = ins // 2
            planar = n_blocks - 2 - i in self.planar_blocks
            _print(f'U{i}: planar = {planar}')

            up_conv = UpConv(
                ins,
                outs,
                up_mode=up_mode,
                merge_mode=merge_mode,
                planar=planar,
                activation=activation,
                batch_norm=batch_norm,
                dim=dim,
                conv_mode=conv_mode,
                adaptive=adaptive
            )
            self.up_convs.append(up_conv)

        self.conv_final = _conv1(outs, self.out_channels, dim=dim)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

        self.pool_error_str = f'Spatial input shape has to be divisible by {2 ** n_blocks}!'

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        sh = x.shape[2:]
        encoder_outs = []

        # Encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            _print(f'D{i}: {module}')
            # Note that this check won't be picked up by PyTorch's tracing JIT compiler,
            #  so it won't impact execution speed if it's jit-compiled.
            # if torch.any(torch.tensor(x.shape[2:]) % 2 != 0):
            #     raise RuntimeError(self.pool_error_str)
            if self.checkpointing:
                x, before_pool = checkpoint(module, x)
            else:
                x, before_pool = module(x)
            _print(before_pool.shape)
            encoder_outs.append(before_pool)

        # Decoding by UpConv and merging with saved outputs of encoder
        for i, module in enumerate(self.up_convs):
            _print(f'U{i}: {module}')
            before_pool = encoder_outs[-(i+2)]
            _print(f'In: {before_pool.shape}')
            if self.checkpointing:
                x = checkpoint(module, before_pool, x)
            else:
                x = module(before_pool, x)
            _print(f'Out: {x.shape}')

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        # Temporarily store output for receptive field estimation using fornoxai/receptivefield
        self.feature_maps = [x]
        if self.conv_mode == 'valid_padback':
            # Workaround to recover spatial shape at the end
            # TODO: Remove this once we support out_shape != in_shape fully in e3
            pad = []
            for a, b in zip(sh, x.shape[2:]):
                diff = (a - b) // 2
                pad.extend([diff, diff])
            pad.reverse()  # Why do PyTorch padding layers suddenly expect (H, W, D) order???

            if x.dim() == 4:
                x = torch.nn.ConstantPad2d(pad, 0)(x)
            elif x.dim() == 5:
                x = torch.nn.ConstantPad3d(pad, 0)(x)
        return x


class UNet3dLite(UNet):
    """(WIP) Re-implementation of the unet3d_lite model from ELEKTRONN2

    See https://github.com/ELEKTRONN/ELEKTRONN2/blob/master/examples/unet3d_lite.py
    (Not yet working due to the AutoCrop node in ELEKTRONN2 working differently)
    """
    def __init__(self):
        super().__init__(
            in_channels=1,
            out_channels=2,
            n_blocks=4,
            start_filts=32,
            up_mode='transpose',
            merge_mode='concat',
            planar_blocks=(0, 1, 2),  # U1 and U2 will later be replaced by non-planar blocks
            activation='relu',
            batch_norm=False,
            dim=3,
            conv_mode='valid',
        )
        # TODO: mrg0 in the original unet3d_lite has upconv_n_f=512, which doesn't appear here
        for i in [1, 2]:
            # Replace planar U1 and U2 blocks with non-planar 3x3x3 versions
            ins = self.up_convs[i].upconv.in_channels
            outs = self.up_convs[i].conv2.out_channels
            self.up_convs[i] = UpConv(
                ins, outs, merge_mode=self.merge_mode, up_mode=self.up_mode,
                planar=False,
                activation=self.activation, batch_norm=self.batch_norm,
                dim=self.dim, conv_mode=self.conv_mode
            )


def test_model(
    batch_size=1,
    in_channels=1,
    out_channels=2,
    n_blocks=3,
    planar_blocks=(),
    merge_mode='concat',
    dim=3
):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_blocks=n_blocks,
        planar_blocks=planar_blocks,
        merge_mode=merge_mode,
        dim=dim,
    ).to(device)

    # Minimal test input
    if dim == 3:
        # Each block in the encoder pathway ends with 2x2x2 downsampling, except
        # planar blocks, which only do 1x2x2 downsampling, so the input has to
        # be larger when using more blocks.
        x = torch.randn(
            batch_size,
            in_channels,
            2 ** n_blocks // (2 ** len(planar_blocks)),
            2 ** n_blocks,
            2 ** n_blocks,
            device=device
        )
        expected_out_shape = (
            batch_size,
            out_channels,
            2 ** n_blocks // (2 ** len(planar_blocks)),
            2 ** n_blocks,
            2 ** n_blocks
        )
    elif dim == 2:
        # Each block in the encoder pathway ends with 2x2 downsampling
        # so the input has to be larger when using more blocks.
        x = torch.randn(
            batch_size,
            in_channels,
            2 ** n_blocks,
            2 ** n_blocks,
            device=device
        )
        expected_out_shape = (
            batch_size,
            out_channels,
            2 ** n_blocks,
            2 ** n_blocks
        )

    # Test forward, autograd, and backward pass with test input
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
    assert out.shape == expected_out_shape


def test_2d_config(max_n_blocks=4):
    for n_blocks in range(1, max_n_blocks + 1):
        print(f'Testing 2D U-Net with n_blocks = {n_blocks}...')
        test_model(n_blocks=n_blocks, dim=2)


def test_planar_configs(max_n_blocks=4):
    for n_blocks in range(1, max_n_blocks + 1):
        planar_combinations = itertools.chain(*[
            list(itertools.combinations(range(n_blocks), i))
            for i in range(n_blocks + 1)
        ])  # [(), (0,), (1,), ..., (0, 1), ..., (0, 1, 2, ..., n_blocks - 1)]

        for p in planar_combinations:
            print(f'Testing 3D U-Net with n_blocks = {n_blocks}, planar_blocks = {p}...')
            test_model(n_blocks=n_blocks, planar_blocks=p)


if __name__ == '__main__':
    # m = UNet3dLite()
    # x = torch.randn(1, 1, 22, 140, 140)
    # m(x)
    test_2d_config()
    print()
    test_planar_configs()
    print('All tests sucessful!')
    # # TODO: Also test valid convolution architecture.
