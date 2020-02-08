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
- Updated for PyTorch 1.0.1 and Python 3.6 (earlier versions unsupported)
- (Optional DEBUG mode for optional printing of debug information)
- Extended documentation
"""

__all__ = ['UNet']

import itertools

from typing import Sequence, Union, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import elektronn3.modules as em


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

        self.conv1 = em.conv3(
            self.in_channels, self.out_channels, planar=planar, dim=dim, padding=padding,
            adaptive=adaptive
        )
        self.conv2 = em.conv3(
            self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding,
            adaptive=adaptive
        )

        if self.pooling:
            kernel_size = 2
            if planar:
                kernel_size = em.planar_kernel(kernel_size)
            self.pool = em.get_maxpool(dim)(kernel_size=kernel_size)

        self.act1 = em.get_activation(activation)
        self.act2 = em.get_activation(activation)

        if self.batch_norm:
            self.bn = em.get_batchnorm(dim)(self.out_channels)

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

        self.upconv = em.upconv2(self.in_channels, self.out_channels,
            mode=self.up_mode, planar=planar, dim=dim, adaptive=adaptive
        )

        if self.merge_mode == 'concat':
            self.conv1 = em.conv3(
                2*self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding,
                adaptive=adaptive
            )
        else:
            # num of input channels to conv2 is same
            self.conv1 = em.conv3(
                self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding,
                adaptive=adaptive
            )
        self.conv2 = em.conv3(
            self.out_channels, self.out_channels, planar=planar, dim=dim, padding=padding,
            adaptive=adaptive
        )

        self.act0 = em.get_activation(activation)
        self.act1 = em.get_activation(activation)
        self.act2 = em.get_activation(activation)

        if self.batch_norm:
            self.bn = em.get_batchnorm(dim)(self.out_channels)

    def forward(self, enc, dec):
        """ Forward pass
        Arguments:
            enc: Tensor from the encoder pathway
            dec: Tensor from the decoder pathway (to be upconv'd)
        """
        updec = self.upconv(dec)
        crenc, upcdec = autocrop(enc, updec)
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

    - Padding is used in size-3-convolutions to prevent loss
      of border pixels.
    - Merging outputs does not require cropping due to (1).
    - Residual connections can be used by specifying
      UNet(merge_mode='add').
    - If non-parametric upsampling is used in the decoder
      pathway (specified by upmode='upsample'), then an
      additional 1x1 convolution occurs after upsampling
      to reduce channel dimensionality by a factor of 2.
      This channel halving happens with the convolution in
      the tranpose convolution (specified by upmode='transpose').

    Additional modifications (@mdraw):

    - Operates on 3D image data (5D tensors) instead of 2D data
    - Uses 3D convolution, 3D pooling etc. by default
    - Each network block pair (the two corresponding submodules in the
      encoder and decoder pathways) can be configured to either work
      in 3D or 2D mode (3D/2D convolution, pooling etc.)
      with the `planar_blocks` parameter.
      This is helpful for dealing with data anisotropy (commonly the
      depth axis has lower resolution in SBEM data sets, so it is not
      as important for convolution/pooling) and can reduce the complexity of
      models (parameter counts, speed, memory usage etc.).
      Note: If planar blocks are used, the input patch size should be
      adapted by reducing depth and increasing height and width of inputs.
    - Configurable activation function.
    - Optional batch normalization

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
            - 'resizeconv_nearest': Use resize-convolution with nearest-
              neighbor interpolation, as proposed in
              https://distill.pub/2016/deconv-checkerboard/
            - 'resizeconv_linear: Same as above, but with (bi-/tri-)linear
              interpolation
            - 'resizeconv_nearest1': Like 'resizeconv_nearest', but using a
              light-weight 1x1 convolution layer instead of a spatial convolution
            - 'resizeconv_linear1': Like 'resizeconv_nearest', but using a
              light-weight 1x1-convolution layer instead of a spatial convolution
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
                ``overlap_shape`` in :py:mod:`elektronn3.inference`).
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
        if up_mode in ('transpose', 'upsample', 'resizeconv_nearest', 'resizeconv_linear',
                       'resizeconv_nearest1', 'resizeconv_linear1'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for upsampling".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        # TODO: Remove merge_mode=add. It's just worse than concat
        if 'resizeconv' in self.up_mode and self.merge_mode == 'add':
            raise ValueError("up_mode \"resizeconv\" is incompatible "
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

        if adaptive:
            print('Warning: adaptive mode is no longer needed on newer PyTorch/CUDA/CuDNN setups and is therefore deprecated.')

        # Indices of blocks that should operate in 2D instead of 3D mode,
        # to save resources
        self.planar_blocks = planar_blocks
        if not planar_blocks:
            # Adaptive Convolutions only make sense if a part of the network
            #  operates in 2D (planar mode)
            self.adaptive = False

        # create the encoder pathway and add to a list
        for i in range(n_blocks):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < n_blocks - 1 else False
            planar = i in self.planar_blocks

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

        self.conv_final = em.conv1(outs, self.out_channels, dim=dim)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        sh = x.shape[2:]
        encoder_outs = []

        # Encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            if self.checkpointing:
                x, before_pool = checkpoint(module, x)
            else:
                x, before_pool = module(x)
            encoder_outs.append(before_pool)

        # Decoding by UpConv and merging with saved outputs of encoder
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            if self.checkpointing:
                x = checkpoint(module, before_pool, x)
            else:
                x = module(before_pool, x)

        # No softmax is used, so you need to apply it in the loss.
        x = self.conv_final(x)
        # Uncomment the following line to temporarily store output for
        #  receptive field estimation using fornoxai/receptivefield:
        # self.feature_maps = [x]  # Currently disabled to save memory
        return x


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
