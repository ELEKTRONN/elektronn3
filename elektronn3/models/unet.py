# Based on https://github.com/jaxony/unet-pytorch
# Modified to support 3d convolutions.

# TODO: Proper attribution to jaxony and paper authors
# TODO: Update docstrings and comments for 3d
# TODO: Find a reasonable default for planar_blocks

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

DEBUG = False
if DEBUG:
    _print = print
else:
    def _print(*args, **kwargs):
        pass


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
           padding=1, bias=True, planar=False):
    if planar:
        stride = planar_kernel(stride)
        padding = planar_pad(padding)
        kernel_size = planar_kernel(kernel_size)
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias
    )


def _upconv2(in_channels, out_channels, mode='transpose', planar=False):
    kernel_size = 2
    stride = 2
    scale_factor = 2
    if planar:
        kernel_size = planar_kernel(kernel_size)
        stride = planar_kernel(stride)
        scale_factor = planar_kernel(scale_factor)
    if mode == 'transpose':
        return nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=scale_factor),
            _conv1(in_channels, out_channels)
        )


def _conv1(in_channels, out_channels):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, planar=False):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = _conv3(self.in_channels, self.out_channels, planar=planar)
        self.conv2 = _conv3(self.out_channels, self.out_channels, planar=planar)

        if self.pooling:
            kernel_size = 2
            if planar:
                kernel_size = planar_kernel(kernel_size)
            self.pool = nn.MaxPool3d(kernel_size=kernel_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose', planar=False):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = _upconv2(self.in_channels, self.out_channels,
            mode=self.up_mode, planar=planar
        )

        if self.merge_mode == 'concat':
            self.conv1 = _conv3(
                2*self.out_channels, self.out_channels, planar=planar
            )
        else:
            # num of input channels to conv2 is same
            self.conv1 = _conv3(self.out_channels, self.out_channels, planar=planar)
        self.conv2 = _conv3(self.out_channels, self.out_channels, planar=planar)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, out_channels=2, in_channels=1, n_blocks=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat', planar_blocks=()):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            n_blocks: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if n_blocks < 1:
            raise ValueError('n_blocks must be > 1.')

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

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = n_blocks

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

            down_conv = DownConv(ins, outs, pooling=pooling, planar=planar)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires n_blocks-1 blocks
        for i in range(n_blocks - 1):
            ins = outs
            outs = ins // 2
            planar = n_blocks - 2 - i in self.planar_blocks
            _print(f'U{i}: planar = {planar}')

            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode, planar=planar)
            self.up_convs.append(up_conv)

        self.conv_final = _conv1(outs, self.out_channels)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv3d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # Encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            _print(f'D{i}: {module}')
            x, before_pool = module(x)
            _print(before_pool.shape)
            encoder_outs.append(before_pool)

        # Decoding by UpConv and merging with saved outputs of encoder
        for i, module in enumerate(self.up_convs):
            _print(f'U{i}: {module}')
            # import IPython; IPython.embed()

            before_pool = encoder_outs[-(i+2)]
            _print(f'In: {before_pool.shape}')
            x = module(before_pool, x)
            _print(f'Out: {x.shape}')

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


def test_model(
    batch_size=1,
    in_channels=1,
    out_channels=2,
    n_blocks=3,
    planar_blocks=(),
    merge_mode='concat'
):
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_blocks=n_blocks,
        planar_blocks=planar_blocks,
        merge_mode=merge_mode
    )

    # Minimal test input
    # Each block in the encoder pathway ends with 2x2x2 downsampling, except
    # planar blocks, which only do 1x2x2 downsampling, so the input has to
    # be larger when using more blocks.
    x = torch.randn(
        batch_size,
        in_channels,
        2 ** n_blocks // (2 ** len(planar_blocks)),
        2 ** (n_blocks - 1),
        2 ** (n_blocks - 1)
    )
    if torch.cuda.is_available():
        model.cuda()
        x = x.cuda()

    # Test forward, autograd, and backward pass with test input
    out = model(x)
    loss = torch.sum(out)
    loss.backward()
    assert out.shape == (
        batch_size,
        out_channels,
        2 ** n_blocks // (2 ** len(planar_blocks)),
        2 ** (n_blocks - 1),
        2 ** (n_blocks - 1)
    )


def test_planar_configs(max_n_blocks=4):
    import itertools

    for n_blocks in range(1, max_n_blocks + 1):
        planar_combinations = itertools.chain(*[
            list(itertools.combinations(range(n_blocks), i))
            for i in range(n_blocks + 1)
        ])  # [(), (0,), (1,), ..., (0, 1), ..., (0, 1, 2, ..., n_blocks - 1)]

        for p in planar_combinations:
            print(f'Testing n_blocks = {n_blocks}, planar_blocks = {p}...')
            test_model(n_blocks=n_blocks, planar_blocks=p)


if __name__ == '__main__':
    test_planar_configs(5)