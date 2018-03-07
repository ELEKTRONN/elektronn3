# 3D variant of the Fully-Convolutional-Densenet ("Tiramisu"),
# based on the MIT-licensed 2D implementation
# https://github.com/bfortuner/pytorch_tiramisu (C) 2017, Brendan Fortuner
# The original (2D) paper was published at https://arxiv.org/abs/1611.09326.

# TODO: Tune for 3D. Even the 57-layer-variant is far too heavy
# TODO: Memory leak? Accidental gradient calculation?
#       (Memory usage grows while training for some reason)

import torch
from torch import nn


def center_crop(layer, d_max, h_max, w_max):
    # https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/merge.py#L162
    # Author does a center crop which crops both inputs (skip and upsample)
    # to size of minimum dimension on both w/h
    d, h, w = layer.shape[2:]
    d_lo = (d - d_max) // 2
    w_lo = (w - w_max) // 2
    h_lo = (h - h_max) // 2
    return layer[:, :, d_lo:(d_lo + d_max), h_lo:(h_lo + h_max), w_lo:(w_lo + w_max)]


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

        # author's impl - lasange 'same' pads with half
        # filter size (rounded down) on "both" sides
        self.add_module('conv', nn.Conv3d(in_channels=in_channels,
            out_channels=growth_rate, kernel_size=3, stride=1,
            padding=1, bias=True))

        self.add_module('drop', nn.Dropout3d(0.2))

    def forward(self, x):
        return super(DenseLayer, self).forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super(DenseBlock, self).__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i * growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            # we pass all previous activations into each dense layer normally
            # But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)  # 1 = channel axis
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super(TransitionDown, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(in_channels=in_channels,
            out_channels=in_channels, kernel_size=1, stride=1,
            padding=0, bias=True))
        self.add_module('drop', nn.Dropout3d(0.2))
        self.add_module('maxpool', nn.MaxPool3d(2))

    def forward(self, x):
        return super(TransitionDown, self).forward(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convTrans = nn.ConvTranspose3d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=2,
            padding=0,
            bias=True)  # crop = 'valid' means padding=0. Padding has reverse effect for transpose conv (reduces output size)
        # http://lasagne.readthedocs.io/en/latest/modules/layers/conv.html#lasagne.layers.TransposedConv2DLayer
        # self.updample3d = nn.UpsamplingBilinear3d(scale_factor=2)

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3), skip.size(4))
        out = torch.cat([out, skip], 1)
        return out


class Bottleneck(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Bottleneck, self).__init__()
        self.add_module('bottleneck', DenseBlock(in_channels, growth_rate, n_layers, upsample=True))

    def forward(self, x):
        return super(Bottleneck, self).forward(x)


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=1, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48, out_channels=2):
        super(FCDenseNet, self).__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks

        cur_channels_count = 0
        skip_connection_channel_counts = []

        #####################
        # First Convolution #
        #####################

        self.add_module('firstconv', nn.Conv3d(in_channels=in_channels,
            out_channels=out_chans_first_conv, kernel_size=3,
            stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck', Bottleneck(cur_channels_count,
            growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        # One final dense block
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]

        #####################
        #      Softmax      #
        #####################

        self.finalConv = nn.Conv3d(in_channels=cur_channels_count,
            out_channels=out_channels, kernel_size=1, stride=1,
            padding=0, bias=True)
        # self.softmax = nn.LogSoftmax()

    def forward(self, x):
        # print("INPUT",x.size())
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            # print("DBD size",out.size())
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        # print ("bnecksize",out.size())
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            # print("DOWN_SKIP_PRE_UPSAMPLE",out.size(),skip.size())
            out = self.transUpBlocks[i](out, skip)
            # print("DOWN_SKIP_AFT_UPSAMPLE",out.size(),skip.size())
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        # out = self.softmax(out)  # LogSoftmax is already computed by loss function
        return out


def FCDenseNet57(in_channels=1, out_channels=2):
    return FCDenseNet(in_channels=in_channels, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
        growth_rate=12, out_chans_first_conv=48, out_channels=out_channels)


def FCDenseNet67(in_channels=1, out_channels=2):
    return FCDenseNet(in_channels=in_channels, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
        growth_rate=16, out_chans_first_conv=48, out_channels=out_channels)


def FCDenseNet103(in_channels=1, out_channels=2):
    return FCDenseNet(in_channels=in_channels, down_blocks=(4, 5, 7, 10, 12),
        up_blocks=(12, 10, 7, 5, 4), bottleneck_layers=15,
        growth_rate=16, out_chans_first_conv=48, out_channels=out_channels)