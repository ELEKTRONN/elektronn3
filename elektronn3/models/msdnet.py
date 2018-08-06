# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Shubham Dokania, Martin Drawitsch

# Modified implementation of http://www.pnas.org/content/early/2017/12/21/1715832114
# Derived from Shubham Dokania's https://github.com/shubham1810/MS-D_Net_PyTorch


# TODO: Mixed 2D/3D mode for anisotropic 3D images


import torch
import torch.nn as nn
import torch.nn.functional as F


def add_conv_block(in_ch=1, out_ch=1, kernel_size=3, dilate=1, last=False, volumetric=True):
    if volumetric:
        Conv = nn.Conv3d
        BatchNorm = nn.BatchNorm3d
    else:
        Conv = nn.Conv2d
        BatchNorm = nn.BatchNorm2d
    pad = dilate if not last else 0
    conv_1 = Conv(in_ch, out_ch, kernel_size, padding=pad, dilation=dilate)
    bn_1 = BatchNorm(out_ch)

    return [conv_1, bn_1]


class MSDNet(nn.Module):
    """
    Paper: A mixed-scale dense convolutional neural network for image analysis
    Published: PNAS, Jan. 2018 
    Paper: http://www.pnas.org/content/early/2017/12/21/1715832114
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal(m, m.weight.data)

    def __init__(self, in_channels=1, out_channels=2, num_layers=40, volumetric=True):

        super().__init__()

        self.layer_list = add_conv_block(in_ch=in_channels, volumetric=volumetric)
        
        current_in_channels = 1
        # Add N layers
        for i in range(num_layers):
            s = i % 10 + 1
            self.layer_list += add_conv_block(
                in_ch=current_in_channels,
                dilate=s,
                volumetric=volumetric
            )
            current_in_channels += 1

        # Add final output block
        self.layer_list += add_conv_block(
            in_ch=current_in_channels + in_channels,
            out_ch=out_channels,
            kernel_size=1,
            last=True,
            volumetric=volumetric
        )

        # Add to Module List
        self.layers = nn.ModuleList(self.layer_list)

        self.apply(self.weight_init)

    def forward(self, x):
        prev_features = []
        inp = x
        
        for i, f in enumerate(self.layers):
            # Check if last conv block
            if i == len(self.layers) - 2:
                x = torch.cat(prev_features + [inp], 1)
            
            x = f(x)

            if (i + 1) % 2 == 0 and not i == (len(self.layers) - 1):
                x = F.relu(x)
                # Append output into previous features
                prev_features.append(x)
                x = torch.cat(prev_features, 1)
        return x


if __name__ == '__main__':
    # Test 2D and 3D inference on random data
    inps = [torch.randn(1, 1, 32, 32), torch.randn(1, 1, 16, 32, 32)]
    for inp in inps:
        m = MSDNet(in_channels=1, out_channels=2, volumetric=inp.dim() == 5)
        o = m(inp)
        print(o.shape)
