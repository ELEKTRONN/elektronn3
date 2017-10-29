import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModule


def passthrough(x, **kwargs):
    return x


def ELUCons(relu, nchan):
    if relu:
        return nn.ReLU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, relu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(relu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        return x


def _make_nConv(nchan, depth, relu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, relu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, relu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(relu, outChans)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, relu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(relu, outChans)
        self.relu2 = ELUCons(relu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, relu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        x = self.do1(down)
        x = self.ops(x)
        x = self.relu2(torch.add(x, down))
        return x


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, relu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(relu, outChans // 2)
        self.relu2 = ELUCons(relu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, relu)

    def forward(self, x, skipx):
        x = self.do1(x)
        skipxdo = self.do2(skipx)
        x = self.relu1(self.bn1(self.up_conv(x)))
        xcat = torch.cat((x, skipxdo), 1)
        x = self.ops(xcat)
        x = self.relu2(torch.add(x, xcat))
        return x


class OutputTransition(nn.Module):
    def __init__(self, inChans, relu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=1, )
        self.bn1 = ContBatchNorm3d(2)
        self.relu1 = ELUCons(relu, 2)
        self.softmax = F.log_softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        x = self.relu1(self.bn1(self.conv1(x)))
        # make channels the last axis
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        x = x.view(x.numel() // 2, 2)
        x = self.softmax(x)
        return x


class VNet(BaseModule):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, relu=True, nll=True, fac=4):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16 // fac, relu)
        self.down_tr32 = DownTransition(16 // fac, 1, relu)
        self.down_tr64 = DownTransition(32 // fac, 2, relu)
        self.down_tr128 = DownTransition(64 // fac, 3, relu, dropout=True)
        self.down_tr256 = DownTransition(128 // fac, 2, relu, dropout=True)
        self.up_tr256 = UpTransition(256 // fac, 256 // fac, 2, relu, dropout=True)
        self.up_tr128 = UpTransition(256 // fac, 128 // fac, 2, relu, dropout=True)
        self.up_tr64 = UpTransition(128 // fac, 64 // fac, 1, relu)
        self.up_tr32 = UpTransition(64 // fac, 32 // fac, 1, relu)
        self.out_tr = OutputTransition(32 // fac, relu, nll)

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        del out256
        del out128
        out = self.up_tr128(out, out64)
        del out64
        out = self.up_tr64(out, out32)
        del out32
        out = self.up_tr32(out, out16)
        del out16
        out = self.out_tr(out)
        return out
