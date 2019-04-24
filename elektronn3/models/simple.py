from torch import nn
import torch
from torch.nn import functional as F

# No useful architectures here, just some small test networks.


class Simple3DNet(nn.Module):
    def __init__(self, n_out_channels=2):
        super().__init__()
        self.n_out_channels = n_out_channels
        self.conv = nn.Sequential(
            nn.Conv3d(1, 10, 3, padding=1), nn.ReLU(),
            nn.Conv3d(10, 10, 3, padding=1), nn.ReLU(),
            nn.Conv3d(10, n_out_channels, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Extended3DNet(nn.Module):
    def __init__(self, n_out_channels=2):
        super().__init__()
        self.n_out_channels = n_out_channels
        self.conv = nn.Sequential(
            nn.Conv3d(1, 64, 5, padding=2), nn.ReLU(),
            nn.Conv3d(64, 64, 5, padding=2), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 64, 3, padding=2), nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=0), nn.ReLU(),
            nn.Conv3d(64, n_out_channels, 1),
        )

    def forward(self, x):
        original_size = x.size()[2:]
        x = self.conv(x)
        x = F.upsample(x, original_size)
        return x


class N3DNet(nn.Module):
    def __init__(self, n_out_channels=2):
        super().__init__()
        self.n_out_channels = n_out_channels
        self.neuro3d_seq = nn.Sequential(
            nn.Conv3d(1, 20, (1,5,5), padding=(0,2,2)), nn.ReLU(),
            nn.Conv3d(20, 30, (1,5,5), padding=(0,2,2)), nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(30, 40, (1,5,5), padding=(0,2,2)), nn.ReLU(),
            nn.Conv3d(40, 80, (3,3,3), padding=(1,1,1)), nn.ReLU(),
            nn.Conv3d(80, 100, (3,3,3), padding=(1,1,1)), nn.ReLU(),
            nn.Conv3d(100, 150, (1,3,3), padding=(0,1,1)), nn.ReLU(),
            nn.Conv3d(150, 50, (1,1,1)), nn.ReLU(),
            nn.Conv3d(50, n_out_channels, (1,1,1))
        )

    def forward(self, x):
        original_size = x.size()[2:]
        x = self.neuro3d_seq(x)
        x = F.upsample(x, original_size)
        return x


class Conv3DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 batch_norm=True, pooling=None, dropout_rate=None,
                 act=None):
        super().__init__()
        if act is None:
            act = nn.ReLU()
        seq = [nn.Conv3d(in_channels, out_channels, kernel_size)]
        if batch_norm:
            seq += [nn.BatchNorm3d(out_channels)]
        seq += [act]
        if pooling is not None:
            seq += [nn.MaxPool3d(pooling)]
        if dropout_rate is not None:
            seq += [nn.Dropout3d(dropout_rate)]
        self.conv3_layer = nn.Sequential(*seq)

    def forward(self, x):
        return self.conv3_layer(x)


class StackedConv2Scalar(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.05, act='relu'):
        super().__init__()
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'leaky_relu':
            act = nn.LeakyReLU()
        self.seq = nn.Sequential(
            Conv3DLayer(in_channels, 20, (1, 5, 5), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(20, 30, (1, 5, 5), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(30, 40, (1, 4, 4), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(40, 50, (1, 4, 4), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(50, 60, (1, 2, 2), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(60, 70, (1, 1, 1), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(70, 70, (1, 1, 1), pooling=(1, 1, 1),
                        dropout_rate=dropout_rate, act=act),
        )
        self.adaptavgpool = nn.AdaptiveAvgPool1d(100)
        self.fc = nn.Sequential(
            nn.Linear(100, 50),
            act,
            nn.Linear(50, 30),
            act,
            nn.Linear(30, n_classes),
        )

    def forward(self, x):
        x = self.seq(x)
        x = x.view(x.size()[0], 1, -1)  # AdaptiveAvgPool1d requires input of shape B C D
        x = self.adaptavgpool(x)
        x = self.fc(x.squeeze(1))  # remove auxiliary axis -> B C with C = n_classes
        return x


class StackedConv2ScalarWithLatentAdd(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.05, act='relu',
                 n_scalar=1):
        super().__init__()
        if act == 'relu':
            act = nn.ReLU()
        elif act == 'leaky_relu':
            act = nn.LeakyReLU()
        self.seq = nn.Sequential(
            Conv3DLayer(in_channels, 20, (1, 5, 5), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(20, 30, (1, 5, 5), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(30, 40, (1, 4, 4), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(40, 50, (1, 4, 4), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(50, 60, (1, 2, 2), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(60, 70, (1, 1, 1), pooling=(1, 2, 2),
                        dropout_rate=dropout_rate, act=act),
            Conv3DLayer(70, 70, (1, 1, 1), pooling=(1, 1, 1),
                        dropout_rate=dropout_rate, act=act),
        )
        self.adaptavgpool = nn.AdaptiveAvgPool1d(100)
        self.fc = nn.Sequential(
            nn.Linear(100 + n_scalar, 50),
            act,
            nn.Linear(50, 30),
            act,
            nn.Linear(30, n_classes),
        )

    def forward(self, x, scal):
        x = self.seq(x)
        x = x.view(x.size()[0], 1, -1)  # AdaptiveAvgPool1d requires input of shape B C D
        x = self.adaptavgpool(x).squeeze(1)
        x = torch.cat((x, scal), 1)
        x = self.fc(x)  # remove auxiliary axis -> B C with C = n_classes
        return x
