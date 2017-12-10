from torch import nn
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
