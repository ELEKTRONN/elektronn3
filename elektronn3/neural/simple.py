from torch import nn
from torch.nn import functional as F

# No useful architectures here, just some small test networks.


class Simple3DNet(nn.Module):
    def __init__(self, n_out_channels=2):
        super().__init__()
        self.n_out_channels = 2
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
        self.n_out_channels = 2
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
