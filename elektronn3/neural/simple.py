from torch import nn
from torch.nn import functional as F

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
        # x = x.permute(1, 0, 2, 3, 4).contiguous()
        # x = x.view(-1, n_out_channels)
        # x = F.log_softmax(x)
        return x
