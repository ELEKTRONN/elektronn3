# modified from https://github.com/rusty1s/pytorch_geometric/blob/16033111a855459b336d61a9902c70bf7dfa7af9/examples/pointnet2_segmentation.py#L1
# and https://github.com/rusty1s/pytorch_geometric/blob/16033111a855459b336d61a9902c70bf7dfa7af9/examples/pointnet2_classification.py#L1

# originally from: https://github.com/dragonbook/pointnet2-pytorch
import numpy as np
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, GroupNorm as GN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, norm=None):
    if norm == 'bn':
        def transf(x): return BN(x, track_running_stats=False)
    elif norm == 'gn':
        def transf(x): return GN(max(1, int(x // 8)), x)
    elif norm is None:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(),)
            for i in range(1, len(channels))
        ])
    else:
        raise NotImplementedError
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), transf(channels[i]))
        for i in range(1, len(channels))
    ])


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2(torch.nn.Module):
    def __init__(self, num_classes, num_features, norm='gn'):
        super(PointNet2, self).__init__()
        self.sa1_module = SAModule(0.2, 0.2, MLP([num_features + 3, 64, 64, 128], norm))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256], norm))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024], norm))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256], norm))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128], norm))
        self.fp1_module = FPModule(3, MLP([128, 128, 128, 128], norm))

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, x, input_pts, output_pts=None):
        # TODO: use our data format here
        num_pts_per_batch = input_pts.size(1)
        num_batches = input_pts.size(0)
        batches = torch.tensor(np.repeat(np.arange(num_batches), num_pts_per_batch), device=x.device)
        sa0_out = (x.view(-1, *(x.size()[2:])), input_pts.view(-1, *(input_pts.size()[2:])), batches)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)

        if output_pts is not None:
            num_pts_per_batch = output_pts.size(1)
            num_batches = output_pts.size(0)
            batches = torch.tensor(np.repeat(np.arange(num_batches), num_pts_per_batch), device=x.device)
            sa0_out = (None, output_pts.view(-1, *(output_pts.size()[2:])), batches)

        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1).view(num_batches, num_pts_per_batch, *(x.size()[1:]))
