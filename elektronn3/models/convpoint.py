# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

"""
This is an implementation of the ConvPoint architecture based on the Github repository
https://github.com/aboulch/ConvPoint. The architecture was described in https://arxiv.org/abs/1904.02375 by Alexandre
Boulch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple
import elektronn3.models.knn.lib.python.nearest_neighbors as nearest_neighbors
from abc import ABC
try:
    from torch_geometric.nn import XConv, fps, global_mean_pool
except ImportError as e:
    # print('XConv layer not available.', e)
    pass
try:
    from pytorch3d.ops import knn

    @torch.jit.script
    def indices_conv_reduction_pt(input_pts: torch.Tensor, K: int, npts: int):
        device = input_pts.device
        # TODO: use a better heuristic
        rand_ixs = torch.randint(input_pts.size(1), (input_pts.size(0), npts, 1), dtype=torch.long, device=device)
        next_pts_ = torch.gather(input_pts, 1, rand_ixs.expand(-1, -1, 3)).to(device)
        _, indices, _ = knn.knn_points(next_pts_, input_pts, K=K)
        return indices, next_pts_


    @torch.jit.script
    def indices_conv_pt(input_pts, K):
        _, indices, _ = knn.knn_points(input_pts, input_pts, K=K)
        return indices, input_pts


    @torch.jit.script
    def indices_deconv_pt(input_pts, next_pts, K):
        _, indices, _ = knn.knn_points(next_pts, input_pts, K=K)
        return indices, next_pts
except ImportError as e:
    # print('pytorch3d.ops not available.', e)
    pass


# STATIC HELPER FUNCTIONS #
def swish(x):
    """https://arxiv.org/pdf/1710.05941.pdf"""
    return x * torch.sigmoid(x)


def identity(x):
    return x


def apply_bn(x, bn):
    return bn(x.transpose(1, 2)).transpose(1, 2).contiguous()


def indices_conv_reduction(input_pts, K, npts):
    indices, queries = \
        nearest_neighbors.knn_batch_distance_pick(input_pts.cpu().detach().numpy(), npts, K, omp=True)
    indices = torch.from_numpy(indices).long()
    queries = torch.from_numpy(queries).float()
    if input_pts.is_cuda:
        indices = indices.to(input_pts.device)
        queries = queries.to(input_pts.device)

    return indices, queries


def indices_conv_reduction_big(input_pts: torch.Tensor, output_pts_num: int, neighbor_num: int, padding: int = None,
                               centroids: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Reduction function used by PtConvBig using possible centroid reductions and padding. This was still
        experimental and will be updated by proper versions in the future.
    """
    input_pts_np = input_pts.cpu().detach().numpy()
    if padding is not None:
        # use only non-padded points for reduction
        output_pts = np.ones((len(input_pts_np), output_pts_num, 3))*padding
        for ix, batch in enumerate(input_pts_np):
            filtered_pts = batch[batch < padding].reshape(-1, 3)
            output_pts[ix, :len(filtered_pts)] = \
                filtered_pts[np.random.choice(len(filtered_pts), min(len(filtered_pts), output_pts_num), replace=False)]
        indices = nearest_neighbors.knn_batch(input_pts_np, output_pts, neighbor_num, omp=True)
    else:
        indices, output_pts = nearest_neighbors.knn_batch_distance_pick(input_pts_np, output_pts_num, neighbor_num,
                                                                        omp=True)

    if centroids:
        # calculate centroid of each neighborhood as new output point
        for ix, batch in enumerate(indices):
            pts_batch = input_pts_np[ix]
            output_pts[ix] = np.mean(pts_batch[batch], axis=1)

    indices = torch.from_numpy(indices).long()
    output_pts = torch.from_numpy(output_pts).float()
    if input_pts.is_cuda:
        indices = indices.to(input_pts.device)
        output_pts = output_pts.to(input_pts.device)
    return indices, output_pts


def indices_conv(input_pts, K):
    indices = \
        nearest_neighbors.knn_batch(input_pts.cpu().detach().numpy(), input_pts.cpu().detach().numpy(), K, omp=True)
    indices = torch.from_numpy(indices).long()
    if input_pts.is_cuda:
        indices = indices.to(input_pts.device)
    return indices, input_pts


def indices_deconv(input_pts, next_pts, K):
    indices = \
        nearest_neighbors.knn_batch(input_pts.cpu().detach().numpy(), next_pts.cpu().detach().numpy(), K, omp=True)
    indices = torch.from_numpy(indices).long()
    if input_pts.is_cuda:
        indices = indices.to(input_pts.device)
    return indices, next_pts


# LAYER DEFINITIONS #


class LayerBase(nn.Module, ABC):

    def __init__(self):
        super(LayerBase, self).__init__()


class PtConv_PT3d(LayerBase):
    def __init__(self, input_features, output_features, n_centers, dim,
                 act=None, use_bias=True):
        if act in [None, 'relu']:
            self.act = F.relu_
        elif act == 'swish':
            self.act = swish
        elif type(act) == str:
            self.act = getattr(F, act)
        elif callable(act):
            self.act = act
        else:
            raise ValueError
        super(PtConv_PT3d, self).__init__()

        # Weight
        self.weight = \
            nn.Parameter(torch.Tensor(input_features, n_centers, output_features), requires_grad=True)
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (input_features + output_features))
        self.weight.data.uniform_(-bound, bound)

        # bias
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_features), requires_grad=True)
            self.bias.data.uniform_(0, 0)

        # centers
        center_data = np.zeros((dim, n_centers))
        for i in range(n_centers):
            coord = np.random.rand(dim)*2 - 1
            while (coord**2).sum() > 1:
                coord = np.random.rand(dim)*2 - 1
            center_data[:, i] = coord
        self.centers = nn.Parameter(torch.from_numpy(center_data).float(), requires_grad=True)

        # MLP
        self.l1 = nn.Linear(dim*n_centers, 2*n_centers)
        self.l2 = nn.Linear(2*n_centers, n_centers)
        self.l3 = nn.Linear(n_centers, n_centers)

    def forward(self, inp, points, K, next_pts=None, normalize=False, dilation=1):
        if normalize:
            raise NotImplementedError
        device = inp.device
        if isinstance(next_pts, int) and points.size(1) != next_pts:
            indices, next_pts_ = indices_conv_reduction_pt(points, K * dilation, next_pts)
        elif (next_pts is None) or (isinstance(next_pts, int) and points.size(1) == next_pts):
            # convolution without reduction
            indices, next_pts_ = indices_conv_pt(points, K * dilation)
        else:
            # convolution with up sampling or projection on given points
            indices, next_pts_ = indices_deconv_pt(points, next_pts, K*dilation)

        if next_pts is None or isinstance(next_pts, int):
            next_pts = next_pts_

        batch_size = inp.size(0)
        n_pts = inp.size(1)

        if dilation > 1:
            indices = indices[:, :, torch.randperm(indices.size(2))]
            indices = indices[:, :, :K]

        # compute indices for indexing points
        add_indices = torch.arange(batch_size, dtype=torch.long).to(device) * n_pts
        indices = indices + add_indices.view(-1, 1, 1)

        # get the features and point cooridnates associated with the indices
        features = inp.view(-1, inp.size(2))[indices]
        pts = points.view(-1, points.size(2))[indices]

        # center the neighborhoods
        pts = pts - next_pts.unsqueeze(2)

        # # normalize to unit ball, or not
        # if normalize:
        #     maxi = torch.sqrt((pts.detach()**2).sum(3).max(2)[0])  # detach is a modificaiton
        #     maxi[maxi == 0] = 1
        #     pts = pts / maxi.view(maxi.size()+(1, 1,))

        # compute the distances
        dists = pts.view(pts.size()+(1,)) - self.centers
        dists = dists.view(dists.size(0), dists.size(1), dists.size(2), -1)
        dists = self.act(self.l1(dists))
        dists = self.act(self.l2(dists))
        dists = self.act(self.l3(dists))

        # compute features
        fs = features.size()
        features = features.transpose(2, 3)
        features = features.view(-1, features.size(2), features.size(3))
        dists = dists.view(-1, dists.size(2), dists.size(3))

        features = torch.bmm(features, dists)

        features = features.view(fs[0], fs[1], -1)

        features = torch.matmul(features, self.weight.view(-1, self.weight.size(2)))
        features = features / fs[2]

        # add a bias
        if self.use_bias:
            features = features + self.bias

        return features, next_pts


class PtConv(LayerBase):
    def __init__(self, input_features, output_features, n_centers, dim,
                 act=None, use_bias=True):
        if act in [None, 'relu']:
            self.act = F.relu_
        elif act == 'swish':
            self.act = swish
        elif type(act) == str:
            self.act = getattr(F, act)
        elif callable(act):
            self.act = act
        else:
            raise ValueError
        super(PtConv, self).__init__()

        # Weight
        self.weight = \
            nn.Parameter(torch.Tensor(input_features, n_centers, output_features), requires_grad=True)
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (input_features + output_features))
        self.weight.data.uniform_(-bound, bound)

        # bias
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_features), requires_grad=True)
            self.bias.data.uniform_(0, 0)

        # centers
        center_data = np.zeros((dim, n_centers))
        for i in range(n_centers):
            coord = np.random.rand(dim)*2 - 1
            while (coord**2).sum() > 1:
                coord = np.random.rand(dim)*2 - 1
            center_data[:, i] = coord
        self.centers = nn.Parameter(torch.from_numpy(center_data).float(), requires_grad=True)

        # MLP
        self.l1 = nn.Linear(dim*n_centers, 2*n_centers)
        self.l2 = nn.Linear(2*n_centers, n_centers)
        self.l3 = nn.Linear(n_centers, n_centers)

    def forward(self, inp, points, K, next_pts=None, normalize=False, indices_=None,
                return_indices=False, dilation=1):
        if indices_ is None:
            if isinstance(next_pts, int) and points.size(1) != next_pts:
                # convolution with reduction
                indices, next_pts_ = indices_conv_reduction(points, K * dilation, next_pts)
            elif (next_pts is None) or (isinstance(next_pts, int) and points.size(1) == next_pts):
                # convolution without reduction
                indices, next_pts_ = indices_conv(points, K * dilation)
            else:
                # convolution with up sampling or projection on given points
                indices, next_pts_ = indices_deconv(points, next_pts, K * dilation)

            if next_pts is None or isinstance(next_pts, int):
                next_pts = next_pts_

            if return_indices:
                indices_ = indices
        else:
            indices = indices_

        batch_size = inp.size(0)
        n_pts = inp.size(1)

        if dilation > 1:
            indices = indices[:, :, torch.randperm(indices.size(2))]
            indices = indices[:, :, :K]

        # compute indices for indexing points
        add_indices = torch.arange(batch_size).type(indices.type()).to(indices.device) * n_pts
        indices = indices + add_indices.view(-1, 1, 1)

        # get the features and point cooridnates associated with the indices
        features = inp.view(-1, inp.size(2))[indices]
        pts = points.view(-1, points.size(2))[indices]

        # center the neighborhoods
        pts = pts - next_pts.unsqueeze(2)

        # normalize to unit ball, or not
        if normalize:
            maxi = torch.sqrt((pts.detach()**2).sum(3).max(2)[0])  # detach is a modificaiton
            maxi[maxi == 0] = 1
            pts = pts / maxi.view(maxi.size()+(1, 1,))

        # compute the distances
        dists = pts.view(pts.size()+(1,)) - self.centers
        dists = dists.view(dists.size(0), dists.size(1), dists.size(2), -1)
        dists = self.act(self.l1(dists))
        dists = self.act(self.l2(dists))
        dists = self.act(self.l3(dists))

        # compute features
        fs = features.size()
        features = features.transpose(2, 3)
        features = features.view(-1, features.size(2), features.size(3))
        dists = dists.view(-1, dists.size(2), dists.size(3))

        features = torch.bmm(features, dists)

        features = features.view(fs[0], fs[1], -1)

        features = torch.matmul(features, self.weight.view(-1, self.weight.size(2)))
        features = features / fs[2]

        # add a bias
        if self.use_bias:
            features = features + self.bias

        if return_indices:
            return features, next_pts, indices_
        else:
            return features, next_pts


class PtConvBig(LayerBase):
    def __init__(self, in_feats: int = 64, out_feats: int = 64, kernel_size: int = 16, dim: int = 3,
                 act: str = None, use_bias: bool = True, padding: int = None, nn_center: bool = True,
                 centroids: bool = False):
        if act in [None, 'relu']:
            self.act = F.relu_
        elif act == 'swish':
            self.act = swish
        elif type(act) == str:
            self.act = getattr(F, act)
        elif callable(act):
            self.act = act
        else:
            raise ValueError
        super(PtConvBig, self).__init__()

        # Kernel depth equals number of input features. Thus, the number of needed weights is
        # Input_Channels x Kernel number x Kernel elements. These weights get initialized randomly.
        self.weight = \
            nn.Parameter(torch.Tensor(in_feats, kernel_size, out_feats), requires_grad=True)
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (in_feats + out_feats))
        self.weight.data.uniform_(-bound, bound)

        # Inititalize bias if necessary
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats), requires_grad=True)
            self.bias.data.uniform_(0, 0)

        self.padding = padding
        self.nn_center = nn_center
        self.centroids = centroids

        # The kernel elements get initialized within the unit sphere. This includes only the elements of one single
        # kernel. The position of these kernel elements are adjusted during training.
        center_data = np.zeros((dim, kernel_size))
        for i in range(kernel_size):
            coord = np.random.rand(dim) * 2 - 1
            while (coord ** 2).sum() > 1:
                coord = np.random.rand(dim) * 2 - 1
            center_data[:, i] = coord
        self.centers = nn.Parameter(torch.from_numpy(center_data).float(), requires_grad=True)

        self.l1 = nn.Linear(dim * kernel_size, 2 * kernel_size)
        self.l2 = nn.Linear(2 * kernel_size, kernel_size)
        self.l3 = nn.Linear(kernel_size, kernel_size)

    def forward(self, features, input_pts, neighbor_num, output_pts=None, normalize=False,
                indices_=None, return_indices=False, dilation=1):
        if indices_ is None:
            if isinstance(output_pts, int) and input_pts.size(1) != output_pts:
                indices, next_pts_ = indices_conv_reduction_big(input_pts, output_pts, neighbor_num * dilation,
                                                                self.padding, self.centroids)
            elif (output_pts is None) or (isinstance(output_pts, int) and input_pts.size(1) == output_pts):
                indices, next_pts_ = indices_conv(input_pts, neighbor_num * dilation)
            else:
                indices, next_pts_ = indices_deconv(input_pts, output_pts, neighbor_num * dilation)

            if output_pts is None or isinstance(output_pts, int):
                output_pts = next_pts_

            if return_indices:
                indices_ = indices
        else:
            indices = indices_

        batch_size = features.size(0)
        n_pts = features.size(1)

        if dilation > 1:
            indices = indices[:, :, torch.randperm(indices.size(2))]
            indices = indices[:, :, :neighbor_num]

        # Compute indices for indexing points (add batch offset to indices)
        add_indices = torch.arange(batch_size).type(indices.type()).to(input_pts.device) * n_pts
        indices = indices + add_indices.view(-1, 1, 1)

        # bs: batchsize, np: point number, nn: neighborhood size
        # [bs, np, nn, 3]
        features = features.view(-1, features.size(2))[indices]
        pts = input_pts.view(-1, input_pts.size(2))[indices]

        # Center each neighboorhood
        if self.nn_center:
            pts = pts - output_pts.unsqueeze(2)

        if normalize:
            # Normalize to unit ball
            maxi = torch.sqrt((pts.detach() ** 2).sum(3).max(2)[0])  # detach is a modification
            maxi[maxi == 0] = 1
            pts = pts / maxi.view(maxi.size() + (1, 1,))

        # Compute the distances between kernel elements and points of centered neighborhoods
        # [bs, np, nn, nn*3]
        dists = pts.view(pts.size() + (1,)) - self.centers
        dists = dists.view(dists.size(0), dists.size(1), dists.size(2), -1)

        # Apply weighting function by the MLP
        dists = F.relu(self.l1(dists))
        dists = F.relu(self.l2(dists))
        dists = F.relu(self.l3(dists))

        fs = features.size()
        features = features.transpose(2, 3)
        # [bs*np, fs, nn]
        features = features.view(-1, features.size(2), features.size(3))
        # [bs*np, nn, nn]
        dists = dists.view(-1, dists.size(2), dists.size(3))

        # Batch matrix multiplications which is essentially equation 2 from the paper
        # (Network weights * Features * Weighting function)
        features = torch.bmm(features, dists)
        # [bs, np, fs*nn]
        features = features.view(fs[0], fs[1], -1)
        features = torch.matmul(features, self.weight.view(-1, self.weight.size(2)))

        # Normalization according to the input size
        features = features / fs[2]

        # add a bias
        if self.use_bias:
            features = features + self.bias

        if return_indices:
            return features, output_pts, indices_
        else:
            return features, output_pts


################################
# SMALL SEGMENTATION NETWORK
################################

class SegSmall3(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, dropout=0,
                 act=None, use_bias=True, track_running_stats=False, use_norm=False):
        super(SegSmall3, self).__init__()
        if act in [None, 'relu']:
            self.act = F.relu_
        elif act == 'swish':
            self.act = swish
        elif type(act) == str:
            self.act = getattr(F, act)
        elif callable(act):
            self.act = act
        else:
            raise ValueError

        n_centers = 24

        pl = 48
        self.cv2 = PtConv(input_channels, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv2_dil = PtConv(input_channels, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv3 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv4 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv5 = PtConv(2 * pl, 4 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv6 = PtConv(4 * pl, 4 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)

        self.cv5d = PtConv(4 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv4d = PtConv(6 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv3d = PtConv(4 * pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv2d = PtConv(3 * pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv1d = PtConv(3 * pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)

        self.cvout = PtConv(pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.fcout = nn.Linear(pl, output_channels)

        if use_norm is False:
            self.bn = identity
            self.bn2 = identity
            self.bn3 = identity
            self.bn4 = identity
            self.bn5 = identity
            self.bn6 = identity

            self.bn5d = identity
            self.bn4d = identity
            self.bn3d = identity
            self.bn2d = identity
            self.bn1d = identity

            self.bnout = identity
        elif use_norm == 'bn':
            self.bn2 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn3 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn4 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn5 = nn.BatchNorm1d(4 * pl, track_running_stats=track_running_stats)
            self.bn6 = nn.BatchNorm1d(4 * pl, track_running_stats=track_running_stats)

            self.bn5d = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn4d = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn3d = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)
            self.bn2d = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)
            self.bn1d = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)

            self.bnout = nn.BatchNorm1d(pl)
        elif use_norm == 'gn':
            self.bn2 = nn.GroupNorm(pl, 2 * pl)
            self.bn3 = nn.GroupNorm(pl, 2 * pl)
            self.bn4 = nn.GroupNorm(pl, 2 * pl)
            self.bn5 = nn.GroupNorm(2 * pl, 4 * pl)
            self.bn6 = nn.GroupNorm(2 * pl, 4 * pl)

            self.bn5d = nn.GroupNorm(pl, 2 * pl)
            self.bn4d = nn.GroupNorm(pl, 2 * pl)
            self.bn3d = nn.GroupNorm(pl // 2, pl)
            self.bn2d = nn.GroupNorm(pl // 2, pl)
            self.bn1d = nn.GroupNorm(pl // 2, pl)

            self.bnout = nn.GroupNorm(pl // 2, pl)
        else:
            raise ValueError

        self.drop = nn.Dropout(dropout)

    def forward(self, x, input_pts, output_pts=None):
        if output_pts is None:
            output_pts = input_pts
        x2, pts2 = self.cv2(x, input_pts, 32, 2048)
        x2_dil, _ = self.cv2_dil(x, input_pts, 24, pts2, dilation=2)
        x2 = torch.cat([x2, x2_dil], dim=2)

        x2 = self.act(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 24, 1024)
        x3 = self.act(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 16, 256)
        x4 = self.act(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 64)
        x5 = self.act(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = self.act(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = self.act(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = self.act(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = self.act(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = self.act(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)

        x1d, _ = self.cv1d(x2d, pts2, 16, output_pts)
        x1d = self.act(apply_bn(x1d, self.bn1d))

        xout, _ = self.cvout(x1d, output_pts, 16)
        xout = self.act(apply_bn(xout, self.bnout))
        xout = self.drop(xout)

        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        return xout


class SegSmall2(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, dropout=0,
                 act=None, use_bias=True, track_running_stats=False, use_norm=False):
        super(SegSmall2, self).__init__()
        if act in [None, 'relu']:
            self.act = F.relu_
        elif act == 'swish':
            self.act = swish
        elif type(act) == str:
            self.act = getattr(F, act)
        elif callable(act):
            self.act = act
        else:
            raise ValueError

        n_centers = 16

        pl = 64
        self.cv2 = PtConv(input_channels, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv4 = PtConv(pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv5 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv6 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)

        self.cv5d = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv4d = PtConv(4 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv3d = PtConv(4 * pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv2d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv1d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)

        self.cvout = PtConv(pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.fcout = nn.Linear(pl, output_channels)

        if use_norm is False:
            self.bn = identity
            self.bn2 = identity
            self.bn3 = identity
            self.bn4 = identity
            self.bn5 = identity
            self.bn6 = identity

            self.bn5d = identity
            self.bn4d = identity
            self.bn3d = identity
            self.bn2d = identity
            self.bn1d = identity

            self.bnout = identity
        elif use_norm == 'bn':
            self.bn2 = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)
            self.bn3 = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)
            self.bn4 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn5 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn6 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)

            self.bn5d = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn4d = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn3d = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)
            self.bn2d = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)

            self.bn1d = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)

            self.bnout = nn.BatchNorm1d(pl)
        elif use_norm == 'gn':
            self.bn2 = nn.GroupNorm(pl // 2, pl)
            self.bn3 = nn.GroupNorm(pl // 2, pl)
            self.bn4 = nn.GroupNorm(pl, 2 * pl)
            self.bn5 = nn.GroupNorm(pl, 2 * pl)
            self.bn6 = nn.GroupNorm(pl, 2 * pl)

            self.bn5d = nn.GroupNorm(pl, 2 * pl)
            self.bn4d = nn.GroupNorm(pl, 2 * pl)
            self.bn3d = nn.GroupNorm(pl // 2, pl)
            self.bn2d = nn.GroupNorm(pl // 2, pl)
            self.bn1d = nn.GroupNorm(pl // 2, pl)

            self.bnout = nn.GroupNorm(pl // 2, pl)
        else:
            raise ValueError

        self.drop = nn.Dropout(dropout)

    def forward(self, x, input_pts, output_pts=None):
        if output_pts is None:
            output_pts = input_pts
        x2, pts2 = self.cv2(x, input_pts, 24, 2048)
        x2 = self.act(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 32, 512)
        x3 = self.act(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 16, 128)
        x4 = self.act(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 32)
        x5 = self.act(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = self.act(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = self.act(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = self.act(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = self.act(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = self.act(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)

        x1d, _ = self.cv1d(x2d, pts2, 16, output_pts)
        x1d = self.act(apply_bn(x1d, self.bn1d))

        xout, _ = self.cvout(x1d, output_pts, 16)
        xout = self.act(apply_bn(xout, self.bnout))
        xout = self.drop(xout)

        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        return xout


class SegSmall(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, dropout=0,
                 act=None, use_bias=True, track_running_stats=False, use_norm=False):
        super(SegSmall, self).__init__()
        if act in [None, 'relu']:
            self.act = F.relu_
        elif act == 'swish':
            self.act = swish
        elif type(act) == str:
            self.act = getattr(F, act)
        elif callable(act):
            self.act = act
        else:
            raise ValueError

        n_centers = 16

        pl = 64
        self.cv2 = PtConv(input_channels, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv4 = PtConv(pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv5 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv6 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)

        self.cv5d = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv4d = PtConv(4 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv3d = PtConv(4 * pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv2d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.cv1d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)

        self.cvout = PtConv(pl, pl, n_centers, dimension, use_bias=use_bias, act=self.act)
        self.fcout = nn.Linear(pl, output_channels)

        if use_norm is False:
            self.bn = identity
            self.bn2 = identity
            self.bn3 = identity
            self.bn4 = identity
            self.bn5 = identity
            self.bn6 = identity

            self.bn5d = identity
            self.bn4d = identity
            self.bn3d = identity
            self.bn2d = identity
            self.bn1d = identity

            self.bnout = identity
        elif use_norm == 'bn':
            self.bn2 = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)
            self.bn3 = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)
            self.bn4 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn5 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn6 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)

            self.bn5d = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn4d = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn3d = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)
            self.bn2d = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)

            self.bn1d = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)

            self.bnout = nn.BatchNorm1d(pl)
        elif use_norm == 'gn':
            self.bn2 = nn.GroupNorm(pl // 2, pl)
            self.bn3 = nn.GroupNorm(pl // 2, pl)
            self.bn4 = nn.GroupNorm(pl, 2 * pl)
            self.bn5 = nn.GroupNorm(pl, 2 * pl)
            self.bn6 = nn.GroupNorm(pl, 2 * pl)

            self.bn5d = nn.GroupNorm(pl, 2 * pl)
            self.bn4d = nn.GroupNorm(pl, 2 * pl)
            self.bn3d = nn.GroupNorm(pl // 2, pl)
            self.bn2d = nn.GroupNorm(pl // 2, pl)
            self.bn1d = nn.GroupNorm(pl // 2, pl)

            self.bnout = nn.GroupNorm(pl // 2, pl)
        else:
            raise ValueError

        self.drop = nn.Dropout(dropout)

    def forward(self, x, input_pts, output_pts=None):
        if output_pts is None:
            output_pts = input_pts
        x2, pts2 = self.cv2(x, input_pts, 16, 2048)
        x2 = self.act(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 512)
        x3 = self.act(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 128)
        x4 = self.act(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 32)
        x5 = self.act(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = self.act(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = self.act(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = self.act(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = self.act(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = self.act(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)

        x1d, _ = self.cv1d(x2d, pts2, 8, output_pts)
        x1d = self.act(apply_bn(x1d, self.bn1d))

        xout, _ = self.cvout(x1d, output_pts, 8)
        xout = self.act(apply_bn(xout, self.bnout))
        xout = self.drop(xout)

        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        return xout


################################
# BIG SEGMENTATION NETWORK
################################

class SegBig(nn.Module):
    def __init__(self, input_channels, output_channels, trs=False, dimension=3, dropout=0, use_bias=False,
                 norm_type='bn', use_norm=True, kernel_size: int = 16, neighbor_nums=None,
                 reductions=None, first_layer=True, padding: int = None, nn_center: bool = True,
                 centroids: bool = False, pl: int = 64, normalize=False):
        super(SegBig, self).__init__()

        n_centers = kernel_size
        self.neighbor_nums = neighbor_nums
        self.reductions = reductions
        self.first_layer = first_layer
        self.normalize = normalize

        self.cv0 = PtConvBig(input_channels, pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                             nn_center=nn_center, centroids=centroids)
        self.cv1 = PtConvBig(pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                             nn_center=nn_center, centroids=centroids)
        self.cv2 = PtConvBig(pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                             nn_center=nn_center, centroids=centroids)
        self.cv3 = PtConvBig(pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                             nn_center=nn_center, centroids=centroids)
        self.cv4 = PtConvBig(pl, 2 * pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                             nn_center=nn_center, centroids=centroids)
        self.cv5 = PtConvBig(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                             nn_center=nn_center, centroids=centroids)
        self.cv6 = PtConvBig(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                             nn_center=nn_center, centroids=centroids)

        self.cv5d = PtConvBig(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                              nn_center=nn_center, centroids=centroids)
        self.cv4d = PtConvBig(4 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                              nn_center=nn_center, centroids=centroids)
        self.cv3d = PtConvBig(4 * pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                              nn_center=nn_center, centroids=centroids)
        self.cv2d = PtConvBig(2 * pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                              nn_center=nn_center, centroids=centroids)
        self.cv1d = PtConvBig(2 * pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                              nn_center=nn_center, centroids=centroids)
        self.cv0d = PtConvBig(2 * pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding,
                              nn_center=nn_center, centroids=centroids)

        self.fcout = nn.Linear(pl + pl, output_channels)

        if not use_norm:
            self.bn0 = nn.Identity()
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
            self.bn4 = nn.Identity()
            self.bn5 = nn.Identity()
            self.bn6 = nn.Identity()

            self.bn5d = nn.Identity()
            self.bn4d = nn.Identity()
            self.bn3d = nn.Identity()
            self.bn2d = nn.Identity()
            self.bn1d = nn.Identity()
            self.bn0d = nn.Identity()
        elif norm_type == 'bn':
            self.bn0 = nn.BatchNorm1d(pl, track_running_stats=trs)
            self.bn1 = nn.BatchNorm1d(pl, track_running_stats=trs)
            self.bn2 = nn.BatchNorm1d(pl, track_running_stats=trs)
            self.bn3 = nn.BatchNorm1d(pl, track_running_stats=trs)
            self.bn4 = nn.BatchNorm1d(2 * pl, track_running_stats=trs)
            self.bn5 = nn.BatchNorm1d(2 * pl, track_running_stats=trs)
            self.bn6 = nn.BatchNorm1d(2 * pl, track_running_stats=trs)

            self.bn5d = nn.BatchNorm1d(2 * pl, track_running_stats=trs)
            self.bn4d = nn.BatchNorm1d(2 * pl, track_running_stats=trs)
            self.bn3d = nn.BatchNorm1d(pl, track_running_stats=trs)
            self.bn2d = nn.BatchNorm1d(pl, track_running_stats=trs)
            self.bn1d = nn.BatchNorm1d(pl, track_running_stats=trs)
            self.bn0d = nn.BatchNorm1d(pl, track_running_stats=trs)
        elif norm_type == 'gn':
            self.bn0 = nn.GroupNorm(pl // 2, pl)
            self.bn1 = nn.GroupNorm(pl // 2, pl)
            self.bn2 = nn.GroupNorm(pl // 2, pl)
            self.bn3 = nn.GroupNorm(pl // 2, pl)
            self.bn4 = nn.GroupNorm(pl, 2 * pl)
            self.bn5 = nn.GroupNorm(pl, 2 * pl)
            self.bn6 = nn.GroupNorm(pl, 2 * pl)

            self.bn5d = nn.GroupNorm(pl, 2 * pl)
            self.bn4d = nn.GroupNorm(pl, 2 * pl)
            self.bn3d = nn.GroupNorm(pl // 2, pl)
            self.bn2d = nn.GroupNorm(pl // 2, pl)
            self.bn1d = nn.GroupNorm(pl // 2, pl)
            self.bn0d = nn.GroupNorm(pl // 2, pl)

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts, return_features=False):
        if self.neighbor_nums is None:
            self.neighbor_nums = [16, 16, 16, 16, 8, 8, 4, 4, 4, 4, 8, 8, 8]
        if self.reductions is None:
            self.reductions = [2048, 1024, 256, 64, 16, 8]

        x0, _ = self.cv0(x, input_pts, self.neighbor_nums[0], normalize=self.normalize)
        x0 = self.relu(apply_bn(x0, self.bn0))

        # Number of output points = 2048, Neighborhood of 16 points
        x1, pts1 = self.cv1(x0, input_pts, self.neighbor_nums[1], self.reductions[0], normalize=self.normalize)
        x1 = self.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, pts1, self.neighbor_nums[2], self.reductions[1], normalize=self.normalize)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, self.neighbor_nums[3], self.reductions[2], normalize=self.normalize)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, self.neighbor_nums[4], self.reductions[3], normalize=self.normalize)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, self.neighbor_nums[5], self.reductions[4], normalize=self.normalize)
        x5 = self.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, self.neighbor_nums[6], self.reductions[5], normalize=self.normalize)
        x6 = self.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, self.neighbor_nums[7], pts5, normalize=self.normalize)
        x5d = self.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, self.neighbor_nums[8], pts4, normalize=self.normalize)
        x4d = self.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, self.neighbor_nums[9], pts3, normalize=self.normalize)
        x3d = self.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, self.neighbor_nums[10], pts2, normalize=self.normalize)
        x2d = self.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)

        x1d, _ = self.cv1d(x2d, pts2, self.neighbor_nums[11], pts1, normalize=self.normalize)
        x1d = self.relu(apply_bn(x1d, self.bn1d))
        x1d = torch.cat([x1d, x1], dim=2)

        x0d, _ = self.cv0d(x1d, pts1, self.neighbor_nums[12], input_pts, normalize=self.normalize)
        x0d = self.relu(apply_bn(x0d, self.bn0d))
        x0d = torch.cat([x0d, x0], dim=2)

        xout = x0d
        xout = self.drop(xout)
        xout = self.fcout(xout)

        if return_features:
            return xout, x0d
        else:
            return xout


class ModelNet40(nn.Module):

    def __init__(self, input_channels, output_channels, dimension=3,
                 dropout=0.1, use_norm='gn', track_running_stats=False,
                 act=None, use_bias=True):
        super(ModelNet40, self).__init__()
        if act in [None, 'relu']:
            self.act = F.relu_
        elif act == 'swish':
            self.act = swish
        elif type(act) == str:
            self.act = getattr(F, act)
        elif callable(act):
            self.act = act
        else:
            raise ValueError
        n_centers = 16
        pl = 64
        # convolutions
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension, act=act, use_bias=use_bias)
        self.cv2 = PtConv(pl, 2 * pl, n_centers, dimension, act=act, use_bias=use_bias)
        self.cv3 = PtConv(2 * pl, 4 * pl, n_centers, dimension, act=act, use_bias=use_bias)
        self.cv4 = PtConv(4 * pl, 4 * pl, n_centers, dimension, act=act, use_bias=use_bias)
        self.cv5 = PtConv(4 * pl, 8 * pl, n_centers, dimension, act=act, use_bias=use_bias)

        # last layer
        self.lin1 = nn.Linear(8 * pl, 2 * pl)
        self.lin2 = nn.Linear(2 * pl, output_channels)

        # normalization
        if not use_norm:
            self.bn1 = identity
            self.bn2 = identity
            self.bn3 = identity
            self.bn4 = identity
            self.bn5 = identity
        elif use_norm == 'bn':
            self.bn1 = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)
            self.bn2 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn3 = nn.BatchNorm1d(4 * pl, track_running_stats=track_running_stats)
            self.bn4 = nn.BatchNorm1d(4 * pl, track_running_stats=track_running_stats)
            self.bn5 = nn.BatchNorm1d(8 * pl, track_running_stats=track_running_stats)
        elif use_norm == 'gn':
            self.bn1 = nn.GroupNorm(pl // 2, pl)
            self.bn2 = nn.GroupNorm(pl, 2 * pl)
            self.bn3 = nn.GroupNorm(2 * pl, 4 * pl)
            self.bn4 = nn.GroupNorm(2 * pl, 4 * pl)
            self.bn5 = nn.GroupNorm(4 * pl, 8 * pl)
        else:
            raise ValueError

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pts):
        x, pts = self.cv1(x, pts, 32, 4096)
        x = self.act(apply_bn(x, self.bn1))

        x, pts = self.cv2(x, pts, 32, 1024)
        x = self.act(apply_bn(x, self.bn2))

        x, pts = self.cv3(x, pts, 16, 512)
        x = self.act(apply_bn(x, self.bn3))

        x, pts = self.cv4(x, pts, 16, 256)
        x = self.act(apply_bn(x, self.bn4))

        x, pts = self.cv5(x, pts, 16, 128)
        x = self.act(apply_bn(x, self.bn5))
        x = x.mean(1)  # calculate mean across points -> aggregate evidence

        x = self.dropout(x)
        x = self.lin1(x)
        x = self.lin2(x)

        return x


class ModelNet40_orig(nn.Module):

    def __init__(self, input_channels, output_channels, dimension=3,
                 dropout=0.1, use_norm='gn', track_running_stats=False,
                 act=None, use_bias=False):
        super(ModelNet40, self).__init__()
        if act in [None, 'relu']:
            self.act = F.relu_
        elif act == 'swish':
            self.act = swish
        elif type(act) == str:
            self.act = getattr(F, act)
        elif callable(act):
            self.act = act
        else:
            raise ValueError
        n_centers = 16
        pl = 64
        # convolutions
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension, act=act, use_bias=use_bias)
        self.cv2 = PtConv(pl, 2 * pl, n_centers, dimension, act=act, use_bias=use_bias)
        self.cv3 = PtConv(2 * pl, 4 * pl, n_centers, dimension, act=act, use_bias=use_bias)
        self.cv4 = PtConv(4 * pl, 4 * pl, n_centers, dimension, act=act, use_bias=use_bias)
        self.cv5 = PtConv(4 * pl, 8 * pl, n_centers, dimension, act=act, use_bias=use_bias)

        # last layer
        self.fcout = nn.Linear(8 * pl, output_channels)

        # normalization
        if not use_norm:
            self.bn1 = identity
            self.bn2 = identity
            self.bn3 = identity
            self.bn4 = identity
            self.bn5 = identity
        elif use_norm == 'bn':
            self.bn1 = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)
            self.bn2 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn3 = nn.BatchNorm1d(4 * pl, track_running_stats=track_running_stats)
            self.bn4 = nn.BatchNorm1d(4 * pl, track_running_stats=track_running_stats)
            self.bn5 = nn.BatchNorm1d(8 * pl, track_running_stats=track_running_stats)
        elif use_norm == 'gn':
            self.bn1 = nn.GroupNorm(pl // 2, pl)
            self.bn2 = nn.GroupNorm(pl, 2 * pl)
            self.bn3 = nn.GroupNorm(2 * pl, 4 * pl)
            self.bn4 = nn.GroupNorm(2 * pl, 4 * pl)
            self.bn5 = nn.GroupNorm(4 * pl, 8 * pl)
        else:
            raise ValueError

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pts):
        x, pts = self.cv1(x, pts, 32, 4096)
        x = self.act(apply_bn(x, self.bn1))

        x, pts = self.cv2(x, pts, 32, 1024)
        x = self.act(apply_bn(x, self.bn2))

        x, pts = self.cv3(x, pts, 16, 512)
        x = self.act(apply_bn(x, self.bn3))

        x, pts = self.cv4(x, pts, 16, 256)
        x = self.act(apply_bn(x, self.bn4))

        x, pts = self.cv5(x, pts, 16, 128)
        x = self.act(apply_bn(x, self.bn5))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fcout(x)

        return x


class ModelNet40_PT3d(nn.Module):

    def __init__(self, input_channels, output_channels, dimension=3,
                 dropout=0.1, use_norm='gn', track_running_stats=False,
                 act=None):
        super(ModelNet40_PT3d, self).__init__()
        if act in [None, 'relu']:
            self.act = F.relu_
        elif act == 'swish':
            self.act = swish
        elif type(act) == str:
            self.act = getattr(F, act)
        elif callable(act):
            self.act = act
        else:
            raise ValueError
        n_centers = 16
        pl = 64
        # convolutions
        self.cv1 = PtConv_PT3d(input_channels, pl, n_centers, dimension, act=act)
        self.cv2 = PtConv_PT3d(pl, 2 * pl, n_centers, dimension, act=act)
        self.cv3 = PtConv_PT3d(2 * pl, 4 * pl, n_centers, dimension, act=act)
        self.cv4 = PtConv_PT3d(4 * pl, 4 * pl, n_centers, dimension, act=act)
        self.cv5 = PtConv_PT3d(4 * pl, 8 * pl, n_centers, dimension, act=act)

        # last layer
        self.lin1 = nn.Linear(8 * pl, 2 * pl)
        self.lin2 = nn.Linear(2 * pl, output_channels)

        # normalization
        if not use_norm:
            self.bn1 = identity
            self.bn2 = identity
            self.bn3 = identity
            self.bn4 = identity
            self.bn5 = identity
        elif use_norm == 'bn':
            self.bn1 = nn.BatchNorm1d(pl, track_running_stats=track_running_stats)
            self.bn2 = nn.BatchNorm1d(2 * pl, track_running_stats=track_running_stats)
            self.bn3 = nn.BatchNorm1d(4 * pl, track_running_stats=track_running_stats)
            self.bn4 = nn.BatchNorm1d(4 * pl, track_running_stats=track_running_stats)
            self.bn5 = nn.BatchNorm1d(8 * pl, track_running_stats=track_running_stats)
        elif use_norm == 'gn':
            self.bn1 = nn.GroupNorm(pl // 2, pl)
            self.bn2 = nn.GroupNorm(pl, 2 * pl)
            self.bn3 = nn.GroupNorm(2 * pl, 4 * pl)
            self.bn4 = nn.GroupNorm(2 * pl, 4 * pl)
            self.bn5 = nn.GroupNorm(4 * pl, 8 * pl)
        else:
            raise ValueError

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pts):
        x, pts = self.cv1(x, pts, 32, 4096)
        x = self.act(apply_bn(x, self.bn1))

        x, pts = self.cv2(x, pts, 32, 1024)
        x = self.act(apply_bn(x, self.bn2))

        x, pts = self.cv3(x, pts, 16, 512)
        x = self.act(apply_bn(x, self.bn3))

        x, pts = self.cv4(x, pts, 16, 256)
        x = self.act(apply_bn(x, self.bn4))

        x, pts = self.cv5(x, pts, 16, 128)
        x = self.act(apply_bn(x, self.bn5))
        x = x.mean(1)  # calculate mean across points -> aggregate evidence

        x = self.dropout(x)
        x = self.lin1(x)
        x = self.lin2(x)

        return x


class ModelNetBig(nn.Module):

    def __init__(self, input_channels, output_channels, dimension=3, dropout=0.1):
        super(ModelNetBig, self).__init__()

        n_centers = 16
        pl = 48

        # convolutions
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension)
        self.cv2 = PtConv(pl, 2 * pl, n_centers, dimension)
        self.cv3 = PtConv(2 * pl, 2 * pl, n_centers, dimension)
        self.cv4 = PtConv(2 * pl, 4 * pl, n_centers, dimension)
        self.cv5 = PtConv(4 * pl, 4 * pl, n_centers, dimension)

        # last layer
        self.fcout = nn.Linear(4 * pl, output_channels)

        # batchnorms
        self.bn1 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(4 * pl, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(4 * pl, track_running_stats=False)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts):
        x1, pts1 = self.cv1(x, input_pts, 32, input_pts.size(1) // 10)
        x1 = self.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, pts1, 32, 1024)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 16, 64)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, _ = self.cv5(x4, pts4, 16, 1)
        x5 = self.relu(apply_bn(x5, self.bn5))
        xout = x5.view(x5.size(0), -1)
        xout = self.dropout(xout)
        xout = self.fcout(xout)

        return xout


class ModelNetAttention(nn.Module):

    def __init__(self, input_channels, output_channels, dimension=3,
                 dropout=0.1, npoints=20000):
        super(ModelNetAttention, self).__init__()

        n_centers = 16
        pl = 32

        # convolutions
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension)
        self.cv2 = PtConv(pl, 2 * pl, n_centers, dimension)
        self.cv3 = PtConv(2 * pl, 2 * pl, n_centers, dimension)
        self.cv4 = PtConv(2 * pl, 4 * pl, n_centers, dimension)
        self.cv5 = PtConv(4 * pl, 4 * pl, n_centers, dimension)

        # last layer
        self.fcout = nn.Linear(4 * pl, output_channels)

        # attention
        self.att1 = Attention(npoints)

        # batchnorms
        self.bn1 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(4 * pl, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(4 * pl, track_running_stats=False)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts):
        x1, pts1 = self.cv1(x, input_pts, 32)
        x1 = self.relu(apply_bn(x1, self.bn1))

        # learn to select the basis points for the first reduction step
        att_w = self.att1(x1.transpose(1, 2))
        att_ixs = torch.argsort(att_w, dim=1)[:, :1024].unsqueeze(2)
        att_ixs = torch.cat([att_ixs, att_ixs, att_ixs], dim=2)
        pts2 = torch.gather(pts1, 1, att_ixs)

        x2, pts2 = self.cv2(x1, pts1, 32, pts2)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 16, 64)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, _ = self.cv5(x4, pts4, 16, 1)
        x5 = self.relu(apply_bn(x5, self.bn5))
        xout = x5.view(x5.size(0), -1)
        xout = self.dropout(xout)
        xout = self.fcout(xout)

        return xout


class ModelNetAttentionBig(nn.Module):

    def __init__(self, input_channels, output_channels, dimension=3,
                 dropout=0.1, npoints=20000):
        super(ModelNetAttentionBig, self).__init__()

        n_centers = 16
        pl = 64
        self.npoints = npoints

        # convolutions
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension)
        self.cv2 = PtConv(pl, 2 * pl, n_centers, dimension)
        self.cv3 = PtConv(2 * pl, 4 * pl, n_centers, dimension)
        self.cv4 = PtConv(4 * pl, 4 * pl, n_centers, dimension)
        self.cv5 = PtConv(4 * pl, 8 * pl, n_centers, dimension)

        # last layer
        self.fcout = nn.Linear(8 * pl, output_channels)

        # attention
        self.att1 = Attention(self.npoints // 3)
        # batchnorms
        self.bn1 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(4 * pl, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(4 * pl, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(8 * pl, track_running_stats=False)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts):
        if input_pts.size(1) != self.npoints:
            raise ValueError(f'Number of input points {input_pts.size(1)} does '
                             f'not match the attention layer {self.npoints}.')
        x1, pts1 = self.cv1(x, input_pts, 32, self.npoints // 3)
        x1 = self.relu(apply_bn(x1, self.bn1))

        # learn to select the basis points for the first reduction step
        att_w = self.att1(x1.transpose(1, 2))
        att_ixs = torch.argsort(att_w, dim=1)[:, :2048].unsqueeze(2)
        att_ixs = torch.cat([att_ixs, att_ixs, att_ixs], dim=2)
        pts2 = torch.gather(pts1, 1, att_ixs)

        x2, pts2 = self.cv2(x1, pts1, 32, pts2)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 512)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 16, 128)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, _ = self.cv5(x4, pts4, 16, 1)
        x5 = self.relu(apply_bn(x5, self.bn5))
        xout = x5.view(x5.size(0), -1)
        xout = self.dropout(xout)
        xout = self.fcout(xout)

        return xout


class ModelNetSelection(nn.Module):

    def __init__(self, input_channels, output_channels, dimension=3,
                 dropout=0.1, npoints=20000):
        super(ModelNetSelection, self).__init__()

        n_centers = 16
        pl = 32

        # convolutions
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension)
        self.cv2 = PtConv(pl, 2 * pl, n_centers, dimension)
        self.cv3 = PtConv(2 * pl, 2 * pl, n_centers, dimension)
        self.cv4 = PtConv(2 * pl, 4 * pl, n_centers, dimension)
        self.cv5 = PtConv(4 * pl, 4 * pl, n_centers, dimension)

        # last layer
        self.fcout = nn.Linear(4 * pl, output_channels)

        # selection
        self.select1 = SelectionLayer(pl)

        # batchnorms
        self.bn1 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(4 * pl, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(4 * pl, track_running_stats=False)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts):
        x1, pts1 = self.cv1(x, input_pts, 32)
        x1 = self.relu(apply_bn(x1, self.bn1))

        # learn to select the basis points for the first reduction step
        pts2 = self.select1(x1, 1024, pts1)

        x2, pts2 = self.cv2(x1, pts1, 32, pts2)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 16, 64)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, _ = self.cv5(x4, pts4, 16, 1)
        x5 = self.relu(apply_bn(x5, self.bn5))
        xout = x5.view(x5.size(0), -1)
        xout = self.dropout(xout)
        xout = self.fcout(xout)

        return xout


class ModelNetSelectionBig(nn.Module):

    def __init__(self, input_channels, output_channels, dimension=3,
                 dropout=0.1):
        super(ModelNetSelectionBig, self).__init__()

        n_centers = 16
        pl = 32
        # convolutions
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension)
        self.cv2 = PtConv(pl, 2 * pl, n_centers, dimension)
        self.cv3 = PtConv(2 * pl, 2 * pl, n_centers, dimension)
        self.cv4 = PtConv(2 * pl, 4 * pl, n_centers, dimension)
        self.cv5 = PtConv(4 * pl, 4 * pl, n_centers, dimension)

        # last layer
        self.fcout = nn.Linear(4 * pl, output_channels)

        # selection
        self.select1 = SelectionLayer(pl)

        # batchnorms
        self.bn1 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(4 * pl, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(4 * pl, track_running_stats=False)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts):
        x1, pts1 = self.cv1(x, input_pts, 32, input_pts.size(1) // 2)
        x1 = self.relu(apply_bn(x1, self.bn1))

        # learn to select 2048 basis points for this reduction step
        pts2 = self.select1(x1, 2048, pts1)

        x2, pts2 = self.cv2(x1, pts1, 32, pts2)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 512)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 16, 128)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, _ = self.cv5(x4, pts4, 16, 1)
        x5 = self.relu(apply_bn(x5, self.bn5))
        xout = x5.view(x5.size(0), -1)
        xout = self.dropout(xout)
        xout = self.fcout(xout)

        return xout


class SelectionLayer(nn.Module):
    def __init__(self, input_channels):
        super(SelectionLayer, self).__init__()
        self.selection = nn.Linear(input_channels, 1)
        self.bn = nn.BatchNorm1d(1, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_in, top_n, x_select=None):
        """
        Linear layer which learns to select `top_n` elements from `x_select`
        based on the features `x_in`.

        Args:
            x_in: Feature tensor used for selection (B x N x C).
            top_n: Number of elements returned for every sample in the batch.
            x_select: If None, will be set to `x_in`.
                Shape required to be (B, N, C)

        Returns:
            The `top_n` selection of x_select (B x top_n x C).
        """
        if x_select is None:
            x_select = x_in
        selection_score = self.selection(x_in)
        selection_score = self.relu(apply_bn(selection_score, self.bn))
        att_ixs = torch.argsort(selection_score, dim=1)[:, :top_n]
        # TODO: any more generic way to create an appropriate index used by torch.gather?
        att_ixs = torch.cat([att_ixs] * x_select.size(2), dim=2)
        return torch.gather(x_select, 1, att_ixs)


def new_parameter(*size):
    """
    # from https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/patterns/attention.html
    """
    out = nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out


class Attention(nn.Module):
    """
    # from https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/patterns/attention.html
    attn = Attention(100)
    x = Variable(torch.randn(16,30,100))
    attn(x).size() == (16,100)
    """
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention = new_parameter(attention_size, 1)

    def forward(self, x_in):
        # after this, we have (batch, dim1) with a diff weight per each cell
        attention_score = torch.matmul(x_in, self.attention).squeeze()
        attention_score = F.softmax(attention_score, dim=-1).view(x_in.size(0), x_in.size(1), 1)
        scored_x = x_in * attention_score

        # now, sum across dim 1 to get the expected feature vector
        condensed_x = torch.sum(scored_x, dim=1)
        return condensed_x


class ModelNet40xConv(nn.Module):

    def __init__(self, input_channels, output_channels, dimension=3, dropout=0.1):
        super(ModelNet40xConv, self).__init__()

        pl = 32

        # convolutions
        self.cv1 = XConv(input_channels, pl, dimension, 8, hidden_channels=32)
        self.cv2 = XConv(pl, 2 * pl, dimension, 8, hidden_channels=64, dilation=2)
        self.cv3 = XConv(2 * pl, 4 * pl, dimension, 12, hidden_channels=128, dilation=2)
        self.cv4 = XConv(4 * pl, 4 * pl, dimension, 16, hidden_channels=128, dilation=2)
        self.cv5 = XConv(4 * pl, 8 * pl, dimension, 16, hidden_channels=256, dilation=2)

        # last layer
        self.lin1 = nn.Linear(8 * pl, 2 * pl)
        self.lin2 = nn.Linear(2 * pl, output_channels)

        # batchnorms
        self.bn1 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(4 * pl, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(4 * pl, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(8 * pl, track_running_stats=False)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, pos):
        nbatches = x.size(0)
        npts = x.size(1)
        x = x.view(-1, x.size(-1))
        pos = pos.view(-1, pos.size(-1))
        batch = torch.tensor(np.repeat(np.arange(nbatches), npts)).to(pos.device)

        x = self.cv1(x, pos, batch)
        idx = fps(pos, batch, ratio=0.33)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        x = self.relu(self.bn1(x))
        x = self.cv2(x, pos, batch)

        x = self.relu(self.bn2(x))
        idx = fps(pos, batch, ratio=0.33)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        x = self.cv3(x, pos, batch)

        x = self.relu(self.bn3(x))
        idx = fps(pos, batch, ratio=0.33)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        x = self.cv4(x, pos, batch)

        x = self.relu(self.bn4(x))
        x = self.cv5(x, pos, batch)

        x = global_mean_pool(x, batch)

        x = self.relu(self.bn5(x))
        x = self.dropout(x)
        x = self.lin1(x)
        x = self.lin2(x)

        return x


class TripletNet(nn.Module):
    """
    adapted from https://github.com/andreasveit/triplet-network-pytorch/blob/master/tripletnet.py
    """
    def __init__(self, rep_net):
        super().__init__()
        self.rep_net = rep_net

    def forward(self, x0, x1, x2):
        if not self.training:
            assert x1 is None and x2 is None
            return self.rep_net(x0[0], x0[1])
        assert x1 is not None, x2 is not None
        z_0 = self.rep_net(x0[0], x0[1])
        z_1 = self.rep_net(x1[0], x1[1])
        z_2 = self.rep_net(x2[0], x2[1])
        dist_a = F.pairwise_distance(z_0, z_1, 2)
        dist_b = F.pairwise_distance(z_0, z_2, 2)
        return dist_a, dist_b, z_0, z_1, z_2
