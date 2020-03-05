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
import elektronn3.models.knn.lib.python.nearest_neighbors as nearest_neighbors
from abc import ABC
try:
    from torch_geometric.nn import XConv, fps, global_mean_pool
except ImportError as e:
    print('XConv layer not available.', e)

# STATIC HELPER FUNCTIONS #


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


class PtConv(LayerBase):
    def __init__(self, input_features, output_features, n_centers, dim, use_bias=True):
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
        dists = F.relu(self.l1(dists))
        dists = F.relu(self.l2(dists))
        dists = F.relu(self.l3(dists))

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


################################
# SMALL SEGMENTATION NETWORK
################################


class SegSmall(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, dropout=0):
        super(SegSmall, self).__init__()

        n_centers = 16

        pl = 48
        self.cv2 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2 * pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4 * pl, 2 * pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4 * pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl, output_channels)

        self.bn2 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn6 = nn.BatchNorm1d(2 * pl, track_running_stats=False)

        self.bn5d = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn4d = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn3d = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn2d = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn1d = nn.BatchNorm1d(pl, track_running_stats=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, input_pts):

        x2, pts2 = self.cv2(x, input_pts, 16, 1024)
        x2 = F.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = F.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)
        x4 = F.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = F.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = F.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = F.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = F.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = F.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = F.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)

        x1d, _ = self.cv1d(x2d, pts2, 8, input_pts)
        x1d = F.relu(apply_bn(x1d, self.bn1d))

        xout = x1d
        xout = xout.view(-1, xout.size(2))
        xout = self.drop(xout)
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        return xout


################################
# BIG SEGMENTATION NETWORK
################################


class SegBig(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, dropout=0):
        super(SegBig, self).__init__()

        n_centers = 16

        pl = 64
        self.cv0 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv1 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv4 = PtConv(pl, 2 * pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=False)

        self.cv5d = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=False)
        self.cv4d = PtConv(4 * pl, 2 * pl, n_centers, dimension, use_bias=False)
        self.cv3d = PtConv(4 * pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=False)
        self.cv0d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=False)

        self.fcout = nn.Linear(pl + pl, output_channels)

        self.bn0 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn1 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn4 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn6 = nn.BatchNorm1d(2 * pl, track_running_stats=False)

        self.bn5d = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn4d = nn.BatchNorm1d(2 * pl, track_running_stats=False)
        self.bn3d = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn2d = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn1d = nn.BatchNorm1d(pl, track_running_stats=False)
        self.bn0d = nn.BatchNorm1d(pl, track_running_stats=False)

        self.drop = nn.Dropout(dropout)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts, return_features=False):

        x0, _ = self.cv0(x, input_pts, 16)
        x0 = self.relu(apply_bn(x0, self.bn0))

        x1, pts1 = self.cv1(x0, input_pts, 16, 2048)
        x1 = self.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, pts1, 16, 1024)
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, 8, 64)
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = self.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = self.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = self.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = self.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = self.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = self.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)

        x1d, _ = self.cv1d(x2d, pts2, 8, pts1)
        x1d = self.relu(apply_bn(x1d, self.bn1d))
        x1d = torch.cat([x1d, x1], dim=2)

        x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        x0d = self.relu(apply_bn(x0d, self.bn0d))

        x0d = torch.cat([x0d, x0], dim=2)

        xout = x0d
        xout = self.drop(xout)
        xout = xout.view(-1, xout.size(2))
        xout = self.fcout(xout)
        xout = xout.view(x.size(0), -1, xout.size(1))

        if return_features:
            return xout, x0d
        else:
            return xout


class ModelNet40(nn.Module):

    def __init__(self, input_channels, output_channels, dimension=3,
                 dropout=0.1, use_bn=True):
        super(ModelNet40, self).__init__()

        n_centers = 16
        pl = 64

        # convolutions
        self.cv1 = PtConv(input_channels, pl, n_centers, dimension)
        self.cv2 = PtConv(pl, 2 * pl, n_centers, dimension)
        self.cv3 = PtConv(2 * pl, 4 * pl, n_centers, dimension)
        self.cv4 = PtConv(4 * pl, 4 * pl, n_centers, dimension)
        self.cv5 = PtConv(4 * pl, 8 * pl, n_centers, dimension)

        # last layer
        self.lin1 = nn.Linear(8 * pl, 2 * pl)
        self.lin2 = nn.Linear(2 * pl, output_channels)

        # batchnorms
        if use_bn:
            self.bn1 = nn.BatchNorm1d(pl, track_running_stats=False)
            self.bn2 = nn.BatchNorm1d(2 * pl, track_running_stats=False)
            self.bn3 = nn.BatchNorm1d(4 * pl, track_running_stats=False)
            self.bn4 = nn.BatchNorm1d(4 * pl, track_running_stats=False)
            self.bn5 = nn.BatchNorm1d(8 * pl, track_running_stats=False)
        else:
            self.bn1 = identity
            self.bn2 = identity
            self.bn3 = identity
            self.bn4 = identity
            self.bn5 = identity

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, pts):
        x, pts = self.cv1(x, pts, 32, 4096)
        x = self.relu(apply_bn(x, self.bn1))

        x, pts = self.cv2(x, pts, 32, 1024)
        x = self.relu(apply_bn(x, self.bn2))

        x, pts = self.cv3(x, pts, 16, 512)
        x = self.relu(apply_bn(x, self.bn3))

        x, pts = self.cv4(x, pts, 16, 256)
        x = self.relu(apply_bn(x, self.bn4))

        x, pts = self.cv5(x, pts, 16, 128)
        x = self.relu(apply_bn(x, self.bn5))
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
