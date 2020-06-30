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
import time
import elektronn3.models.knn.lib.python.nearest_neighbors as nearest_neighbors
from abc import ABC
from typing import Tuple


# STATIC HELPER FUNCTIONS #

def swish(x):
    """https://arxiv.org/pdf/1710.05941.pdf"""
    return x * torch.sigmoid(x)


def identity(x):
    return x


def apply_bn(x, bn):
    return bn(x.transpose(1, 2)).transpose(1, 2).contiguous()


def indices_conv_reduction(input_pts: torch.Tensor, output_pts_num: int, neighbor_num: int, padding: int = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """ This function picks output_pts_num random points from input_pts and returns these points (queries) and
        their neighbor_num nearest neighbors from input_pts (indices).
    """
    if padding is not None:
        input_pts_np = input_pts.cpu().detach().numpy()
        output_pts = np.ones((len(input_pts_np), output_pts_num, 3))*padding
        for ix, batch in enumerate(input_pts_np):
            filtered_pts = batch[batch < padding].reshape(-1, 3)
            output_pts[ix, :len(filtered_pts)] = \
                filtered_pts[np.random.choice(len(filtered_pts), min(len(filtered_pts), output_pts_num), replace=False)]

        indices = nearest_neighbors.knn_batch(input_pts_np, output_pts, neighbor_num, omp=True)
    else:
        indices, output_pts = nearest_neighbors.knn_batch_distance_pick(input_pts.cpu().detach().numpy(),
                                                                        output_pts_num, neighbor_num, omp=True)
    indices = torch.from_numpy(indices).long()
    output_pts = torch.from_numpy(output_pts).float()
    if input_pts.is_cuda:
        indices = indices.to(input_pts.device)
        output_pts = output_pts.to(input_pts.device)
    return indices, output_pts


def indices_conv(input_pts: torch.Tensor, neighbor_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns the indices of neighbor_num nearest neighbors for each point in input_pts. """
    indices = nearest_neighbors.knn_batch(input_pts.cpu().detach().numpy(),
                                          input_pts.cpu().detach().numpy(), neighbor_num, omp=True)
    indices = torch.from_numpy(indices).long()
    if input_pts.is_cuda:
        indices = indices.to(input_pts.device)
    return indices, input_pts


def indices_deconv(input_pts: torch.Tensor, output_pts: torch.Tensor, neighbor_num: int) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns the indices of neighbor_num nearest neighbors for each point in output_pts. """
    indices = nearest_neighbors.knn_batch(input_pts.cpu().detach().numpy(),
                                          output_pts.cpu().detach().numpy(), neighbor_num, omp=True)
    indices = torch.from_numpy(indices).long()
    if input_pts.is_cuda:
        indices = indices.to(input_pts.device)
    return indices, output_pts


# LAYER DEFINITIONS #


class LayerBase(nn.Module, ABC):

    def __init__(self):
        super(LayerBase, self).__init__()


class PtConv(LayerBase):
    def __init__(self, input_features, output_features, kernele_num, dim, act=None, use_bias=True, padding: int = None):
        """
        Args:
            input_features: Number of input channels (e.g. with RGB this would be 3)
            output_features: Number of convolutional kernels used in this layer. It defines the output
                feature dimension, i.e. the size of y.
            kernele_num: Number of elements per convolutional kernel.
            dim: The dimensions of the problem. For 3D point clouds this is set to 3.
        """
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

        # Kernel depth equals number of input features. Thus, the number of needed weights is
        # Input_Channels x Kernel number x Kernel elements. These weights get initialized randomly.
        self.weight = \
            nn.Parameter(torch.Tensor(input_features, kernele_num, output_features), requires_grad=True)
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (input_features + output_features))
        self.weight.data.uniform_(-bound, bound)

        # The model is built around RELU(Wx+b) where b is the bias. If Batch Normalization is used, the bias term is
        # not needed as it is included in the normalization. Thus, it is set to 0 when calling this function.
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_features), requires_grad=True)
            self.bias.data.uniform_(0, 0)

        self.padding = padding

        # The kernel elements get initialized within the unit sphere. This includes only the elements of one single
        # kernel. The position of theses kernel elements are adjusted during training.
        center_data = np.zeros((dim, kernele_num))
        for i in range(kernele_num):
            coord = np.random.rand(dim) * 2 - 1
            while (coord ** 2).sum() > 1:
                coord = np.random.rand(dim) * 2 - 1
            center_data[:, i] = coord
        self.centers = nn.Parameter(torch.from_numpy(center_data).float(), requires_grad=True)

        # Neural Network for adjusting the kernel element positions. First layer consists of 2*n_centers linear units,
        # second and third layer have n_centers linear units. Each linear unit computes y = Wx+b.
        self.l1 = nn.Linear(dim * kernele_num, 2 * kernele_num)
        self.l2 = nn.Linear(2 * kernele_num, kernele_num)
        self.l3 = nn.Linear(kernele_num, kernele_num)

    def forward(self, features, input_pts, neighbor_num, output_pts=None, normalize=False,
                indices_=None, return_indices=False, dilation=1):
        """
        Args:
            features: Batch of features.
            input_pts: Batch of points.
            neighbor_num: Size of neighborhood.
            output_pts: Number of output points.
            normalize: Normalization to unit sphere.
            indices_: neighbor indices.
            return_indices: Flag for returning the neighbor indices.
            dilation: factor for dilated convolutions
        """
        if indices_ is None:
            if isinstance(output_pts, int) and input_pts.size(1) != output_pts:
                indices, next_pts_ = indices_conv_reduction(input_pts, output_pts, neighbor_num * dilation, self.padding)
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


################################
# BIG SEGMENTATION NETWORK
################################


class SegBig(nn.Module):
    def __init__(self, input_channels, output_channels, trs=False, dimension=3, dropout=0, use_bias=False,
                 norm_type='bn', use_norm=True, kernel_size: int = 16, neighbor_nums=None, dilations=None,
                 reductions=None, first_layer=True, padding: int = None):
        super(SegBig, self).__init__()

        n_centers = kernel_size
        self.neighbor_nums = neighbor_nums
        if dilations is None:
            dilations = [1, 1, 1, 1]
        self.dilations = dilations
        self.reductions = reductions
        self.first_layer = first_layer

        # Number of convolutional kernels to start with
        pl = 64

        # 64 convolutional kernels
        self.cv0 = PtConv(input_channels, pl, n_centers, dimension, use_bias=use_bias, padding=padding)
        self.cv1 = PtConv(pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding)
        self.cv2 = PtConv(pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding)

        # 128 convolutional kernels
        self.cv4 = PtConv(pl, 2 * pl, n_centers, dimension, use_bias=use_bias, padding=padding)
        self.cv5 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, padding=padding)
        self.cv6 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, padding=padding)
        self.cv5d = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, padding=padding)
        # Inputs get concatenated with previous inputs
        self.cv4d = PtConv(4 * pl, 2 * pl, n_centers, dimension, use_bias=use_bias, padding=padding)

        # 64 Convolutional kernels + Concatenated inputs
        self.cv3d = PtConv(4 * pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding)
        self.cv2d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding)
        self.cv1d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding)
        self.cv0d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=use_bias, padding=padding)

        # Fully connected layer
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
        """
        Args:
            x: Batch of features
            input_pts: Batch of points
            return_features: Flag for returning calculated features.
        """
        if self.neighbor_nums is None:
            self.neighbor_nums = [16, 16, 16, 16, 8, 8, 4, 4, 4, 4, 8, 8, 8]
        if self.reductions is None:
            self.reductions = [2048, 1024, 256, 64, 16, 8]

        x0, _ = self.cv0(x, input_pts, self.neighbor_nums[0], dilation=self.dilations[0])
        x0 = self.relu(apply_bn(x0, self.bn0))

        # Number of output points = 2048, Neighborhood of 16 points
        x1, pts1 = self.cv1(x0, input_pts, self.neighbor_nums[1], self.reductions[0], dilation=self.dilations[1])
        x1 = self.relu(apply_bn(x1, self.bn1))

        x2, pts2 = self.cv2(x1, pts1, self.neighbor_nums[2], self.reductions[1], dilation=self.dilations[2])
        x2 = self.relu(apply_bn(x2, self.bn2))

        x3, pts3 = self.cv3(x2, pts2, self.neighbor_nums[3], self.reductions[2], dilation=self.dilations[3])
        x3 = self.relu(apply_bn(x3, self.bn3))

        x4, pts4 = self.cv4(x3, pts3, self.neighbor_nums[4], self.reductions[3])
        x4 = self.relu(apply_bn(x4, self.bn4))

        x5, pts5 = self.cv5(x4, pts4, self.neighbor_nums[5], self.reductions[4])
        x5 = self.relu(apply_bn(x5, self.bn5))

        x6, pts6 = self.cv6(x5, pts5, self.neighbor_nums[6], self.reductions[5])
        x6 = self.relu(apply_bn(x6, self.bn6))

        x5d, _ = self.cv5d(x6, pts6, self.neighbor_nums[7], pts5)
        x5d = self.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)

        x4d, _ = self.cv4d(x5d, pts5, self.neighbor_nums[8], pts4)
        x4d = self.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)

        x3d, _ = self.cv3d(x4d, pts4, self.neighbor_nums[9], pts3)
        x3d = self.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)

        x2d, _ = self.cv2d(x3d, pts3, self.neighbor_nums[10], pts2)
        x2d = self.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)

        x1d, _ = self.cv1d(x2d, pts2, self.neighbor_nums[11], pts1)
        x1d = self.relu(apply_bn(x1d, self.bn1d))
        x1d = torch.cat([x1d, x1], dim=2)

        x0d, _ = self.cv0d(x1d, pts1, self.neighbor_nums[12], input_pts)
        x0d = self.relu(apply_bn(x0d, self.bn0d))
        x0d = torch.cat([x0d, x0], dim=2)

        xout = x0d
        xout = self.drop(xout)
        xout = self.fcout(xout)

        if return_features:
            return xout, x0d
        else:
            return xout

################################
# TEST SEGMENTATION NETWORK
################################


class SegTest(nn.Module):
    def __init__(self, input_channels, output_channels, dimension=3, dropout=0):
        super(SegTest, self).__init__()

        n_centers = 16

        # Number of convolutional kernels to start with
        pl = 64

        # 64 convolutional kernels
        self.cv0 = PtConv(input_channels, pl, n_centers, dimension, use_bias=False)
        self.cv1 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv2 = PtConv(pl, pl, n_centers, dimension, use_bias=False)
        self.cv3 = PtConv(pl, pl, n_centers, dimension, use_bias=False)

        # 128 convolutional kernels
        self.cv4 = PtConv(pl, 2 * pl, n_centers, dimension, use_bias=False)
        self.cv5 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=False)
        self.cv6 = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=False)
        self.cv5d = PtConv(2 * pl, 2 * pl, n_centers, dimension, use_bias=False)
        # Inputs get concatenated with previous inputs
        self.cv4d = PtConv(4 * pl, 2 * pl, n_centers, dimension, use_bias=False)

        # 64 Convolutional kernels + Concatenated inputs
        self.cv3d = PtConv(4 * pl, pl, n_centers, dimension, use_bias=False)
        self.cv2d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=False)
        self.cv1d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=False)
        self.cv0d = PtConv(2 * pl, pl, n_centers, dimension, use_bias=False)

        # Fully connected layer
        self.fcout = nn.Linear(pl + pl, output_channels)

        self.bn0 = nn.BatchNorm1d(pl)
        self.bn1 = nn.BatchNorm1d(pl)
        self.bn2 = nn.BatchNorm1d(pl)
        self.bn3 = nn.BatchNorm1d(pl)
        self.bn4 = nn.BatchNorm1d(2 * pl)
        self.bn5 = nn.BatchNorm1d(2 * pl)
        self.bn6 = nn.BatchNorm1d(2 * pl)

        self.bn5d = nn.BatchNorm1d(2 * pl)
        self.bn4d = nn.BatchNorm1d(2 * pl)
        self.bn3d = nn.BatchNorm1d(pl)
        self.bn2d = nn.BatchNorm1d(pl)
        self.bn1d = nn.BatchNorm1d(pl)
        self.bn0d = nn.BatchNorm1d(pl)

        self.drop = nn.Dropout(dropout)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts, return_features=False, output_mask=None):
        """
        Args:
            x: Batch of features
            input_pts: Batch of points
            return_features: Flag for returning calculated features.
            output_mask: Mask for excluding points from prediction
        """

        if output_mask is None:
            output_mask = torch.ones((input_pts.shape[0], input_pts.shape[1]), dtype=torch.bool)

        start = time.time()
        x0, _ = self.cv0(x, input_pts, 16)
        x0 = self.relu(apply_bn(x0, self.bn0))
        print(f"cv0: {time.time() - start} s")

        # Number of output points = 2048, Neighborhood of 16 points
        start = time.time()
        x1, pts1 = self.cv1(x0, input_pts, 16, 2048)
        x1 = self.relu(apply_bn(x1, self.bn1))
        print(f"cv1: {time.time() - start} s")

        start = time.time()
        x2, pts2 = self.cv2(x1, pts1, 16, 1024)
        x2 = self.relu(apply_bn(x2, self.bn2))
        print(f"cv2: {time.time() - start} s")

        start = time.time()
        x3, pts3 = self.cv3(x2, pts2, 16, 256)
        x3 = self.relu(apply_bn(x3, self.bn3))
        print(f"cv3: {time.time() - start} s")

        start = time.time()
        x4, pts4 = self.cv4(x3, pts3, 8, 64)
        x4 = self.relu(apply_bn(x4, self.bn4))
        print(f"cv4: {time.time() - start} s")

        start = time.time()
        x5, pts5 = self.cv5(x4, pts4, 8, 16)
        x5 = self.relu(apply_bn(x5, self.bn5))
        print(f"cv5: {time.time() - start} s")

        start = time.time()
        x6, pts6 = self.cv6(x5, pts5, 4, 8)
        x6 = self.relu(apply_bn(x6, self.bn6))
        print(f"cv6: {time.time() - start} s")

        start = time.time()
        x5d, _ = self.cv5d(x6, pts6, 4, pts5)
        x5d = self.relu(apply_bn(x5d, self.bn5d))
        x5d = torch.cat([x5d, x5], dim=2)
        print(f"cv5d: {time.time() - start} s")

        start = time.time()
        x4d, _ = self.cv4d(x5d, pts5, 4, pts4)
        x4d = self.relu(apply_bn(x4d, self.bn4d))
        x4d = torch.cat([x4d, x4], dim=2)
        print(f"cv4d: {time.time() - start} s")

        start = time.time()
        x3d, _ = self.cv3d(x4d, pts4, 4, pts3)
        x3d = self.relu(apply_bn(x3d, self.bn3d))
        x3d = torch.cat([x3d, x3], dim=2)
        print(f"cv3d: {time.time() - start} s")

        start = time.time()
        x2d, _ = self.cv2d(x3d, pts3, 8, pts2)
        x2d = self.relu(apply_bn(x2d, self.bn2d))
        x2d = torch.cat([x2d, x2], dim=2)
        print(f"cv2d: {time.time() - start} s")

        start = time.time()
        x1d, _ = self.cv1d(x2d, pts2, 8, pts1)
        x1d = self.relu(apply_bn(x1d, self.bn1d))
        x1d = torch.cat([x1d, x1], dim=2)
        print(f"cv1d: {time.time() - start} s")

        start = time.time()
        x0d, _ = self.cv0d(x1d, pts1, 8, input_pts)
        x0d = self.relu(apply_bn(x0d, self.bn0d))
        x0d = torch.cat([x0d, x0], dim=2)
        print(f"cv0d: {time.time() - start} s")

        xout = x0d
        xout = self.drop(xout)
        start = time.time()
        xout = self.fcout(xout)
        print(f"fc: {time.time() - start} s")

        if return_features:
            return xout, x0d
        else:
            return xout


################################
# CLASSIFICATION NETWORKS
################################


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
