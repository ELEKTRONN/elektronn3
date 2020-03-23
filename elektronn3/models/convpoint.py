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
import gpustat
import elektronn3.models.knn.lib.python.nearest_neighbors as nearest_neighbors
from abc import ABC
from typing import Tuple


# STATIC HELPER FUNCTIONS #


def apply_bn(x, bn):
    return bn(x.transpose(1, 2)).transpose(1, 2).contiguous()


def indices_conv_reduction(input_pts: torch.Tensor, output_pts_num: int, neighbor_num: int) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """ This function picks output_pts_num random points from input_pts and returns these points (queries) and
        their neighbor_num nearest neighbors from input_pts (indices).
    """
    indices, queries = nearest_neighbors.knn_batch_distance_pick(input_pts.cpu().detach().numpy(),
                                                                 output_pts_num, neighbor_num, omp=True)
    indices = torch.from_numpy(indices).long()
    queries = torch.from_numpy(queries).float()
    if input_pts.is_cuda:
        indices = indices.to(input_pts.device)
        queries = queries.to(input_pts.device)
    return indices, queries


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
    def __init__(self, input_features, output_features, kernele_num, dim, use_bias=True):
        """
        Args:
            input_features: Number of input channels (e.g. with RGB this would be 3)
            output_features: Number of convolutional kernels used in this layer. It defines the output
                feature dimension, i.e. the size of y.
            kernele_num: Number of elements per convolutional kernel.
            dim: The dimensions of the problem. For 3D point clouds this is set to 3.
        """
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

        # The kernel elements get initialized within the unit sphere. This includes only the elements of one single
        # kernel. The position of theses kernel elements are adjusted during training.
        center_data = np.zeros((dim, kernele_num))
        for i in range(kernele_num):
            coord = np.random.rand(dim)*2 - 1
            while (coord**2).sum() > 1:
                coord = np.random.rand(dim)*2 - 1
            center_data[:, i] = coord
        self.centers = nn.Parameter(torch.from_numpy(center_data).float(), requires_grad=True)

        # Neural Network for adjusting the kernel element positions. First layer consists of 2*n_centers linear units,
        # second and third layer have n_centers linear units. Each linear unit computes y = Wx+b.
        self.l1 = nn.Linear(dim * kernele_num, 2 * kernele_num)
        self.l2 = nn.Linear(2 * kernele_num, kernele_num)
        self.l3 = nn.Linear(kernele_num, kernele_num)

    def forward(self, features, input_pts, neighbor_num, output_pts=None, normalize=False,
                indices_=None, return_indices=False):
        """
        Args:
            features: Batch of features.
            input_pts: Batch of points.
            neighbor_num: Size of neighborhood.
            output_pts: Number of output points.
            normalize: Normalization to unit sphere.
            indices_: neighbor indices.
            return_indices: Flag for returning the neighbor indices.
        """

        if indices_ is None:
            # Convolution with reduction if number of output points is given as int and shape of output points is
            # not equal to the shape of input points (is also applied if given number of output points is larger than
            # number of input points)
            if isinstance(output_pts, int) and input_pts.size(1) != output_pts:
                indices, next_pts_ = indices_conv_reduction(input_pts, output_pts, neighbor_num)

            # Convolution without reduction if no info about output points is given or given number of output points
            # equals number of input points
            elif (output_pts is None) or (isinstance(output_pts, int) and input_pts.size(1) == output_pts):
                indices, next_pts_ = indices_conv(input_pts, neighbor_num)

            # Convolution with up sampling or projection on given points
            else:
                indices, next_pts_ = indices_deconv(input_pts, output_pts, neighbor_num)

            if output_pts is None or isinstance(output_pts, int):
                output_pts = next_pts_

            if return_indices:
                indices_ = indices
        else:
            indices = indices_

        batch_size = features.size(0)
        n_pts = features.size(1)

        # Compute indices for indexing points (add batch offset to indices)
        add_indices = torch.arange(batch_size).type(indices.type()).to(input_pts.device) * n_pts
        indices = indices + add_indices.view(-1, 1, 1)

        # Get the features and point cooridnates associated with the indices (flatten batches and use
        # indices with offset
        features = features.view(-1, features.size(2))[indices]
        pts = input_pts.view(-1, input_pts.size(2))[indices]

        # Center each neighboorhood
        pts = pts - output_pts.unsqueeze(2)

        if normalize:
            # Normalize to unit ball
            maxi = torch.sqrt((pts.detach()**2).sum(3).max(2)[0])  # detach is a modificaiton
            maxi[maxi == 0] = 1
            pts = pts / maxi.view(maxi.size()+(1, 1,))

        # Compute the distances between kernel elements and points of centered neighborhoods
        dists = pts.view(pts.size()+(1,)) - self.centers
        dists = dists.view(dists.size(0), dists.size(1), dists.size(2), -1)

        # Learn the weighting function by the MLP
        dists = F.relu(self.l1(dists))
        dists = F.relu(self.l2(dists))
        dists = F.relu(self.l3(dists))

        # Compute features
        fs = features.size()
        features = features.transpose(2, 3)
        features = features.view(-1, features.size(2), features.size(3))
        dists = dists.view(-1, dists.size(2), dists.size(3))

        # Batch matrix multiplications which is essentially equation 2 from the paper
        # (Network weights * Features * Weighting function)
        features = torch.bmm(features, dists)
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
    def __init__(self, input_channels, output_channels, dimension=3):
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

        self.drop = nn.Dropout(0.5)

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
        xout = self.drop(xout)
        xout = self.fcout(xout)

        return xout


################################
# BIG SEGMENTATION NETWORK
################################


class SegBig(nn.Module):
    def __init__(self, input_channels, output_channels, trs=False, dimension=3, dropout=0):
        super(SegBig, self).__init__()

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

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, input_pts, return_features=False):
        """
        Args:
            x: Batch of features
            input_pts: Batch of points
            return_features: Flag for returning calculated features.
        """
        x0, _ = self.cv0(x, input_pts, 16)
        x0 = self.relu(apply_bn(x0, self.bn0))

        # Number of output points = 2048, Neighborhood of 16 points
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
        print(f"cv0: {time.time()-start} s")

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
