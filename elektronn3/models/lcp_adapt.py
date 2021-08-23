# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Philipp Schubert, Jonathan Klimesch

"""
This is an implementation based on LightConvPoint from Alexandre Boulch et al. (https://github.com/valeoai/LightConvPoint,
https://github.com/aboulch/ConvPoint).
"""


import torch
import torch.nn as nn
import lightconvpoint.nn as lcp_nn


class ConvAdaptSeg(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ConvNet,
                 Search,
                 kernel_num,
                 architecture,
                 activation,
                 norm,
                 track_running_stats=False,
                 ):
        """Adaptable ConvPoint segmentation network.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output  channels.
            ConvNet: The convolutional class to be used in the network.
            Search: The search class to be used in the network.
            kernel_num: Initial number of kernels.
            architecture: Short description of the architecture which should get used. It has the following format:
                [{in_channels: ic, out_channels: oc, kernel_size: ks, neighbor_num: nn, npoints: np}, ...]
                in_channels = -1 if first layer
                npoints = -1 if no reduction / = 'd' if deconvolution
            activation: Activation function to use throughout the network.
            normalization: Type of normalization, currently either 'bn' or 'gn'
            track_running_stats: Flag for BatchNorm only.
        """

        super().__init__()

        if architecture is None:
            # Standard architecture if none is given
            architecture = [dict(ic=-1, oc=1, ks=16, nn=16, np=-1),
                            dict(ic=1, oc=1, ks=16, nn=16, np=2048),
                            dict(ic=1, oc=1, ks=16, nn=16, np=1024),
                            dict(ic=1, oc=1, ks=16, nn=16, np=256),
                            dict(ic=1, oc=2, ks=16, nn=16, np=64),
                            dict(ic=2, oc=2, ks=16, nn=16, np=16),
                            dict(ic=2, oc=2, ks=16, nn=16, np=8),
                            dict(ic=2, oc=2, ks=16, nn=4, np='d'),
                            dict(ic=4, oc=2, ks=16, nn=4, np='d'),
                            dict(ic=4, oc=1, ks=16, nn=4, np='d'),
                            dict(ic=2, oc=1, ks=16, nn=8, np='d'),
                            dict(ic=2, oc=1, ks=16, nn=8, np='d'),
                            dict(ic=2, oc=1, ks=16, nn=8, np='d')]

        self.architecture = architecture
        self.layers = nn.ModuleList()

        for layer in self.architecture:
            # set Normalization
            if norm.lower() == 'bn':
                normalization = nn.BatchNorm1d(layer['oc'] * kernel_num, track_running_stats=track_running_stats)
            elif norm.lower() == 'gn':
                normalization = nn.GroupNorm(layer['oc'] * kernel_num // 2, layer['oc'] * kernel_num)
            else:
                raise NotImplementedError

            # set search function
            if layer['np'] == -1 or layer['np'] == 'd':
                search = Search(K=layer['nn'])
            else:
                search = Search(K=layer['nn'], npoints=layer['np'])

            # set layers
            if layer['ic'] == -1:
                self.layers.append(lcp_nn.Conv(ConvNet(in_channels, layer['oc'] * kernel_num, layer['ks']), search))
            else:
                cv = lcp_nn.Conv(
                    ConvNet(layer['ic'] * kernel_num, layer['oc'] * kernel_num, layer['ks']),
                    search,
                    activation=activation(),
                    normalization=normalization,
                )
                self.layers.append(cv)

        self.fcout = nn.Conv1d(2 * kernel_num, out_channels, 1)
        self.drop = nn.Dropout(0)

    def forward(self, x, input_pts, support_points=None, indices=None, return_features=False):

        if support_points is None:
            support_points = [None for _ in range(len(self.architecture))]
        if indices is None:
            indices = [None for _ in range(len(self.architecture))]

        # cache tensors during downpath for concat during uppath
        x_arr = []
        xd_num = 0
        pts_arr = [input_pts]
        ids_arr = []
        pts_return = []

        for ix, layer in enumerate(self.architecture):
            if layer['ic'] == -1:
                # First layer
                x, pts, ids = self.layers[ix](x, input_pts, input_pts, indices=indices[ix])
                x_arr.append(x)
                pts_arr.append(pts)
                ids_arr.append(ids)
            elif layer['np'] != 'd':
                # Layers in downpath
                x, pts, ids = self.layers[ix](x_arr[-1], pts_arr[-1], support_points[ix - 1], indices=indices[ix])
                x_arr.append(x)
                pts_arr.append(pts)
                ids_arr.append(ids)
            else:
                # Layers in uppath
                xd, _, ids = self.layers[ix](x_arr[-1], pts_arr[-1], pts_arr[-2], indices=indices[ix])
                xd = torch.cat([xd, x_arr[-2 - xd_num * 2]], dim=1) if xd is not None else None
                xd_num += 1
                pts_return.append(pts_arr.pop())
                x_arr.append(xd)
                ids_arr.append(ids)

        if x_arr[-1] is not None:
            x0d = x_arr[-1]
            xout = self.drop(x0d)
            xout = self.fcout(xout)

            if return_features:
                return xout, x0d
            else:
                return xout
        else:
            return None, ids_arr, pts_return.reverse()
