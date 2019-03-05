# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

import torch
from torch import nn


# TODO: ScriptModule
# @torch._jit_internal.weak_module
# class L1BatchNorm(torch.jit.ScriptModule):
class L1BatchNorm(nn.Module):
    """L1-Norm-based Batch Normalization module.

    Use with caution. This code is not extensively tested.

    References:
    - https://arxiv.org/abs/1802.09769
    - https://arxiv.org/abs/1803.01814
    """
    __constants__ = ['l2factor', 'eps', 'momentum']

    def __init__(self, num_features: int, momentum: float = 0.9):
        super().__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, num_features))
        self.eps = 1e-5
        self.l2factor = (3.1416 / 2) ** 0.5

    # @torch._jit_internal.weak_script
    # @torch.jit.script_method
    def forward(self, x):
        ndim = x.dim()  # If this is known statically, this module can be a ScriptModule
        reduce_dims = (0, 2, 3, 4)[:ndim]
        b_sh = (1, x.shape[1], 1, 1, 1)[:ndim]  # Broadcastable shape
        if self.training:
            mean = x.mean(dim=reduce_dims, keepdim=True)
            x_minus_mean = x - mean
            absdiff = x_minus_mean.abs()
            l1mean = absdiff.mean(dim=reduce_dims, keepdim=True)
            l1scaled = l1mean * self.l2factor + self.eps
            mom = self.momentum
            self.running_mean.mul_(mom).add_(mean.flatten() * (1 - mom))
            self.running_var.mul_(mom).add_(l1scaled.flatten() * (1 - mom))
        else:
            mean = self.running_mean.view(b_sh)
            l1scaled = self.running_var.view(b_sh)
            x_minus_mean = x - mean
        gamma = self.gamma.view(b_sh)
        beta = self.beta.view(b_sh)
        return gamma * x_minus_mean / l1scaled + beta
