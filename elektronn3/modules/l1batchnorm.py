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
            meandiff = x - mean
            absdiff = meandiff.abs()
            l1mean = absdiff.mean(dim=reduce_dims, keepdim=True)
            l1scaled = l1mean * self.l2factor + self.eps
            with torch.no_grad():  # Update running stats
                mom = self.momentum
                self.running_mean.mul_(mom).add_(mean.flatten() * (1 - mom))
                self.running_var.mul_(mom).add_(l1scaled.flatten() * (1 - mom))
        else:
            mean = self.running_mean.view(b_sh)
            l1scaled = self.running_var.view(b_sh)
            meandiff = x - mean
        gamma = self.gamma.view(b_sh)
        beta = self.beta.view(b_sh)
        return gamma * meandiff / l1scaled + beta


# @torch._jit_internal.weak_script
def l1_group_norm(x, num_groups, weight, bias, eps):
    l2factor = 1.2533  # == (pi / 2) ** 0.5
    ndim = x.dim()
    sh = x.shape
    g = num_groups
    n, c = sh[:2]
    # grouped_sh = (n, g, c // g, d, h, w)
    grouped_sh = (n, g, c // g, *sh[2:])  # Split C dimension into groups
    grouped = x.view(grouped_sh)
    reduce_dims = (2, 3, 4, 5)[:ndim - 1]  # Reduce over grouped channels and spatial dimensions
    mean = grouped.mean(dim=reduce_dims, keepdim=True)
    meandiff = grouped - mean
    absdiff = meandiff.abs()
    l1mean = absdiff.mean(dim=reduce_dims, keepdim=True)
    l1scaled = l1mean * l2factor + eps
    normalized = meandiff / l1scaled
    normalized = normalized.view(sh)
    broadcast_sh = (1, c, 1, 1, 1)[:ndim]  # Shape broadcastable over all dims of x
    weight = weight.view(broadcast_sh)
    bias = bias.view(broadcast_sh)
    return weight * normalized + bias


# @torch._jit_internal.weak_module
class L1GroupNorm(nn.GroupNorm):
    r"""Applies L1 Group Normalization over a mini-batch of inputs.

    This works in the same way as `torch.nn.GroupNorm`, but uses the
    scaled L1 norm instead of the L2 norm for better numerical stability,
    performance and half precision support.
    L1 *batch* normalization was proposed in

    - https://arxiv.org/abs/1802.09769
    - https://arxiv.org/abs/1803.01814

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine', 'weight', 'bias']

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__(num_groups, num_channels, eps, affine)
        print('Warning: L1 Group norm is experimental and may have issues.')

    @torch._jit_internal.weak_script_method
    def forward(self, input):
        return l1_group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
