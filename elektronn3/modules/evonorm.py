# Adapted from https://github.com/digantamisra98/EvoNorm/blob/a4946004/evonorm2d.py
# Supports 2D and 3D via the dim argument.

import torch
import torch.nn as nn


def instance_std(x, eps=1e-5):
    if x.ndim == 5:
        dims = (2, 3, 4)
    else:
        dims = (2, 3)
    var = torch.var(x, dim=dims, keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)


def group_std(x, groups=32, eps=1e-5):
    sh = x.shape
    if x.ndim == 5:
        dims = (2, 3, 4, 5)
        n, c, d, h, w = sh
        x = torch.reshape(x, (n, groups, c // groups, d, h, w))
    else:
        dims = (2, 3, 4)
        n, c, h, w = sh
        x = torch.reshape(x, (n, groups, c // groups, h, w))
    var = torch.var(x, dim=dims, keepdim=True).expand_as(x)
    return torch.reshape(torch.sqrt(var + eps), sh)


class EvoNorm(nn.Module):

    def __init__(self, input, non_linear=True, version='S0', affine=True, momentum=0.9, eps=1e-5,
                 groups=32, training=True, dim=3):
        super().__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.silu = nn.SiLU()
        self.groups = groups
        self.eps = eps
        if self.version not in ['B0', 'S0']:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        self.affine = affine
        self.dim = dim
        if self.dim == 3:
            rs_shape = (1, self.insize, 1, 1, 1)  # 5D
        elif self.dim == 2:
            rs_shape = (1, self.insize, 1, 1)  # 4D
        else:
            raise ValueError('Invalid dim. 2 or 3 expected.')

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(rs_shape))
            self.beta = nn.Parameter(torch.zeros(rs_shape))
            if self.non_linear:
                self.v = nn.Parameter(torch.ones(rs_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
            self.register_buffer('v', None)
        self.register_buffer('running_var', torch.ones(rs_shape))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.ndim != self.dim + 2:
            raise ValueError(f'Expected {self.dim + 2}D input but got {x.ndim} input.')

    def forward(self, x):
        self._check_input_dim(x)
        if self.version == 'S0':
            if self.non_linear:
                num = self.silu(x)
                return num / group_std(x, groups=self.groups, eps=self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == 'B0':
            if self.training:
                if x.ndim == 5:
                    dims = (0, 2, 3, 4)
                else:
                    dims = (0, 2, 3)
                var = torch.var(x, dim=dims, unbiased=False, keepdim=True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                den = torch.max((var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
