"""
Normalization layer recalibration tools

Based on https://github.com/mdraw/contrib/blob/bdf4da5/torchcontrib/optim/swa.py
"""

import copy

import torch


class NoApplicableLayersException(Exception):
    pass


def recalibrate_bn(loader, model, device=None):
    r"""Returns a model with running_mean, running_var buffers of normalization layers recalibrated.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.

    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.

        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.

        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.
    """
    calmodel = copy.deepcopy(model)  # Copy so the original model is not changed
    if device is None:  # Figure out device by looking where one of the parameters lies
        device = next(iter(calmodel.parameters())).device
    if not _check_bn(calmodel):
        raise NoApplicableLayersException('Model does not have any batchnorm layers.')
    calmodel.train()
    calmodel.apply(_set_bn_cma)
    calmodel.apply(_reset_bn)
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            inp = batch[0]
        elif isinstance(batch, dict):
            inp = batch['inp']
        inp = inp.to(device)
        calmodel(inp)
    calmodel.eval()
    return calmodel


def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _set_bn_cma(module):
    """Configure all BN layers to use cumulative moving average stats"""
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = None
