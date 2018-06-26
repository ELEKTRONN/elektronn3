# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""Utilities for neural network model manipulation"""

from typing import Tuple

import torch
from torch import nn


def find_first(model: nn.Module, module_type: type = nn.Conv2d) -> Tuple[str, nn.Module]:
    """Return the first submodule of an nn.Module with a certain type."""
    for name, mod in model.named_modules():
        if isinstance(mod, module_type):
            return name, mod
    # Loop finished, nothing found...
    raise LookupError(f'Module does\'nt have any layers of type {module_type}.')


def find_first_conv(model: nn.Module, in_channels=3) -> Tuple[str, nn.Module]:
    """Return the first Conv2d submodule with a certain in_channels value."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == in_channels:
            return name, mod
    # Loop finished, nothing found...
    raise LookupError(f'Module does\'nt have any Conv2d layers with in_channels={in_channels}.')


# TODO: Layer initialization?
# TODO: Is it possible to re-use original conv1 weights in some way?
def change_conv1_input_channels(
        model: nn.Module,
        old_in_channels: int = 3,
        new_in_channels: int = 1
) -> None:
    """Change input channels of a convnet model (in-place).

    This can be used to turn RGB models (``in_channels=3``) into single-channel
    models (``in_channels=1``) automatically by replacing the first Conv layer,
    while preserving pretrained weights in all other layers.."""
    name, conv1 = find_first_conv(model, in_channels=old_in_channels)
    model.__dict__[name] = nn.Conv2d(
        new_in_channels, conv1.out_channels, conv1.kernel_size,
        conv1.stride, conv1.padding, conv1.dilation, conv1.groups, conv1.bias
    )
