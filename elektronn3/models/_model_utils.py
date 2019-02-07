# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""Utilities for neural network model manipulation"""

from typing import Tuple

import torch
from torch import nn


def visualize_receptive_field(
        model: torch.nn.Module,
        input_shape: Tuple[int, ...] = (96, 96, 1),  # (H, W, C) order!
        interactive=False
) -> None:
    """Visualize analytical and effective receptive filelds of a network.

    Requires https://github.com/fornaxai/receptivefield to work
    (``pip install receptivefield``).

    Limitations:
    - Only works for 2D inputs (tensor layout (N, C, H, W)!
    - Doesn't work with ScriptModules (yet). As a workaround, you can
      construct a new model (``pymodel``) with the same code (but without
      compiling it) and set its ``state_dict`` to the ``state_dict`` of the
      compiled model (``scriptmodel``) as follows:
      >>> # Python-defined standard PyTorch model
      >>> pymodel: torch.nn.Module
      >>> # Equivalent TorchScript representation of pymodel
      >>> scriptmodel: torch.jit.ScriptModule
      >>> # Update pymodel state
      >>> pymodel.load_state_dict(scriptmodel.state_dict())
      Then you can safely pass ``pymodel`` to ``visualize_receptive_field``.
    - Requires that the ``model`` stores its output (or an intermediate
      layer output of interest) as ``self.feature_maps[0]`` before returning,
      (see bottom of :py:meth:`elektronn3.models.unet.UNet.forward`).

    Example::
    >>> from elektronn3.models._model_utils import visualize_receptive_field
    >>> from elektronn3.models.unet import UNet
    >>> model = UNet(
    ...     in_channels=1,
    ...     out_channels=2,
    ...     n_blocks=3,
    ...     activation='lin',
    ...     dim=2,
    ... )
    >>> # (Train model for a few iterations here)
    >>> input_shape = (96, 96, 1)
    >>> visualize_receptive_field(model, input_shape, interactive=True)

    input_shape = (96, 96, 1)
    visualize_receptive_field(model, input_shape, interactive=True)"""
    from receptivefield.pytorch import PytorchReceptiveField
    import matplotlib.pyplot as plt

    def model_fn():
        return model.eval()

    rf = PytorchReceptiveField(model_fn)
    rf_params = rf.compute(input_shape=input_shape)
    center = (input_shape[0] // 2, input_shape[1] // 2)

    fig, ax = plt.subplots()
    rf.plot_gradient_at(fm_id=0, point=center, axis=ax)
    if interactive:
        plt.show()
    return fig


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


def num_params(model: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
