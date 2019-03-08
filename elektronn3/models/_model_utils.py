# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch and others (see individual docstrings for more author info)

"""Utilities for neural network model manipulation"""

from typing import Tuple, Union, List

import numpy as np
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


# TODO: Support passing example tensors directly instead of shapes
def model_summary(
        model: nn.Module,
        in_shapes: Union[Tuple[int, ...], List[Tuple[int, ...]]],
        device: Union[torch.device, str] = 'cpu'
) -> Tuple[dict, str]:
    """Get a summ_dict of what a model does

    Based on https://github.com/sksq96/pytorch-summ_dict/blob/b50f21/torchsummary/torchsummary.py
    by Shubham Chandel and others
    (see contributors listed in https://github.com/sksq96/pytorch-summary).

    but  with many changes such as
    - torch.device support
    - Cleaner code using PyTorch 1.0 API and requiring Python 3.6
    - Reporting of final output shape
    - Not printing results directly, but returning them as a string and a dict
    - No more special treatment of the batch dimension
    """

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summ_dict)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summ_dict[m_key] = {}  # OrderedDict()
            summ_dict[m_key]['input_shape'] = list(input[0].shape)
            if isinstance(output, (list, tuple)):
                summ_dict[m_key]['output_shape'] = [
                    list(o.shape) for o in output
                ]
            else:
                summ_dict[m_key]['output_shape'] = list(output.shape)

            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
                params += module.weight.numel()
                summ_dict[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'shape'):
                params += module.bias.numel()
            summ_dict[m_key]['nb_params'] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(in_shapes, tuple):
        in_shapes = [in_shapes]

    x = [torch.randn(in_shape, device=device) for in_shape in in_shapes]

    summ_dict = {}  # OrderedDict()
    hooks = []

    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summ_list = []  # For building the string representation

    summ_list.append('----------------------------------------------------------------------------')
    line_new = '{:>25}  {:>28} {:>20}'.format('Layer (type)', 'Output Shape', 'Param #')
    summ_list.append(line_new)
    summ_list.append('============================================================================')
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summ_dict:
        # input_shape, output_shape, trainable, nb_params
        line_new = '{:>28}  {:>28} {:>15}'.format(
            layer,
            str(summ_dict[layer]['output_shape']),
            '{0:,}'.format(summ_dict[layer]['nb_params']),
        )
        total_params += summ_dict[layer]['nb_params']
        total_output += np.prod(summ_dict[layer]['output_shape'])
        if 'trainable' in summ_dict[layer]:
            if summ_dict[layer]['trainable']:
                trainable_params += summ_dict[layer]['nb_params']
        summ_list.append(line_new)

    # assume 4 bytes/number (float on cuda).
    total_in_shape = abs(np.prod(in_shapes) * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_in_shape
    last_key = list(summ_dict.keys())[-1]
    output_shape = summ_dict[last_key]['output_shape']

    summ_dict['total_in_shape'] = total_in_shape
    summ_dict['total_output_size'] = total_output_size
    summ_dict['total_params_size'] = total_params_size
    summ_dict['total_size'] = total_size
    summ_dict['output_shape'] = output_shape

    summ_list.append('============================================================================')
    summ_list.append('Total params: {0:,}'.format(total_params))
    summ_list.append('Trainable params: {0:,}'.format(trainable_params))
    summ_list.append('Non-trainable params: {0:,}'.format(total_params - trainable_params))
    summ_list.append('----------------------------------------------------------------------------')
    summ_list.append('Input size (MB): %0.2f' % total_in_shape)
    summ_list.append('Forward/backward pass size (MB): %0.2f' % total_output_size)
    summ_list.append('Params size (MB): %0.2f' % total_params_size)
    summ_list.append('Estimated Total Size (MB): %0.2f' % total_size)
    summ_list.append('Output shape: {}'.format(output_shape))
    summ_list.append('----------------------------------------------------------------------------')

    summ_str = '\n'.join(summ_list)

    return summ_dict, summ_str
