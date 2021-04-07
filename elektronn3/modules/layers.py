# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""Neural network layers"""

from typing import Optional, Tuple

import torch
from torch import nn


class GatherExcite(nn.Module):
    """Gather-Excite module (https://arxiv.org/abs/1810.12348),

    a generalization of the Squeeze-and-Excitation module
    (https://arxiv.org/abs/1709.01507).

    Args:
        channels: Number of input channels (= number of output channels)
        extent: extent factor that determines how much the gather operator
            output is smaller than its input. The special value ``extent=0``
            activates global gathering (so the gathered information has no
            spatial extent).
        param_gather: If ``True``, the gather operator is parametrized
            according to https://arxiv.org/abs/1810.12348.
        param_excite: If ``True``, the excitation operator is parametrized
            according to https://arxiv.org/abs/1810.12348 (also equivalent to
            the original excitation operator proposed in
            https://arxiv.org/abs/1709.01507).
        reduction:  Channel reduction rate of the parametrized excitation
            operator.
        spatial_shape: Spatial shape of the module input. This needs to be
            specified if ``param_gather=0 and extent=0`` (parametrized global
            gathering).
    """
    def __init__(
            self,
            channels: int,
            extent: int = 0,
            param_gather: bool = False,
            param_excite: bool = True,
            reduction: int = 16,
            spatial_shape: Optional[Tuple[int, ...]] = None
    ):
        super().__init__()
        if extent == 1:
            raise NotImplementedError('extent == 1 doesn\'t make sense.')
        if param_gather:
            if extent == 0:  # Global parametrized gather operator
                if spatial_shape is None:
                    raise ValueError(
                        'With param_gather=True, extent=0, you will need to specify spatial_shape.')
                self.gather = nn.Sequential(
                    nn.Conv3d(channels, channels, spatial_shape),
                    nn.BatchNorm3d(channels),
                    nn.ReLU()
                )
            else:
                # This will make the model much larger with growing extent!
                # TODO: This is ugly and I'm not sure if it should even be supported
                assert extent in [2, 4, 8, 16]
                num_convs = int(torch.log2(torch.tensor(extent, dtype=torch.float32)))
                self.gather = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv3d(channels, channels, 3, stride=2, padding=1),
                        nn.BatchNorm3d(channels),
                        nn.ReLU()
                    ) for _ in range(num_convs)
                ])
        else:
            if extent == 0:
                self.gather = nn.AdaptiveAvgPool3d(1)  # Global average pooling
            else:
                self.gather = nn.AvgPool3d(extent)
        if param_excite:
            self.excite = nn.Sequential(
                nn.Conv3d(channels, channels // reduction, 1),
                nn.ReLU(),
                nn.Conv3d(channels // reduction, channels, 1)
            )
        else:
            self.excite = nn.Identity()

        if extent == 0:
            self.interpolate = nn.Identity()  # Use broadcasting instead of interpolation
        else:
            self.interpolate = torch.nn.functional.interpolate

    def forward(self, x):
        y = self.gather(x)
        y = self.excite(y)
        y = torch.sigmoid(self.interpolate(y, x.shape[2:]))
        return x * y
