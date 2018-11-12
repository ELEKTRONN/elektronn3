"""Utilities for large-scale neural network inference on
n-dimensional image data.

This module does not depend on any other part of elektronn3, so it can be used
by other projects without problem.

Important note: Currently this module assumes that the inference model
outputs tensors of the same spatial shape as its inputs.
If your model output is strided w.r.t. the inputs (e.g. due to pooling or
strided convolution without subsequent upsampling) or part of the borders
is cut off (e.g. due to "valid" convolutions without shape-preserving padding),
you will need to alter/wrap it to take this into account.

For example, if your model reduces spatial resolution by factor 8,
you can wrap it as follows to make it work with this module:

>>> model = torch.nn.Sequential(model, torch.nn.Upsample(scale_factor=8))

This limitation might be lifted some day, but it's currently low priority.
"""

import torch

from .inference import Predictor
