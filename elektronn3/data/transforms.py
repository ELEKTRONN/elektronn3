# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""
Transformations (data augmentation, normalization etc.) for semantic segmantation.

Important note: The transformations here have a similar interface to
 torchvsion.transforms, but there are two key differences:
 1. They all map (inp, target) pairs to (transformed_inp, transformed_target)
    pairs instead of just inp to inp.
    Most transforms don't change the target, though.
 2. They exclusively operate on numpy.ndarray data instead of PIL or
    torch.Tensor data.
"""

from typing import Sequence, Tuple, Optional, Dict, Any, Union

import numpy as np

from elektronn3.data.random_blurring import apply_random_blurring, check_random_data_blurring_config


# Transformation = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


class Identity:
    def __call__(self, inp, target):
        return inp, target


class Compose:
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> Compose([
        >>>     Normalize(mean=(155.291411,), std=(41.812504,)),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inp, target):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize:
    """Normalizes inputs with supplied per-channel means and stds."""
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
    ) -> Tuple[np.ndarray, np.ndarray]:
        normalized = np.empty_like(inp)
        if not inp.shape[0] == self.mean.shape[0] == self.std.shape[0]:
            raise ValueError('mean and std must have the same length as the C '
                             'axis (number of channels) of the input.')
        for c in range(inp.shape[0]):
            normalized[c] = (inp[c] - self.mean[c]) / self.std[c]
        return normalized, target


class RandomBlurring:  # Warning: This operates in-place!
    def __init__(
            self,
            config: Dict[str, Any],
            patch_shape: Optional[Sequence[int]] = None
    ):
        self.config = config
        if patch_shape is not None:
            check_random_data_blurring_config(patch_shape, **config)

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
    ) -> Tuple[np.ndarray, np.ndarray]:
        # In-place, overwrites inp!
        assert inp.ndim == 4, 'Currently only (C, D, H, W) inputs are supported.'
        apply_random_blurring(inp_sample=inp, **self.config)
        return inp, target


class RandomCrop:
    def __init__(self, size: Sequence[int]):
        self.size = np.array(size)

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
    ) -> Tuple[np.ndarray, np.ndarray]:
        ndim_spatial = len(self.size)  # Number of spatial axes E.g. 3 for (C,D,H.W)
        img_shape = inp.shape[-ndim_spatial:]
        # Number of nonspatial axes (like the C axis). Usually this is one
        ndim_nonspatial = inp.ndim - ndim_spatial
        # Calculate the "lower" corner coordinate of the slice
        coords_lo = np.array([
            np.random.randint(0, img_shape[i] - self.size[i] + 1)
            for i in range(ndim_spatial)
        ])
        coords_hi = coords_lo + self.size  # Upper ("high") corner coordinate.
        # Calculate necessary slice indices for reading the file
        nonspatial_slice = [  # Slicing all available content in these dims.
            slice(0, inp.shape[i]) for i in range(ndim_nonspatial)
        ]
        spatial_slice = [  # Slice only the content within the coordinate bounds
            slice(coords_lo[i], coords_hi[i]) for i in range(ndim_spatial)
        ]
        full_slice = nonspatial_slice + spatial_slice
        inp_cropped = inp[full_slice]
        if target is None:
            return inp_cropped, target

        if target.ndim == inp.ndim - 1:  # inp: (C, [D,], H, W), target: ([D,], H, W)
            full_slice = full_slice[1:]  # Remove C axis from slice because target doesn't have it
        target_cropped = target[full_slice]
        return inp_cropped, target_cropped
