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
  pairs instead of just inp to inp. Most transforms don't change the target, though.
2. They exclusively operate on numpy.ndarray data instead of PIL or torch.Tensor data.
"""

from typing import Sequence, Tuple, Optional, Dict, Any, Union

import numpy as np
import skimage.exposure

from elektronn3.data import random_blurring


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
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        normalized = np.empty_like(inp)
        if not inp.shape[0] == self.mean.shape[0] == self.std.shape[0]:
            raise ValueError('mean and std must have the same length as the C '
                             'axis (number of channels) of the input.')
        for c in range(inp.shape[0]):
            normalized[c] = (inp[c] - self.mean[c]) / self.std[c]
        return normalized, target


# TODO: Per-channel gamma_std, infer channels from gamma_std shape? Same with AdditiveGaussianNoise
# TODO: Sample from a more suitable distribution. Due to clipping to gamma_min,
#       there is currently a strong bias towards this value.
class RandomGammaCorrection:
    """Applies random gamma correction to the input.

    Args:
        gamma_std: standard deviation of the gamma value.
        channels: If ``channels`` is ``None``, the noise is applied to
            all channels of the input tensor.
            If ``channels`` is a ``Sequence[int]``, noise is only applied
            to the specified channels.
        prob: probability (between 0 and 1) with which to perform this
            augmentation. The input is returned unmodified with a probability
            of ``1 - prob``.
        rng: Optional random state for deterministic execution
    """
    def __init__(
            self,
            gamma_std: float = 0.5,
            channels: Optional[Sequence[int]] = None,
            prob: float = 1.0,
            rng: Optional[np.random.RandomState] = None
    ):
        self.gamma_std = gamma_std
        self.channels = channels
        self.prob = prob
        self.rng = np.random.RandomState() if rng is None else rng
        self.gamma_min = 0.25

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.rand() > self.prob:
            return inp, target
        channels = range(inp.shape[0]) if self.channels is None else self.channels
        gcorr = np.empty_like(inp)
        for c in channels:
            gamma = self.rng.normal(1.0, self.gamma_std)
            gamma = max(self.gamma_min, gamma)  # Prevent gamma <= 0 (0 causes zero division)
            # adjust_gamma() requires inputs in the (0, 1) range, so the
            #  image intensity values are rescaled to (0, 1) and after
            #  applying gamma correction they are rescaled back to the original
            #  intensity range.
            orig_intensity_range = inp[c].min(), inp[c].max()
            rescaled = skimage.exposure.rescale_intensity(inp[c], out_range=(0, 1))
            gcorr01c = skimage.exposure.adjust_gamma(rescaled, gamma)  # still in (0, 1) range
            # Rescale to original (normalized) intensity range
            gcorr[c] = skimage.exposure.rescale_intensity(gcorr01c, out_range=orig_intensity_range)

        return gcorr, target


# TODO: [Random]GaussianBlur


class AdditiveGaussianNoise:
    """Adds random gaussian noise to the input.

    Args:
        sigma: Sigma parameter of the gaussian distribution to draw from
        channels: If ``channels`` is ``None``, the noise is applied to
            all channels of the input tensor.
            If ``channels`` is a ``Sequence[int]``, noise is only applied
            to the specified channels.
        prob: probability (between 0 and 1) with which to perform this
            augmentation. The input is returned unmodified with a probability
            of ``1 - prob``.
        rng: Optional random state for deterministic execution
    """
    def __init__(
            self,
            sigma: float = 0.1,
            channels: Optional[Sequence[int]] = None,
            prob: float = 1.0,
            rng: Optional[np.random.RandomState] = None
    ):
        self.sigma = sigma
        self.channels = channels
        self.prob = prob
        self.rng = np.random.RandomState() if rng is None else rng

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.rand() > self.prob:
            return inp, target
        noise = np.empty_like(inp)
        channels = range(inp.shape[0]) if self.channels is None else self.channels
        for c in channels:
            noise[c] = self.rng.normal(0, self.sigma, inp[c].shape)
        noisy_inp = inp + noise
        return noisy_inp, target


class RandomBlurring:  # Warning: This operates in-place!

    _default_scheduler = random_blurring.ScalarScheduler(
        value=0.1,
        max_value=0.5,
        growth_type="lin",
        interval=500000,
        steps_per_report=1000
    )
    _default_config = {
        "probability": 0.5,
        "threshold": _default_scheduler,
        "lower_lim_region_size": [3, 6, 6],
        "upper_lim_region_size": [8, 16, 16],
        "verbose": False,
    }

    def __init__(
            self,
            config: Dict[str, Any],
            patch_shape: Optional[Sequence[int]] = None
    ):
        self.config = {**self._default_config, **config}
        # TODO: support random state
        if patch_shape is not None:
            random_blurring.check_random_data_blurring_config(patch_shape, **config)

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
    ) -> Tuple[np.ndarray, np.ndarray]:
        # In-place, overwrites inp!
        assert inp.ndim == 4, 'Currently only (C, D, H, W) inputs are supported.'
        random_blurring.apply_random_blurring(inp_sample=inp, **self.config)
        return inp, target


class RandomCrop:
    def __init__(self, size: Sequence[int]):
        # TODO: support random state
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


class SqueezeTarget:
    """Squeeze a specified dimension in target tensors.

    (This is just needed as a workaround for the example neuro_data_cdhw data
    set, because its targets have a superfluous first dimension.)"""
    def __init__(self, dim, inplace=True):
        self.dim = dim

    def __call__(
            self,
            inp: np.ndarray,  # Returned without modifications
            target: np.ndarray,
    ):
        return inp, target.squeeze(axis=self.dim)


# TODO: Handle target striding and offsets via transforms?

# TODO: Meta-transform that performs a wrapped transform with a certain
#       probability, replacing prob params?

# TODO: Functional API (transforms.functional).
#       The current object-oriented interface should be rewritten as a wrapper
#       for the functional API (see implementation in torchvision).

# TODO: Extract a non-random version from each random transform.
#       E.g. RandomGammaCorrection should wrap GammaCorrection, which takes
#       the actual gamma value as an argument instead of a parametrization
#       of the random distribution from which gamma is sampled.
