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

from typing import Sequence, Tuple, Optional, Dict, Any, Callable

import numpy as np
import skimage.exposure

import scipy
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from elektronn3.data.transforms import random_blurring
from elektronn3.data.transforms.random import Normal, HalfNormal

Transform = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


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

    def __init__(self, transforms: Sequence[Transform]):
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


class Lambda:
    """Wraps a function of the form f(x, y) = (x', y') into a transform.

    Args:
        func: A function that takes two arrays and returns a
            tuple of two arrays.

    Example:
        >>> # Multiplies inputs (x) by 255, leaves target (y) unchanged
        >>> t = Lambda(lambda x, y: (x * 255, y))

        >>> # You can also pass regular Python functions to Lambda
        >>> def flatten(x, y):
        >>>     return x.reshape(-1), y.reshape(-1)
        >>> t = Lambda(flatten)
    """
    def __init__(
            self,
            func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ):
        self.func = func

    def __call__(
            self,
            inp: np.ndarray,
            target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.func(inp, target)


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
            gamma_min: float = 0.25,  # Prevent gamma <= 0 (0 causes zero division)
            channels: Optional[Sequence[int]] = None,
            prob: float = 1.0,
            rng: Optional[np.random.RandomState] = None
    ):
        self.channels = channels
        self.prob = prob
        self.rng = np.random.RandomState() if rng is None else rng
        self.gamma_generator = Normal(
            mean=1.0, sigma=gamma_std, bounds=(gamma_min, np.inf), rng=rng
        )

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
            gamma = self.gamma_generator()
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


# TODO: The current necessity of intensity rescaling for normalized
#       (zero mean, unit std) inputs is really uncool. Can we circumvent this?
class RandomGrayAugment:
    r"""Performs gray value augmentations in the same way as ELEKTRONN2's
    ``greyAugment()`` function, but with additional support for inputs
    outside of the :math:`[0, 1]` intensity value range. Targets are not
    modified.

    This augmentation method randomly selects the values ``alpha``, ``beta``
    and ``gamma`` within *sensible* ranges and subsequently performs:

    - Temporarily rescaling image intensities to the :math:`[0, 1]` range
      (necessary for gamma correction).
    - Linear intensity scaling (contrast) by multiplying the input with
      :math:`\alpha = 1 + 0.3r`, where :math:`r \in \mathcal{U}[-0.5, 0.5]`.
      Value range: :math:`\alpha \in [0.85, 1.15]`.
    - Adding a constant value :math:`\beta = 0.3r`, where
      :math:`r \in \mathcal{U}[-0.5, 0.5]`.
      Value range: :math:`\beta \in [-0.15, 0.15]`.
    - Gamma correction with :math:`\gamma = 2^r`, where
      :math:`r \in \mathcal{U}[-1, 1]`.
    - Clipping all image intensity values to the range :math:`[0, 1]`.
    - Re-rescaling intensities back to the original input value range.
    """
    def __init__(
            self,
            channels: Optional[Sequence[int]] = None,
            prob: float = 1.0,
            rng: Optional[np.random.RandomState] = None
    ):
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

        channels = range(inp.shape[0]) if self.channels is None else self.channels
        nc = len(channels)
        aug = inp.copy()  # Copy so we don't overwrite inp
        # The calculations below have to be performed on inputs that have a
        #  value range of (0, 1), so they have to be rescaled.
        #  The augmented image will be re-rescaled to the original input value
        #  range at the end of the function.
        orig_intensity_ranges = [(inp[c].min(), inp[c].max()) for c in channels]
        for c in channels:  # TODO: Can we vectorize this?
            aug[c] = skimage.exposure.rescale_intensity(inp[c], out_range=(0, 1))

        alpha = 1 + (self.rng.rand(nc) - 0.5) * 0.3  # ~ contrast
        beta = (self.rng.rand(nc) - 0.5) * 0.3  # Mediates whether values are clipped for shadows or lights
        gamma = 2.0 ** (self.rng.rand(nc) * 2 - 1)  # Sample from [0.5, 2]

        aug[channels] = aug[channels] * alpha[:, None, None] + beta[:, None, None]
        aug[channels] = np.clip(aug[channels], 0, 1)
        aug[channels] = aug[channels] ** gamma[:, None, None]

        for c in channels:  # Rescale to original (normalized) intensity range
            aug[c] = skimage.exposure.rescale_intensity(aug[c], out_range=orig_intensity_ranges[c])

        return aug, target


# TODO: [Random]GaussianBlur
class RandomGaussianBlur:
    """Adds random gaussian blur to the input.
        Args:
            sigma:Sigma parameter of the HalfNormal distribution to draw from
            prob: probability (between 0 and 1) with which to perform this
                augmentation. The input is returned unmodified with a probability
                of ``1 - prob``.
            rng: Optional random state for deterministic execution
            aniso_factor: a tuple or an array to apply the anisotropy, must
                match the dimension of the input

        """

    def __init__(
            self,
            sigma: float = 1,
            channels: Optional[Sequence[int]] = None,
            prob: float = 1.0,
            rng: Optional[np.random.RandomState] = None,
            aniso_factor: Optional = None,
    ):
        self.channels = channels
        self.prob = prob
        self.rng = np.random.RandomState() if rng is None else rng
        self.gaussian_std= HalfNormal(sigma=sigma, rng=rng)
        self.aniso_factor = (1,1,1) if aniso_factor is None or aniso_factor == 1.0 else aniso_factor

        if aniso_factor  is not None:
            if isinstance(aniso_factor, (list, tuple)):
                self.aniso_factor = np.array(aniso_factor)
            elif isinstance(aniso_factor, np.ndarray):
                self.aniso_factor = aniso_factor
            else:
                raise ValueError("aniso_factor not understood")

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.rand() > self.prob:
            return inp, target

        channels = range(inp.shape[0]) if self.channels is None else self.channels
        blurred_inp = np.empty_like(inp)
        for c in channels:
            shape = inp[c].shape
            if inp[c].ndim ==2:
                self.aniso_factor = (self.aniso_factor[0], self.aniso_factor[1])
            sigma = self.gaussian_std(shape=inp[c].ndim)
            aniso_sigma = np.divide(sigma, self.aniso_factor)
            blurred_inp[c] = gaussian_filter(inp[c], sigma=aniso_sigma)

        return blurred_inp, target



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
        self.channels = channels
        self.prob = prob
        self.rng = np.random.RandomState() if rng is None else rng
        self.noise_generator = Normal(mean=0, sigma=sigma, rng=rng)

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
            noise[c] = self.noise_generator(shape=inp[c].shape)
        noisy_inp = inp + noise
        return noisy_inp, target



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


class ElasticTransform:
    """
    Based on https://gist.github.com/fmder/e28813c1e8721830ff9c


    Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

        Args:
            sigma: Sigma parameter of the gaussian distribution to draw from
            alpha: Strength of the elastic transform

            rng: Optional random state for deterministic execution

        The input image should be of dimensions (C, H, W) or (C, D, H, W).
        C must be included.

    """

    def __init__(
            self,
            sigma: float = 4,
            alpha: float = 10,
            channels: Optional[Sequence[int]] = None,
            prob: float = 0.25,
            rng: Optional[np.random.RandomState] = None,

    ):
        self.sigma = sigma
        self.alpha = alpha
        self.channels = channels
        self.prob = prob
        self.rng = np.random.RandomState() if rng is None else rng

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications

    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.rand() > self.prob:
            return inp, target
        channels = range(inp.shape[0]) if self.channels is None else self.channels
        deformed_img = np.empty_like(inp)
        for c in channels:
            shape = inp[c].shape
            if inp[c].ndim ==3 :
                dx = gaussian_filter((self.rng.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
                dy = gaussian_filter((self.rng.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
                dz = gaussian_filter((self.rng.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
                x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
                indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

            elif inp[c].ndim == 2 :
                dx = gaussian_filter((self.rng.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
                dy = gaussian_filter((self.rng.rand(*shape)* 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
                x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
                indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

            else:
                raise ValueError("Image dimension not understood!")

            deformed_img[c] = map_coordinates(inp[c], indices, order=1).reshape(shape)

        return deformed_img, target


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
