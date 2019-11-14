# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""
Transformations (data augmentation, normalization etc.) for semantic segmantation.

Important note: The transformations here have a similar interface to
torchvision.transforms, but there are two key differences:

1. They all map (inp, target) pairs to (transformed_inp, transformed_target)
  pairs instead of just inp to transformed_inp. Most transforms don't change the target, though.
2. They exclusively operate on numpy.ndarray data instead of PIL or torch.Tensor data.
"""

from typing import Sequence, Tuple, Optional, Dict, Any, Callable, Union

import numpy as np
import skimage.exposure
import skimage.transform

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from elektronn3.data.transforms import random_blurring
from elektronn3.data.transforms.random import Normal, HalfNormal, RandInt

Transform = Callable[
    [np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, Optional[np.ndarray]]
]


class _DropSample(Exception):
    """Sample will be dropped and won't be fed into a DataLoader"""
    pass


class Identity:
    def __call__(self, inp, target=None):
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


class RandomSlicewiseTransform:
    """Wraps any 2D transform and applies it per 2D slice independently in a 3D
    input-target pair.

    Works with any (..., D, H, W) memory layout given that if the ``target``
    is not ``None``, the last three dimensions of target and input tensors
    match.

    Args:
        transform: transform that works on 2D slices
        prob: Probability with which each slice is chosen to transformed by the
            specified ``transform``.

    Example::

        Here we replace each slice by zeros with p=0.1. This has an effect
        similar to the "missing section" augmentation described in
        https://arxiv.org/abs/1706.00120.

        >>> def zero_out(inp, target): return inp * 0, target
        >>> t = RandomSlicewiseTransform(zero_out, prob=0.1)
    """
    def __init__(
            self,
            transform: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
            prob: float = 0.1,
            inplace: bool = True
    ):
        self.transform = transform
        self.prob = prob
        assert inplace, 'Only inplace operation is supported currently'

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert inp.ndim == 4, 'Input must be of shape (C, D, H, W)'
        if target is not None:
            assert inp.shape[-3:] == target.shape[-3:], 'Spatial shape of target must match input'
        D = inp.shape[-3]  # Extent in z dimension
        for z in range(D):
            # TODO: rand can be vectorized
            if np.random.rand() < self.prob:  # Only apply to this z slice with p=prob
                if target is None:
                    # For support of (..., D, H, W) memory layout:
                    # Indexing with [..., z, :, :] is necessary to stay agnostic
                    # to the number of pre-depth dimensions (we only know for
                    # certain that depth is the third from last dimension).
                    inp[..., z, :, :], _ = self.transform(inp[..., z, :, :], None)
                else:
                    inp[..., z, :, :], target[..., z, :, :] = self.transform(inp[..., z, :, :], target[..., z, :, :])
        return inp, target


class DropIfTooMuchBG:
    """Filter transform that skips a sample if the background class is too
    dominant in the target."""
    def __init__(self, bg_id=0, threshold=0.9):
        self.bg_id = bg_id
        self.threshold = threshold

    def __call__(
            self,
            inp: np.ndarray,
            target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.sum(target == self.bg_id) / target.size > self.threshold:
            raise _DropSample
        return inp, target  # Return inp, target unmodified


class SmoothOneHotTarget:
    """Converts target tensors to one-hot encoding, with optional label smoothing.

    Args:
        num_classes: Number of classes (C) in the data set.
        smooth_eps: Label smoothing strength. If ``smooth_eps=0`` (default), no
            smoothing is applied and regular one-hot tensors are returned.
            See section 7 of https://arxiv.org/abs/1512.00567
    """
    # TODO: Add example to docstring
    def __init__(self, num_classes: int, smooth_eps: float = 0.):
        assert 0 <= smooth_eps < 0.5
        self.num_classes = num_classes
        self.smooth_eps = smooth_eps

    def __call__(
            self,
            inp: np.ndarray,  # returned without modifications
            target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.smooth_eps == 0.:
            eye = np.eye(self.num_classes)
        else:
            # Create a "soft" eye where  0 is replaced by smooth_eps and 1 by (1 - smooth_eps)
            eye = np.full((self.num_classes, self.num_classes), self.smooth_eps)
            np.fill_diagonal(eye, 1. - self.smooth_eps)
        onehot = np.moveaxis(eye[target], -1, 0)
        assert np.all(onehot.argmax(0) == target)
        return inp, onehot


class Normalize:
    """Normalizes inputs with supplied per-channel means and stds.

    Args:
        mean: Global mean value(s) of the inputs. Can either be a sequence
            of float values where each value corresponds to a channel
            or a single float value (only for single-channel data).
        std: Global standard deviation value(s) of the inputs. Can either
            be a sequence of float values where each value corresponds to a
            channel or a single float value (only for single-channel data).
        inplace: Apply in-place (works faster, needs less memory but overwrites
            inputs).
    """
    def __init__(
            self,
            mean: Union[Sequence[float], float],
            std: Union[Sequence[float], float],
            inplace: bool = False
    ):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.inplace = inplace
        # Unsqueeze first dimensions if mean and scalar are passed as scalars
        if self.mean.ndim == 0:
            self.mean = self.mean[None]
        if self.std.ndim == 0:
            self.std = self.std[None]

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.inplace:
            normalized = inp  # Refer to the same memory space
        else:
            normalized = np.empty_like(inp)
        if not inp.shape[0] == self.mean.shape[0] == self.std.shape[0]:
            raise ValueError('mean and std must have the same length as the C '
                             'axis (number of channels) of the input.')
        for c in range(inp.shape[0]):
            normalized[c] = (inp[c] - self.mean[c]) / self.std[c]
        return normalized, target


# TODO: Support uniform distribution (or any distribution in general)
class RandomBrightnessContrast:
    """Randomly changes brightness and contrast of the input image.

    Args:
        brightness_std: Standard deviation of contrast change strength (bias)
        contrast_std: Standard deviation of brightness change strength (scale)
        channels: If ``channels`` is ``None``, the change is applied to
            all channels of the input tensor.
            If ``channels`` is a ``Sequence[int]``, change is only applied
            to the specified channels.
        prob: probability (between 0 and 1) with which to perform this
            augmentation. The input is returned unmodified with a probability
            of ``1 - prob``.
    """


    def __init__(
            self,
            brightness_std: float = 0.5,
            contrast_std: float = 0.5,
            channels: Optional[Sequence[int]] = None,
            prob: float = 1.0,
    ):
        if not channels:  # Support empty sequences as an alias for None
            channels = None
        self.channels = channels
        self.prob = prob
        self.brightness_gen = Normal(mean=0.0, sigma=brightness_std)
        self.contrast_gen = Normal(mean=1.0, sigma=contrast_std)

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.prob:
            return inp, target
        channels = range(inp.shape[0]) if self.channels is None else self.channels
        augmented = inp.copy()
        for c in channels:
            a = self.contrast_gen()
            b = self.brightness_gen()
            # Formula based on tf.image.{adjust_contrast,adjust_brightness}
            # See https://www.tensorflow.org/api_docs/python/tf/image
            m = np.mean(inp[c])
            augmented[c] = a * (inp[c] - m) + m + b
        return augmented, target


# TODO: Per-channel gamma_std, infer channels from gamma_std shape? Same with AdditiveGaussianNoise
# TODO: Sample from a more suitable distribution. Due to clipping to gamma_min,
#       there is currently a strong bias towards this value.
class RandomGammaCorrection:
    """Applies random gamma correction to the input.

    Args:
        gamma_std: standard deviation of the gamma value.
        channels: If ``channels`` is ``None``, the change is applied to
            all channels of the input tensor.
            If ``channels`` is a ``Sequence[int]``, change is only applied
            to the specified channels.
        prob: probability (between 0 and 1) with which to perform this
            augmentation. The input is returned unmodified with a probability
            of ``1 - prob``.
    """
    def __init__(
            self,
            gamma_std: float = 0.5,
            gamma_min: float = 0.25,  # Prevent gamma <= 0 (0 causes zero division)
            channels: Optional[Sequence[int]] = None,
            prob: float = 1.0,
    ):
        if not channels:  # Support empty sequences as an alias for None
            channels = None
        self.channels = channels
        self.prob = prob
        self.gamma_generator = Normal(
            mean=1.0, sigma=gamma_std, bounds=(gamma_min, np.inf)
        )

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.prob:
            return inp, target
        channels = range(inp.shape[0]) if self.channels is None else self.channels
        gcorr = inp.copy()
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
    ):
        if not channels:  # Support empty sequences as an alias for None
            channels = None
        self.channels = channels
        self.prob = prob

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.prob:
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

        alpha = 1 + (np.random.rand(nc) - 0.5) * 0.3  # ~ contrast
        beta = (np.random.rand(nc) - 0.5) * 0.3  # Mediates whether values are clipped for shadows or lights
        gamma = 2.0 ** (np.random.rand(nc) * 2 - 1)  # Sample from [0.5, 2]

        aug[channels] = aug[channels] * alpha[:, None, None] + beta[:, None, None]
        aug[channels] = np.clip(aug[channels], 0, 1)
        aug[channels] = aug[channels] ** gamma[:, None, None]

        for c in channels:  # Rescale to original (normalized) intensity range
            aug[c] = skimage.exposure.rescale_intensity(aug[c], out_range=orig_intensity_ranges[c])

        return aug, target


class RandomGaussianBlur:
    """Adds random gaussian blur to the input.

    Args:
        distsigma: Sigma parameter of the half-normal distribution from
            which sigmas for the gaussian blurring are drawn.
            To clear up possible confusion: The ``distsigma`` parameter does
            **not** directly parametrize the gaussian blurring, but the
            random distribution from which the blurring sigmas are drawn
            from.
        prob: probability (between 0 and 1) with which to perform this
            augmentation. The input is returned unmodified with a probability
            of ``1 - prob``.
        aniso_factor: a tuple or an array to apply the anisotropy, must
            match the dimension of the input.
    """

    def __init__(
            self,
            distsigma: float = 1,
            channels: Optional[Sequence[int]] = None,
            prob: float = 1.0,
            aniso_factor: Optional = None,
    ):
        if not channels:  # Support empty sequences as an alias for None
            channels = None
        self.channels = channels
        self.prob = prob
        self.gaussian_std = HalfNormal(sigma=distsigma)
        if aniso_factor is None or aniso_factor == 1:
            aniso_factor = np.array([1, 1, 1])
        self.aniso_factor = aniso_factor

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.prob:
            return inp, target

        channels = range(inp.shape[0]) if self.channels is None else self.channels
        blurred_inp = np.empty_like(inp)
        for c in channels:
            self.aniso_factor = self.aniso_factor[:inp[c].ndim]
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
    """

    def __init__(
            self,
            sigma: float = 0.1,
            channels: Optional[Sequence[int]] = None,
            prob: float = 1.0,
    ):
        self.channels = channels
        self.prob = prob
        self.noise_generator = Normal(mean=0, sigma=sigma)

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
            # TODO: fast in-place version
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.prob:
            return inp, target
        noise = np.empty_like(inp)
        channels = range(inp.shape[0]) if self.channels is None else self.channels
        for c in channels:
            noise[c] = self.noise_generator(shape=inp[c].shape)
        noisy_inp = inp + noise
        return noisy_inp, target


class RandomCrop:
    def __init__(self, size: Sequence[int]):
        self.size = np.array(size)

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None  # returned without modifications
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
        full_slice = tuple(nonspatial_slice + spatial_slice)
        inp_cropped = inp[full_slice]
        if target is None:
            return inp_cropped, target

        if target.ndim == inp.ndim - 1:  # inp: (C, [D,], H, W), target: ([D,], H, W)
            full_slice = full_slice[1:]  # Remove C axis from slice because target doesn't have it
        target_cropped = target[full_slice]
        return inp_cropped, target_cropped


def _draw_debug_grid(
        inp: np.ndarray,
        target: Optional[np.ndarray] = None,
        s: int = 16,
        v: float = 0.,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Draw an ``s``-spaced grid of ``v`` values to visualize deformations."""
    if target is not None and target.ndim == inp.ndim - 1:
        target[::s] = v
    if inp.ndim == 4:
        inp[:, ::s] = v
        inp[:, :, ::s] = v
        inp[:, :, :, ::s] = v
        if target is not None:
            target[:, ::s] = v
            target[:, :, ::s] = v
            if target.ndim == 4:
                target[:, :, :, ::s] = v
    elif inp.ndim == 3:
        inp[:, ::s] = v
        inp[:, :, ::s] = v
        if target is not None:
            target[:, ::s] = v
            if target.ndim == 3:
                target[:, :, ::s] = v
    return inp, target


class ElasticTransform:
    """
    Based on https://gist.github.com/fmder/e28813c1e8721830ff9c


    Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.

        Args:
            sigma: Sigma parameter of the gaussian smoothing performed on the
                random displacement field. High ``sigma`` values (> 4) lead
                to less randomness and more spatial consistency.
                Lower values
            alpha: Factor by which all random displacements are multiplied.
                Each local displacement is in the range ``[0, alpha]``, so e.g.
                for ``alpha=1`` you won't see much of an effect.
            channels: If ``channels`` is ``None``, the change is applied to
                all channels of the input tensor.
                If ``channels`` is a ``Sequence[int]``, change is only applied
                to the specified channels.
            prob: probability (between 0 and 1) with which to perform this
                augmentation. The input is returned unmodified with a probability
                of ``1 - prob``
            target_discrete_ix: list
                List of target channels that contain discrete values.
                By default (``None``), every channel is is seen as discrete (this is
                generally the case for classification tasks).
                This information is used to decide what kind of interpolation should
                be used for reading target data:

                    - discrete targets are obtained by nearest-neighbor interpolation
                    - non-discrete (continuous) targets are linearly interpolated.
            aniso_factor: Factor by which to divide the deformation strength in the
                z axis. E.g. if the data has half resolution in the z dimension, set
                ``aniso_factor = 2``. By default it is ``1``, so every spatial
                dimension is treated equally.
            draw_debug_grid: If ``True``, draw a 16-spaced grid into the image to
                visualize deformations. This is only for debugging purposes and
                should never be enabled during training.

        The input image should be of dimensions (C, H, W) or (C, D, H, W).
        C must be included.

    """

    def __init__(
            self,
            sigma: float = 4,
            alpha: float = 40,
            channels: Optional[Sequence[int]] = None,
            prob: float = 0.25,
            target_discrete_ix: Optional[Sequence[int]] = None,
            aniso_factor: float = 1.,
            draw_debug_grid: bool = False
    ):
        self.sigma = sigma
        self.alpha = alpha
        self.channels = channels
        self.prob = prob
        self.target_discrete_ix = target_discrete_ix
        self.aniso_factor = aniso_factor
        self.draw_debug_grid = draw_debug_grid

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None

    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if np.random.rand() > self.prob:
            return inp, target

        channels = range(inp.shape[0]) if self.channels is None else self.channels

        # TODO (low priority): This could be written for n-d without explicit dimensions.
        if inp.ndim == 4:
            if self.draw_debug_grid:
                inp, target = _draw_debug_grid(inp, target)
            ish, tsh = np.array(inp.shape[-3:]), np.array(target.shape[-3:])
            dz = gaussian_filter((np.random.rand(*ish) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
            dy = gaussian_filter((np.random.rand(*ish) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
            dx = gaussian_filter((np.random.rand(*ish) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
            z, y, x = np.array(
                np.meshgrid(np.arange(ish[0]), np.arange(ish[1]), np.arange(ish[2]), indexing='ij'),
                dtype=np.float64
            )
            dz /= self.aniso_factor
            z += dz
            y += dy
            x += dx
            indices = np.reshape(z, (-1, 1)), np.reshape(y, (-1, 1)), np.reshape(x, (-1, 1))

            # If there is a target, apply the same deformation field to the target
            if target is not None and np.any(ish != tsh):
                if self.draw_debug_grid:
                    inp, target = _draw_debug_grid(inp, target)
                # Crop input re-indexing arrays to the target region and transform coordinates
                #  to the target' own frame by subtracting the input-target offset
                lo = (ish - tsh) // 2
                hi = ish - lo
                tcrop = tuple([slice(lo[i], hi[i]) for i in range(3)])
                target_indices = (
                    np.reshape(z[tcrop] - lo[0], (-1, 1)),
                    np.reshape(y[tcrop] - lo[1], (-1, 1)),
                    np.reshape(x[tcrop] - lo[2], (-1, 1))
                )
            else:
                target_indices = indices
        elif inp.ndim == 3:
            ish, tsh = np.array(inp.shape[-2:]), np.array(target.shape[-2:])
            dy = gaussian_filter((np.random.rand(*ish) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
            dx = gaussian_filter((np.random.rand(*ish) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
            y, x = np.array(
                np.meshgrid(np.arange(ish[0]), np.arange(ish[1])),
                dtype=np.float64
            )
            y += dy
            x += dx
            indices = np.reshape(y, (-1, 1)), np.reshape(x, (-1, 1))

            # If there is a target, apply the same deformation field to the target
            if target is not None and np.any(ish != tsh):
                # Crop input re-indexing arrays to the target region and transform coordinates
                #  to the target' own frame by subtracting the input-target offset
                lo = (ish - tsh) // 2
                hi = ish - lo
                tcrop = tuple([slice(lo[i], hi[i]) for i in range(2)])
                target_indices = (
                    np.reshape(y[tcrop] - lo[0], (-1, 1)),
                    np.reshape(x[tcrop] - lo[1], (-1, 1))
                )
            else:
                target_indices = indices
        else:
            raise ValueError("Input dimension not understood!")

        deformed_img = np.empty_like(inp)
        for c in channels:
            deformed_img[c] = map_coordinates(inp[c], indices, order=1).reshape(ish)

        if target is None:
            return deformed_img, target
        else:
            target_c = True  # True if the first dim of target is the number of channels
            if target.ndim == 4:  # (C, D, H, W)
                target_channels = target.shape[0]
                target_shape = target[0].shape
            elif target.ndim == 3:  # (C, H, W) or (D, H, W)
                if inp.ndim == 3:  # (C, H, W) target, (C, H, W) input
                    target_channels = target.shape[0]
                    target_shape = target[0].shape
                elif inp.ndim == 4:  # (D, H, W) target, (C, D, H, W) input
                    target_c = False
                    target_channels = 1
                    target_shape = target.shape
                else:
                    raise ValueError("Input dimension not understood!")
            elif target.ndim == 2:  # (H, W)
                target_c = False
                target_channels = 1
                target_shape = target.shape
            else:
                raise ValueError("Target dimension not understood!")

            if self.target_discrete_ix is None:
                self.target_discrete_ix = [True for i in range(target_channels)]
            else:
                self.target_discrete_ix = [i in self.target_discrete_ix for i in range(target_channels)]

            deformed_target = np.empty_like(target)
            if target_c:
                for tc in range(target_channels):
                    target_order = 0 if self.target_discrete_ix[tc] is True else 1
                    deformed_target[tc] = map_coordinates(target[tc], target_indices, order=target_order).reshape(target_shape)
            else:
                target_order = 0 if self.target_discrete_ix[0] is True else 1
                deformed_target = map_coordinates(target, target_indices, order=target_order).reshape(target_shape)
            return deformed_img, deformed_target


class SqueezeTarget:
    """Squeeze a specified dimension in target tensors.

    (This is just needed as a workaround for the example neuro_data_cdhw data
    set, because its targets have a superfluous first dimension.)"""
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(
            self,
            inp: np.ndarray,  # Returned without modifications
            target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return inp, target.squeeze(axis=self.dim)


class RandomFlip:
    """Randomly flips spatial input and target dimensions respectively. Spatial
    dimensions are considered to occur last in the input/target shape and are
    flipped with probability p=0.5 (iid).

    Args:
        ndim_spatial: Number of spatial dimension in input, e.g.
            ``ndim_spatial=2`` for input shape (N, C, H, W)
    """
    def __init__(
            self,
            ndim_spatial: int = 2,
    ):
        self.randint = RandInt()
        self.ndim_spatial = ndim_spatial

    def __call__(
            self,
            inp: np.ndarray,
            target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: np.flip now supports multi-dimensional flipping as of numpy 1.15
        #       So we can rewrite this with np.flip to make it more readable.
        flip_dims = self.randint(self.ndim_spatial)
        # flip all images at once
        slices_inp = tuple(
            [slice(None, None, 1) for _ in range(len(inp.shape) - self.ndim_spatial)] +
            [slice(None, None, (-1)**flip_d) for flip_d in flip_dims]
        )
        inp_flipped = inp[slices_inp].copy()
        if target is not None:
            slices_target = tuple(
                [slice(None, None, 1) for _ in range(len(target.shape) - self.ndim_spatial)] +
                [slice(None, None, (-1)**flip_d) for flip_d in flip_dims]
            )
            target_flipped = target[slices_target].copy()
        else:
            target_flipped = None
        return inp_flipped, target_flipped


# TODO: Make rotation border mode configurable
class RandomRotate2d:
    """Random rotations in the xy plane, based on scikit-image.

    If inputs are 3D images ([C,] D, H, W) Rotate them as a stack of 2D images
    by the same angle, constraining the rotation direction to the xy plane"""
    def __init__(self, angle_range=(-180, 180), prob=1):
        self.angle_range = angle_range
        self.prob = prob

    def __call__(self, inp, target):
        assert inp.ndim in (3, 4)
        if np.random.rand() > self.prob:
            return inp, target
        angle = np.random.uniform(*self.angle_range)
        rot_opts = {'angle': angle, 'preserve_range': True, 'mode': 'reflect'}

        if target.ndim == inp.ndim - 1:  # Implicit (no) channel dimension
            target_c = False
        elif target.ndim == inp.ndim:  # Explicit channel dimension
            target_c = True
        else:
            raise ValueError('Target dimension not understood.')

        def rot(inp, target):
            """Rotate in 2D space"""
            for c in range(inp.shape[0]):
                inp[c] = skimage.transform.rotate(inp[c], **rot_opts).astype(inp.dtype)[None]
            if target is None:
                return inp, target  # Return early if target shouldn't be transformed
            # Otherwise, transform target:
            if target_c:
                for c in range(target.shape[0]):
                    target[c] = skimage.transform.rotate(target[c], **rot_opts).astype(target.dtype)
            else:
                target = skimage.transform.rotate(target, **rot_opts).astype(target.dtype)
            return inp, target
        
        if inp.ndim == 3:  # 2D case
            rinp, rtarget = rot(inp, target)
        else:  # 3D case: Rotate each z slice separately by the same angle
            rinp = np.empty_like(inp)
            rtarget = np.empty_like(target)
            for z in range(rinp.shape[1]):
                if target_c:
                    rinp[:, z], rtarget[:, z] = rot(inp[:, z], target[:, z])
                else:
                    rinp[:, z], rtarget[z] = rot(inp[:, z], target[z])

        return rinp, rtarget


# TODO: Support other image shapes
class AlbuSeg2d:
    """Wrapper for albumentations' segmentation-compatible 2d augmentations.

    Wraps an augmentation so it can be used within elektronn3's transform pipeline.
    See https://github.com/albu/albumentations.

    If ``target`` is ``None``, it is ignored. Else, it is passed to the wrapped
    albumentations augmentation as the ``mask`` argument.

    Args:
        albu: albumentation object of type `DualTransform`.

    Example::

        >>> import albumentations
        >>> transform = AlbuSeg2d(albumentations.ShiftScaleRotate(
        ...     p=0.98, rotate_limit=180, scale_limit=0.1, interpolation=3
        ... ))
    """

    def __init__(self, albu: 'albumentations.core.transforms_interface.DualTransform'):
        self.albu = albu

    def __call__(self, inp, target):
        assert inp.ndim == 3 and inp.shape[0] == 1
        if target is not None:
            assert target.ndim == 2 and target.shape == inp.shape[1:]
            augmented = self.albu(image=inp[0], mask=target)  # Strip C dimension
            atarget = np.array(augmented['mask'], dtype=target.dtype)
        else:
            augmented = self.albu(image=inp[0])  # Strip C dimension
            atarget = None
        ainp = np.array(augmented['image'], dtype=inp.dtype)[None]  # Re-attach C dimension
        return ainp, atarget


# TODO: Functional API (transforms.functional).
#       The current object-oriented interface should be rewritten as a wrapper
#       for the functional API (see implementation in torchvision).

# TODO: Extract a non-random version from each random transform.
#       E.g. RandomGammaCorrection should wrap GammaCorrection, which takes
#       the actual gamma value as an argument instead of a parametrization
#       of the random distribution from which gamma is sampled.

# TODO: Albumentations wrapper for img-to-scalar scenarios like classification
