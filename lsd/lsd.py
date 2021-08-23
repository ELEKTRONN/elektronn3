from typing import Sequence, Tuple, Optional, Dict, Any, Callable, Union

import warnings
import numpy as np
import skimage.exposure
import skimage.transform

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import distance_transform_edt


from elektronn3.data.transforms import random_blurring
from elektronn3.data.transforms.random import Normal, HalfNormal, RandInt

Transform = Callable[
    [np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, Optional[np.ndarray]]
]
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


class DistanceTransformTarget:
    """Converts discrete binary label target tensors to their (signed)
    euclidean distance transform (EDT) representation.
    
    Based on the method proposed in https://arxiv.org/abs/1805.02718.
    
    Args:
    scale: Scalar value to divide distances before applying normalization
    normalize_fn: Function to apply to distance map for normalization.
    inverted: Invert target labels before computing transform if ``True``.
         This means the distance map will show the distance to the nearest
         foreground pixel at each background pixel location (which is the
         opposite behavior of standard distance transform).
    signed: Compute signed distance transform (SEDT), where foreground
        regions are not 0 but the negative distance to the nearest
        foreground border.
    vector: Return distance vector map instead of scalars.
    """
    def __init__(
        self,
        scale: Optional[float] = 50.,
        normalize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = np.tanh,
        inverted: bool = True,
        signed: bool = True,
        vector: bool = False
    ):
        self.scale = scale
        self.normalize_fn = normalize_fn
        self.inverted = inverted
        self.signed = signed
        self.vector = vector
        
    def edt(self, target: np.ndarray) -> np.ndarray:
        sh = target.shape
        if target.min() == 1:  # If everything is 1, the EDT should be inf for every pixel
            nc = target.ndim if self.vector else 1
            return np.full((nc, *sh), np.inf, dtype=np.float32)
        
        if self.vector:
            if target.ndim == 2:
                coords = np.mgrid[:sh[0], :sh[1]]
            elif target.ndim == 3:
                coords = np.mgrid[:sh[0], :sh[1], :sh[2]]
            else:
                raise RuntimeError(f'Target shape {sh} not understood.')
            inds = distance_transform_edt(
                target, return_distances=False, return_indices=True
            ).astype(np.float32)
            dist = inds - coords
            # assert np.isclose(np.sqrt(dist[0] ** 2 + dist[1] ** 2), distance_transform_edt(target))
            return dist
            
        # Else: Regular scalar edt
        dist = distance_transform_edt(target).astype(np.float32)[None]
        return dist
        
    def __call__(
        self,
        inp: np.ndarray,  # returned without modifications
        target: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if target is None:
            return inp, target
        # assert target.max() <= 1
        # Ensure np.bool dtype, invert if needed
        if self.inverted:
            target = target == 0
        else:
            target = target > 0
        dist = self.edt(target)
        if self.signed:
            # Compute same transform on the inverted target. The inverse transform can be
            #  subtracted from the original transform to get a signed distance transform.
            invdist = self.edt(~target)
            dist -= invdist
        if self.normalize_fn is not None:
            dist = self.normalize_fn(dist / self.scale)
        return inp, dist
        
class LSDTarget:
    def __init__(
        self,
        scale: Optional[float] = 50.,
        normalize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = np.tanh,
        inverted: bool = True,
        signed: bool = True,
        vector: bool = False
    ):
        self.scale = scale
        self.normalize_fn = normalize_fn
        self.inverted = inverted
        self.signed = signed
        self.vector = vector
        
    def edt(self, target: np.ndarray) -> np.ndarray:
        sh = target.shape
        if target.min() == 1:  # If everything is 1, the EDT should be inf for every pixel
            nc = target.ndim if self.vector else 1
            return np.full((nc, *sh), np.inf, dtype=np.float32)
        
        if self.vector:
            if target.ndim == 2:
                coords = np.mgrid[:sh[0], :sh[1]]
            elif target.ndim == 3:
                coords = np.mgrid[:sh[0], :sh[1], :sh[2]]
            else:
                raise RuntimeError(f'Target shape {sh} not understood.')
            inds = distance_transform_edt(
                target, return_distances=False, return_indices=True
            ).astype(np.float32)
            dist = inds - coords
            # assert np.isclose(np.sqrt(dist[0] ** 2 + dist[1] ** 2), distance_transform_edt(target))
            return dist
            
        # Else: Regular scalar edt
        dist = distance_transform_edt(target).astype(np.float32)[None]
        return dist
        
    def __call__(
        self,
        inp: np.ndarray,  # returned without modifications
        target: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if target is None:
            return inp, target
        # assert target.max() <= 1
        # Ensure np.bool dtype, invert if needed
        if self.inverted:
            target = target == 0
        else:
            target = target > 0
        dist = self.edt(target)
        if self.signed:
            # Compute same transform on the inverted target. The inverse transform can be
            #  subtracted from the original transform to get a signed distance transform.
            invdist = self.edt(~target)
            dist -= invdist
        if self.normalize_fn is not None:
            dist = self.normalize_fn(dist / self.scale)
        return inp, dist
        
