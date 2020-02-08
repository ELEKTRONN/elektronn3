"""Code related to data sources (HDF5 etc.)"""

# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

import os
import traceback
from typing import Union, Any, Sequence

import h5py
import numpy as np

from elektronn3.data.utils import logger


class DataSource:  #(Protocol):  # Protocol requires Python 3.8 or typing_extensions...
    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray: ...

    # Expected properties: size, shape, dtype, fname, in_memory


class HDF5DataSource(DataSource):
    """An h5py.Dataset wrapper for safe multiprocessing. Opens the file and
    the dataset on each read/property access and then immediately closes it.

    This is a workaround for this issue and related data corruptions:
    https://github.com/pytorch/pytorch/issues/11929.

    By avoiding open file handles before worker processes are forked,
    concurrency issues with HDF5's global state do not apply."""

    def __init__(self, fname: str, key: str, in_memory: bool = False):
        self.fname = os.path.expanduser(fname)
        self.key = key
        self.in_memory = in_memory

        if self.in_memory:
            self._data: np.ndarray
            self._initialize_memory()

    def _initialize_memory(self) -> None:
        with h5py.File(self.fname, 'r') as f:
            h5data = f[self.key]
            self._data = h5data[()]

    # Wraps direct attribute, property and method access
    def __getattr__(self, attr: str) -> Any:
        if self.in_memory:
            h5data = self._data
            return getattr(h5data, attr)
        with h5py.File(self.fname, 'r') as f:
            h5data = f[self.key]
            return getattr(h5data, attr)

    # But dunder methods have to be wrapped manually: https://stackoverflow.com/a/3700899
    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        if self.in_memory:
            h5data = self._data
            return h5data[idx]
        with h5py.File(self.fname, 'r') as f:
            h5data = f[self.key]
            return h5data[idx]


def slice_3d(
        src: DataSource,
        coords_lo: Sequence[int],
        coords_hi: Sequence[int],
        dtype: type = np.float32,
        prepend_empty_axis: bool = False,
        max_retries: int = 5,
        check_bounds=True,
) -> np.ndarray:
    """ Slice a patch of 3D image data out of a data source.

    Args:
        src: Source data set from which to read data.
            The expected data shapes are (C, D, H, W) or (D, H, W).
        coords_lo: Lower bound of the coordinates where data should be read
            from in ``src``.
        coords_hi: Upper bound of the coordinates where data should be read
            from in ``src``.
        dtype: NumPy ``dtype`` that the sliced array will be cast to if it
            doesn't already have this dtype.
        prepend_empty_axis: Prepends a new empty (1-sized) axis to the sliced
            array before returning it.
        max_retries: Maximum retries if a read error occurs when reading from
            the HDF5 file.
        check_bounds: If ``True`` (default), only indices that are within the
            bounds of ``src`` will be allowed (no negative indices or slices
            to indices that exceed the shape of ``src``, which would normally
            just be ignored).

    Returns:
        Sliced image array.
    """
    if check_bounds:
        if np.any(np.array(coords_lo) < 0):
            raise RuntimeError(f'coords_lo={coords_lo} exceeds src shape {src.shape[-3:]}')
        if np.any(np.array(coords_hi) > np.array(src.shape[-3:])):
            raise RuntimeError(f'coords_hi={coords_hi} exceeds src shape {src.shape[-3:]}')
    if max_retries <= 0:
        logger.error(
            f'slice_h5(): max_retries exceeded at {coords_lo}, {coords_hi}. Aborting...'
        )
        raise ValueError

    try:
        # Generalized n-d slicing code (temporarily disabled because of the
        #  performance issue described in the comment below):
        ## full_slice = calculate_nd_slice(src, coords_lo, coords_hi)
        ## # # TODO: Use a better workaround or fix this in h5py:
        ## srcv = src.value  # Workaround for hp5y indexing limitation. The `.value` call is very unfortunate! It loads the entire cube to RAM.
        ## cut = srcv[full_slice]

        if src.ndim == 4:
            cut = src[
                :,
                coords_lo[0]:coords_hi[0],
                coords_lo[1]:coords_hi[1],
                coords_lo[2]:coords_hi[2]
            ]
        elif src.ndim == 3:
            cut = src[
                coords_lo[0]:coords_hi[0],
                coords_lo[1]:coords_hi[1],
                coords_lo[2]:coords_hi[2]
            ]
        else:
            raise ValueError(f'Expected src.ndim to be 3 or 4, but got {src.ndim} instead.')
    # Work around mysterious random HDF5 read errors by recursively calling
    #  this function from within itself until it works again or until
    #  max_retries is exceeded.
    except OSError:
        traceback.print_exc()
        logger.warning(
            f'Read error. Retrying at the same location ({max_retries} attempts remaining)...'
        )
        # Try slicing from the same coordinates, but with max_retries -= 1.
        #  (Overriding prepend_empty_axis to False because the initial (outer)
        #  call will prepend the axis and propagating it to the recursive
        #  (inner) calls could lead to multiple axes being prepended.)
        cut = slice_3d(
            src=src,
            coords_lo=coords_lo,
            coords_hi=coords_hi,
            dtype=dtype,
            prepend_empty_axis=False,  # See comment above
            max_retries=(max_retries - 1)
        )
        # If the recursive call above was sucessful, use its result `cut`
        # as if it was the immediate result of the first slice attempt.
    if prepend_empty_axis:
        cut = cut[None]
    if cut.dtype != dtype:
        cut = cut.astype(dtype)
    return cut
