# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert, Marius Killinger


import logging
import os
import signal
from typing import Sequence, Tuple

import h5py
import numpy as np

from elektronn3 import floatX
from elektronn3.data.sources import DataSource

logger = logging.getLogger("elektronn3log")


eps = 0.0001  # To avoid divisions by zero


def _to_full_numpy(seq) -> np.ndarray:
    if isinstance(seq, np.ndarray):
        return seq
    elif isinstance(seq, DataSource):
        return seq[()]
    elif isinstance(seq[0], np.ndarray):
        return np.array(seq)
    elif isinstance(seq[0], h5py.Dataset) or isinstance(seq[0], DataSource):
        # Explicitly pre-load all dataset values into ndarray format
        return np.array([x[()] for x in seq])
    else:
        raise ValueError('inputs must be an ndarray, a sequence of ndarrays '
                         'or a sequence of h5py.Datasets.')


def calculate_means(inputs: Sequence) -> Tuple[float]:
    inputs = [
        _to_full_numpy(inp).reshape(inp.shape[0], -1)  # Flatten every dim except C
        for inp in inputs
    ]  # Necessary if shapes don't match
    # Preserve C, but concatenate everything else into one flat dimension
    inputs = np.concatenate(inputs, axis=1)
    means = np.mean(inputs, axis=1)
    return tuple(means)


def calculate_stds(inputs: Sequence) -> Tuple[float]:
    inputs = [
        _to_full_numpy(inp).reshape(inp.shape[0], -1)  # Flatten every dim except C
        for inp in inputs
    ]  # Necessary if shapes don't match
    # Preserve C, but concatenate everything else into one flat dimension
    inputs = np.concatenate(inputs, axis=1)
    stds = np.std(inputs, axis=1)
    return tuple(stds)


def calculate_class_weights(
        targets: Sequence[np.ndarray],
        mode='inverse'
) -> np.ndarray:
    """Calulate class weights that assign more weight to less common classes.

    The weights can then be used for loss function rebalancing (e.g. for
    CrossEntropyLoss it's very important to do this when training on
    datasets with high class imbalance."""

    targets = np.concatenate([
        _to_full_numpy(target).flatten()  # Flatten every dim except C
        for target in targets
    ])  # Necessary if shapes don't match

    def __inverse(targets):
        """The weight of each class c1, c2, c3, ... with labeled-element
        counts n1, n2, n3, ... is assigned by weight[i] = N / n[i],
        where N is the total number of all elements in all ``targets``.
        (We could achieve the same relative weight proportions by
        using weight[i] = 1 / n[i], as proposed in
        https://arxiv.org/abs/1707.03237, but we multiply by N to prevent
        very small values that could lead to numerical issues."""
        classes = np.arange(0, targets.max() + 1)
        # Count total number of labeled elements per class
        num_labeled = np.array([
            np.sum(np.equal(targets, c))
            for c in classes
        ], dtype=np.float32)
        class_weights = (targets.size / (num_labeled + eps)).astype(np.float32)
        return class_weights

    def __norpf_inverse(targets):
        classes = np.arange(0, targets.max() + 1)
        # Count total number of labeled elements per class
        num_labeled = np.array([
            np.sum(np.equal(targets, c))
            for c in classes
        ], dtype=np.float32)
        class_weights = (num_labeled / targets.size).astype(np.float32)
        return class_weights

    def __binmean(targets):
        """Use the mean of the targets to determine class weights.

        This assumes a binary segmentation problem (background/foreground) and
        breaks in a multi-class setting."""
        target_mean = np.mean(targets)
        bg_weight = target_mean / (1. + target_mean)
        fg_weight = 1. - bg_weight
        # class_weights = torch.tensor([bg_weight, fg_weight])
        class_weights = np.array([bg_weight, fg_weight], dtype=np.float32)
        return class_weights

    if mode == 'inverse':
        return __inverse(targets)
    elif mode == 'inversesquared':
        return __inverse(targets) ** 2
    elif mode == 'norpf_inverse':
        return __norpf_inverse(targets)
    elif mode == 'binmean':
        return __binmean(targets)


def calculate_nd_slice(src, coords_lo, coords_hi):
    """Calculate the ``slice`` object list that is used as indices for
    reading from a data source.

    Unfortunately, this kind of slice list is not yet supported by h5py.
    It only works with numpy arrays."""
    # Separate spatial dimensions (..., H, W) from nonspatial dimensions (C, ...)
    spatial_dims = len(coords_lo)  # Assuming coords_lo addresses all spatial dims
    nonspatial_dims = src.ndim - spatial_dims  # Assuming every other dim is nonspatial

    # Calculate necessary slice indices for reading the file
    nonspatial_slice = [  # Slicing all available content in these dims.
        slice(0, src.shape[i]) for i in range(nonspatial_dims)
    ]
    spatial_slice = [  # Slice only the content within the coordinate bounds
        slice(coords_lo[i], coords_hi[i]) for i in range(spatial_dims)
    ]
    full_slice = nonspatial_slice + spatial_slice
    return full_slice


def save_to_h5(data, path, hdf5_names=None, overwrite=False, compression=True):
    """
    Saves data to HDF5 File.

    Parameters
    ----------
    data: list or dict of np.arrays
        if list, hdf5_names has to be set.
    path: str
        forward-slash separated path to file
    hdf5_names: list of str
        has to be the same length as data
    overwrite : bool
        determines whether existing files are overwritten
    compression : bool
        True: compression='gzip' is used which is recommended for sparse and
        ordered data

    Returns
    -------
    nothing

    """
    if (not type(data) is dict) and hdf5_names is None:
        raise Exception("hdf5names has to be set if data is a list")
    if os.path.isfile(path) and overwrite:
        os.remove(path)
    f = h5py.File(path, "w")
    if type(data) is dict:
        for key in data.keys():
            if compression:
                f.create_dataset(key, data=data[key], compression="gzip")
            else:
                f.create_dataset(key, data=data[key])
    else:
        if len(hdf5_names) != len(data):
            f.close()
            raise Exception("Not enough or to much hdf5-names given!")
        for nb_data in range(len(data)):
            if compression:
                f.create_dataset(hdf5_names[nb_data], data=data[nb_data],
                                 compression="gzip")
            else:
                f.create_dataset(hdf5_names[nb_data], data=data[nb_data])
    f.close()


def as_floatX(x):
    if not hasattr(x, '__len__'):
        return np.array(x, dtype=floatX)
    return np.ascontiguousarray(x, dtype=floatX)


def squash01(img: np.ndarray) -> np.ndarray:
    """Squash image array to the value range [0, 1] (no clipping).

    This can be used to prepare network outputs or normalized inputs
    for plotting and generic image processing functions.
    """
    img = img.astype(np.float32)
    squashed = (img - np.min(img)) / np.ptp(img)
    return squashed


# https://gist.github.com/tcwalther/ae058c64d5d9078a9f333913718bba95
# class based on: http://stackoverflow.com/a/21919644/487556
class DelayedInterrupt:
    def __init__(self, signals):
        if not isinstance(signals, list) and not isinstance(signals, tuple):
            signals = [signals]
        self.sigs = signals

    def __enter__(self):
        self.signal_received = {}
        self.old_handlers = {}
        for sig in self.sigs:
            self.signal_received[sig] = False
            self.old_handlers[sig] = signal.getsignal(sig)
            def handler(s, frame):
                logger.warning('Signal %s received. Delaying KeyboardInterrupt.' % sig)
                self.signal_received[sig] = (s, frame)
                # Note: in Python 3.5, you can use signal.Signals(sig).name
            self.old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)

    def __exit__(self, type, value, traceback):
        for sig in self.sigs:
            signal.signal(sig, self.old_handlers[sig])
            if self.signal_received[sig] and self.old_handlers[sig]:
                self.old_handlers[sig](*self.signal_received[sig])


class CleanExit:
    # https://stackoverflow.com/questions/4205317/capture-keyboardinterrupt-in-python-without-try-except
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is KeyboardInterrupt:
            logger.warning('Delaying KeyboardInterrupt.')
            return True
        return exc_type is None


class GracefulInterrupt:
    # by https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
    now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, sig, frame):
        logger.warning('Signal %s received. Delaying KeyboardInterrupt.' % sig)
        self.now = True
