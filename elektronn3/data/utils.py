# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert, Marius Killinger


import logging
import os
import signal
import traceback
from typing import Sequence

import h5py
import numpy as np

from elektronn3 import floatX

logger = logging.getLogger("elektronn3log")


def slice_h5(
        src: h5py.Dataset,
        coords_lo: Sequence,
        coords_hi: Sequence,
        dtype: type = np.float32,
        prepend_batch_axis: bool = False,
        max_retries: int = 5,
) -> np.ndarray:
    if max_retries <= 0:
        logger.error(
            f'slice_h5(): max_retries exceeded at {coords_lo}, {coords_hi}. Aborting...'
        )
        raise ValueError
    assert len(coords_lo) == len(coords_hi) == 3
    try:
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
            cut = cut[None]  # Prepend a new C axis
        else:
            raise ValueError(  # TODO: Other (custom) exception type
                f'src has wrong shape {src.shape}. Only 3D and 4D shapes are supported.'
            )
    # Work around mysterious random HDF5 read errors by recursively calling
    # this function from within itself until it works again or until
    # max_retries is exceeded.
    except OSError:
        traceback.print_exc()
        logger.warning(
            f'Read error. Retrying at the same location ({max_retries} attempts remaining)...'
        )
        # Try slicing from the same coordinates, but with max_retries -= 1.
        # (Overriding prepend_batch_axis to False because the initial (outer)
        #  call will prepend the axis and propagating it to the recursive
        #  (inner) calls could lead to multiple axes being prepended.)
        cut = slice_h5(
            src=src,
            coords_lo=coords_lo,
            coords_hi=coords_hi,
            dtype=dtype,
            prepend_batch_axis=False,  # See comment above
            max_retries=(max_retries - 1)
        )
        # If the recursive call above was sucessfull, use its result `cut`
        # as if it was the immediate result of the first slice attempt.
    if prepend_batch_axis:
        cut = cut[None]
        assert cut.ndim == 5
    else:
        assert cut.ndim == 4
    if cut.dtype != dtype:
        cut = cut.astype(dtype)
    return cut


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
