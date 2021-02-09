# -*- coding: utf-8 -*-
# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Marius Killinger, Philipp Schubert, Martin Drawitsch

import logging
import time
import warnings
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from elektronn3.training import plotting
from elektronn3 import floatX

logger = logging.getLogger('elektronn3log')


def create_preview_batch_from_knossos(knossos_preview_config: Dict[str, str]) -> torch.Tensor:
    from knossos_utils import KnossosDataset
    config = knossos_preview_config
    required_keys = ['dataset', 'size', 'offset', 'mag', 'target_mags']
    for k in required_keys:
        if k not in config.keys():
            raise ValueError(f'Required key {k} missing from knossos_preview_config.')
    logger.info(f'Loading preview region from dataset {config["dataset"]}')
    datasets = config['dataset']
    if isinstance(datasets, str):
        datasets = [datasets]
    inp_np = []
    for idx, dataset_path in enumerate(datasets):
        ds = KnossosDataset(dataset_path)
        inp_np.append(ds.load_raw(
            offset=config['offset'],
            size=config['size'],
            mag=config['mag'],
            datatype=np.float32
        ))  # (D, H, W)
    inp_np = np.stack(inp_np, axis=0) # C, D, H, W
    inp_np = inp_np[None]  # (N, C, D, H, W)
    inp_np = inp_np / config.get('scale_brightness', 1.)
    inp = torch.from_numpy(inp_np)
    return inp


class HistoryTracker:
    def __init__(self):
        self.plotting_proc = None
        self.debug_outputs = None
        self.regression_track = None
        self.debug_output_names = None

        self.timeline = AccumulationArray(n_init=int(1e5), dtype=dict(
            names=[u'time', u'loss', u'batch_char', ], formats=[u'f4', ] * 3))

        self.history = AccumulationArray(n_init=int(1e4), dtype=dict(
            names=[u'steps', u'time', u'train_loss', u'valid_loss',
                   u'loss_gain', u'train_err', u'valid_err', u'lr', u'mom',
                   u'gradnetrate'], formats=[u'i4', ] + [u'f4', ] * 9))
        self.loss = AccumulationArray(n_init=int(1e5), data=[])

    def update_timeline(self, vals):
        self.timeline.append(vals)
        self.loss.append(vals[1])

    def register_debug_output_names(self, names):
        self.debug_output_names = names

    def update_history(self, vals):
        self.history.append(vals)

    def update_debug_outputs(self, vals):
        if self.debug_outputs is None:
            self.debug_outputs = AccumulationArray(n_init=int(1e5),
                                                   right_shape=len(vals))

        self.debug_outputs.append(vals)

    def update_regression(self, pred, target):
        if self.regression_track is None:
            assert len(pred)==len(target)
            p = AccumulationArray(n_init=int(1e5), right_shape=len(pred))
            t = AccumulationArray(n_init=int(1e5), right_shape=len(pred))
            self.regression_track = [p, t]

        self.regression_track[0].append(pred)
        self.regression_track[1].append(target)

    def plot(self, save_path=None, autoscale=True, close=True, loss_smoothing_len=200):
        plotting.plot_hist(self.timeline, self.history, save_path,
                           loss_smoothing_len, autoscale)

        if self.debug_output_names and self.debug_outputs.length:
            plotting.plot_debug(self.debug_outputs, self.debug_output_names,
                                save_path)

        if self.regression_track:
            plotting.plot_regression(self.regression_track[0],
                                     self.regression_track[1], save_path)
            plotting.plot_kde(self.regression_track[0],
                              self.regression_track[1], save_path)

        if close:
            plt.close('all')


# TODO: Try to remove this thing (or document/rewrite it)
class AccumulationArray:
    def __init__(self, right_shape=(), dtype=floatX, n_init=100, data=None,
                 ema_factor=0.95):
        if isinstance(dtype, dict) and right_shape!=():
            raise ValueError("If dict is used as dtype, right shape must be"
                             "unchanged (i.e it is 1d)")

        if data is not None and len(data):
            n_init += len(data)
            right_shape = data.shape[1:]
            dtype = data.dtype

        self._n_init = n_init
        if isinstance(right_shape, int):
            self._right_shape = (right_shape,)
        else:
            self._right_shape = tuple(right_shape)
        self.dtype = dtype
        self.length = 0
        self._buffer = self._alloc(n_init)
        self._min = +np.inf
        self._max = -np.inf
        self._sum = 0
        self._ema = None
        self._ema_factor = ema_factor

        if data is not None and len(data):
            self.length = len(data)
            self._buffer[:self.length] = data
            self._min = data.min(0)
            self._max = data.max(0)
            self._sum = data.sum(0)

    def __repr__(self):
        return repr(self.data)

    def _alloc(self, n):
        if isinstance(self._right_shape, (tuple, list, np.ndarray)):
            ret = np.zeros((n,) + tuple(self._right_shape), dtype=self.dtype)
        elif isinstance(self.dtype, dict):  # rec array
            ret = np.zeros(n, dtype=self.dtype)
        else:
            raise ValueError("dtype not understood")
        return ret

    def append(self, data):
        # data = self.normalise_data(data)
        if len(self._buffer)==self.length:
            tmp = self._alloc(len(self._buffer) * 2)
            tmp[:self.length] = self._buffer
            self._buffer = tmp

        if isinstance(self.dtype, dict):
            for k, val in enumerate(data):
                self._buffer[self.length][k] = data[k]
        else:
            self._buffer[self.length] = data
            if self._ema is None:
                self._ema = self._buffer[self.length]
            else:
                f = self._ema_factor
                fc = 1 - f
                self._ema = self._ema * f + self._buffer[self.length] * fc

        self.length += 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self._min = np.minimum(data, self._min)
            self._max = np.maximum(data, self._max)
            self._sum = self._sum + np.asanyarray(data)

    def add_offset(self, off):
        self.data[:] += off
        if off.ndim>np.ndim(self._sum):
            off = off[0]
        self._min += off
        self._max += off
        self._sum += off * self.length

    def clear(self):
        self.length = 0
        self._min = +np.inf
        self._max = -np.inf
        self._sum = 0

    def mean(self):
        return np.asarray(self._sum, dtype=floatX) / self.length

    def sum(self):
        return self._sum

    def max(self):
        return self._max

    def min(self):
        return self._min

    def __len__(self):
        return self.length

    @property
    def data(self):
        return self._buffer[:self.length]

    @property
    def ema(self):
        return self._ema

    def __getitem__(self, slc):
        return self._buffer[:self.length][slc]


class Timer:
    def __init__(self):
        self.origin = time.time()
        self.t0 = self.origin

    @property
    def t_passed(self):
        return time.time() - self.origin


def pretty_string_time(t):
    """Custom printing of elapsed time"""
    if t > 4000:
        s = 't=%.1fh' % (t / 3600)
    elif t > 300:
        s = 't=%.0fm' % (t / 60)
    else:
        s = 't=%.0fs' % (t)
    return s
