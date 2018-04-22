# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

__all__ = ['PatchCreator']

import logging
import os
import sys
import time
import traceback
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data

from elektronn3.data import transformations
from elektronn3.data.utils import slice_h5
from elektronn3.data.data_erasing import check_random_data_blurring_config

logger = logging.getLogger('elektronn3log')


class PatchCreator(data.Dataset):
    def __init__(self, input_path=None, target_path=None,
                 input_h5data=None, target_h5data=None, cube_prios=None, valid_cube_indices=None,
                 border_mode='crop', aniso_factor=2, target_vec_ix=None,
                 target_discrete_ix=None, mean=None, std=None, normalize=True,
                 source='train', patch_shape=None, preview_shape=None,
                 grey_augment_channels=None, warp=False, warp_args=None,
                 ignore_thresh=False, force_dense=False, class_weights=False,
                 epoch_size=100, eager_init=True, cuda_enabled='auto',
                 random_blurring_config=None):
        assert (input_path and target_path and input_h5data and target_h5data)
        if len(input_h5data)!=len(target_h5data):
            raise ValueError("input_h5data and target_h5data must be lists of same length!")
        input_path = os.path.expanduser(input_path)
        target_path = os.path.expanduser(target_path)
        if cuda_enabled == 'auto':
            cuda_enabled = torch.cuda.is_available()
            device = 'GPU' if cuda_enabled else 'CPU'
            logger.info(f'Using {device}.')
        self.cuda_enabled = cuda_enabled
        # batch properties
        self.source = source
        self.grey_augment_channels = grey_augment_channels
        self.warp = warp
        self.warp_args = warp_args
        self.ignore_thresh = ignore_thresh
        self.force_dense = force_dense

        # general properties
        # TODO: Merge *_path with *_h5data, i.e. *_h5data should contain tuples (<full/path/to/hdf5.h5>, <hdf5datasetkey>).
        self.input_path = input_path
        self.target_path = target_path
        # TODO: "*_files" is a bit misleading, because those are actually tuples (filename, h5_key).
        self.input_h5data = input_h5data
        self.target_h5data = target_h5data
        self.cube_prios = cube_prios
        # TODO: Support separate validation data? (Not using indices, but an own validation list)
        self.valid_cube_indices = valid_cube_indices if valid_cube_indices is not None else []
        self.aniso_factor = aniso_factor
        self.border_mode = border_mode
        self.target_vec_ix = target_vec_ix
        self.target_discrete_ix = target_discrete_ix
        self.epoch_size = epoch_size
        self._epoch_size = epoch_size

        # Infer geometric info from input/target shapes
        # HACK
        self.patch_shape = np.array(patch_shape, dtype=np.int)
        self.ndim = self.patch_shape.ndim
        # TODO: Strides and offsets are currently hardcoded. Try to calculate them or at least make them configurable.
        self.strides = np.array([1, 1, 1], dtype=np.int) #np.array(target_node.shape.strides, dtype=np.int)
        self.offsets = np.array([0, 0, 0], dtype=np.int) #np.array(target_node.shape.offsets, dtype=np.int)
        self.target_ps = self.patch_shape - self.offsets * 2
        self.target_dtype = np.int64
        self.mode = 'img-img'  # TODO: what would change for img-scalar? Is that even neccessary?
        # The following will be inferred when reading data
        self.n_labelled_pixels = 0
        self.c_input = None  # Number of input channels
        self.c_target = None  # Number of target channels

        # Actual data fields
        self.valid_inputs = []
        self.valid_targets = []

        self.train_inputs = []
        self.train_targets = []

        if preview_shape is None:
            self.preview_shape = self.patch_shape
        else:
            self.preview_shape = preview_shape
        self._preview_batch = None


        # Setup internal stuff
        self.rng = np.random.RandomState(
            np.uint32((time.time() * 0.0001 - int(time.time() * 0.0001)) * 4294967295)
        )
        self.pid = os.getpid()
        self.gc_count = 1

        self._sampling_weight = None
        self._training_count = None
        self._valid_count = None
        self.n_successful_warp = 0
        self.n_failed_warp = 0
        self.n_read_failures = 0

        self.load_data()
        self._mean = mean
        self._std = std
        self.normalize = normalize
        if eager_init:
            if self.normalize:
                # Pre-compute to prevent later redundant computation in multiple processes.
                _, _ = self.mean, self.std
            # Load preview data on initialization so read errors won't occur late
            # and reading doesn't have to be done by each background worker process separately.
            _ = self.preview_batch
        if class_weights:
            # TODO: This target mean calculation can be expensive. Add support for pre-calculated values, similar to `mean` param. # Not quite sure what you mean, this is done once only anyway
            target_mean = np.mean(self.train_targets)
            bg_weight = target_mean / (1. + target_mean)
            fg_weight = 1. - bg_weight
            self.class_weights = torch.FloatTensor([bg_weight, fg_weight])
            logger.info(f'Calculated class weights: {[bg_weight, fg_weight]}')
            if self.cuda_enabled:
                self.class_weights = self.class_weights.cuda()
        else:
            self.class_weights = None

        self.random_blurring_config = random_blurring_config
        if self.random_blurring_config:
            check_random_data_blurring_config(patch_shape,
                                              **self.random_blurring_config)

    def __getitem__(self, index):
        # use index just as counter, subvolumes will be chosen randomly

        if self.grey_augment_channels is None:
            self.grey_augment_channels = []
        self._reseed()
        input_src, target_src = self._getcube(self.source)  # get cube randomly
        while True:
            try:
                inp, target = self.warp_cut(input_src, target_src, self.warp, self.warp_args)
                target = target.astype(self.target_dtype)
                # Arbitrarily choosing 100 as the threshold here, because we
                # currently can't find out the total number of classes in the
                # data set automatically. The assumption here is that no one
                # wants to use elektronn3 with a data set that actually
                # contains more than 100 classes for the target labels.
                # TODO: Remove this stupid check ASAP once https://github.com/ELEKTRONN/elektronn3/issues/10 is fixed.
                if target.max() > 100:
                    # TODO: Find out where to catch this early / prevent this issue from happening
                    logger.warning(f'invalid target: max = {target.max()}. Skipping batch...')
                    continue
            except transformations.WarpingOOBError:
                self.n_failed_warp += 1
                if self.n_failed_warp > 20 and self.n_failed_warp > 2 * self.n_successful_warp:
                    fail_ratio = self.n_failed_warp / (self.n_failed_warp + self.n_successful_warp)
                    fail_percentage = int(round(100 * fail_ratio))
                    # Note that this warning will be spammed once the conditions are met.
                    # Better than logging it once and risking that it stays unnoticed IMO.
                    logger.warning(
                        f'{fail_percentage}% of warping attempts are failing. '
                        'Consider lowering the warping strength.'
                    )
                continue
            # TODO: Actually find out what's causing those.
            except OSError:
                if self.n_read_failures > self.n_successful_warp:
                    logger.error(
                        'Encountered more OSErrors than successful samples\n'
                        f'(Counted {self.n_read_failures} errors.)\n'
                        'There is probably something wrong with your HDF5 '
                        'files. Aborting...'
                    )
                    raise RuntimeError
                self.n_read_failures += 1
                traceback.print_exc()
                logger.warning(
                    '\nUnhandled OSError while reading data from HDF5 file.\n'
                    f'  input: {input_src.file.filename}\n'
                    f'  target: {target_src.file.filename}\n'
                    'Continuing with next sample. For details, see the '
                    'traceback above.\n'
                )
                continue
            self.n_successful_warp += 1
            if self.normalize:
                inp = (inp - self.mean) / self.std
            if self.grey_augment_channels and self.source == "train":  # grey augmentation only for training
                inp = transformations.grey_augment(inp, self.grey_augment_channels, self.rng)
            break

        # Final modification of targets: striding and replacing nan
        if not (self.force_dense or np.all(self.strides == 1)):
            target = self._stridedtargets(target)

        return inp, target

    def __len__(self):
        return self.epoch_size

    def __repr__(self):
        s = "{0:,d}-target Data Set with {1:,d} input channel(s):\n" + \
            "#train cubes: {2:,d} and #valid cubes: {3:,d}, {4:,d} labelled " + \
            "pixels."
        s = s.format(self.c_target, self.c_input, self._training_count,
                     self._valid_count, self.n_labelled_pixels)
        return s

    # TODO: Support individual per-cube mean/std (in case of high differences
    #       between cubes)? Not sure if this is important.

    # TODO: Respect separate channels
    @property
    def mean(self):
        if self._mean is None:
            logger.warning(
                'Calculating mean of training inputs. This is potentially slow. Please supply\n'
                'it manually when initializing the data set to make startup faster.'
            )
            means = [np.mean(x) for x in self.train_inputs]
            self._mean = np.mean(means)
            logger.info(f'mean = {self._mean:.6f}')
        return self._mean

    # TODO: Respect separate channels
    @property
    def std(self):
        if self._std is None:
            logger.warning(
                'Calculating std of training inputs. This is potentially slow. Please supply\n'
                'it manually when initializing the data set to make startup faster.'
            )
            stds = [np.std(x) for x in self.train_inputs]
            # Note that this is not the same as the std of all train_inputs
            # together. The mean of stds of the individual input data cubes
            # is different because it only acknowledges intra-cube variance,
            # not variance between training cubes.
            # TODO: Does it make sense to have the actual global std of all
            #       training inputs? If yes, how can it be computed without
            #       loading everything into RAM at once?
            self._std = np.mean(stds)
            logger.info(f'std = {self._std:.6f}')
        return self._std

    def validate(self):
        self.source = "valid"
        self.epoch_size = 10

    def train(self):
        self.source = "train"
        self.epoch_size = self._epoch_size

    def _create_preview_batch(
            self,
            inp_source: h5py.Dataset,
            target_source: h5py.Dataset,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:

        # Central slicing
        halfshape = np.array(self.preview_shape) // 2
        if inp_source.ndim == 4:
            inp_shape = np.array(inp_source.shape[1:])
            target_shape = np.array(target_source.shape[1:])
        elif inp_source.ndim == 3:
            inp_shape = np.array(inp_source.shape)
            target_shape = np.array(target_source.shape)
        inp_center = inp_shape // 2
        inp_lo = inp_center - halfshape
        inp_hi = inp_center + halfshape
        target_center = target_shape // 2
        target_lo = target_center - halfshape
        target_hi = target_center + halfshape
        if np.any(inp_center < halfshape):
            raise ValueError(
                'preview_shape is too big for shape of input source.'
                f'Requested {self.preview_shape}, but can only deliver {tuple(inp_shape)}.'
            )
        elif np.any(target_center < halfshape):
            raise ValueError(
                'preview_shape is too big for shape of target source.'
                f'Requested {self.preview_shape}, but can only deliver {tuple(target_shape)}.'
            )
        inp_np = slice_h5(inp_source, inp_lo, inp_hi, prepend_batch_axis=True)
        target_np = slice_h5(
            target_source, target_lo, target_hi,
            dtype=self.target_dtype, prepend_batch_axis=True
        )

        if self.normalize:
            inp_np = ((inp_np - self.mean) / self.std).astype(np.float32)

        inp = torch.from_numpy(inp_np)
        target = torch.from_numpy(target_np)
        if self.cuda_enabled:
            inp = inp.cuda()
            target = target.cuda()
        return inp, target

    # In this implementation the preview batch is always kept in GPU memory.
    # This means much better inference speed when using it, but this may be
    # a bad decision if GPU memory is limited.
    # --> TODO: Document this concern and decide how to deal with it.
    #           (E.g. suggest a smaller preview shape if catching OOM,
    #            or keep the batch in main memory ("cpu") and only move it
    #            to GPU when needed, freeing up GPU memory afterwards
    #            -> first evaluate cost of moving?...)
    #
    # TODO: Make targets optional so we can have larger previews without ground truth targets?

    @property
    def preview_batch(self) -> Tuple[Variable, Variable]:
        if self._preview_batch is None:
            inp, target = self._create_preview_batch(
                self.valid_inputs[0], self.valid_targets[0]
            )  # TODO: Don't hardcode valid_*[0]

            self._preview_batch = (inp, target)
        return self._preview_batch

    @property
    def warp_stats(self):
        return "Warp stats: successful: %i, failed %i, quota: %.1f" %(
            self.n_successful_warp, self.n_failed_warp,
            float(self.n_successful_warp)/(self.n_failed_warp+self.n_successful_warp))

    def _reseed(self):
        """Reseeds the rng if the process ID has changed!"""
        current_pid = os.getpid()
        if current_pid != self.pid:
            logger.debug(f'New worker process started (PID {current_pid})')
            self.pid = current_pid
            self.rng.seed(
                np.uint32((time.time()*0.0001 - int(time.time()*0.0001))*4294967295+self.pid)
            )

    def warp_cut(self, inp_src, target_src, warp, warp_params):
        """
        (Wraps :py:meth:`elektronn3.data.transformations.get_warped_slice()`)

        Cuts a warped slice out of the input and target arrays.
        The same random warping transformation is each applied to both input
        and target.

        Warping is randomly applied with the probability defined by the ``warp``
        parameter (see below).

        Parameters
        ----------
        inp_src: h5py.Dataset
            Input image source (in HDF5)
        target_src: h5py.Dataset
            Target image source (in HDF5)
        warp: float or bool
            False/True disable/enable warping completely.
            If ``warp`` is a float, it is used as the ratio of inputs that
            should be warped.
            E.g. 0.5 means approx. every second call to this function actually
            applies warping to the image-target pair.
        warp_params: dict
            kwargs that are passed through to
            :py:meth:`elektronn2.data.transformations.get_warped_slice()`.
            Can be empty.

        Returns
        -------
        inp: np.ndarray
            (Warped) input image slice
        target_src: np.ndarray
            (Warped) target slice
        """
        if (warp is True) or (warp == 1):  # always warp
            do_warp = True
        elif 0 < warp < 1:  # warp only a fraction of examples
            do_warp = True if (self.rng.rand() < warp) else False
        else:  # never warp
            do_warp = False

        if not do_warp:
            warp_params = dict(warp_params)
            warp_params['warp_amount'] = 0

        inp, target = transformations.get_warped_slice(
            inp_src,
            self.patch_shape,
            aniso_factor=self.aniso_factor,
            target_src=target_src,
            target_ps=self.target_ps,
            target_vec_ix=self.target_vec_ix,
            target_discrete_ix=self.target_discrete_ix,
            rng=self.rng,
            **warp_params
        )

        return inp, target

    def _getcube(self, source):
        """
        Draw an example cube according to sampling weight on training data,
        or randomly on valid data
        """
        if source == 'train':
            p = self.rng.rand()
            i = np.flatnonzero(self._sampling_weight <= p)[-1]
            inp_source, target_source = self.train_inputs[i], self.train_targets[i]
        elif source == "valid":
            if len(self.valid_inputs) == 0:
                raise ValueError("No validation set")

            i = self.rng.randint(0, len(self.valid_inputs))
            inp_source = self.valid_inputs[i]
            target_source = self.valid_targets[i]
        else:
            raise ValueError("Unknown data source")

        return inp_source, target_source

    def _stridedtargets(self, target):
        return target[:, :, ::self.strides[0], ::self.strides[1], ::self.strides[2]]

    def load_data(self):
        inp_files, target_files = self.open_files()

        prios = []
        # Distribute Cubes into training and valid list
        for k, (inp, target) in enumerate(zip(inp_files, target_files)):
            if k in self.valid_cube_indices:
                self.valid_inputs.append(inp)
                self.valid_targets.append(target)
            else:
                self.train_inputs.append(inp)
                self.train_targets.append(target)
                # If no priorities are given: sample proportional to cube size
                prios.append(target.size)

        if self.cube_prios is None:
            prios = np.array(prios, dtype=np.float)
        else:  # If priorities are given: sample irrespective of cube size
            prios = np.array(self.cube_prios, dtype=np.float)

        # sample example i if: batch_prob[i] < p
        self._sampling_weight = np.hstack((0, np.cumsum(prios / prios.sum())))
        self._training_count = len(self.train_inputs)
        self._valid_count = len(self.valid_inputs)

    def check_files(self):  # TODO: Update for cdhw version
        """
        Check if all files are accessible.
        """
        notfound = False
        give_neuro_data_hint = False
        fullpaths = [os.path.join(self.input_path, f) for f, _ in self.input_h5data] + \
                    [os.path.join(self.target_path, f) for f, _ in self.target_h5data]
        for p in fullpaths:
            if not os.path.exists(p):
                print('{} not found.'.format(p))
                notfound = True
                if 'neuro_data_zxy' in p:
                    give_neuro_data_hint = True
        if give_neuro_data_hint:
            print('\nIt looks like you are referencing the neuro_data_zxy dataset.\n'
                  'To install the neuro_data_xzy dataset to the default location, run:\n'
                  '  $ wget http://elektronn.org/downloads/neuro_data_zxy.zip\n'
                  '  $ unzip neuro_data_zxy.zip -d ~/neuro_data_zxy')
        if notfound:
            print('\nPlease fetch the necessary dataset and/or '
                  'change the relevant file paths in the network config.')
            sys.stdout.flush()
            sys.exit(1)

    def open_files(self):
        self.check_files()
        inp_h5sets, target_h5sets = [], []

        print('\nUsing data sets:')
        for (inp_fname, inp_key), (target_fname, target_key) in zip(self.input_h5data, self.target_h5data):
            inp_h5 = h5py.File(os.path.join(self.input_path, inp_fname), 'r')[inp_key]
            target_h5 = h5py.File(os.path.join(self.target_path, target_fname), 'r')[target_key]

            # assert inp_h5.ndim == 4
            # assert target_h5.ndim == 4
            if inp_h5.ndim == 4:
                self.c_input = inp_h5.shape[0]
                self.c_target = target_h5.shape[0]
                self.n_labelled_pixels += target_h5[0].size
            elif inp_h5.ndim == 3:
                self.c_input = 1
                self.c_target = 1
                self.n_labelled_pixels += target_h5.size
            print(f'  input:       {inp_fname}[{inp_key}]: {inp_h5.shape} ({inp_h5.dtype})')
            print(f'  with target: {target_fname}[{target_key}]: {target_h5.shape} ({target_h5.dtype})')
            inp_h5sets.append(inp_h5)
            target_h5sets.append(target_h5)
        print()

        return inp_h5sets, target_h5sets
