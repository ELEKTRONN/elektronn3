# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

__all__ = ['PatchCreator', 'SimpleNeuroData2d']

import logging
import os
import sys
import time
import traceback
from os.path import expanduser
from typing import Tuple, Dict, Optional, Union, Sequence, Any, List, Callable

import h5py
import numpy as np
import torch
from torch.utils import data

from elektronn3.data import transformations, transforms  # TODO: Rename transformations module
from elektronn3.data.utils import slice_h5

logger = logging.getLogger('elektronn3log')


class PatchCreator(data.Dataset):
    """Dataset iterator class that creates 3D image patches from HDF5 files.

    It implements the PyTorch ``Dataset`` interface and is meant to be used
    with a PyTorch ``DataLoader`` (or the modified
    :py:class:`elektronn3.training.trainer.train_utils.DelayedDataLoader``, if it is
    used with :py:class:`elektronn3.training.trainer.Trainer``).

    The main idea of this class is to automate input and target patch creation
    for training convnets for semantic image segmentation. Patches are sliced
    from random locations in the supplied HDF5 files (``input_h5data``,
    ``target_h5data``).
    Optionally, the source coordinates from which patches
    are sliced are obtained by random warping with affine or perspective
    transformations for efficient augmentation and avoiding border artifacts
    (see ``warp``, ``warp_kwargs``).
    Note that whereas other warping-based image augmentation systems usually
    warp images themselves, elektronn3 performs warping transformations on
    the **coordinates** from which image patches are sliced and obtains voxel
    values by interpolating between actual image voxels at the warped source
    locations (which are not confined to the original image's discrete
    coordinate grid).
    (TODO: A visualization would be very helpful here to make this more clear)
    For more information about this warping mechanism see
    :py:meth:`elektronn3.data.cnndata.warp_slice()`.

    Currently, only 3-dimensional image data sets are supported, but 2D
    support is also planned.

    Args:
        input_h5data: Sequence of ``(filename, hdf5_key)`` tuples, where
            each item specifies the filename and
            the HDF5 dataset key under which the input data is stored.
        target_h5data: Sequence of ``(filename, hdf5_key)`` tuples, where
            each item specifies the filename and
            the HDF5 dataset key under which the target data is stored.
        patch_shape: Desired spatial shape of the samples that the iterator
            delivers by slicing from the data set files.
            Since this determines the size of input samples that are fed
            into the neural network, this is a very important value to tune.
            Making it too large can result in slow training and excessive
            memory consumption, but if it is too small, it can hinder the
            perceptive ability of the neural network because the samples it
            "sees" get too small to extract meaningful features.
            Adequate values for ``patch_shape`` are highly dependent on the
            data set ("How large are typical ROIs? How large does an image
            patch need to be so you can understand the input?") and also
            depend on the neural network architecture to be used (If the
            effective receptive field of the network is small, larger patch
            sizes won't help much).
        cube_prios: List of per-cube priorities, where a higher priority
            means that it is more likely that a sample comes from this cube.
        aniso_factor: Depth-anisotropy factor of the data set. E.g.
            if your data set has half resolution in the depth dimension,
            set ``aniso_factor=2``. If all dimensions have the same
            resolution, set ``aniso_factor=1``.
        target_discrete_ix: List of target channels that contain discrete values.
            By default (``None``), every channel is is seen as discrete (this is
            generally the case for classification tasks).
            This information is used to decide what kind of interpolation should
            be used for reading target data:
            - discrete targets are obtained by nearest-neighbor interpolation
            - non-discrete (continuous) targets are linearly interpolated.
        train: Determines if samples come from training or validation
            data.
            If ``True``, training data is returned.
            If ``False``, validation data is returned.
        preview_shape: Desired spatial shape of the dedicated preview batch.
            The preview batch is obtained by slicing a patch of this
            shape out of the center of the preview cube.
            If it is ``None`` (default), preview batch functionality will be
            disabled.
        warp: ratio of training samples that should be obtained using
            geometric warping augmentations.
        warp_kwargs: kwargs that are passed through to
            :py:meth:`elektronn3.data.transformations.get_warped_slice()`.
            See the docs of this function for information on kwargs options.
            Can be empty.
        epoch_size: Determines the length (``__len__``) of the ``Dataset``
            iterator. ``epoch_size`` can be set to an arbitrary value and
            doesn't have any effect on the content of produced training
            samples. It is recommended to set it to a suitable value for
            one "training phase", so after each ``epoch_size`` batches,
            validation/logging/plotting are performed by the training loop
            that uses this data set (e.g.
            ``elektronn3.training.trainer.Trainer``).
        squeeze_target: If ``True``, target tensors will be squeezed in their
            channel axis if it is empty. This workaround and will be removed
            later. It is currently needed to support targets that have an
            extra channel axis which doesn't exist in the network outputs.
    """
    def __init__(
            self,
            input_h5data: List[Tuple[str, str]],
            target_h5data: List[Tuple[str, str]],
            patch_shape: Sequence[int],
            cube_prios: Optional[Sequence[float]] = None,
            aniso_factor: int = 2,
            target_discrete_ix: Optional[List[int]] = None,
            train: bool = True,
            preview_shape: Optional[Sequence[int]] = None,
            warp: Union[bool, float] = False,
            warp_kwargs: Optional[Dict[str, Any]] = None,
            epoch_size: int = 100,
            squeeze_target: bool = False,
            transform: Callable = transforms.Identity(),
    ):
        # Early checks
        if len(input_h5data) != len(target_h5data):
            raise ValueError("input_h5data and target_h5data must be lists of same length!")
        if not train:
            if warp:
                logger.warning(
                    'Augmentations should not be used on validation data.'
                )
        else:
            if preview_shape is not None:
                raise ValueError()

        # batch properties
        self.train = train
        self.warp = warp
        self.warp_kwargs = warp_kwargs
        self.squeeze_target = squeeze_target
        # TODO: Instead of overly specific hacks like squeeze_target, we should
        #       make "transformations" like this fully customizable, similar to
        #       the `torchvision.transforms` interface.
        #       E.g. squeeze_target could then be implemented as a
        #       `lambda x: x.squeeze(0)` transformation that can be combined
        #       with others. Non-geometric augmentations and normalization could
        #       also be implemented as pluggable transformations.

        # general properties
        input_h5data = [(expanduser(fn), key) for (fn, key) in input_h5data]
        target_h5data = [(expanduser(fn), key) for (fn, key) in target_h5data]
        self.input_h5data = input_h5data
        self.target_h5data = target_h5data
        self.cube_prios = cube_prios
        # TODO: Support separate validation data? (Not using indices, but an own validation list)
        self.aniso_factor = aniso_factor
        self.target_discrete_ix = target_discrete_ix
        self.epoch_size = epoch_size
        self._orig_epoch_size = epoch_size  # Store original epoch_size so it can be reset later.

        self.patch_shape = np.array(patch_shape, dtype=np.int)
        self.ndim = self.patch_shape.ndim
        # TODO: Make strides and offsets for targets configurable
        # self.strides = ...
        #  strides will need to be applied *during* dataset iteration now
        #  (-> strided reading in slice_h5()... or should strides be applied
        #   with some fancy downscaling operator? Naively strided reading
        #   could mess up targets in unfortunate cases:
        #   e.g. ``[0, 1, 0, 1, 0, 1][::2] == [0, 0, 0]``, discarding all 1s).
        self.offsets = np.array([0, 0, 0])
        self.target_ps = self.patch_shape  - self.offsets * 2
        self._target_dtype = np.int64
        self.mode = 'img-img'  # TODO: what would change for img-scalar? Is that even neccessary?
        # The following will be inferred when reading data
        self.n_labelled_pixels = 0
        self.c_input = None  # Number of input channels
        self.c_target = None  # Number of target channels

        # Actual data fields
        self.inputs = []
        self.targets = []

        self.preview_shape = preview_shape
        self._preview_batch = None


        # Setup internal stuff
        self.rng = np.random.RandomState(
            np.uint32((time.time() * 0.0001 - int(time.time() * 0.0001)) * 4294967295)
        )
        self.pid = os.getpid()

        self._sampling_weight = None
        self._training_count = None
        self._count = None
        self.n_successful_warp = 0
        self.n_failed_warp = 0
        self.n_read_failures = 0

        self.load_data()  # Open dataset files

        if transform is None:
            transform = lambda x: x
        self.transform = transform

        # Load preview data on initialization so read errors won't occur late
        # and reading doesn't have to be done by each background worker process separately.
        _ = self.preview_batch

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Note that the index is ignored. Samples are always random
        return self._get_random_sample()

    def _get_random_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        #                                      np.float32, self._target_dtype
        # use index just as counter, subvolumes will be chosen randomly

        self._reseed()
        input_src, target_src = self._getcube()  # get cube randomly
        while True:
            try:
                # TODO: Limit validation data warping
                inp, target = self.warp_cut(input_src, target_src, self.warp, self.warp_kwargs)
                target = target.astype(self._target_dtype)
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
                        f'{fail_percentage}% of warping attempts are failing.\n'
                        'Consider lowering lowering warp_kwargs[\'warp_amount\']).'
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
            inp, target = self.transform(inp, target)
            break

        # target is now of shape (K, D, H, W), where K is the number of
        #  target channels (not to be confused with the number of classes
        #  for the classification problem, C.
        if self.squeeze_target:
            # If K == 1, K is squeezed here to match common network output
            #  shapes (which usually lack a K axis).
            # Make sure it's actually the channel axis we're squeezing here,
            #  not a spatial dimension that's coincidentally of size 1:
            assert len(self.target_ps) == target_src.ndim - 1
            target = target.squeeze(0)  # (K, (D,) H, W) -> ((D,) H, W)

        # inp, target are still numpy arrays here. Relying on auto-conversion to
        #  torch Tensors by the ``collate_fn`` of the ``DataLoader``.
        return inp, target

    def __len__(self) -> int:
        return self.epoch_size

    def __repr__(self) -> str:
        s = "{0:,d}-target Data Set with {1:,d} input channel(s):\n" + \
            "#train cubes: {2:,d} and #valid cubes: {3:,d}, {4:,d} labelled " + \
            "pixels."
        s = s.format(self.c_target, self.c_input, self._training_count,
                     self._valid_count, self.n_labelled_pixels)
        return s

    def _create_preview_batch(
            self,
            inp_source: h5py.Dataset,
            target_source: h5py.Dataset,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
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
        inp_np = slice_h5(inp_source, inp_lo, inp_hi, prepend_empty_axis=True)
        target_np = slice_h5(
            target_source, target_lo, target_hi,
            dtype=self._target_dtype, prepend_empty_axis=True
        )
        inp_np, target_np = self.transform(inp_np, target_np)

        inp = torch.from_numpy(inp_np)
        target = torch.from_numpy(target_np)

        # See comments at the end of PatchCreator.__getitem__()
        # Note that here it's the dimension index 1 that we're squeezing,
        #  because index 0 is the batch dimension.
        if self.squeeze_target:
            assert len(self.target_ps) == target_source.ndim - 1
            target = target.squeeze(1)  # (N, K, (D,), H, W) -> (N, (D,) H, W)

        return inp, target

    # TODO: Make targets optional so we can have larger previews without ground truth targets?

    @property
    def preview_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._preview_batch is None and self.preview_shape is not None:
            inp, target = self._create_preview_batch(
                self.inputs[0], self.targets[0]
            )  # TODO: Don't hardcode [0]

            self._preview_batch = (inp, target)
        return self._preview_batch

    @property
    def warp_stats(self) -> str:
        return "Warp stats: successful: %i, failed %i, quota: %.1f" %(
            self.n_successful_warp, self.n_failed_warp,
            float(self.n_successful_warp)/(self.n_failed_warp+self.n_successful_warp))

    def _reseed(self) -> None:
        """Reseeds the rng if the process ID has changed!"""
        current_pid = os.getpid()
        if current_pid != self.pid:
            logger.debug(f'New worker process started (PID {current_pid})')
            self.pid = current_pid
            self.rng.seed(
                np.uint32((time.time()*0.0001 - int(time.time()*0.0001))*4294967295+self.pid)
            )

    def warp_cut(
            self,
            inp_src: h5py.Dataset,
            target_src: h5py.Dataset,
            warp: Union[float, bool],
            warp_kwargs: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        warp_kwargs: dict
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
            warp_kwargs = dict(warp_kwargs)
            warp_kwargs['warp_amount'] = 0

        inp, target = transformations.get_warped_slice(
            inp_src,
            self.patch_shape,
            aniso_factor=self.aniso_factor,
            target_src=target_src,
            target_ps=self.target_ps,
            target_discrete_ix=self.target_discrete_ix,
            rng=self.rng,
            **warp_kwargs
        )

        return inp, target

    def _getcube(self) -> Tuple[h5py.Dataset, h5py.Dataset]:
        """
        Draw an example cube according to sampling weight on training data,
        or randomly on valid data
        """
        if self.train:
            p = self.rng.rand()
            i = np.flatnonzero(self._sampling_weight <= p)[-1]
            inp_source, target_source = self.inputs[i], self.targets[i]
        else:
            if len(self.inputs) == 0:
                raise ValueError("No validation set")

            # TODO: Sampling weight for validation data?
            i = self.rng.randint(0, len(self.inputs))
            inp_source, target_source = self.inputs[i], self.targets[i]

        return inp_source, target_source

    def load_data(self) -> None:
        inp_files, target_files = self.open_files()

        prios = []
        # Distribute Cubes into training and valid list
        for k, (inp, target) in enumerate(zip(inp_files, target_files)):
            self.inputs.append(inp)
            self.targets.append(target)
            # If no priorities are given: sample proportional to cube size
            prios.append(target.size)

        if self.cube_prios is None:
            prios = np.array(prios, dtype=np.float)
        else:  # If priorities are given: sample irrespective of cube size
            prios = np.array(self.cube_prios, dtype=np.float)

        # sample example i if: batch_prob[i] < p
        self._sampling_weight = np.hstack((0, np.cumsum(prios / prios.sum())))
        self._count = len(self.inputs)

    def check_files(self) -> None:  # TODO: Update for cdhw version
        """
        Check if all files are accessible.
        """
        notfound = False
        give_neuro_data_hint = False
        fullpaths = [f for f, _ in self.input_h5data] + \
                    [f for f, _ in self.target_h5data]
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

    def open_files(self) -> Tuple[List[h5py.Dataset], List[h5py.Dataset]]:
        self.check_files()
        inp_h5sets, target_h5sets = [], []
        modestr = 'Training' if self.train else 'Validation'
        print(f'\n{modestr} data set:')
        for (inp_fname, inp_key), (target_fname, target_key) in zip(self.input_h5data, self.target_h5data):
            inp_h5 = h5py.File(inp_fname, 'r')[inp_key]
            target_h5 = h5py.File(target_fname, 'r')[target_key]

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


class SimpleNeuroData2d(data.Dataset):
    """ 2D Dataset class for neuro_data_cdhw, reading from a single HDF5 file.

    Delivers 2D image slices from the (H, W) plane at given D indices.
    Not scalable, keeps everything in memory.
    This is just a minimalistic proof of concept.
    """

    def __init__(
            self,
            inp_path=None,
            target_path=None,
            train=True,
            inp_key='raw', target_key='lab',
            # offset=(0, 0, 0),
            pool=(1, 1, 1)
    ):
        super().__init__()
        cube_id = 0 if train else 2
        if inp_path is None:
            inp_path = expanduser(f'~/neuro_data_cdhw/raw_{cube_id}.h5')
        if target_path is None:
            target_path = expanduser(f'~/neuro_data_cdhw/barrier_int16_{cube_id}.h5')
        self.inp_file = h5py.File(os.path.expanduser(inp_path), 'r')
        self.target_file = h5py.File(os.path.expanduser(target_path), 'r')
        self.inp = self.inp_file[inp_key].value.astype(np.float32) / 255
        self.target = self.target_file[target_key].value.astype(np.int64)
        self.target = self.target[0]  # Squeeze superfluous first dimension

        self.target = self.target[::pool[0], ::pool[1], ::pool[2]]  # Handle pooling (dirty hack TODO)

        # Cut inp and target to same size
        inp_shape = np.array(self.inp.shape[1:])
        target_shape = np.array(self.target.shape)
        diff = inp_shape - target_shape
        offset = diff // 2  # offset from image boundaries

        self.inp = self.inp[
            :,
            offset[0]: inp_shape[0] - offset[0],
            offset[1]: inp_shape[1] - offset[1],
            offset[2]: inp_shape[2] - offset[2],
        ]

        self.close_files()  # Using file contents from memory -> no need to keep the file open.

    def __getitem__(self, index):
        # Get z slices
        inp = self.inp[:, index]
        target = self.target[index]
        return inp, target

    def __len__(self):
        return self.target.shape[0]

    def close_files(self):
        self.inp_file.close()
        self.target_file.close()
