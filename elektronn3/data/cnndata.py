# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

__all__ = ['PatchCreator', 'SimpleNeuroData2d', 'Segmentation2d', 'Reconstruction2d']

import logging
import os
import sys
import time
import traceback
from os.path import expanduser
from typing import Tuple, Dict, Optional, Union, Sequence, Any, List, Callable

import h5py
import imageio
import numpy as np
import torch
from torch.utils import data

from elektronn3.data import coord_transforms, transforms
from elektronn3.data.utils import slice_h5

logger = logging.getLogger('elektronn3log')


class PatchCreator(data.Dataset):
    """Dataset iterator class that creates 3D image patches from HDF5 files.

    It implements the PyTorch ``Dataset`` interface and is meant to be used
    with a PyTorch ``DataLoader`` (or the modified
    :py:class:`elektronn3.training.trainer.train_utils.DelayedDataLoader`, if it is
    used with :py:class:`elektronn3.training.trainer.Trainer``).

    The main idea of this class is to automate input and target patch creation
    for training convnets for semantic image segmentation. Patches are sliced
    from random locations in the supplied HDF5 files (``input_h5data``,
    ``target_h5data``).
    Optionally, the source coordinates from which patches
    are sliced are obtained by random warping with affine or perspective
    transformations for efficient augmentation and avoiding border artifacts
    (see ``warp_prob``, ``warp_kwargs``).
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
        offset: Shape of the offset by which each the targets are cropped
            on each side. This needs to be set if the outputs of the network
            you train with are smaller than its inputs.
            For example, if the spatial shape of your inputs is
            ``patch_shape=(48, 96, 96)`` the spatial shape of your outputs is
            ``out_shape=(32, 56, 56)``, you should set ``offset=(8, 20, 20)``,
            because ``offset = (patch_shape - out_shape) / 2`` should always
            hold true.
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
        target_dtype: dtype that target tensors should be cast to.
        train: Determines if samples come from training or validation
            data.
            If ``True``, training data is returned.
            If ``False``, validation data is returned.
        warp_prob: ratio of training samples that should be obtained using
            geometric warping augmentations.
        warp_kwargs: kwargs that are passed through to
            :py:meth:`elektronn3.data.coord_transforms.get_warped_slice()`.
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
        transform: Transformation function to be applied to ``(inp, target)``
            samples (for normalization, data augmentation etc.). The signature
            is always ``inp, target = transform(inp, target)``, where ``inp``
            and ``target`` both are ``numpy.ndarray``s.
            To combine multiple transforms, use
            :py:class:`elektronn3.data.transforms.Compose`.
            See :py:mod:`elektronn3.data.transforms`. for some implementations.
        num_classes: The total number of different target classes that exist
            in the data set. Setting this is optional, but some features might
            only work if this is specified.
        in_memory: If ``True``, all data set files are immediately loaded
            into host memory and are permanently kept there as numpy arrays.
            If this is disabled (default), file contents are always read from
            the HDF5 files to produce samples. (Note: This does not mean it's
            slower, because file contents are transparently cached by h5py,
            see http://docs.h5py.org/en/latest/high/file.html#chunk-cache).

    """
    def __init__(
            self,
            input_h5data: List[Tuple[str, str]],
            target_h5data: List[Tuple[str, str]],
            patch_shape: Sequence[int],
            offset: Sequence[int] = (0, 0, 0),
            cube_prios: Optional[Sequence[float]] = None,
            aniso_factor: int = 2,
            target_discrete_ix: Optional[List[int]] = None,
            target_dtype: np.dtype = np.int64,
            train: bool = True,
            warp_prob: Union[bool, float] = False,
            warp_kwargs: Optional[Dict[str, Any]] = None,
            epoch_size: int = 100,
            transform: Callable = transforms.Identity(),
            num_classes: Optional[int] = None,
            in_memory: bool = False,
            cube_meta = 1,
    ):
        # Early checks
        if len(input_h5data) != len(target_h5data):
            raise ValueError("input_h5data and target_h5data must be lists of same length!")
        if not train:
            if warp_prob > 0:
                logger.warning(
                    'Augmentations should not be used on validation data.'
                )

        # batch properties
        self.train = train
        self.warp_prob = warp_prob
        self.warp_kwargs = warp_kwargs if warp_kwargs is not None else {}

        # general properties
        input_h5data = [(expanduser(fn), key) for (fn, key) in input_h5data]
        target_h5data = [(expanduser(fn), key) for (fn, key) in target_h5data]
        self.input_h5data = input_h5data
        self.target_h5data = target_h5data
        self.cube_meta = cube_meta
        self.cube_prios = cube_prios
        self.aniso_factor = aniso_factor
        self.target_discrete_ix = target_discrete_ix
        self.epoch_size = epoch_size
        self._orig_epoch_size = epoch_size  # Store original epoch_size so it can be reset later.
        self.num_classes = num_classes
        self.in_memory = in_memory

        self.patch_shape = np.array(patch_shape, dtype=np.int)
        self.ndim = self.patch_shape.ndim
        self.offset = np.array(offset)
        self.target_patch_size = self.patch_shape - self.offset * 2
        self._target_dtype = target_dtype
        self.transform = transform

        # Setup internal stuff
        self.pid = os.getpid()

        # The following fields will be filled when reading data
        self.n_labelled_pixels = 0
        self.inputs = []
        self.targets = []
        self._sampling_weight = None

        self.load_data()  # Open dataset files

        self.n_successful_warp = 0
        self.n_failed_warp = 0
        self.n_read_failures = 0

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Note that the index is ignored. Samples are always random
        return self._get_random_sample()

    def _get_random_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        #                                 np.float32, self._target_dtype
        # use index just as counter, subvolumes will be chosen randomly

        input_src, target_src, i = self._getcube()  # get cube randomly
        warp_prob = self.warp_prob
        while True:
            try:
                inp, target = self.warp_cut(input_src, target_src, warp_prob, self.warp_kwargs)
                target = target.astype(self._target_dtype)
            except coord_transforms.WarpingOOBError:
                # Temporarily set warp_prob to 1 to make sure that the next attempt
                #  will also try to use warping. Otherwise, self.warp_prob would not
                #  reflect the actual probability of a sample being obtained by warping.
                warp_prob = 1
                self.n_failed_warp += 1
                if self.n_failed_warp > 20 and self.n_failed_warp > 8 * self.n_successful_warp:
                    fail_ratio = self.n_failed_warp / (self.n_failed_warp + self.n_successful_warp)
                    fail_percentage = int(round(100 * fail_ratio))
                    # Note that this warning will be spammed once the conditions are met.
                    # Better than logging it once and risking that it stays unnoticed IMO.
                    logger.warning(
                        f'{fail_percentage}% of warping attempts are failing.\n'
                        'Consider lowering lowering warp_kwargs[\'warp_amount\']).'
                    )
                continue
            except coord_transforms.WarpingSanityError:
                logger.exception('Invalid coordinate values encountered while warping. Retrying...')
                continue
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
            try:
                inp, target = self.transform(inp, target)
            except transforms._DropSample:
                # A filter transform has chosen to drop this sample, so skip it
                logger.debug('Sample dropped.')
                continue
            break

        # inp, target are still numpy arrays here. Relying on auto-conversion to
        #  torch Tensors by the ``collate_fn`` of the ``DataLoader``.
        multi_class_target = target.argmax(axis=0) if len(target.shape) > 3 else target
        return inp, target, multi_class_target, self.cube_meta[i], os.path.basename(self.input_h5data[i][0])

    def __len__(self) -> int:
        return self.epoch_size

    # TODO: Write a good __repr__(). The version below is completely outdated.
    # def __repr__(self) -> str:
    #     s = "{0:,d}-target Data Set with {1:,d} input channel(s):\n" + \
    #         "#train cubes: {2:,d} and #valid cubes: {3:,d}, {4:,d} labelled " + \
    #         "pixels."
    #     s = s.format(self.c_target, self.c_input, self._training_count,
    #                  self._valid_count, self.n_labelled_pixels)
    #     return s

    @property
    def warp_stats(self) -> str:
        return "Warp stats: successful: %i, failed %i, quota: %.1f" %(
            self.n_successful_warp, self.n_failed_warp,
            float(self.n_successful_warp)/(self.n_failed_warp+self.n_successful_warp))

    def warp_cut(
            self,
            inp_src: h5py.Dataset,
            target_src: h5py.Dataset,
            warp_prob: Union[float, bool],
            warp_kwargs: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        (Wraps :py:meth:`elektronn3.data.coord_transforms.get_warped_slice()`)

        Cuts a warped slice out of the input and target arrays.
        The same random warping transformation is each applied to both input
        and target.

        Warping is randomly applied with the probability defined by the ``warp_prob``
        parameter (see below).

        Parameters
        ----------
        inp_src: h5py.Dataset
            Input image source (in HDF5)
        target_src: h5py.Dataset
            Target image source (in HDF5)
        warp_prob: float or bool
            False/True disable/enable warping completely.
            If ``warp_prob`` is a float, it is used as the ratio of inputs that
            should be warped.
            E.g. 0.5 means approx. every second call to this function actually
            applies warping to the image-target pair.
        warp_kwargs: dict
            kwargs that are passed through to
            :py:meth:`elektronn2.data.coord_transforms.get_warped_slice()`.
            Can be empty.

        Returns
        -------
        inp: np.ndarray
            (Warped) input image slice
        target_src: np.ndarray
            (Warped) target slice
        """
        if (warp_prob is True) or (warp_prob == 1):  # always warp
            do_warp = True
        elif 0 < warp_prob < 1:  # warp only a fraction of examples
            do_warp = True if (np.random.rand() < warp_prob) else False
        else:  # never warp
            do_warp = False

        if not do_warp:
            warp_kwargs = dict(warp_kwargs)
            warp_kwargs['warp_amount'] = 0

        M = coord_transforms.get_warped_coord_transform(
            inp_src_shape=inp_src.shape,
            patch_shape=self.patch_shape,
            aniso_factor=self.aniso_factor,
            target_src_shape=target_src.shape,
            target_patch_shape=self.target_patch_size,
            **warp_kwargs
        )

        inp, target = coord_transforms.warp_slice(
            inp_src=inp_src,
            patch_shape=self.patch_shape,
            M=M,
            target_src=target_src,
            target_patch_shape=self.target_patch_size,
            target_discrete_ix=self.target_discrete_ix
        )

        return inp, target

    def _getcube(self) -> Tuple[h5py.Dataset, h5py.Dataset]:
        """
        Draw an example cube according to sampling weight on training data,
        or randomly on valid data
        """
        if self.train:
            i = np.random.choice(np.arange(self._sampling_weight.size), p=self._sampling_weight)
            inp_source, target_source = self.inputs[i], self.targets[i]
        else:
            if len(self.inputs) == 0:
                raise ValueError("No validation set")

            # TODO: Sampling weight for validation data?
            i = np.random.randint(0, len(self.inputs))
            inp_source, target_source = self.inputs[i], self.targets[i]

        return inp_source, target_source, i

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

        self._sampling_weight = prios / prios.sum()
        logger.debug(f'prios = {prios}, sampling_weight = {self._sampling_weight}')

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
                if 'neuro_data_cdhw' in p:
                    give_neuro_data_hint = True
        if give_neuro_data_hint:
            print('\nIt looks like you are referencing the neuro_data_cdhw dataset.\n'
                  'To install the neuro_data_xzy dataset to the default location, run:\n'
                  '  $ wget https://github.com/ELEKTRONN/elektronn.github.io/releases/download/neuro_data_cdhw/neuro_data_cdhw.zip\n'
                  '  $ unzip neuro_data_cdhw.zip -d ~/neuro_data_cdhw')
        if notfound:
            print('\nPlease fetch the necessary dataset and/or '
                  'change the relevant file paths in the network config.')
            sys.stdout.flush()
            sys.exit(1)

    def open_files(self) -> Tuple[List[h5py.Dataset], List[h5py.Dataset]]:
        self.check_files()
        inp_h5sets, target_h5sets = [], []
        modestr = 'Training' if self.train else 'Validation'
        memstr = ' (in memory)' if self.in_memory else ''
        logger.info(f'\n{modestr} data set{memstr}:')
        for (inp_fname, inp_key), (target_fname, target_key), cube_meta in zip(self.input_h5data, self.target_h5data, self.cube_meta):
            inp_h5_file = h5py.File(inp_fname, 'r')
            inp_h5_data = inp_h5_file[inp_key]#[:, 50:-50, 100:-100, 100:-100]
            target_h5_file = h5py.File(target_fname, 'r')
            target_h5_data = target_h5_file[target_key]
            if self.in_memory:
                # Get copies of the dataset contents as in-memory numpy arrays
                inp_h5_val = inp_h5_data[()]
                inp_h5_file.close()
                inp_h5_data = inp_h5_val
                target_h5_val = target_h5_data[()]
                target_h5_file.close()
                target_h5_data = target_h5_val

            logger.info(f'  input:       {inp_fname}[{inp_key}]: {inp_h5_data.shape} ({inp_h5_data.dtype})')
            logger.info(f'  with target: {target_fname}[{target_key}]: {target_h5_data.shape} ({target_h5_data.dtype})')
            logger.info(f'  cube_meta:   {cube_meta}')
            inp_h5sets.append(inp_h5_data)
            target_h5sets.append(target_h5_data)
        print()

        return inp_h5sets, target_h5sets


def get_preview_batch(
        h5data: Tuple[str, str],
        preview_shape: Optional[Tuple[int, ...]] = None,
        transform: Callable = transforms.Identity(),
        in_memory: bool = False
) -> torch.Tensor:
    fname, key = h5data
    inp_h5 = h5py.File(fname, 'r')[key]
    if in_memory:
        inp_h5 = inp_h5.value
    dim = len(preview_shape)  # 2D or 3D
    inp_shape = np.array(inp_h5.shape[-dim:])
    if preview_shape is None:  # Slice everything
        inp_lo = np.zeros_like(inp_shape)
        inp_hi = inp_shape
    else:  # Slice only a preview_shape-sized region from the center of the input
        halfshape = np.array(preview_shape) // 2
        inp_center = inp_shape // 2
        inp_lo = inp_center - halfshape
        inp_hi = inp_center + halfshape
        if np.any(inp_center < halfshape):
            raise ValueError(
                'preview_shape is too big for shape of input source.'
                f'Requested {preview_shape}, but can only deliver {tuple(inp_shape)}.'
            )
    memstr = ' (in memory)' if in_memory else ''
    logger.info(f'\nPreview data{memstr}:')
    logger.info(f'  input:       {fname}[{key}]: {inp_h5.shape} ({inp_h5.dtype})\n')
    inp_np = slice_h5(inp_h5, inp_lo, inp_hi, prepend_empty_axis=True)
    if inp_np.ndim == dim + 1:  # Should be dim + 2 for (N, C) dims
        inp_np = inp_np[:, None]  # Add missing C dim
    inp_np, _ = transform(inp_np, None)
    inp = torch.from_numpy(inp_np)
    return inp


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
            inp_key='raw',
            target_key='lab',
            # offset=(0, 0, 0),
            pool=(1, 1, 1),
            transform: Callable = transforms.Identity(),
            num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.transform = transform
        self.num_classes = num_classes
        cube_id = 0 if train else 2
        if inp_path is None:
            inp_path = expanduser(f'~/neuro_data_cdhw/raw_{cube_id}.h5')
        if target_path is None:
            target_path = expanduser(f'~/neuro_data_cdhw/barrier_int16_{cube_id}.h5')
        self.inp_file = h5py.File(os.path.expanduser(inp_path), 'r')
        self.target_file = h5py.File(os.path.expanduser(target_path), 'r')
        self.inp = self.inp_file[inp_key].value.astype(np.float32)
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
        inp, target = self.transform(inp, target)
        return inp, target

    def __len__(self):
        return self.target.shape[0]

    def close_files(self):
        self.inp_file.close()
        self.target_file.close()


# TODO: docs, types
class Segmentation2d(data.Dataset):
    """Simple dataset for 2d segmentation.

    Expects a list of ``input_paths`` and ``target_paths`` where
    ``target_paths[i]`` is the target of ``input_paths[i]`` for all i.
    """
    def __init__(
            self,
            inp_paths,
            target_paths,
            transform=transforms.Identity(),
            in_memory=True,
            inp_dtype=np.float32,
            target_dtype=np.int64,
            epoch_multiplier=1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        self.inp_paths = inp_paths
        self.target_paths = target_paths
        self.transform = transform
        self.in_memory = in_memory
        self.inp_dtype = inp_dtype
        self.target_dtype = target_dtype
        self.epoch_multiplier = epoch_multiplier

        if self.in_memory:
            self.inps = [
                np.array(imageio.imread(fname)).astype(np.float32)[None]
                for fname in self.inp_paths
            ]
            self.targets = [
                np.array(imageio.imread(fname)).astype(np.int64)
                for fname in self.target_paths
            ]

    def __getitem__(self, index):
        index %= len(self.inp_paths)  # Wrap around to support epoch_multiplier
        if self.in_memory:
            inp = self.inps[index]
            target = self.targets[index]
        else:
            inp = np.array(imageio.imread(self.inp_paths[index]), dtype=self.inp_dtype)
            if inp.ndim == 2:  # (H, W)
                inp = inp[None]  # (C=1, H, W)
            target = np.array(imageio.imread(self.target_paths[index]), dtype=self.target_dtype)
        while True:  # Only makes sense if RandomCrop is used
            try:
                inp, target = self.transform(inp, target)
                break
            except transforms._DropSample:
                pass
        return inp, target

    def __len__(self):
        return len(self.target_paths) * self.epoch_multiplier


# TODO: Document
class Reconstruction2d(data.Dataset):
    """Simple dataset for 2d reconstruction for auto-encoders etc..
    """
    def __init__(
            self,
            inp_paths,
            transform=transforms.Identity(),
            in_memory=True,
            inp_dtype=np.float32,
            epoch_multiplier=1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        self.inp_paths = inp_paths
        self.transform = transform
        self.in_memory = in_memory
        self.inp_dtype = inp_dtype
        self.epoch_multiplier = epoch_multiplier

        if self.in_memory:
            self.inps = [
                np.array(imageio.imread(fname)).astype(np.float32)[None]
                for fname in self.inp_paths
            ]

    def __getitem__(self, index):
        index %= len(self.inp_paths)  # Wrap around to support epoch_multiplier
        if self.in_memory:
            inp = self.inps[index]
        else:
            inp = np.array(imageio.imread(self.inp_paths[index]), dtype=self.inp_dtype)
            if inp.ndim == 2:  # (H, W)
                inp = inp[None]  # (C=1, H, W)
        while True:  # Only makes sense if RandomCrop is used
            try:
                inp, _ = self.transform(inp, None)
                break
            except transforms._DropSample:
                pass
        return inp, inp

    def __len__(self):
        return len(self.inp_paths) * self.epoch_multiplier
