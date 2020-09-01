# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert

__all__ = ['PatchCreator', 'SimpleNeuroData2d', 'Segmentation2d', 'Reconstruction2d']

import logging
import os
import sys
import traceback
from os.path import expanduser
from typing import Tuple, Dict, Optional, Union, Sequence, Any, List, Callable

import h5py
import imageio
import numpy as np
import torch
from torch.utils import data

from elektronn3.data import coord_transforms, transforms
from elektronn3.data.sources import DataSource, HDF5DataSource, slice_3d

logger = logging.getLogger('elektronn3log')


class _DefaultCubeMeta:
    def __getitem__(self, *args, **kwargs): return np.inf


# TODO: Document passing DataSources directly
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
        input_sources: Sequence of ``(filename, hdf5_key)`` tuples, where
            each item specifies the filename and
            the HDF5 dataset key under which the input data is stored.
        target_sources: Sequence of ``(filename, hdf5_key)`` tuples, where
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
            In some transforms ``target`` can also be set to ``None``. In this
            case it is ignored and only ``inp`` is processed.
            To combine multiple transforms, use
            :py:class:`elektronn3.data.transforms.Compose`.
            See :py:mod:`elektronn3.data.transforms`. for some implementations.
        in_memory: If ``True``, all data set files are immediately loaded
            into host memory and are permanently kept there as numpy arrays.
            If this is disabled (default), file contents are always read from
            the HDF5 files to produce samples. (Note: This does not mean it's
            slower, because file contents are transparently cached by h5py,
            see http://docs.h5py.org/en/latest/high/file.html#chunk-cache).

    """
    def __init__(
            self,
            input_sources: List[Tuple[str, str]],
            patch_shape: Sequence[int],
            target_sources: Optional[List[Tuple[str, str]]] = None,
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
            in_memory: bool = False,
            cube_meta=_DefaultCubeMeta(),
    ):
        # Early checks
        if target_sources is not None and len(input_sources) != len(target_sources):
            raise ValueError(
                'If target_sources is not None, input_sources and '
                'target_sources must be lists of same length.'
            )
        if not train:
            if warp_prob > 0:
                logger.warning('Augmentations should not be used on validation data.')

        # batch properties
        self.train = train
        self.warp_prob = warp_prob
        self.warp_kwargs = warp_kwargs if warp_kwargs is not None else {}

        # general properties
        self.input_sources = input_sources
        self.target_sources = target_sources
        self.cube_meta = cube_meta
        self.cube_prios = cube_prios
        self.aniso_factor = aniso_factor
        self.target_discrete_ix = target_discrete_ix
        self.epoch_size = epoch_size
        self._orig_epoch_size = epoch_size  # Store original epoch_size so it can be reset later.
        self.in_memory = in_memory

        self.patch_shape = np.array(patch_shape, dtype=np.int)
        self.ndim = self.patch_shape.ndim
        self.offset = np.array(offset)
        self.target_patch_shape = self.patch_shape - self.offset * 2
        self._target_dtype = target_dtype
        self.transform = transform

        # Setup internal stuff
        self.pid = os.getpid()

        # The following fields will be filled when reading data
        self.n_labelled_pixels = 0
        self.inputs: List[DataSource] = []
        self.targets: List[DataSource] = []

        self.load_data()  # Open dataset files

        self.n_successful_warp = 0
        self.n_failed_warp = 0
        self._failed_warp_warned = False

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Note that the index is ignored. Samples are always random
        return self._get_random_sample()

    def _get_random_sample(self) -> Dict[str, Any]:
        input_src, target_src, i = self._getcube()  # get cube randomly
        warp_prob = self.warp_prob
        while True:
            try:
                inp, target = self.warp_cut(input_src, target_src, warp_prob, self.warp_kwargs)
                if target is not None:
                    target = target.astype(self._target_dtype)
            except coord_transforms.WarpingOOBError as e:
                # Temporarily set warp_prob to 1 to make sure that the next attempt
                #  will also try to use warping. Otherwise, self.warp_prob would not
                #  reflect the actual probability of a sample being obtained by warping.
                warp_prob = 1 if warp_prob > 0 else 0
                self.n_failed_warp += 1
                if self.n_failed_warp > 20 and self.n_failed_warp > 8 * self.n_successful_warp and not self._failed_warp_warned:
                    fail_ratio = self.n_failed_warp / (self.n_failed_warp + self.n_successful_warp)
                    fail_percentage = int(round(100 * fail_ratio))
                    print(e)
                    logger.warning(
                        f'{fail_percentage}% of warping attempts are failing.\n'
                        'Consider lowering lowering your input patch shapes or warp_kwargs[\'warp_amount\']).'
                    )
                    self._failed_warp_warned = True
                continue
            except coord_transforms.WarpingSanityError:
                logger.exception('Invalid coordinate values encountered while warping. Retrying...')
                continue
            self.n_successful_warp += 1
            try:
                inp, target = self.transform(inp, target)
            except transforms._DropSample:
                # A filter transform has chosen to drop this sample, so skip it
                logger.debug('Sample dropped.')
                continue
            break

        inp = torch.as_tensor(inp)
        cube_meta = torch.as_tensor(self.cube_meta[i])
        fname = os.path.basename(self.inputs[i].fname)
        sample = {
            'inp': inp,
            'cube_meta': cube_meta,  # TODO: Make cube_meta completely optional again
            'fname': fname
        }
        if target is not None:
            sample['target'] = torch.as_tensor(target)

        return sample

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
            inp_src: DataSource,
            target_src: Optional[DataSource],
            warp_prob: Union[float, bool],
            warp_kwargs: Dict[str, Any]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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

        if target_src is None:
            target_src_shape = None
            target_patch_shape = None
        else:
            target_src_shape = target_src.shape
            target_patch_shape = self.target_patch_shape

        M = coord_transforms.get_warped_coord_transform(
            inp_src_shape=inp_src.shape,
            patch_shape=self.patch_shape,
            aniso_factor=self.aniso_factor,
            target_src_shape=target_src_shape,
            target_patch_shape=target_patch_shape,
            **warp_kwargs
        )

        inp, target = coord_transforms.warp_slice(
            inp_src=inp_src,
            patch_shape=self.patch_shape,
            M=M,
            target_src=target_src,
            target_patch_shape=target_patch_shape,
            target_discrete_ix=self.target_discrete_ix
        )

        return inp, target

    def _getcube(self) -> Tuple[DataSource, DataSource, int]:
        """
        Draw an example cube according to sampling weight on training data,
        or randomly on valid data
        """
        i = np.random.choice(
            np.arange(len(self.cube_prios)),
            p=self.cube_prios / np.sum(self.cube_prios)
        )
        inp_source = self.inputs[i]
        target_source = None if self.targets is None else self.targets[i]
        return inp_source, target_source, i

    def load_data(self) -> None:
        if len(self.inputs) == len(self.targets) == 0:
            inp_files, target_files = self.open_files()
            self.inputs.extend(inp_files)
            if target_files is None:
                self.targets = None
            else:
                self.targets.extend(target_files)
        else:
            logger.info('Using directly specified data sources.')

        if self.cube_prios is None:
            # If no priorities are given: sample proportionally to target sizes
            #  if available, or else w.r.t. input sizes (voxel counts)
            self.cube_prios = []
            if self.targets is None:
                self.cube_prios = [inp.size for inp in self.inputs]
            else:
                self.cube_prios = [target.size for target in self.targets]
            self.cube_prios = np.array(self.cube_prios, dtype=np.float32) / np.sum(self.cube_prios)

        logger.debug(f'cube_prios = {self.cube_prios}')

    def check_files(self) -> None:
        """
        Check if all files are accessible.
        """
        notfound = False
        give_neuro_data_hint = False
        fullpaths = [f for f, _ in self.input_sources]
        if self.target_sources is not None:
            fullpaths.extend([f for f, _ in self.target_sources])
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

    def open_files(self) -> Tuple[List[DataSource], Optional[List[DataSource]]]:
        self.check_files()
        inp_sources, target_sources = [], []
        modestr = 'Training' if self.train else 'Validation'
        memstr = ' (in memory)' if self.in_memory else ''
        logger.info(f'\n{modestr} data set{memstr}:')
        if self.target_sources is None:
            for (inp_fname, inp_key), cube_meta in zip(self.input_sources, self.cube_meta):
                inp_source = HDF5DataSource(fname=inp_fname, key=inp_key, in_memory=self.in_memory)
                logger.info(f'  input:       {inp_fname}[{inp_key}]: {inp_source.shape} ({inp_source.dtype})')
                if not np.all(cube_meta == np.inf):
                    logger.info(f'  cube_meta:   {cube_meta}')
                inp_sources.append(inp_source)
                target_sources = None
        else:
            for (inp_fname, inp_key), (target_fname, target_key), cube_meta in zip(self.input_sources, self.target_sources, self.cube_meta):
                inp_source = HDF5DataSource(fname=inp_fname, key=inp_key, in_memory=self.in_memory)
                target_source = HDF5DataSource(fname=target_fname, key=target_key, in_memory=self.in_memory)
                logger.info(f'  input:       {inp_fname}[{inp_key}]: {inp_source.shape} ({inp_source.dtype})')
                logger.info(f'  with target: {target_fname}[{target_key}]: {target_source.shape} ({target_source.dtype})')
                if not np.all(cube_meta == np.inf):
                    logger.info(f'  cube_meta:   {cube_meta}')
                inp_sources.append(inp_source)
                target_sources.append(target_source)
        logger.info('')

        return inp_sources, target_sources


def get_preview_batch(
        h5data: Tuple[str, str],
        preview_shape: Optional[Tuple[int, ...]] = None,
        transform: Optional[Callable] = None,
        in_memory: bool = False,
        dim: Optional[float] = None,
) -> torch.Tensor:
    fname, key = h5data
    inp_h5 = h5py.File(fname, 'r')[key]
    if in_memory:
        inp_h5 = inp_h5.value
    if dim is None:
        if preview_shape is None:
            raise ValueError('At least one of preview_shape, dim must be defined.')
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
    inp_np = slice_3d(inp_h5, inp_lo, inp_hi, prepend_empty_axis=True)
    if inp_np.ndim == dim + 1:  # Should be dim + 2 for (N, C) dims
        inp_np = inp_np[:, None]  # Add missing C dim
    if transform is not None:
        for n in range(inp_np.shape[0]):  # N is usually 1, so this is only iterated once with n=0
            inp_np[0], _ = transform(inp_np[0], None)
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
            out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.transform = transform
        self.out_channels = out_channels
        cube_id = 0 if train else 2
        if inp_path is None:
            inp_path = expanduser(f'~/neuro_data_cdhw/raw_{cube_id}.h5')
        if target_path is None:
            target_path = expanduser(f'~/neuro_data_cdhw/barrier_int16_{cube_id}.h5')
        self.inp_file = h5py.File(os.path.expanduser(inp_path), 'r')
        self.target_file = h5py.File(os.path.expanduser(target_path), 'r')
        self.inp = self.inp_file[inp_key][()].astype(np.float32)
        self.target = self.target_file[target_key][()].astype(np.int64)
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
        inp = torch.as_tensor(inp)
        target = torch.as_tensor(target)
        sample = {
            'inp': inp,
            'target': target,
            'cube_meta': np.inf,
            'fname': str(index)
        }
        return sample

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
            offset=None,
            in_memory=True,
            inp_dtype=np.float32,
            target_dtype=np.int64,
            epoch_multiplier=1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        self.inp_paths = inp_paths
        self.target_paths = target_paths
        self.transform = transform
        self.offset = offset
        self.in_memory = in_memory
        self.inp_dtype = inp_dtype
        self.target_dtype = target_dtype
        self.epoch_multiplier = epoch_multiplier

        if self.in_memory:
            self.inps = []
            multichannel_image = None
            for fname in self.inp_paths:
                img = imageio.imread(fname).astype(np.float32)
                if multichannel_image:
                    assert len(img.shape) == 3, f'Mixed multi-channel {multichannel_image} and single-channel images {fname} in gt.'
                if len(img.shape) == 3:
                    multichannel_image = fname
                    # bring color channel to front
                    self.inps.append(img.swapaxes(0,2).swapaxes(1,2))
                else:
                    self.inps.append(img[None])
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
            inp = np.array(imageio.imread(self.inp_paths[index]), dtype=np.float32)
            if inp.ndim == 2:  # (H, W)
                inp = inp[None]  # (C=1, H, W)
            target = np.array(imageio.imread(self.target_paths[index]), dtype=np.int64)
        while True:  # Only makes sense if RandomCrop is used
            try:
                inp, target = self.transform(inp, target)
                break
            except transforms._DropSample:
                pass
        if self.offset is not None:
            off = self.offset
            target = target[:, off[0]:-off[0], off[1]:-off[1]]
        sample = {
            'inp': torch.as_tensor(inp.astype(self.inp_dtype)),
            'target': torch.as_tensor(target.astype(self.target_dtype)),
            'cube_meta': np.inf,
            'fname': str(self.inp_paths[index])
        }
        return sample

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
        inp = torch.as_tensor(inp)
        sample = {
            'inp': inp,
            'target': inp,
            'cube_meta': np.inf,
            'fname': str(self.inp_paths[index])
        }
        return sample

    def __len__(self):
        return len(self.inp_paths) * self.epoch_multiplier


class TripletData2d(data.Dataset):
    """Simple dataset for 2D triplet loss training.
    """
    def __init__(
            self,
            inp_paths,
            transform=transforms.Identity(),
            invariant_transform=None,
            in_memory=True,
            inp_dtype=np.float32,
            epoch_multiplier=1,  # Pretend to have more data in one epoch
    ):
        super().__init__()
        self.inp_paths = inp_paths
        self.transform = transform
        self.invariant_transform = invariant_transform
        self.in_memory = in_memory
        self.inp_dtype = inp_dtype
        self.epoch_multiplier = epoch_multiplier

        if self.in_memory:
            self.inps = [
                np.array(imageio.imread(fname)).astype(np.float32)[None]
                for fname in self.inp_paths
            ]

    def _get(self, index):
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
        return inp

    def _randidx_excluding(self, exclude):
        while True:
            idx = np.random.randint(0, len(self.inp_paths) // self.epoch_multiplier)
            if idx != exclude:
                return idx

    def __getitem__(self, index):
        index %= len(self.inp_paths)  # Wrap around to support epoch_multiplier
        anchor = self._get(index)
        if self.invariant_transform is None:
            # Assuming a random augmentation transform, the positive image will be different than
            #  the anchor, but it will originate from the same image file.
            #  If random cropping and geometrical transforms are used, make sure that the loss is
            #  not calculated on localized/spatial outputs!
            pos = self._get(index)
        else:
            # Apply an additional transform against which the network should learn invariant behavior
            pos, _ = self.invariant_transform(anchor, None)
        # Sample a negative image from a random different index -> different image
        neg_idx = self._randidx_excluding(index)
        neg = self._get(neg_idx)
        if self.invariant_transform is not None:
            # Also apply the invariant transform to the negative image because otherwise
            #  the model could "cheat" by detecting that the inherent features of this
            #  transform only exist in the positive image.
            neg, _ = self.invariant_transform(neg, None)
        sample = {
            'anchor': torch.as_tensor(anchor),
            'pos': torch.as_tensor(pos),
            'neg': torch.as_tensor(neg),
            'fname': f'ap{index}n{neg_idx}'
        }
        return sample

    def __len__(self):
        return len(self.inp_paths) * self.epoch_multiplier

# TODO: Warn if datasets have no content
