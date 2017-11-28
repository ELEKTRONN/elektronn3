from __future__ import absolute_import, division, print_function
from builtins import int, zip
__all__ = ['BatchCreatorImage']
import gc
import os
import sys
import time
import logging
import h5py
import numpy as np
import tqdm
from . import transformations
from . import utils
import torch
import signal
from torch.utils import data
from .. import floatX
from .utils import DelayedInterrupt

logger = logging.getLogger('elektronn3log')


class BatchCreatorImage(data.Dataset):
    def __init__(self, d_path=None, l_path=None,
                 d_files=None, l_files=None, cube_prios=None, valid_cubes=None,
                 border_mode='crop', aniso_factor=2, target_vec_ix=None,
                 target_discrete_ix=None, h5stream=False, zxy=True,
                 source='train', patch_size=None,
                 grey_augment_channels=None, warp=False, warp_args=None,
                 ignore_thresh=False, force_dense=False, class_weights=False,
                 epoch_size=100, cuda_enabled='auto'):
        assert (d_path and l_path and d_files and l_files)
        if len(d_files)!=len(l_files):
            raise ValueError("d_files and l_files must be lists of same length!")
        d_path = os.path.expanduser(d_path)
        l_path = os.path.expanduser(l_path)
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
        self.zxy = zxy
        self.h5stream = h5stream
        self.d_path = d_path
        self.l_path = l_path
        self.d_files = d_files
        self.l_files = l_files
        self.cube_prios = cube_prios
        self.valid_cubes = valid_cubes if valid_cubes is not None else []
        self.aniso_factor = aniso_factor
        self.border_mode = border_mode
        self.target_vec_ix = target_vec_ix
        self.target_discrete_ix = target_discrete_ix
        self.epoch_size = epoch_size
        self._epoch_size = epoch_size

        # Infer geometric info from input/target shapes
        # HACK
        self.patch_size = np.array(patch_size, dtype=np.int)
        self.ndim = self.patch_size.ndim
        self.strides = np.array([1, 1, 1], dtype=np.int) #np.array(target_node.shape.strides, dtype=np.int)
        self.offsets = np.array([0, 0, 0], dtype=np.int) #np.array(target_node.shape.offsets, dtype=np.int)
        self.target_ps = self.patch_size - self.offsets * 2
        self.t_dtype = np.int64
        self.mode = 'img-img'
        # The following will be inferred when reading data
        self.n_labelled_pixel = 0
        self.n_f = None  # number of channels/feature in input
        self.t_n_f = None  # the shape of the returned label batch at index 1

        # Actual data fields
        self.valid_d = []
        self.valid_l = []
        self.valid_extra = []

        self.train_d = []
        self.train_l = []
        self.train_extra = []

        # Setup internal stuff
        self.rng = np.random.RandomState(np.uint32((time.time() * 0.0001 -
                                                    int(
                                                        time.time() * 0.0001)) * 4294967295))
        self.pid = os.getpid()
        self.gc_count = 1

        self._sampling_weight = None
        self._training_count = None
        self._valid_count = None
        self.n_successful_warp = 0
        self.n_failed_warp = 0

        self.load_data()
        if class_weights:
            target_mean = np.mean(self.train_l)
            bg_weight = target_mean / (1. + target_mean)
            fg_weight = 1. - bg_weight
            self.class_weights = torch.FloatTensor([bg_weight, fg_weight])
            if self.cuda_enabled:
                self.class_weights = self.class_weights.cuda()
        else:
            self.class_weights = None

    def __getitem__(self, index):
        # use index just as counter, subvolumes will be chosen randomly
        d, l = self.getbatch(1)
        # squeeze first axis because getitem needs ch, z, x, y; without explicit batch axis
        return torch.from_numpy(d[0]), torch.from_numpy(l[0])

    def __len__(self):
        return self.epoch_size
        if self.source == "train":
            return len(self.d_files) - len(self.valid_cubes)
        elif self.source == "valid":
            return len(self.valid_cubes)
        else:
            raise NotImplementedError

    def close(self):
        return

    def __repr__(self):
        s = "{0:,d}-target Data Set with {1:,d} input channel(s):\n" + \
            "#train cubes: {2:,d} and #valid cubes: {3:,d}, {4:,d} labelled " + \
            "pixels."
        s = s.format(self.t_n_f, self.n_f, self._training_count,
                     self._valid_count, self.n_labelled_pixel)
        return s

    def validate(self):
        self.source = "valid"
        self.epoch_size = 10

    def train(self):
        self.source = "train"
        self.epoch_size = self._epoch_size

    @property
    def warp_stats(self):
        return "Warp stats: successful: %i, failed %i, quota: %.1f" %(
            self.n_successful_warp, self.n_failed_warp,
            float(self.n_successful_warp)/(self.n_failed_warp+self.n_successful_warp))

    def _reseed(self):
        """Reseeds the rng if the process ID has changed!"""
        current_pid = os.getpid()
        if current_pid!=self.pid:
            self.pid = current_pid
            self.rng.seed(np.uint32((time.time()*0.0001 -
                                     int(time.time()*0.0001))*4294967295+self.pid))

    def _allocbatch(self, batch_size):
        images = np.zeros((batch_size, self.n_f,)+tuple(self.patch_size), dtype='float32')
        sh = self.patch_size - self.offsets * 2
        target = np.zeros((batch_size, self.t_n_f)+tuple(sh), dtype=self.t_dtype)
        return images, target

    def getbatch(self, batch_size=1):
        """
        Prepares a batch by randomly sampling, shifting and augmenting
        patches from the data

        Parameters
        ----------
        batch_size: int
            Number of examples in batch (for CNNs often just 1)
        source: str
            Data set to draw data from: 'train'/'valid'
        grey_augment_channels: list
            List of channel indices to apply grey-value augmentation to
        warp: bool or float
            Whether warping/distortion augmentations are applied to examples
            (slow --> use multiprocessing). If this is a float number,
            warping is applied to this fraction of examples e.g. 0.5 --> every
            other example.
        warp_args: dict
            Additional keyword arguments that get passed through to
            elektronn2.data.transformations.get_warped_slice()
        ignore_thresh: float
            If the fraction of negative targets in an example patch exceeds this
            threshold, this example is discarded (Negative targets are ignored
            for training [but could be used for unsupervised target propagation]).
        force_dense: bool
            If True the targets are *not* sub-sampled according to the CNN output\
            strides. Dense targets requires MFP in the CNN!

        Returns
        -------
        data: np.ndarray
            [bs, ch, x, y] or [bs, ch, z, y, x] for 2D and 3D CNNS
        target: np.ndarray
            [bs, ch, x, y] or [bs, ch, z, y, x]
        ll_mask1: np.ndarray
            (optional) [bs, n_target]
        ll_mask2: np.ndarray
            (optional) [bs, n_target]
        """
        # This is especially required for multiprocessing
        if self.grey_augment_channels is None:
            self.grey_augment_channels = []
        self._reseed()
        images, target = self._allocbatch(batch_size)
        ll_masks = []
        patch_count = 0
        while patch_count < batch_size:  # Loop to fill up batch with examples
            d, t, ll_mask = self._getcube(self.source)  # get cube randomly

            try:
                d, t = self.warp_cut(d, t,  self.warp, self.warp_args)
                self.n_successful_warp += 1

            except transformations.WarpingOOBError:
                self.n_failed_warp += 1
                continue

            # Check only if a ignore_thresh is set and the cube is labelled
            if (self.ignore_thresh is not False) and (not np.any(ll_mask[1])):
                if (t < 0).mean() > self.ignore_thresh:
                    continue  # do not use cubes which have no information

            if self.source == "train":  # no grey augmentation for testing
                d = transformations.greyAugment(d, self.grey_augment_channels, self.rng)

            target[patch_count] = t
            images[patch_count] = d
            ll_masks.append(ll_mask)
            patch_count += 1

        # Final modification of targets: striding and replacing nan
        if not (self.force_dense or np.all(self.strides == 1)):
            target = self._stridedtargets(target)

        ret = [images, target]  # The "normal" batch
        self.gc_count += 1
        if self.gc_count % 1000 == 0:
            gc.collect()
        return tuple(ret)

    def warp_cut(self, img, target, warp, warp_params):
        """
        (Wraps :py:meth:`elektronn2.data.transformations.get_warped_slice()`)

        Cuts a warped slice out of the input and target arrays.
        The same random warping transformation is each applied to both input
        and target.

        Warping is randomly applied with the probability defined by the ``warp``
        parameter (see below).

        Parameters
        ----------
        img: np.ndarray
            Input image
        target: np.ndarray
            Target image
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
        d: np.ndarray
            (Warped) input image slice
        t: np.ndarray
            (Warped) target slice
        """
        if (warp is True) or (warp == 1):  # always warp
            do_warp = True
        elif (0 < warp < 1):  # warp only a fraction of examples
            do_warp = True if (self.rng.rand() < warp) else False
        else:  # never warp
            do_warp = False

        if not do_warp:
            warp_params = dict(warp_params)
            warp_params['warp_amount'] = 0

        d, t = transformations.get_warped_slice(img, self.patch_size,
                                                aniso_factor=self.aniso_factor,
                                                target=target,
                                                target_ps=self.target_ps,
                                                target_vec_ix=self.target_vec_ix,
                                                target_discrete_ix=self.target_discrete_ix,
                                                rng=self.rng, **warp_params)

        return d, t

    def _getcube(self, source):
        """
        Draw an example cube according to sampling weight on training data,
        or randomly on valid data
        """
        if source == 'train':
            p = self.rng.rand()
            i = np.flatnonzero(self._sampling_weight <= p)[-1]
            d, t, ll_mask = self.train_d[i], self.train_l[i], self.train_extra[i]
        elif source == "valid":
            if len(self.valid_d) == 0:
                raise ValueError("No validation set")

            i = self.rng.randint(0, len(self.valid_d))
            d = self.valid_d[i]
            t = self.valid_l[i]
            ll_mask = self.valid_extra[i]
        else:
            raise ValueError("Unknown data source")

        return d, t, ll_mask

    def _stridedtargets(self, lab):
        if self.ndim == 3:
            return lab[:, :, ::self.strides[0], ::self.strides[1], ::self.strides[2]]
        elif self.ndim == 2:
            return lab[:, :, ::self.strides[0], ::self.strides[1]]

    def load_data(self):
        """
        Parameters
        ----------

        d_path/l_path: string
          Directories to load data from
        d_files/l_files: list
          List of data/target files in <path> directory (must be in the same order!).
          Each list element is a tuple in the form
          **(<Name of h5-file>, <Key of h5-dataset>)**
        cube_prios: list
          (not normalised) list of sampling weights to draw examples from
          the respective cubes. If None the cube sizes are taken as priorities.
        valid_cubes: list
          List of indices for cubes (from the file-lists) to use as validation
          data and exclude from training, may be empty list to skip performance
          estimation on validation data.
        """
        # returns lists of cubes, ll_mask is a tuple per cube
        data, target, extras = self.read_files()

        if self.mode == 'img-scalar':
            data = transformations.border_treatment(data, self.patch_size, self.border_mode,
                                    self.ndim)

        default_extra = (np.ones(self.t_n_f), np.zeros(self.t_n_f))
        extras = [default_extra if x is None else x for x in extras]

        prios = []
        # Distribute Cubes into training and valid list
        for k, (d, t, e) in enumerate(zip(data, target, extras)):
            if k in self.valid_cubes:
                self.valid_d.append(d)
                self.valid_l.append(t)
                self.valid_extra.append(e)
            else:
                self.train_d.append(d)
                self.train_l.append(t)
                self.train_extra.append(e)
                # If no priorities are given: sample proportional to cube size
                prios.append(t.size)

        if self.cube_prios is None:
            prios = np.array(prios, dtype=np.float)
        else:  # If priorities are given: sample irrespective of cube size
            prios = np.array(self.cube_prios, dtype=np.float)

        # sample example i if: batch_prob[i] < p
        self._sampling_weight = np.hstack((0, np.cumsum(prios / prios.sum())))
        self._training_count = len(self.train_d)
        self._valid_count = len(self.valid_d)

    def check_files(self):
        """
        Check if file paths in the network config are available.
        """
        notfound = False
        give_neuro_data_hint = False
        fullpaths = [os.path.join(self.d_path, f) for f, _ in self.d_files] + \
                    [os.path.join(self.l_path, f) for f, _ in self.l_files]
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

    def read_files(self):
        """
        Image files on disk are expected to be in order (ch,x,y,z) or (x,y,z)
        But image stacks are returned as (z,ch,x,y) and target as (z,x,y,)
        irrespective of the order in the file. If the image files have no
        channel this dimension is extended to a singleton dimension.
        """
        self.check_files()
        data, target, extras = [], [], []
        pbar = tqdm.tqdm(total=len(self.d_files))

        for (d_f, d_key), (l_f, l_key) in zip(self.d_files, self.l_files):
            pbar.write('Loading %s and %s' % (d_f, l_f))
            if self.h5stream:
                d = h5py.File(os.path.join(self.d_path, d_f), 'r')[d_key]
                t = h5py.File(os.path.join(self.l_path, l_f), 'r')[l_key]
                assert d.compression == t.compression == None
                assert len(d.shape) == len(t.shape) == 4
                assert d.dtype == floatX
                assert t.dtype == self.t_dtype

            else:
                d = utils.h5load(os.path.join(self.d_path, d_f), d_key)
                t = utils.h5load(os.path.join(self.l_path, l_f), l_key)

            try:
                ll_mask_1 = utils.h5load(os.path.join(self.l_path, l_f),
                                         'll_mask')
                if not self.zxy:
                    ll_mask_1 = transformations.xyz2zxy(ll_mask_1)
                extras.append(ll_mask_1)
            except KeyError:
                extras.append(None)
            if not self.zxy:
                d = transformations.xyz2zxy(d)
                t = transformations.xyz2zxy(t)
            if self.mode == 'img-scalar':
                assert t.ndim == 1, "Scalar targets must be 1d"

            if len(d.shape) == 4:  # h5 dataset has no ndim
                self.n_f = d.shape[0]
            elif len(d.shape) == 3:  # We have no channels in data
                self.n_f = 1
                d = d[None]  # add (empty) 0-axis

            if len(t.shape) == 3:  # If labels not empty add first axis
                t = t[None]

            if self.t_n_f is None:
                self.t_n_f = t.shape[0]
            else:
                assert self.t_n_f == t.shape[0]

            self.n_labelled_pixel += t[0].size

            # determine normalisation depending on int or float type
            if d.dtype.kind in ('u', 'i'):
                m = 255.
                d = np.ascontiguousarray(d, dtype=floatX) / m

            if (np.dtype(self.t_dtype) is not np.dtype(t.dtype)) and \
                self.t_dtype not in ['float32']:
                m = t.max()
                M = np.iinfo(self.t_dtype).max
                if m  > M:
                    raise ValueError("Loading of data: targets must be cast "
                                     "to %s, but %s cannot store value %g, "
                                     "maximum allowed value: %g. You may try "
                                     "to renumber targets." %(self.t_dtype,
                                                             self.t_dtype, m, M))
            if not self.h5stream:
                d = np.ascontiguousarray(d, dtype=floatX)
                t = np.ascontiguousarray(t, dtype=self.t_dtype)

            pbar.write('Shapes (means): data %s (%0.3f), targets %s (%0.3f)' %
                       (d.shape,  d.mean(), t.shape, t.mean()))

            data.append(d)
            target.append(t)
            gc.collect()
            pbar.update()
        print()

        return data, target, extras
