# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert
import datetime
import pprint
from collections import deque

import gc
import logging
import os
import shutil
import warnings
import zipfile

from itertools import islice
from math import nan
from pickle import PickleError
from textwrap import dedent
from typing import Tuple, Dict, Optional, Callable, Any, Sequence, List, Union

import inspect
import IPython
import numpy as np
import tensorboardX
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.cuda import amp
from tqdm import tqdm


import elektronn3
from elektronn3.training import handlers
from elektronn3.training.swa import SWA
from elektronn3.training.train_utils import pretty_string_time, create_preview_batch_from_knossos
from elektronn3.training.train_utils import Timer, HistoryTracker

from torch.utils import collect_env
from elektronn3.inference import Predictor
from elektronn3 import __file__ as arch_src

logger = logging.getLogger('elektronn3log')


class NaNException(RuntimeError):
    """When a NaN value is detected"""
    pass


def _worker_init_fn(worker_id: int) -> None:
    """Sets a unique but deterministic random seed for background workers.

    Only sets the seed for NumPy because PyTorch and Python's own RNGs
    take care of reseeding on their own.
    See https://github.com/numpy/numpy/issues/9650."""
    # Modulo 2**32 because np.random.seed() only accepts values up to 2**32 - 1
    initial_seed = torch.initial_seed() % 2**32
    worker_seed = initial_seed + worker_id
    np.random.seed(worker_seed)


# Be careful from where you call this! Not sure if this is concurrency-safe.
def _change_log_file_to(
        new_path: str,
        transfer_old_logs: bool = True,
        delete_old_file: bool = True
) -> None:
    """Transfer the current log file to a new location and redirect logs."""

    def _get_first_file_handler() -> logging.FileHandler:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return handler
        return RuntimeError('logger has no FileHandler.')

    # Getting the first (and presumably only) file handler
    file_handler = _get_first_file_handler()
    if transfer_old_logs:
        with open(file_handler.baseFilename) as f:
            old_logs = f.read()
        with open(new_path, 'w') as f:
            f.write(old_logs)
    file_handler.close()
    if delete_old_file:
        os.remove(file_handler.baseFilename)
    file_handler.baseFilename = new_path


class Trainer:
    """ General training loop abstraction for supervised training.

    Args:
        model: PyTorch model (``nn.Module``) that shall be trained.
            Please make sure that the output shape of the ``model``
            matches the shape of targets that are delivered by the
            ``train_dataset``.
        criterion: PyTorch loss that shall be used as the optimization
            criterion.
        optimizer: PyTorch optimizer that shall be used to update
            ``model`` weights according to the ``criterion`` in each
            iteration.
        device: The device on which the network shall be trained.
        train_dataset: PyTorch dataset (``data.Dataset``) which produces
            training samples when iterated over.
            :py:class:`elektronn3.data.cnndata.PatchCreator` is currently
            recommended for constructing datasets.
        valid_dataset: PyTorch dataset (``data.Dataset``) which produces
            validation samples when iterated over.
            The length (``len(valid_dataset)``) of it determines how many
            samples are used for one validation metric calculation.
        unlabeled_dataset: Unlabeled dataset (only inputs) for
            semi-supervised training. If this is supplied, ``ss_criterion``
            needs to be set to the loss that should be computed on unlabeled
            inputs.
        valid_metrics: Validation metrics to be calculated on
            validation data after each training epoch. All metrics are logged
            to tensorboard.
        ss_criterion: Loss criterion for the self-supervised part of
            semi-supervised training. The ``ss_criterion`` loss is computed
            on batches from the ``unlabeled_dataset`` and added to the
            supervised loss in each training step.
        save_root: Root directory where training-related files are
            stored. Files are always written to the subdirectory
            ``save_root/exp_name/``.
        exp_name: Name of the training experiment. Determines the subdirectory
            to which files are written and should uniquely identify one
            training experiment.
            If ``exp_name`` is not set, it is auto-generated from the model
            name and a time stamp in the format ``'%y-%m-%d_%H-%M-%S'``.
        example_input: An example input tensor that can be fed to the
            ``model``. This is used for JIT tracing during model serialization.
        save_jit: Chooses if/how a JIT version (.pts file) of the
            ``model`` should always be saved in addition to regular model
            snapshots. Choices:

            - ``None`` (default): Disable saving JIT models.
            - ``'script'`` (recommended if possible): The model is compiled
              directly with ``torch.jit.script()`` and saved as a .pts file
            - ``'trace'``: The model is JIT-traced with ``example_input``
              and saved as a .pts file
        batch_size: Desired batch size of training samples.
        preview_batch: Set a fixed input batch for preview predictions.
            If it is ``None`` (default), preview batch functionality will be
            disabled. As a more powerful alternative for KNOSSOS datasets, consider using
            the ``knossos_preview_config`` option instead.
        knossos_preview_config: Configures preview batch creation and preview inferences
            based on a KNOSSOS dataset region. Here is an example of how it should look like:

            >>> knossos_preview_config = {
            ...     'dataset': 'path/to/knossos/dataset.conf',
            ...     'offset': [0, 0, 0],  # Offset (min) coordinates
            ...     'size': [256, 256, 64],  # Size (shape) of the region
            ...     'mag': 1,  # source mag
            ...     'target_mags': [1, 2, 3],  # List of target mags to which the inference results should be written
            ...     'remap_ids': None  # Config for class ID remapping (optional). See transforms.RemapTargetIDs
            ... }

            Periodic preview inference results are written to .k.zip annotation files that can be
            loaded with KNOSSOS and overlayed over the original data. .k.zip files are saved in the
            training directory, with file names reflecting their training step.
        preview_interval: Determines how often to perform preview inference.
            Preview inference is performed every ``preview_interval`` epochs
            during training. Regardless of this value, preview predictions
            will also be performed once after epoch 1.
            (To disable preview predictions altogether, just set
            ``preview_batch = None``).
        inference_kwargs: Additional options that are supplied to the
            :py:class:`elektronn3.inference.Predictor` instance
            that is used for periodic preview inference on the
            ``preview_batch``.
        extra_save_steps: Permanent model snapshots are saved at the
            training steps specified here. E.g. with
            ``extra_save_steps = (0, 30, 3000)``, a snapshot is made at
            steps 0 (before training begins), step 30 and step 3000.
        num_workers: Number of background processes that are used to produce
            training samples without blocking the main training loop.
            See :py:class:`torch.utils.data.DataLoader`
            For normal training, you can mostly set ``num_workers=1``.
            Only use more workers if you notice a data loader bottleneck.
            Set ``num_workers=0`` if you want to debug the datasets
            implementation, to avoid mulitprocessing-specific issues.
        schedulers: Dictionary of schedulers for training hyperparameters,
            e.g. learning rate schedulers that can be found in
            `py:mod:`torch.optim.lr_scheduler`.
        overlay_alpha: Alpha (transparency) value for alpha-blending of
            overlay image plots.
        enable_videos: Enables video visualizations for 3D image data
            in tensorboard. Requires the moviepy package.
            Warning: Videos are stored as GIFs and can get very large,
            so only use this if you log seldomly or have a lot of storage
            capacity.
        enable_tensorboard: If ``True``, tensorboard logging/plotting is
            enabled during training.
        tensorboard_root_path: Path to the root directory under which
            tensorboard log directories are created. Log ("event") files are
            written to a subdirectory that has the same name as the
            ``exp_name``.
            If ``tensorboard_root_path`` is not set, tensorboard logs are
            written to ``save_path`` (next to model checkpoints, plots etc.).
        ignore_errors: If ``True``, the training process tries to ignore
            all errors and continue with the next batch if it encounters
            an error on the current batch.
            It's not recommended to use this. It's only helpful for certain
            debugging scenarios.
        ipython_shell: If ``True`` keyboard interrupts (Ctrl-C) won't exit
            the process but only pause training and enter an IPython shell.
            Additionally, errors during training (except
            C-level segfaults etc.) won't crash the whole training process,
            but drop to an IPython shell so errors can be inspected with
            access to the current training state.
        out_channels: Optionally specifies the total number of different target
            classes for classification tasks. If this is not set manually,
            the ``Trainer`` checks if the ``train_dataset`` provides this
            value. If available, ``self.out_channels`` is set to
            ``self.train_dataset.out_channels``. Otherwise, it is set to
            ``None``.
            The ``out_channels`` attribute is used for plotting purposes and is
            not strictly required for training.
        sample_plotting_handler: Function that receives training and
            validation samples and is responsible for visualizing them by
            e.g. plotting them to tensorboard and/or writing them to files.
            It is called once after each training epoch and once after each
            validation run.
            If ``None``, a tensorboard-based default handler is used that
            works for most classification scenarios and for 3-channel
            regression.
        preview_plotting_handler: Function that is responsible for producing
            previews and visualizing/plotting/logging them.
            It is called once each ``preview_interval`` epochs.
            If ``None``, a tensorboard-based default handler is used that
            works for most classification scenarios.
        mixed_precision: If ``True``, enable Automated Mixed Precision training
            powered by https://github.com/NVIDIA/apex to reduce memory usage
            and (if a GPU with Tensor Cores is used) make training much faster.
            This is currently experimental and might cause instabilities.
        tqdm_kwargs: Extra arguments to be passed to tqdm progress bars.
            For example, to disable tqdm outputs completely, pass
            ``tqdm_kwargs={'disable': True}``.
    """

    tb: tensorboardX.SummaryWriter
    terminate: bool
    step: int
    epoch: int
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    exp_name: str
    save_path: str  # Full path to where training files are stored
    out_channels: Optional[int]  # Number of channels of the network outputs

    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            save_root: str,
            train_dataset: torch.utils.data.Dataset,
            valid_dataset: Optional[torch.utils.data.Dataset] = None,
            unlabeled_dataset: Optional[torch.utils.data.Dataset] = None,
            valid_metrics: Optional[Dict] = None,
            ss_criterion: Optional[torch.nn.Module] = None,
            preview_batch: Optional[torch.Tensor] = None,
            knossos_preview_config: Optional[Dict[str, str]] = None,
            preview_interval: int = 5,
            inference_kwargs: Optional[Dict[str, Any]] = None,
            hparams: Optional[Dict[str, Any]] = None,
            extra_save_steps: Sequence[int] = (),
            exp_name: Optional[str] = None,
            example_input: Optional[torch.Tensor] = None,
            enable_save_trace: bool = False,
            save_jit: Optional[str] = None,
            batch_size: int = 1,
            num_workers: int = 0,
            schedulers: Optional[Dict[Any, Any]] = None,
            overlay_alpha: float = 0.4,
            enable_videos: bool = False,
            enable_tensorboard: bool = True,
            tensorboard_root_path: Optional[str] = None,
            ignore_errors: bool = False,
            ipython_shell: bool = False,
            out_channels: Optional[int] = None,
            sample_plotting_handler: Optional[Callable] = None,
            preview_plotting_handler: Optional[Callable] = None,
            mixed_precision: bool = False,
            tqdm_kwargs: Optional[Dict] = None
    ):
        inference_kwargs = {} if inference_kwargs is None else inference_kwargs
        if preview_batch is not None and (
                'tile_shape' not in inference_kwargs or (
                    'overlap_shape' not in inference_kwargs and 'offset' not in inference_kwargs)):
            raise ValueError(
                'If preview_batch is set, you will also need to specify '
                'tile_shape and overlap_shape or offset in inference_kwargs!'
            )
        if knossos_preview_config is not None:
            if preview_batch is not None:
                raise ValueError('If you set a preview_knossos_source, you cannot also set a preview batch.')
            preview_batch = create_preview_batch_from_knossos(knossos_preview_config)
        if enable_save_trace:
            logger.warning('enable_save_trace is deprecated. Please use the save_jit option instead.')
            assert save_jit in [None, 'trace']
            save_jit = 'trace'

        # Ensure that all nn.Modules are on the right device
        model.to(device)
        if isinstance(criterion, torch.nn.Module):
            criterion.to(device)
        if isinstance(ss_criterion, torch.nn.Module):
            ss_criterion.to(device)

        self.ignore_errors = ignore_errors
        self.ipython_shell = ipython_shell
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.valid_metrics = valid_metrics
        self.ss_criterion = ss_criterion
        self.preview_batch = preview_batch
        self.knossos_preview_config = knossos_preview_config
        self.preview_interval = preview_interval
        self.inference_kwargs = inference_kwargs
        self.extra_save_steps = extra_save_steps
        self.overlay_alpha = overlay_alpha
        self.save_root = os.path.expanduser(save_root)
        self.example_input = example_input
        self.save_jit = save_jit
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_plotting_handler = sample_plotting_handler
        self.preview_plotting_handler = preview_plotting_handler
        self.mixed_precision = mixed_precision
        self.tqdm_kwargs = {} if tqdm_kwargs is None else tqdm_kwargs

        self._tracker = HistoryTracker()
        self._timer = Timer()
        self._first_plot = True
        self._shell_info = dedent("""
            Entering IPython training shell. To continue, hit Ctrl-D twice.
            To terminate, set self.terminate = True and then hit Ctrl-D twice.
        """).strip()

        self.inference_kwargs.setdefault('batch_size', 1)
        self.inference_kwargs.setdefault('verbose', True)
        self.inference_kwargs.setdefault('apply_softmax', True)

        if self.unlabeled_dataset is not None and self.ss_criterion is None:
            raise ValueError('If an unlabeled_dataset is supplied, you must also set ss_criterion.')

        if hparams is None:
            hparams = {}
        else:
            for k, v in hparams.items():
                if isinstance(v, (tuple, list)):
                    # Convert to str because tensorboardX doesn't support
                    # tuples and lists in add_hparams()
                    hparams[k] = str(v)
        self.hparams = hparams

        self.scaler = amp.GradScaler(enabled=self.mixed_precision)

        if exp_name is None:  # Auto-generate a name based on model name and ISO timestamp
            timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
            exp_name = model.__class__.__name__ + '__' + timestamp
        self.exp_name = exp_name
        self.save_path = os.path.join(save_root, exp_name)
        if os.path.isdir(self.save_path):
            raise RuntimeError(
                f'{self.save_path} already exists.\nPlease choose a '
                'different combination of save_root and exp_name.'
            )
        os.makedirs(self.save_path)
        _change_log_file_to(f'{self.save_path}/elektronn3.log')
        logger.info(f'Writing files to save_path {self.save_path}/\n')

        self.terminate = False
        self.step = 0
        self.epoch = 0
        if schedulers is None:
            schedulers = {'lr': StepLR(optimizer, 1000, 1)}  # No-op scheduler
        self.schedulers = schedulers
        self.__lr_closetozero_alreadytriggered = False  # Used in periodic scheduler handling
        self._lr_nhood = deque(maxlen=3)  # Keeps track of the last, current and next learning rate

        self.out_channels = out_channels
        self.max_plot_id = None

        try:
            self.max_plot_id = max(self.out_channels, self.criterion.ignore_index + 1)
        except AttributeError: # no ignore_idx
            self.max_plot_id = self.out_channels
        except TypeError: # no out_channels
            self.max_plot_id = None

        if enable_videos:
            try:
                import moviepy
            except:
                logger.warning('moviepy is not installed. Disabling video logs.')
                enable_videos = False
        self.enable_videos = enable_videos
        self.tb = None  # Tensorboard handler
        if enable_tensorboard:
            if self.sample_plotting_handler is None:
                self.sample_plotting_handler = handlers._tb_log_sample_images
            if self.preview_plotting_handler is None:
                self.preview_plotting_handler = handlers._tb_log_preview

            if tensorboard_root_path is None:
                tb_path = self.save_path
            else:
                tensorboard_root_path = os.path.expanduser(tensorboard_root_path)
                tb_path = os.path.join(tensorboard_root_path, self.exp_name)
                os.makedirs(tb_path, exist_ok=True)
            self.tb = tensorboardX.SummaryWriter(logdir=tb_path, flush_secs=20)

            if self.hparams:
                self.tb.add_hparams(hparam_dict=self.hparams, metric_dict={})

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            timeout=60 if self.num_workers > 0 else 0,
            worker_init_fn=_worker_init_fn
        )
        if valid_dataset is not None:
            self.valid_loader = DataLoader(
                self.valid_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers,
                pin_memory=True, worker_init_fn=_worker_init_fn
            )
        if self.unlabeled_dataset is not None:
            self.unlabeled_loader = DataLoader(
                self.unlabeled_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, pin_memory=True,
                timeout=60 if self.num_workers > 0 else 0, worker_init_fn=_worker_init_fn
            )

        self.best_val_loss = np.inf  # Best recorded validation loss
        self.best_tr_loss = np.inf

        self.valid_metrics = {} if valid_metrics is None else valid_metrics

    def run(self, max_steps: int = 1, max_runtime=3600 * 24 * 7) -> None:
        """Train the network for ``max_steps`` steps.
        After each training epoch, validation performance is measured and
        visualizations are computed and logged to tensorboard."""
        self.start_time = datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(seconds=max_runtime)
        self._save_model(suffix='_initial', verbose=False)
        self._lr_nhood.clear()
        self._lr_nhood.append(self.optimizer.param_groups[0]['lr'])  # LR of the first training step
        while not self.terminate:
            try:
                stats, misc, tr_sample_images = self._train(max_steps, max_runtime)
                self.epoch += 1

                if self.valid_dataset is None:
                    stats['val_loss'] = nan
                    val_sample_images = None
                else:
                    valid_stats, val_sample_images = self._validate()
                    stats.update(valid_stats)

                # Log to stdout and text log file
                self._log_basic(stats, misc)
                # Render visualizations and log to tensorboard
                self._log_to_tensorboard(stats, misc, tr_sample_images, val_sample_images)
                # Legacy non-tensorboard logging to files
                self._log_to_history_tracker(stats, misc)

                # Save trained model state
                self._save_model(val_loss=stats['val_loss'], verbose=False)  # Not verbose because it can get spammy.
                # TODO: Support other metrics for determining what's the "best" model?
                if stats['val_loss'] < self.best_val_loss:
                    self.best_val_loss = stats['val_loss']
                    self._save_model(suffix='_best', val_loss=stats['val_loss'])
            except KeyboardInterrupt:
                if self.ipython_shell:
                    IPython.embed(header=self._shell_info)
                else:
                    break
                if self.terminate:
                    break
            except Exception as e:
                logger.exception('Unhandled exception during training:')
                if self.ignore_errors:
                    # Just print the traceback and try to carry on with training.
                    # This can go wrong in unexpected ways, so don't leave the training unattended.
                    pass
                elif self.ipython_shell:
                    print("\nEntering Command line such that Exception can be "
                          "further inspected by user.\n\n")
                    IPython.embed(header=self._shell_info)
                    if self.terminate:
                        break
                else:
                    raise e
        self._save_model(suffix='_final')
        if self.tb is not None:
            self.tb.close()  # Ensure that everything is flushed

    def _train_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core training step on self.device"""
        inp = batch['inp']
        target = batch.get('target')
        target_class = batch.get('class')
        # Everything with a "d" prefix refers to tensors on self.device (i.e. probably on GPU)
        dinp = inp.to(self.device, non_blocking=True)
        dtarget = target.to(self.device, non_blocking=True) if target is not None else None
        dtarget_class = target_class.to(self.device, non_blocking=True) if target_class is not None else None
        # forward pass
        with amp.autocast(enabled=self.mixed_precision):
            dout = self.model(dinp)
            if dtarget_class is not None:
                dloss = self.criterion(dout, dtarget, dtarget_class)
            else:
                dloss = self.criterion(dout, dtarget)

        unlabeled = batch.get('unlabeled')
        if unlabeled is not None:  # Add a simple consistency loss
            u_inp = unlabeled['inp']
            du_inp = u_inp.to(self.device, non_blocking=True)
            with amp.autocast(enabled=self.mixed_precision):
                du_loss = self.ss_criterion(du_inp)
                dloss += du_loss
            self.tb.add_scalar('stats/tr_uloss', float(du_loss), global_step=self.step)

        if torch.isnan(dloss):
            logger.error('NaN loss detected! Aborting training.')
            raise NaNException
        # update step
        self.scaler.scale(dloss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return dloss, dout

    def _train(self, max_steps, max_runtime):
        """Train for one epoch or until max_steps or max_runtime is reached"""
        self.model.train()

        # Scalar training stats that should be logged and written to tensorboard later
        stats: Dict[str, Union[float, List[float]]] = {stat: [] for stat in ['tr_loss']}
        # Other scalars to be logged
        misc: Dict[str, Union[float, List[float]]] = {misc: [] for misc in ['mean_target']}
        # Hold image tensors for real-time training sample visualization in tensorboard
        images: Dict[str, np.ndarray] = {}

        running_vx_size = 0  # Counts input sizes (number of pixels/voxels) of training batches
        timer = Timer()
        batch_iter = tqdm(
            self.train_loader,
            'Training',
            total=len(self.train_loader),
            dynamic_ncols=True,
            **self.tqdm_kwargs
        )
        unlabeled_iter = None if self.unlabeled_dataset is None else iter(self.unlabeled_loader)
        for i, batch in enumerate(batch_iter):
            if self.step in self.extra_save_steps:
                self._save_model(f'_step{self.step}', verbose=True)

            if unlabeled_iter is not None:
                batch['unlabeled'] = next(unlabeled_iter)
            dloss, dout = self._train_step(batch)

            with torch.no_grad():
                loss = float(dloss)
                target = batch.get('target')
                mean_target = float(target.to(torch.float32).mean()) if target is not None else 0.
                misc['mean_target'].append(mean_target)
                stats['tr_loss'].append(loss)
                batch_iter.set_description(f'Training (loss {loss:.4f})')
                self._tracker.update_timeline([self._timer.t_passed, loss, mean_target])

            # Not using .get_lr()[-1] because ReduceLROnPlateau does not implement get_lr()
            misc['learning_rate'] = self.optimizer.param_groups[0]['lr']  # LR for the this iteration
            self._scheduler_step(loss)

            running_vx_size += batch['inp'].numel()

            self._incr_step(max_runtime, max_steps)
            if i == len(self.train_loader) - 1 or self.terminate:
                # Last step in this epoch or in the whole training
                # Preserve last training batch and network output for later visualization
                images['inp'] = batch['inp'].numpy()
                images['fname'] = batch.get('fname')
                if 'target' in batch:
                    images['target'] = batch['target'].numpy()
                if 'unlabeled' in batch:
                    images['unlabeled'] = batch['unlabeled']
                images['out'] = dout.detach().cpu().numpy()
                self._put_current_attention_maps_into(images)

            if self.terminate:
                break

        stats['tr_loss_std'] = np.std(stats['tr_loss'])
        misc['tr_speed'] = len(self.train_loader) / timer.t_passed
        misc['tr_speed_vx'] = running_vx_size / timer.t_passed / 1e6  # MVx

        return stats, misc, images

    def _put_current_attention_maps_into(self, images):
        if getattr(self.model, 'attention', None):
            for i in range(len(self.model.up_convs)):
                att = self.model.up_convs[i].att[0][0].detach().cpu().numpy()
                if att.ndim == 3:
                    att = att[att.shape[0] // 2]
                images[f'att{i}'] = att

    def _incr_step(self, max_runtime, max_steps):
        """Increment the current training step counter"""
        self.step += 1
        if self.step >= max_steps:
            logger.info(f'max_steps ({max_steps}) exceeded. Terminating...')
            self.terminate = True
        if datetime.datetime.now() >= self.end_time:
            logger.info(f'max_runtime ({max_runtime} seconds) exceeded. Terminating...')
            self.terminate = True

    def _scheduler_step(self, loss):
        """Update schedules"""
        for sched in self.schedulers.values():
            # support ReduceLROnPlateau; doc. uses validation loss instead
            # http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
            if 'metrics' in inspect.signature(sched.step).parameters:
                sched.step(metrics=loss)
            else:
                sched.step()
        # Append LR of the next iteration (after sched.step()) for local LR minima detection
        self._lr_nhood.append(self.optimizer.param_groups[0]['lr'])
        self._handle_lr()

    def _handle_lr(self) -> None:
        r"""Handle quasi-periodic learning rate schedulers that lower the
        learning rate to local minima but then ramp it up again
        (Cosine Annealing, SGDR, Cyclical LRs etc.).

        Model saving is triggered when a local minimum of learning rates is
        detected. For the motivation of this behavior, see
        https://arxiv.org/abs/1704.00109. The saved models can be used to build
        an ensemble.

        Local minima are found by checking for the simple criterion
        :math:`\lr_{t-1}` > \lr{t} < lr{t+1}`.

        If an SWA (Stochastic Weight Averaging) optimizer is detected, the SWA
        algorithm is performed (see https://arxiv.org/abs/1803.05407) and the
        resulting model is also saved, marked by the "_swa" file name suffix.

        .. note::
            The saved SWA model performs batch norm statistics correction
            only on a limited number of batches from the ``self.train_loader``
            (currently hardcoded to 10),
            so if the model uses batch normalization with running statistics and
            you suspect that this amount of batches won't be representative
            enough for your input data distribution, you might want to ensure a
            good estimate yourself by running
            :py:meth:`elektronn3.trainer.SWA.bn_update()` on the model with a
            larger number of input batches after loading the model for
            inference.
        """
        if len(self._lr_nhood) < 3:
            return  # Can't get lrs, but at this early stage it's also not relevant
        last_lr = self._lr_nhood[-3]
        curr_lr = self._lr_nhood[-2]
        next_lr = self._lr_nhood[-1]
        if last_lr > curr_lr < next_lr:
            logger.info(
                f'Local learning rate minimum {curr_lr:.2e} detected at step '
                f'{self.step}. Saving model...')
            self._save_model(suffix=f'_minlr_step{self.step}')
            # Handle Stochastic Weight Averaging optimizer if SWA is used
            if isinstance(self.optimizer, SWA):
                # TODO: Make bn_update configurable (esp. number of batches)
                self.optimizer.update_swa()  # Put current model params into SWA buffer
                self.optimizer.swap_swa_sgd()  # Perform SWA and write results into model params
                has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in self.model.modules())
                if has_bn:  # Perform batch norm correction
                    try:
                        max_bn_corr_batches = 10  # Batches to use to correct SWA batchnorm stats
                        # We're assuming here that len(self.train_loader), which is an upper bound for
                        #  len(swa_loader), is sufficient for a good stat estimation
                        swa_loader = islice(self.train_loader, max_bn_corr_batches)
                        # This may be expensive (comparable to validation computations)
                        SWA.bn_update(swa_loader, self.model, device=self.device)
                        self._save_model(suffix='_swa', verbose=False)
                    except:
                        logger.exception(
                            'SWA helper bn_update has failed. SWA model will be saved with incorrect '
                            'batchnorm statistics. Please make sure to manually correct the BN stats '
                            'before deploying the model.'
                        )
                        self._save_model(suffix='_swa_todo_batchnorm_corr', verbose=False)
                else:  # No batch norm -> save model directly
                    self._save_model(suffix='_swa', verbose=False)
                self.optimizer.swap_swa_sgd()  # Swap back model to the original state before SWA

    @torch.no_grad()
    def _validate(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        self.model.eval()  # Set dropout and batchnorm to eval mode

        val_loss = []
        outs = []
        targets = []
        stats = {name: [] for name in self.valid_metrics.keys()}
        batch_iter = tqdm(
            enumerate(self.valid_loader),
            'Validating',
            total=len(self.valid_loader),
            dynamic_ncols=True,
            **self.tqdm_kwargs
        )
        for i, batch in batch_iter:
            # Everything with a "d" prefix refers to tensors on self.device (i.e. probably on GPU)
            inp = batch['inp']
            target = batch.get('target')
            target_class = batch.get('class')
            dinp = inp.to(self.device, non_blocking=True)
            dtarget = target.to(self.device, non_blocking=True) if target is not None else None
            dtarget_class = target_class.to(self.device, non_blocking=True) if target_class is not None else None
            with amp.autocast(enabled=self.mixed_precision):
                dout = self.model(dinp)
                if dtarget is None:  # Use self-supervised unary loss function
                    val_loss.append(self.ss_criterion(dout).item())
                elif dtarget_class is not None:
                    val_loss.append(self.criterion(dout, dtarget, dtarget_class).item())
                else:
                    val_loss.append(self.criterion(dout, dtarget).item())
            out = dout.detach().cpu()
            outs.append(out)
            targets.append(target)

        images = {
            'inp': inp.numpy(),
            'out': out.numpy(),
            'target': None if target is None else target.numpy(),
            'fname': batch.get('fname'),
        }
        self._put_current_attention_maps_into(images)

        stats['val_loss'] = np.mean(val_loss)
        stats['val_loss_std'] = np.std(val_loss)

        for name, evaluator in self.valid_metrics.items():
            mvals = [evaluator(target, out) for target, out in zip(targets, outs)]
            if np.all(np.isnan(mvals)):
                stats[name] = np.nan
            else:
                stats[name] = np.nanmean(mvals)

        # # This code is currently commented out because it's quite slow. TODO: Speed up by computing softmax on GPU above
        # # Plot per-class PR curves if a supported classification scenario is detected.
        # if out.ndim == target.ndim + 1 and self.inference_kwargs.get('apply_softmax'):
        #     softmax_outs = torch.stack(outs).softmax(2)  # Apply softmax in dim=2 because of additional stack dim
        #     for c in range(out.shape[1]):
        #         self.tb.add_pr_curve(
        #             f'pr_c{c}',
        #             labels=torch.stack(targets),
        #             predictions=torch.stack([so[:, c] for so in softmax_outs]),
        #             global_step=self.step
        #         )

        self.model.train()  # Reset model to training mode

        return stats, images

# TODO: Instead of using specific keys like val_loss, enable passing info as an
#       extra dict whose contents will be added to the state_dict
    def _save_model(
            self,
            suffix: str = '',
            unwrap_parallel: bool = True,
            verbose: bool = True,
            val_loss=np.nan
    ) -> None:
        """Save/serialize trained model state to files.

        Writes the following files in the ``self.save_path``:

        - ``state_dicts.pth`` contains the a dict that holds the ``state_dict``
          of the trained model, the ``state_dict`` of the optimizer and
          some meta information (global step, epoch, best validation loss)
          The included parameters can be read and used to overwrite another
          model's ``state_dict``. The model code (architecture) itself is not
          included in this file.
        - ``model.pt`` contains a pickled version of the complete model,
          including the trained weights. You can simply
          ``model = torch.load('model.pt')`` to obtain the full model and its
          training state. This will not work if the source code relevant to de-
          serializing the model object has changed! If this is is the case,
          you will need to use the ``state_dict.pth`` to extract parameters and
          manually load them into a model.
        - ``model.pts`` contains the model in the ``torch.jit`` ScriptModule
          serialization format. If ``model`` is not already a ``ScriptModule``
          and ``self.save_jit`` is not ``None``, a ScriptModule form of the
          ``model`` will be created on demand.

        Args:
            suffix: If defined, this string will be added before the file
                extensions of the respective files mentioned above.
            unwrap_parallel: If ``True`` (default) and the model uses a parallel
                module wrapper like ``torch.nn.DataParallel``, this is
                automatically detected and the wrapped model is saved directly
                to make later deserialization easier. This can be disabled by
                setting ``unwrap_parallel=False``.
            verbose: If ``True`` (default), log infos about saved models at
                log-level "INFO" (which appears in stdout). Else, only silently
                log with log-level "DEBUG".
            val_loss: Stores the validation loss
                (default value if not supplied: NaN)
        """
        log = logger.info if verbose else logger.debug

        model = self.model

        model_trainmode = model.training

        # We do this awkard check because there are too many different
        # parallel wrappers in PyTorch and some of them have changed names
        # in different releases (DataParallel, DistributedDataParallel{,CPU}).
        is_wrapped = (
            hasattr(model, 'module') and
            'parallel' in str(type(model)).lower() and
            isinstance(model.module, torch.nn.Module)
        )
        if is_wrapped and unwrap_parallel:
            # If a parallel wrapper was used, the only thing we should save
            # is the model.module, which contains the actual model and params.
            # If we saved the wrapped module directly, deserialization would
            # get unnecessarily difficult.
            model = model.module

        state_dict_path = os.path.join(self.save_path, f'state_dict{suffix}.pth')
        model_path = os.path.join(self.save_path, f'model{suffix}.pt')

        try:
            lr_sched_state = self.schedulers['lr'].state_dict()
        except:  # No valid scheduler in use
            lr_sched_state = None

        info = {
            'global_step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'val_loss': val_loss,
            'inference_kwargs': self.inference_kwargs,
            'elektronn3.__version__': elektronn3.__version__,
            'env_info': collect_env.get_pretty_env_info()
        }
        # Make sure everything is a string (if inference_kwargs contains a
        #  transform object, it may not be picklable)
        info = {k: str(v) for k, v in info.items()}

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_sched_state_dict': lr_sched_state,
            'scaler_state_dict': self.scaler.state_dict(),
            'info': info
        }, state_dict_path)
        log(f'Saved state_dict as {state_dict_path}')
        pts_model_path = f'{model_path}s'
        try:
            # Try saving directly as an uncompiled nn.Module
            torch.save(model, model_path)
            log(f'Saved model as {model_path}')
            if self.save_jit == 'script':  # Compile directly for serialization
                jitmodel = torch.jit.script(model)
            elif self.save_jit == 'trace':  # Trace and serialize the model in eval mode
                if self.example_input is None:
                    raise ValueError('If save_jit="trace", example_input needs to be specified.')
                with warnings.catch_warnings():
                    # It's enough to be warned once during initial tracing
                    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                    jitmodel = torch.jit.trace(model.eval(), self.example_input.to(self.device))
            if self.save_jit is not None:  # Save jit model, either from script or trace
                jitmodel.save(pts_model_path)
                log(f'Saved jitted model ({self.save_jit}) as {pts_model_path}')
        except (TypeError, PickleError) as exc:
            # If model is already a ScriptModule, it can't be saved with torch.save()
            # Use ScriptModule.save() instead in this case.
            # Using the file extension '.pts' to show it's a ScriptModule.
            if isinstance(model, torch.jit.ScriptModule):
                model_path += 's'
                model.save(pts_model_path)
                log(f'Saved jitted model as {pts_model_path}')
            else:
                raise exc
        finally:
            # Reset training state to the one it had before this function call,
            # because it could have changed with the model.eval() call above.
            model.training = model_trainmode
        if os.path.isfile(pts_model_path):
            with zipfile.ZipFile(pts_model_path, 'a', compression=zipfile.ZIP_DEFLATED) as zfile:
                infostr = pprint.pformat(info, indent=2, width=120)
                zfile.writestr('info.txt', infostr)

    def _log_basic(self, stats, misc):
        """Log to stdout and text log file"""
        tr_loss = np.mean(stats['tr_loss'])
        val_loss = np.mean(stats['val_loss'])
        lr = misc['learning_rate']
        tr_speed = misc['tr_speed']
        tr_speed_vx = misc['tr_speed_vx']
        t = pretty_string_time(self._timer.t_passed)
        text = f'step={self.step:06d}, tr_loss={tr_loss:.3f}, val_loss={val_loss:.3f}, '
        text += f'lr={lr:.2e}, {tr_speed:.2f} it/s, {tr_speed_vx:.2f} MVx/s, {t}'
        logger.info(text)

    def _log_to_tensorboard(
            self,
            stats: Dict,
            misc: Dict,
            tr_images: Dict,
            val_images: Optional[Dict] = None,
            file_stats: Optional[Dict] = None,
    ) -> None:
        """Create visualizations, make preview predictions, log and plot to tensorboard"""
        if self.tb:
            try:
                self._tb_log_scalars(stats, 'stats')
                self._tb_log_scalars(misc, 'misc')
                if self.preview_batch is not None:
                    if self.epoch % self.preview_interval == 0 or self.epoch == 1:
                        # TODO: Also save preview inference results in a (3D) HDF5 file
                        self.preview_plotting_handler(self)
                self.sample_plotting_handler(self, tr_images, group='tr_samples')
                if val_images is not None:
                    self.sample_plotting_handler(self, val_images, group='val_samples')
                if file_stats is not None:
                    self._tb_log_scalars(file_stats, 'file_stats')
                self._tb_log_histograms()
            except Exception:
                logger.exception('Error occured while logging to tensorboard:')

    def _log_to_history_tracker(self, stats: Dict, misc: Dict) -> None:
        """Update history tracker and plot stats (kind of made obsolete by tensorboard)"""
        # TODO: Decide what to do with this, now that most things are already in tensorboard.
        if self._tracker.history.length > 0:
            tr_loss_gain = self._tracker.history[-1][2] - np.mean(stats['tr_loss'])
        else:
            tr_loss_gain = 0
        if not stats.get('tr_accuracy'):
            tr_accuracy = nan
        else:
            tr_accuracy = np.nanmean(stats['tr_accuracy'])
        val_accuracy = stats.get('val_accuracy', nan)
        self._tracker.update_history([
            self.step, self._timer.t_passed, np.mean(stats['tr_loss']), np.mean(stats['val_loss']),
            tr_loss_gain, tr_accuracy, val_accuracy, misc['learning_rate'], 0, 0
        ])
        # Plot tracker stats to pngs in save_path
        self._tracker.plot(self.save_path)

    def _tb_log_scalars(
            self,
            scalars: Dict[str, float],
            tag: str = 'default'
    ) -> None:
        for key, value in scalars.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                for i in range(len(value)):
                    if not np.isnan(value[i]):
                        self.tb.add_scalar(f'{tag}/{key}', value[i], self.step - len(value) + i)
            elif not np.isnan(value):
                self.tb.add_scalar(f'{tag}/{key}', value, self.step)

    def _tb_log_histograms(self) -> None:
        """Log histograms of model parameters and their current gradients.

        Make sure to run this between ``backward()`` and ``zero_grad()``,
        because otherwise gradient histograms will only consist of zeros.
        """
        for name, param in self.model.named_parameters():
            self.tb.add_histogram(f'param/{name}', param, self.step)
            grad = param.grad if param.grad is not None else torch.tensor(0)
            self.tb.add_histogram(f'grad/{name}', grad, self.step)

    def _preview_inference(
            self,
            inp: np.ndarray,
            inference_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        if self.out_channels is None:
            raise RuntimeError('Can\'t do preview prediction if Trainer.out_channels is not set.')
        out_shape = (self.out_channels, *inp.shape[2:])
        predictor = Predictor(
            model=self.model,
            device=self.device,
            out_shape=out_shape,
            **inference_kwargs,
        )
        out = predictor.predict(inp)
        return out


class Backup:
    """ Backup class for archiving training script, src folder and environment info.
    Should be used for any future archiving needs.

    Args:
        script_path: The path to the training script. Eg. train_unet_neurodata.py
        save_path: The path where the information is archived.
        extra_content: Dictionary of {filename: content} entries, where content
            is a string that should be written to a file with the specified name.

    """
    def __init__(self, script_path, save_path, extra_content=None):
        self.script_path = script_path
        self.save_path = save_path
        self.extra_content = extra_content

    def archive_backup(self):
        """Archiving the source folder, the training script and environment info.

        The training script is saved with the prefix "0-" to distinguish from regular scripts.
        Environment information equivalent to the output of ``python -m torch.utils.collect_env``
        is saved in a file named "env_info.txt".
        """

        # Archiving the Training script
        shutil.copyfile(self.script_path, self.save_path + '/0-' + os.path.basename(self.script_path))
        os.chmod(self.save_path + '/0-' + os.path.basename(self.script_path), 0o755)
        # Archiving the src folder
        pkg_path = os.path.dirname(arch_src)
        backup_path = os.path.join(self.save_path, 'src_backup')
        shutil.make_archive(backup_path, 'gztar', pkg_path)

        # Archiving the Environment Info
        env_info = collect_env.get_pretty_env_info()
        with open(self.save_path + '/env_info.txt', 'w') as f:
            f.write(env_info)
        if self.extra_content is not None:
            for fname, content in self.extra_content.items():
                with open(f'{self.save_path}/{fname}', 'w') as f:
                    f.write(content)


def findcudatensors() -> Tuple[int, List[torch.Tensor]]:
    """Find currently living tensors that are allocated on cuda device memory.
    This can be used for debugging memory leaks:
    If ``findcudatensors()[0]`` grows unexpectedly between GPU computations,
    you can look at the returned ``tensors`` list to find out what tensors
    are currently allocated, for example
    ``print([x.shape for x in findcudatensors()[1])``.

    Returns a tuple of

    - total memory usage of found tensors in MiB
    - a list of all of those tensors, ordered by size."""
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) \
                    and obj.device.type == 'cuda' \
                    and not isinstance(obj, torch.nn.Parameter):  # Exclude model params
                tensors.append(obj)
        except:
            pass
    tensors.sort(key=lambda x: x.numel())
    total_mib = sum(x.numel() * 32 for x in tensors) / 1024**2  # Assuming float32
    return total_mib, tensors
