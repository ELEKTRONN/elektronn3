# -*- coding: utf-8 -*-
# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert
import datetime
import logging
import os
import traceback
import shutil

from textwrap import dedent
from typing import Tuple, Dict, Optional, Callable, Any, Sequence

import inspect
import IPython
import numpy as np
import torch
import torch.utils.data
from skimage.color import label2rgb
from torch.optim.lr_scheduler import StepLR

from elektronn3.training.train_utils import Timer, pretty_string_time
from elektronn3.training.train_utils import DelayedDataLoader
from elektronn3.training.train_utils import HistoryTracker
from elektronn3.training import collect_env
from elektronn3.training import metrics
from elektronn3.data.utils import squash01
from elektronn3 import __file__ as arch_src

logger = logging.getLogger('elektronn3log')

try:
    from .tensorboard import TensorBoardLogger
    tensorboard_available = True
except:
    tensorboard_available = False
    logger.exception('Tensorboard not available.')


class NaNException(RuntimeError):
    """When a NaN value is detected"""
    pass


class Trainer:
    """ Training loop abstraction with IPython and tensorboard integration.

    Hitting Ctrl-C anytime during the training will drop you to the IPython
    training shell where you can access training data and make interactive
    changes.
    To continue training, hit Ctrl-D twice.
    If you want the process to terminate after leaving the shell, set
    ``self.terminate = True`` inside it and then hit Ctrl-D twice.


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
        valid_metrics: Validation metrics to be calculated on
            validation data after each training epoch. All metrics are logged
            to tensorboard.
        save_root: Root directory where training-related files are
            stored. Files are always written to the subdirectory
            ``save_root/exp_name/``.
        exp_name: Name of the training experiment. Determines the subdirectory
            to which files are written and should uniquely identify one
            training experiment.
            If ``exp_name`` is not set, it is auto-generated from the model
            name and a time stamp in the format ``'%y-%m-%d_%H-%M-%S'``.
        batchsize: Desired batch size of training samples.
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
        enable_tensorboard: If ``True``, tensorboard logging/plotting is
            enabled during training.
        tensorboard_root_path: Path to the root directory under which
            tensorboard log directories are created. Log ("event") files are
            written to a subdirectory that has the same name as the
            ``exp_name``.
            If ``tensorboard_root_path`` is not set, tensorboard logs are
            written to ``save_path`` (next to model checkpoints, plots etc.).
        model_has_softmax_outputs: If ``False`` (default),
            the softmax operation is performed on network outputs before
            plotting them, so raw network outputs get converted into predicted
            class probabilities.
            Set this to ``True`` if the output of ``model`` is already a
            softmax output.
        ignore_errors: If ``True``, the training process tries to ignore
            all errors and continue with the next batch if it encounters
            an error on the current batch.
            It's not recommended to use this. It's only helpful for certain
            debugging scenarios.
        ipython_on_error: If ``True``, errors during training (except
            C-level segfaults etc.) won't crash the whole training process,
            but drop to an IPython shell so errors can be inspected with
            access to the current training state.
        classes: Optionally specifies the different target
            classes for classification tasks. If this is not set manually,
            the ``Trainer`` checks if the ``train_dataset`` provides this
            value. If available, ``self.num_classes`` is set to
            ``self.train_dataset.classes``. Otherwise, it is set to
            ``None``.
            The ``classes`` attribute is used for plotting purposes and is
            not strictly required for training.
    """
    # TODO: Write logs of the text logger to a file in save_root. The file
    #       handler should be replaced (see elektronn3.logger module).
    # TODO: Maybe there should be an option to completely disable exception
    #       hooks and IPython integration, so Ctrl-C directly terminates.

    tb: TensorBoardLogger
    terminate: bool
    step: int
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    exp_name: str
    save_path: str  # Full path to where training files are stored
    num_classes: Optional[int]  # Number of different target classes in the train_dataset

    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            save_root: str,
            train_dataset: torch.utils.data.Dataset,
            valid_dataset: Optional[torch.utils.data.Dataset] = None,
            valid_metrics: Optional[Dict] = None,
            exp_name: Optional[str] = None,
            batchsize: int = 1,
            num_workers: int = 0,
            schedulers: Optional[Dict[Any, Any]] = None,
            overlay_alpha: float = 0.2,
            enable_tensorboard: bool = True,
            tensorboard_root_path: Optional[str] = None,
            model_has_softmax_outputs: bool = False,
            ignore_errors: bool = False,
            ipython_on_error: bool = False,
            classes: Optional[Sequence[int]] = None,
    ):
        self.ignore_errors = ignore_errors
        self.ipython_on_error = ipython_on_error
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.valid_metrics = valid_metrics
        self.overlay_alpha = overlay_alpha
        self.save_root = os.path.expanduser(save_root)
        self.batchsize = batchsize
        self.num_workers = num_workers
        # TODO: This could be automatically determined by parsing the model
        self.model_has_softmax_outputs = model_has_softmax_outputs

        self._tracker = HistoryTracker()
        self._timer = Timer()
        self._first_plot = True
        self._shell_info = dedent("""
            Entering IPython training shell. To continue, hit Ctrl-D twice.
            To terminate, set self.terminate = True and then hit Ctrl-D twice.
        """).strip()

        if exp_name is None:  # Auto-generate a name based on model name and ISO timestamp
            timestamp = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
            exp_name = model.__class__.__name__ + '__' + timestamp
        self.exp_name = exp_name
        self.save_path = os.path.join(save_root, exp_name)
        os.makedirs(self.save_path, exist_ok=True)  # TODO: Warn if directory already exists

        self.terminate = False
        self.step = 0
        if schedulers is None:
            schedulers = {'lr': StepLR(optimizer, 1000, 1)}  # No-op scheduler
        self.schedulers = schedulers

        # Determine optional dataset properties
        self.classes = classes
        self.num_classes = None
        if hasattr(self.train_dataset, 'classes'):
            self.classes = self.train_dataset.classes
            self.num_classes = len(self.train_dataset.classes)
        self.previews_enabled = hasattr(valid_dataset, 'preview_batch')\
            and valid_dataset.preview_shape is not None

        if not tensorboard_available and enable_tensorboard:
            enable_tensorboard = False
            logger.warning('Tensorboard is not available, so it has to be disabled.')
        self.tb = None  # Tensorboard handler
        if enable_tensorboard:
            if tensorboard_root_path is None:
                tb_path = self.save_path
            else:
                tensorboard_root_path = os.path.expanduser(tensorboard_root_path)
                tb_path = os.path.join(tensorboard_root_path, self.exp_name)
                os.makedirs(tb_path, exist_ok=True)
            # TODO: Make always_flush user-configurable here:
            self.tb = TensorBoardLogger(log_dir=tb_path, always_flush=False)

        self.train_loader = DelayedDataLoader(
            self.train_dataset, batch_size=self.batchsize, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            timeout=30  # timeout arg requires https://github.com/pytorch/pytorch/commit/1661370ac5f88ef11fedbeac8d0398e8369fc1f3
        )
        # num_workers is set to 0 for valid_loader because validation background processes sometimes
        # fail silently and stop responding, bringing down the whole training process.
        # This issue might be related to https://github.com/pytorch/pytorch/issues/1355.
        # The performance impact of disabling multiprocessing here is low in normal settings,
        # because the validation loader doesn't perform expensive augmentations, but just reads
        # data from hdf5s.
        if valid_dataset is not None:
            self.valid_loader = DelayedDataLoader(
                self.valid_dataset, self.batchsize, num_workers=0, pin_memory=False,
                timeout=30
            )
        self.best_val_loss = np.inf  # Best recorded validation loss

        self.valid_metrics = {} if valid_metrics is None else valid_metrics

    # TODO: Modularize, make some general parts reusable for other trainers.
    def train(self, max_steps: int = 1) -> None:
        """Train the network for ``max_steps`` steps.

        After each training epoch, validation performance is measured and
        visualizations are computed and logged to tensorboard."""
        while self.step < max_steps:
            try:
                # --> self.train()
                self.model.train()

                # Scalar training stats that should be logged and written to tensorboard later
                stats: Dict[str, float] = {'tr_loss': 0.0}
                # Other scalars to be logged
                misc: Dict[str, float] = {}
                # Hold image tensors for real-time training sample visualization in tensorboard
                images: Dict[str, torch.Tensor] = {}

                running_acc = 0
                running_mean_target = 0
                running_vx_size = 0
                timer = Timer()
                for inp, target in self.train_loader:
                    inp, target = inp.to(self.device), target.to(self.device)

                    # forward pass
                    out = self.model(inp)
                    loss = self.criterion(out, target)
                    if torch.isnan(loss):
                        logger.error('NaN loss detected! Aborting training.')
                        raise NaNException

                    # update step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Prevent accidental autograd overheads after optimizer step
                    inp.detach_()
                    target.detach_()
                    out.detach_()
                    loss.detach_()

                    # get training performance
                    stats['tr_loss'] += float(loss)
                    acc = metrics.bin_accuracy(target, out)  # TODO
                    mean_target = target.to(torch.float32).mean()
                    print(f'{self.step:6d}, loss: {loss:.4f}', end='\r')
                    self._tracker.update_timeline([self._timer.t_passed, float(loss), mean_target])

                    # Preserve training batch and network output for later visualization
                    images['inp'] = inp
                    images['target'] = target
                    images['out'] = out
                    # this was changed to support ReduceLROnPlateau which does not implement get_lr
                    misc['learning_rate'] = self.optimizer.param_groups[0]["lr"] # .get_lr()[-1]
                    # update schedules
                    for sched in self.schedulers.values():
                        # support ReduceLROnPlateau; doc. uses validation loss instead
                        # http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
                        if "metrics" in inspect.signature(sched.step).parameters:
                            sched.step(metrics=float(loss))
                        else:
                            sched.step()

                    running_acc += acc
                    running_mean_target += mean_target
                    running_vx_size += inp.numel()

                    self.step += 1
                    if self.step >= max_steps:
                        break
                stats['tr_accuracy'] = running_acc / len(self.train_loader)
                stats['tr_loss'] /= len(self.train_loader)
                misc['tr_speed'] = len(self.train_loader) / timer.t_passed
                misc['tr_speed_vx'] = running_vx_size / timer.t_passed / 1e6  # MVx
                mean_target = running_mean_target / len(self.train_loader)
                if self.valid_dataset is None:
                    stats['val_loss'], stats['val_accuracy'] = float('nan'), float('nan')
                else:
                    valid_stats = self.validate()
                    stats.update(valid_stats)


                # Update history tracker (kind of made obsolete by tensorboard)
                # TODO: Decide what to do with this, now that most things are already in tensorboard.
                if self.step // len(self.train_dataset) > 1:
                    tr_loss_gain = self._tracker.history[-1][2] - stats['tr_loss']
                else:
                    tr_loss_gain = 0
                self._tracker.update_history([
                    self.step, self._timer.t_passed, stats['tr_loss'], stats['val_loss'],
                    tr_loss_gain, stats['tr_accuracy'], stats['val_accuracy'], misc['learning_rate'], 0, 0
                ])  # 0's correspond to mom and gradnet (?)
                t = pretty_string_time(self._timer.t_passed)
                loss_smooth = self._tracker.loss._ema

                # Logging to stdout, text log file
                text = "%05i L_m=%.3f, L=%.2f, tr_acc=%05.2f%%, " % (self.step, loss_smooth, stats['tr_loss'], stats['tr_accuracy'])
                text += "val_acc=%05.2f%s, prev=%04.1f, L_diff=%+.1e, " % (stats['val_accuracy'], "%", mean_target * 100, tr_loss_gain)
                text += "LR=%.2e, %.2f it/s, %.2f MVx/s, %s" % (misc['learning_rate'], misc['tr_speed'], misc['tr_speed_vx'], t)
                logger.info(text)

                # Plot tracker stats to pngs in save_path
                self._tracker.plot(self.save_path)

                # Reporting to tensorboard logger
                if self.tb:
                    self.tb_log_scalars(stats, 'stats')
                    self.tb_log_scalars(misc, 'misc')
                    if self.previews_enabled:
                        self.tb_log_preview()
                    self.tb_log_sample_images(images, group='tr_samples')
                    self.tb.writer.flush()

                # Save trained model state
                self.save_model()
                if stats['val_loss'] < self.best_val_loss:
                    self.best_val_loss = stats['val_loss']
                    self.save_model(suffix='_best')
            except KeyboardInterrupt:
                IPython.embed(header=self._shell_info)
                if self.terminate:
                    return
            except Exception as e:
                traceback.print_exc()
                if self.ignore_errors:
                    # Just print the traceback and try to carry on with training.
                    # This can go wrong in unexpected ways, so don't leave the training unattended.
                    pass
                elif self.ipython_on_error:
                    print("\nEntering Command line such that Exception can be "
                          "further inspected by user.\n\n")
                    IPython.embed(header=self._shell_info)
                    if self.terminate:
                        return
                else:
                    raise e
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_path, f'model-final-{self.step:06d}.pth')
        )

    def save_model(self, suffix=''):
        """Save/serialize trained model state to files.

        Writes to two files in the ``self.save_path``:

        - ``state_dict.pth`` contains the ``state_dict`` of the trained model.
          The included parameters can be read and used to overwrite another
          model's ``state_dict``. The model code (architecture) itself is not
          included in this file.
        - ``model.pt`` contains a pickled version of the complete model, including
          the trained weights. You can simply
          ``model = torch.load('model.pt')`` to obtain the full model and its
          training state. This will not work if the source code relevant to de-
          serializing the model object has changed! If this is is the case,
          you will need to use the ``state_dict.pth`` to extract parameters and
          manually load them into a model.

        If ``suffix`` is defined, it will be added before the file extension.
        """
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_path, f'state_dict{suffix}.pth')
        )
        torch.save(self.model, os.path.join(self.save_path, f'model{suffix}.pt'))

    def validate(self) -> Dict[str, float]:
        self.model.eval()  # Set dropout and batchnorm to eval mode

        val_loss = 0
        stats = {name: 0 for name in self.valid_metrics.keys()}
        for inp, target in self.valid_loader:
            inp, target = inp.to(self.device), target.to(self.device)
            with torch.no_grad():
                out = self.model(inp)
                val_loss += self.criterion(out, target).item() / len(self.valid_loader)
                for name, evaluator in self.valid_metrics.items():
                    stats[name] += evaluator(target, out) / len(self.valid_loader)

        self.tb_log_sample_images(
            {'inp': inp, 'out': out, 'target': target},
            group='val_samples'
        )

        stats['val_loss'] = val_loss

        self.model.train()  # Reset model to training mode

        # TODO: Refactor: Either remove side effects (plotting)
        return stats

    def tb_log_scalars(
            self,
            scalars: Dict[str, float],
            tag: str = 'default'
    ) -> None:
        for key, value in scalars.items():
            self.tb.log_scalar(f'{tag}/{key}', value, self.step)

    @staticmethod
    def _get_batch2img_function(
            batch: torch.Tensor,
            z_plane: Optional[int] = None
    ) -> Callable[[torch.Tensor], np.ndarray]:
        """
        Defines ``batch2img`` function dynamically, depending on tensor shapes.

        ``batch2img`` slices a 4D or 5D tensor to (C, H, W) shape, moves it to
        host memory and converts it to a numpy array.
        By arbitrary choice, the first element of a batch is always taken here.
        In the 5D case, the D (depth) dimension is sliced at z_plane.

        This function is useful for plotting image samples during training.

        Args:
            batch: 4D or 5D tensor, used for shape analysis.
            z_plane: Index of the spatial plane where a 5D image tensor should
                be sliced. If not specified, this is automatically set to half
                the size of the D dimension.

        Returns:
            Function that slices a plottable 2D image out of a torch.Tensor
            with batch and channel dimensions.
        """
        if batch.dim() == 5:  # (N, C, D, H, W)
            if z_plane is None:
                z_plane = batch.shape[2] // 2
            assert z_plane in range(batch.shape[2])
            return lambda x: x[0, :, z_plane].cpu().numpy()
        elif batch.dim() == 4:  # (N, C, H, W)
            return lambda x: x[0, :].cpu().numpy()
        else:
            raise ValueError('Only 4D and 5D tensors are supported.')

    def tb_log_preview(
            self,
            z_plane: Optional[int] = None,
            group: str = 'preview_batch'
    ) -> None:
        """ Preview from constant region of preview batch data.

        This only works for datasets that have a ``preview_batch`` attribute.
        """
        inp_batch = self.valid_dataset.preview_batch[0].to(self.device)
        out_batch = preview_inference(self.model, inp_batch=inp_batch)
        if not self.model_has_softmax_outputs:
            out_batch = out_batch.softmax(1)  # Apply softmax before plotting

        batch2img = self._get_batch2img_function(out_batch, z_plane)

        out = batch2img(out_batch)
        pred = out.argmax(0)

        for c in range(out.shape[0]):
            self.tb.log_image(f'{group}/c{c}', out[c], self.step, cmap='gray')
        self.tb.log_image(f'{group}/pred', pred, self.step, num_classes=self.num_classes)

        # This is only run once per training, because the ground truth for
        # previews is constant (always the same preview inputs/targets)
        if self._first_plot:
            preview_inp, preview_target = self.valid_dataset.preview_batch
            inp = batch2img(preview_inp)[0]
            if preview_target.dim() == preview_inp.dim() - 1:
                # Unsqueeze C dimension in target so it matches batch2dim()'s expected shape
                preview_target = preview_target[:, None]
            target = batch2img(preview_target)[0]
            self.tb.log_image(f'{group}/inp', inp, step=0, cmap='gray')
            # Ground truth target for direct comparison with preview prediction
            self.tb.log_image(f'{group}/target', target, step=0, num_classes=self.num_classes)
            self._first_plot = False

    # TODO: There seems to be an issue with inp-target mismatches when batch_size > 1
    def tb_log_sample_images(
            self,
            images: Dict[str, torch.Tensor],
            z_plane: Optional[int] = None,
            group: str = 'sample'
    ) -> None:
        """Preview from last training/validation sample

        Since the images are chosen randomly from the training/validation set
        they come from random regions in the data set.

        Note: Training images are possibly augmented, so the plots may look
            distorted/weirdly colored.
        """

        out_batch = images['out']
        if not self.model_has_softmax_outputs:
            out_batch = out_batch.softmax(1)  # Apply softmax before plotting

        batch2img = self._get_batch2img_function(out_batch, z_plane)

        inp = batch2img(images['inp'])[0]
        target_batch_with_c = images['target'][:, None]
        target = batch2img(target_batch_with_c)[0]

        out = batch2img(out_batch)
        pred = out.argmax(0)

        self.tb.log_image(f'{group}/inp', inp, step=self.step, cmap='gray')
        self.tb.log_image(f'{group}/target', target, step=self.step, num_classes=self.num_classes)

        for c in range(out.shape[0]):
            self.tb.log_image(f'{group}/c{c}', out[c], step=self.step, cmap='gray')
        self.tb.log_image(f'{group}/pred', pred, step=self.step, num_classes=self.num_classes)

        inp01 = squash01(inp)  # Squash to [0, 1] range for label2rgb and plotting
        target_ov = label2rgb(target, inp01, bg_label=0, alpha=self.overlay_alpha)
        pred_ov = label2rgb(pred, inp01, bg_label=0, alpha=self.overlay_alpha)
        self.tb.log_image(f'{group}/target_overlay', target_ov, step=self.step, colorbar=False)
        self.tb.log_image(f'{group}/pred_overlay', pred_ov, step=self.step, colorbar=False)
        # TODO: Synchronize overlay colors with pred- and target colors
        # TODO: What's up with the colorbar in overlay plots?
        # TODO: When plotting overlay images, they appear darker than they should.
        #       This normalization issue gets worse with higher alpha values
        #       (i.e. with more contribution of the overlayed label map).
        #       Don't know how to fix this currently.


def preview_inference(
        model: torch.nn.Module,
        inp_batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()  # Set dropout and batchnorm to eval mode
    # Attention: Inference on Tensors with unexpected shapes can lead to errors!
    # Staying with multiples of 16 for lengths seems to work.
    with torch.no_grad():
        out_batch = model(inp_batch)
    model.train()  # Reset model to training mode

    return out_batch


class Backup:
    """ Backup class for archiving training script, src folder and environment info.
    Should be used for any future archiving needs.

    Args:
        script_path: The path to the training script. Eg. train_unet_neurodata.py
        save_path: The path where the information is archived.

    """
    def __init__(self, script_path, save_path):
        self.script_path = script_path
        self.save_path = save_path

    def archive_backup(self):
        """Archiving the source folder, the training script and environment info.

        The training script is saved with the prefix '0-' to distinguish from regular scripts.
        Some of the information saved in the env info is:
        PyTorch version: 0.4.0
        Is debug build: No
        CUDA used to build PyTorch: 8.0.61
        OS: CentOS Linux release 7.3.1611 (Core)
        GCC version: (GCC) 5.2.0
        CMake version: Could not collect
        Python version: 3.6
        Is CUDA available: Yes
        CUDA runtime version: 8.0.44
        GPU models and configuration:
        GPU 0: GeForce GTX 980 Ti
        GPU 1: GeForce GTX 980 Ti
        .
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
