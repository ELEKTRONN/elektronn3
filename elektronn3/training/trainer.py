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

from textwrap import dedent
from typing import Tuple, Dict, Optional, Union, Any

import inspect
import IPython
import numpy as np
import torch
import torch.utils.data
from skimage.color import label2rgb
from torch.optim.lr_scheduler import ExponentialLR, StepLR

from elektronn3.training.train_utils import Timer, pretty_string_time
from elektronn3.training.train_utils import DelayedDataLoader
from elektronn3.training.train_utils import HistoryTracker
from elektronn3.data.utils import save_to_h5, squash01
from elektronn3.data.cnndata import PatchCreator

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


class StoppableTrainer:
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
            ``dataset``.
        criterion: PyTorch loss that shall be used as the optimization
            criterion.
        optimizer: PyTorch optimizer that shall be used to update
            ``model`` weights according to the ``criterion`` in each
            iteration.
        dataset: PyTorch dataset (``data.Dataset``) which produces
            training samples when iterated over.
            ``StoppableTrainer`` currently has some assumptions about
            the behavior of the ``dataset``, e.g. that the length of
            the ``dataset`` has no special meaning except controlling how
            often validation, plotting etc. are performed during training.
            Currently only instances of
            :py:class:`elektronn3.data.cnndata.PatchCreator` are supported as
            the ``dataset``.
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
            Set ``num_workers=0`` if you want to debug the ``dataset``
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
        device: The device on which the network shall be trained.
        ignore_errors: If ``True``, the training process tries to ignore
            all errors and continue with the next batch if it encounters
            an error on the current batch.
            It's not recommended to use this. It's only helpful for certain
            debugging scenarios.
        ipython_on_error: If ``True``, errors during training (except
            C-level segfaults etc.) won't crash the whole training process,
            but drop to an IPython shell so errors can be inspected with
            access to the current training state.
    """
    # TODO: Consider merging tensorboard_root_path with save_root so we have everything in one place.
    # TODO: Write logs of the text logger to a file in save_root. The file
    #       handler should be replaced (see elektronn3.logger module).
    # TODO: Maybe there should be an option to completely disable exception
    #       hooks and IPython integration, so Ctrl-C directly terminates.
    # TODO: Try to support dataset implementations other than PatchCreator?
    #       (The problem is currently that the *one* dataset is expected to
    #       handle both training and validation via the ``.train()`` and
    #       ``.validate()`` switches and a preview batch is expected to be
    #       present.

    tb: TensorBoardLogger
    terminate: bool
    step: int
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader
    exp_name: str
    save_path: str  # Full path to where training files are stored

    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            device,  # torch.Device type is not available
            dataset: PatchCreator,
            save_root: str,
            exp_name: Optional[str] = None,
            batchsize: int = 1,
            num_workers: int = 0,
            schedulers: Optional[Dict[Any, Any]] = None,  # TODO: Define a Scheduler protocol. This needs typing_extensions.
            overlay_alpha: float = 0.2,
            enable_tensorboard: bool = True,
            tensorboard_root_path: Optional[str] = None,
            ignore_errors: bool = False,
            ipython_on_error: bool = True
    ):
        self.ignore_errors = ignore_errors
        self.ipython_on_error = ipython_on_error
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.dataset = dataset
        self.overlay_alpha = overlay_alpha
        self.save_root = os.path.expanduser(save_root)
        self.batchsize = batchsize
        self.num_workers = num_workers

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
            self.dataset, batch_size=self.batchsize, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            timeout=30  # timeout arg requires https://github.com/pytorch/pytorch/commit/1661370ac5f88ef11fedbeac8d0398e8369fc1f3
        )
        # num_workers is set to 0 for valid_loader because validation background processes sometimes
        # fail silently and stop responding, bringing down the whole training process.
        # This issue might be related to https://github.com/pytorch/pytorch/issues/1355.
        # The performance impact of disabling multiprocessing here is low in normal settings,
        # because the validation loader doesn't perform expensive augmentations, but just reads
        # data from hdf5s.
        self.valid_loader = DelayedDataLoader(
            self.dataset, self.batchsize, num_workers=0, pin_memory=False,
            timeout=30
        )

    # Yeah I know this is an abomination, but this monolithic function makes
    # it possible to access all important locals from within the
    # KeyboardInterrupt-triggered IPython shell.
    # TODO: Try to modularize it as well as possible while keeping the
    #       above-mentioned requirement. E.g. the history tracking stuff can
    #       be refactored into smaller functions
    def train(self, max_steps: int = 1) -> None:
        while self.step < max_steps:
            try:
                # --> self.train()
                self.model.train()
                self.dataset.train()

                # Scalar training stats that should be logged and written to tensorboard later
                stats: Dict[str, float] = {'tr_loss': 0.0}
                # Other scalars to be logged
                misc: Dict[str, float] = {}
                # Hold image tensors for real-time training sample visualization in tensorboard
                images: Dict[str, torch.Tensor] = {}

                numel = 0
                target_sum = 0
                incorrect = 0
                vx_size = 0
                timer = Timer()
                for inp, target in self.train_loader:
                    inp, target = inp.to(self.device), target.to(self.device)

                    # forward pass
                    out = self.model(inp)
                    loss = self.criterion(out, target)
                    if torch.isnan(loss):
                        logger.error('NaN loss detected! Check your hyperparams.')
                        raise NaNException

                    # update step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # get training performance
                    numel += int(target.numel())
                    target_sum += int(target.sum())
                    vx_size += inp.numel()
                    maxcl = maxclass(out)
                    # .ne() creates a ByteTensor, which leads to integer
                    # overflows when it is sum-reduced. Therefore it's
                    # necessary to cast to a LongTensor before reducing.
                    incorrect += int(maxcl.ne(target).long().sum())
                    stats['tr_loss'] += float(loss)
                    print(f'{self.step:6d}, loss: {loss:.4f}', end='\r')
                    self._tracker.update_timeline([self._timer.t_passed, float(loss), target_sum / numel])

                    # Preserve training batch and network output for later
                    # visualization (detached from the implicit autograd
                    # graph, so operations on them are not recorded and
                    # differentiated).
                    images['inp'] = inp.detach()
                    images['target'] = target.detach()
                    images['out'] = out.detach()
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
                    self.step += 1
                stats['tr_err'] = 100. * incorrect / numel
                stats['tr_loss'] /= len(self.train_loader)
                mean_target = target_sum / numel
                misc['tr_speed'] = len(self.train_loader) / timer.t_passed
                misc['tr_speed_vx'] = vx_size / timer.t_passed / 1e6  # MVx

                stats['val_loss'], stats['val_err'] = self.validate()

                if self.step // len(self.dataset) > 1:
                    tr_loss_gain = self._tracker.history[-1][2] - stats['tr_loss']
                else:
                    tr_loss_gain = 0
                self._tracker.update_history([self.step, self._timer.t_passed, stats['tr_loss'],
                                              stats['val_loss'], tr_loss_gain,
                                              stats['tr_err'], stats['val_err'], misc['learning_rate'], 0, 0])  # 0's correspond to mom and gradnet (?)
                t = pretty_string_time(self._timer.t_passed)
                loss_smooth = self._tracker.loss._ema
                text = "%05i L_m=%.3f, L=%.2f, tr=%05.2f%%, " % \
                       (self.step, loss_smooth, stats['tr_loss'],
                        stats['tr_err'])
                text += "vl=%05.2f%s, prev=%04.1f, L_diff=%+.1e, " \
                    % (stats['val_err'], "%", mean_target * 100, tr_loss_gain)
                text += "LR=%.2e, %.2f it/s, %.2f MVx/s, %s" \
                        % (misc['learning_rate'], misc['tr_speed'],
                           misc['tr_speed_vx'], t)
                # TODO: Log voxels/s
                logger.info(text)
                if self.tb:
                    self.tb_log_scalars(stats, misc)
                    self.tb_log_preview()
                    self.tb_log_sample_images(images, group='tr_samples')
                    self.tb.writer.flush()
                if self.save_path is not None:
                    self._tracker.plot(self.save_path)
                if self.save_path is not None:
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.save_path, f'model-{self.step:06d}.pth')
                    )
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

    def validate(self) -> Tuple[float, float]:
        self.dataset.validate()  # Switch dataset to validation sources
        self.model.eval()  # Set dropout and batchnorm to eval mode

        val_loss = 0
        incorrect = 0
        numel = 0
        for inp, target in self.valid_loader:
            inp, target = inp.to(self.device), target.to(self.device)
            with torch.no_grad():
                out = self.model(inp)
                numel += int(target.numel())
                val_loss += float(self.criterion(out, target))
                maxcl = maxclass(out)  # get the index of the max log-probability
                incorrect += int(maxcl.ne(target).long().sum())
        val_loss /= len(self.valid_loader)  # loss function already averages over batch size
        val_err = 100. * incorrect / numel
        # TODO: Plot some validation images (inp, pred, target, overlay)
        #       See tb_log_sample_images
        self.tb_log_sample_images(
            {'inp': inp, 'out': out, 'target': target},
            group='val_samples'
        )


        # Reset dataset and model to training mode
        self.dataset.train()
        self.model.train()

        return val_loss, val_err

    def tb_log_scalars(
            self,
            stats: Dict[str, float],
            misc: Dict[str, float]
    ) -> None:
        for key, value in stats.items():
            self.tb.log_scalar(f'stats/{key}', value, self.step)
        for key, value in misc.items():
            self.tb.log_scalar(f'misc/{key}', value, self.step)

    def tb_log_preview(
            self,
            z_plane: Optional[int] = None,
            group: str = 'preview_batch'
    ) -> None:
        """Preview from constant region of preview batch data"""
        _, out = preview_inference(self.model, device=self.device, dataset=self.dataset)

        if z_plane is None:
            z_plane = out.shape[2] // 2
        assert z_plane in range(out.shape[2])

        mcl = maxclass(out)
        pred = mcl[0, z_plane, ...].cpu().numpy()

        for c in range(out.shape[1]):
            c_out = out[0, c, z_plane, ...].cpu().numpy()
            self.tb.log_image(f'{group}/c{c}', c_out, self.step)
        self.tb.log_image(f'{group}/pred', pred, self.step)

        # This is only run once per training, because the ground truth for
        # previews is constant (always the same preview inputs/targets)
        if self._first_plot:
            preview_inp, preview_target = self.dataset.preview_batch
            inp = preview_inp[0, 0, z_plane, ...].cpu().numpy()
            target = preview_target[0, z_plane, ...].cpu().numpy()
            self.tb.log_image(f'{group}/inp', inp, step=0)
            # Ground truth target for direct comparison with preview prediction
            self.tb.log_image(f'{group}/target', target, step=0)
            self._first_plot = False

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

        out = images['out']

        if z_plane is None:
            z_plane = out.shape[2] // 2
        assert z_plane in range(out.shape[2])

        inp = images['inp'][0, 0, z_plane, ...].cpu().numpy()
        target = images['target'][0, z_plane].cpu().numpy()
        mcl = maxclass(out)
        pred = mcl[0, z_plane, ...].cpu().numpy()

        self.tb.log_image(f'{group}/inp', inp, step=self.step)
        self.tb.log_image(f'{group}/target', target, step=self.step)

        for c in range(out.shape[1]):
            c_out = out[0, c, z_plane, ...].cpu().numpy()
            self.tb.log_image(f'{group}/c{c}', c_out, step=self.step)
        self.tb.log_image(f'{group}/pred', pred, step=self.step)

        inp01 = squash01(inp)  # Squash to [0, 1] range for label2rgb and plotting
        target_ov = label2rgb(target, inp01, bg_label=0, alpha=self.overlay_alpha)
        pred_ov = label2rgb(pred, inp01, bg_label=0, alpha=self.overlay_alpha)
        self.tb.log_image(f'{group}/target_overlay', target_ov, step=self.step)
        self.tb.log_image(f'{group}/pred_overlay', pred_ov, step=self.step)
        # TODO: When plotting overlay images, they appear darker than they should.
        #       This normalization issue gets worse with higher alpha values
        #       (i.e. with more contribution of the overlayed label map).
        #       Don't know how to fix this currently.

# TODO: Move all the functions below out of trainer.py


def maxclass(class_predictions: torch.Tensor):
    """For each point in a tensor, determine the class with max. probability.

    Args:
        class_predictions: Tensor of shape (N, C, ...)

    Returns:
        Tensor of shape (N, ...)
    """
    maxcl = class_predictions.max(dim=1)[1]
    return maxcl


# TODO
def save_to_h5(fname: str, model_output: torch.Tensor):
    raise NotImplementedError

    maxcl = maxclass(model_output)  # TODO: Ensure correct shape
    save_to_h5(
        [maxcl, dataset.valid_d[0][0, :shape[0], :shape[1], :shape[2]].astype(np.float32)],
        fname,
        hdf5_names=["pred", "raw"]
    )
    save_to_h5(
        [np.exp(model_output.view([1, 2, shape[0], shape[1], shape[2]])[0, 1].cpu().numpy(), dtype=np.float32)],
        fname+"prob.h5",
        hdf5_names=["prob"]
    )


def preview_inference(
        model: torch.nn.Module,
        device,
        dataset: Optional[PatchCreator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()  # Set dropout and batchnorm to eval mode
    # Attention: Inference on Tensors with unexpected shapes can lead to errors!
    # Staying with multiples of 16 for lengths seems to work.
    if dataset is None:
        d, h, w = dataset.preview_shape
        inp = torch.rand(1, dataset.c_input, d, h, w, device=device)
    else:
        inp = dataset.preview_batch[0]
    with torch.no_grad():
        out = model(inp)
    model.train()  # Reset model to training mode

    return inp, out
