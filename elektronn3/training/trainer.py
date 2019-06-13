# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert
import datetime
import gc
import logging
import os
import shutil

from pickle import PickleError
from textwrap import dedent
from typing import Tuple, Dict, Optional, Callable, Any, Sequence, List

import inspect
import IPython
import numpy as np
import tensorboardX
import torch
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from elektronn3.training import handlers
from elektronn3.training.train_utils import Timer, pretty_string_time
from elektronn3.training.train_utils import DelayedDataLoader
from elektronn3.training.train_utils import HistoryTracker

from torch.utils import collect_env
from elektronn3.training import metrics
from elektronn3.inference import Predictor
from elektronn3 import __file__ as arch_src

logger = logging.getLogger('elektronn3log')


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
        example_input: An example input tensor that can be fed to the
            ``model``. This is used for JIT tracing during model serialization.
        enable_save_trace: If ``True``, the model is JIT-traced with
            ``example_input`` every time it is serialized to disk.
        batchsize: Desired batch size of training samples.
        preview_batch: Set a fixed input batch for preview predictions.
            If it is ``None`` (default), preview batch functionality will be
            disabled.
        preview_tile_shape
        preview_overlap_shape
        preview_interval: Determines how often to perform preview inference.
            Preview inference is performed every ``preview_interval`` epochs
            during training. Regardless of this value, preview predictions
            will also be performed once after epoch 1.
            (To disable preview predictions altogether, just set
            ``preview_batch = None``).
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
        enable_tensorboard: If ``True``, tensorboard logging/plotting is
            enabled during training.
        tensorboard_root_path: Path to the root directory under which
            tensorboard log directories are created. Log ("event") files are
            written to a subdirectory that has the same name as the
            ``exp_name``.
            If ``tensorboard_root_path`` is not set, tensorboard logs are
            written to ``save_path`` (next to model checkpoints, plots etc.).
        apply_softmax_for_prediction: If ``True`` (default),
            the softmax operation is performed on network outputs before
            plotting them, so raw network outputs get converted into predicted
            class probabilities.
            Set this to ``False`` if the output of ``model`` is already a
            softmax output or if you don't want softmax outputs at all.
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
        num_classes: Optionally specifies the total number of different target
            classes for classification tasks. If this is not set manually,
            the ``Trainer`` checks if the ``train_dataset`` provides this
            value. If available, ``self.num_classes`` is set to
            ``self.train_dataset.num_classes``. Otherwise, it is set to
            ``None``.
            The ``num_classes`` attribute is used for plotting purposes and is
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
    """
    # TODO: Write logs of the text logger to a file in save_root. The file
    #       handler should be replaced (see elektronn3.logger module).
    # TODO: Log useful info, like ELEKTRONN2 does
    # TODO: Support logging non-binary metrics

    tb: tensorboardX.SummaryWriter
    terminate: bool
    step: int
    epoch: int
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
            preview_batch: Optional[torch.Tensor] = None,
            preview_tile_shape: Optional[Tuple[int, ...]] = None,
            preview_overlap_shape: Optional[Tuple[int, ...]] = None,
            preview_interval: int = 5,
            offset: Optional[Sequence[int]] = None,
            exp_name: Optional[str] = None,
            example_input: Optional[torch.Tensor] = None,
            enable_save_trace: bool = False,
            batchsize: int = 1,
            num_workers: int = 0,
            schedulers: Optional[Dict[Any, Any]] = None,
            overlay_alpha: float = 0.2,
            enable_videos: bool = False,
            enable_tensorboard: bool = True,
            tensorboard_root_path: Optional[str] = None,
            apply_softmax_for_prediction: bool = True,
            ignore_errors: bool = False,
            ipython_shell: bool = True,
            num_classes: Optional[int] = None,
            sample_plotting_handler: Optional[Callable] = None,
            preview_plotting_handler: Optional[Callable] = None,
            mixed_precision: bool = False,
    ):
        if preview_batch is not None and\
                (preview_tile_shape is None or preview_overlap_shape is None):
            raise ValueError(
                'If preview_batch is set, you will also need to specify '
                'preview_tile_shape and preview_overlap_shape!'
            )
        if num_workers > 1 and 'PatchCreator' in str(type(train_dataset)):
            logger.warning(
                'Training with num_workers > 1 can cause instabilities if '
                'you are using PatchCreator.\nBe advised that PatchCreator '
                'might randomly deliver broken batches in your training and '
                'can crash it at any point of time.\n'
                'Please set num_workers to 1 or 0.\n'
            )
        self.ignore_errors = ignore_errors
        self.ipython_shell = ipython_shell
        self.device = device
        try:
            model.to(device)
        except RuntimeError as exc:
            if isinstance(model, torch.jit.ScriptModule):
                # "RuntimeError: to is not supported on TracedModules"
                # But .cuda() works for some reason. Using this messy
                # workaround in the hope that we can drop it soon.
                # TODO: Remove this when ScriptModule.to() is supported
                # See https://github.com/pytorch/pytorch/issues/7354
                if 'cuda' in str(self.device):  # (Ignoring device number!)
                    model.cuda()
            else:
                raise exc
        self.model = model
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.valid_metrics = valid_metrics
        self.preview_batch = preview_batch
        self.preview_tile_shape = preview_tile_shape
        self.preview_overlap_shape = preview_overlap_shape
        self.preview_interval = preview_interval
        self.offset = offset
        self.overlay_alpha = overlay_alpha
        self.save_root = os.path.expanduser(save_root)
        self.example_input = example_input
        self.enable_save_trace = enable_save_trace
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.apply_softmax_for_prediction = apply_softmax_for_prediction
        self.sample_plotting_handler = sample_plotting_handler
        self.preview_plotting_handler = preview_plotting_handler
        self.mixed_precision = mixed_precision

        self._tracker = HistoryTracker()
        self._timer = Timer()
        self._first_plot = True
        self._shell_info = dedent("""
            Entering IPython training shell. To continue, hit Ctrl-D twice.
            To terminate, set self.terminate = True and then hit Ctrl-D twice.
        """).strip()

        if self.mixed_precision:
            from apex import amp
            self.amp_handle = amp.init()

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
        logger.info(f'Writing files to save_path {self.save_path}/\n')

        self.terminate = False
        self.step = 0
        self.epoch = 0
        if schedulers is None:
            schedulers = {'lr': StepLR(optimizer, 1000, 1)}  # No-op scheduler
        self.schedulers = schedulers

        self.num_classes = num_classes
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
            # TODO: Make always_flush user-configurable here:
            self.tb = tensorboardX.SummaryWriter(log_dir=tb_path)

        self.train_loader = DelayedDataLoader(
            self.train_dataset, batch_size=self.batchsize, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            timeout=60
        )
        # num_workers is set to 0 for valid_loader because validation background processes sometimes
        # fail silently and stop responding, bringing down the whole training process.
        # This issue might be related to https://github.com/pytorch/pytorch/issues/1355.
        # The performance impact of disabling multiprocessing here is low in normal settings,
        # because the validation loader doesn't perform expensive augmentations, but just reads
        # data from hdf5s.
        if valid_dataset is not None:
            self.valid_loader = DelayedDataLoader(
                self.valid_dataset, self.batchsize, shuffle=True, num_workers=0, pin_memory=True,
                timeout=60
            )
        self.best_val_loss = np.inf  # Best recorded validation loss

        self.valid_metrics = {} if valid_metrics is None else valid_metrics

    # TODO: Modularize, make some general parts reusable for other trainers.
    def run(self, max_steps: int = 1, max_runtime=3600 * 24 * 7) -> None:
        """Train the network for ``max_steps`` steps.

        After each training epoch, validation performance is measured and
        visualizations are computed and logged to tensorboard."""
        self.start_time = datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(seconds=max_runtime)
        while not self.terminate:
            try:
                stats, misc, images = self._train(max_steps, max_runtime)
                self.epoch += 1

                if self.valid_dataset is None:
                    stats['val_loss'], stats['val_accuracy'] = float('nan'), float('nan')
                else:
                    valid_stats = self._validate()
                    stats.update(valid_stats)


                # Update history tracker (kind of made obsolete by tensorboard)
                # TODO: Decide what to do with this, now that most things are already in tensorboard.
                if self.step // len(self.train_dataset) > 1:
                    tr_loss_gain = self._tracker.history[-1][2] - np.mean(stats['tr_loss'])
                else:
                    tr_loss_gain = 0
                self._tracker.update_history([
                    self.step, self._timer.t_passed, np.mean(stats['tr_loss']), np.mean(stats['val_loss']),
                    tr_loss_gain, np.nanmean(stats['tr_accuracy']), stats['val_accuracy'], misc['learning_rate'], 0, 0
                ])  # 0's correspond to mom and gradnet (?)
                t = pretty_string_time(self._timer.t_passed)
                loss_smooth = self._tracker.loss._ema

                # Logging to stdout, text log file
                text = "%05i L_m=%.3f, L=%.2f, tr_acc=%05.2f%%, " % (self.step, loss_smooth, np.mean(stats['tr_loss']), np.nanmean(stats['tr_accuracy']))
                text += "val_acc=%05.2f%s, prev=%04.1f, L_diff=%+.1e, " % (stats['val_accuracy'], "%", np.mean(misc['mean_target']) * 100, tr_loss_gain)
                text += "LR=%.2e, %.2f it/s, %.2f MVx/s, %s" % (misc['learning_rate'], misc['tr_speed'], misc['tr_speed_vx'], t)
                logger.info(text)

                # Plot tracker stats to pngs in save_path
                self._tracker.plot(self.save_path)

                # Reporting to tensorboard logger
                if self.tb:
                    try:
                        self._tb_log_scalars(stats, 'stats')
                        self._tb_log_scalars(misc, 'misc')
                        if self.preview_batch is not None:
                            if self.epoch % self.preview_interval == 0 or self.epoch == 1:
                                # TODO: Also save preview inference results in a (3D) HDF5 file
                                self.preview_plotting_handler(self)
                        self.sample_plotting_handler(self, images, group='tr_samples')
                    except Exception:
                        logger.exception('Error occured while logging to tensorboard:')

                # Save trained model state
                self._save_model()
                # TODO: Support other metrics for determining what's the "best" model?
                if stats['val_loss'] < self.best_val_loss:
                    self.best_val_loss = stats['val_loss']
                    self._save_model(suffix='_best')
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

    def _train(self, max_steps, max_runtime):
        self.model.train()

        # Scalar training stats that should be logged and written to tensorboard later
        stats: Dict[str, list] = {stat: [] for stat in ['tr_loss', 'tr_accuracy']}
        # Other scalars to be logged
        misc: Dict[str, float] = {misc: [] for misc in ['mean_target']}
        # Hold image tensors for real-time training sample visualization in tensorboard
        images: Dict[str, np.ndarray] = {}

        running_vx_size = 0
        timer = Timer()
        pbar = tqdm(enumerate(self.train_loader), 'Training', total=len(self.train_loader))
        for i, (inp, target) in pbar:
            # Everything with a "d" prefix refers to tensors on self.device (i.e. probably on GPU)
            dinp = inp.to(self.device, non_blocking=True)
            dtarget = target.to(self.device, non_blocking=True)
            weight = cube_meta[0].to(device=self.device, dtype=self.criterion.weight.dtype, non_blocking=True)
            prev_weight = self.criterion.weight.clone()
            self.criterion.weight *= weight
            #self.criterion.weight = None
            #self.criterion.pos_weight = prev_weight * weight
            #self.criterion.pos_weight = self.criterion.pos_weight.view(-1,1,1,1)
            #self.criterion.weight = self.criterion.weight.view(-1,1,1,1)
            #self.criterion.pos_weight = self.criterion.weight

            # forward pass
            dout = self.model(dinp)

            #print(dout.dtype, dout.shape, dtarget.dtype, dtarget.shape, dout.min(), dout.max())
            dloss = self.criterion(dout, dtarget)
            #dcumloss = dloss if i == 0 else dcumloss + dloss
            #print(dloss, dloss.size())
            #dloss = (dloss * prev_weight * weight).mean()
            if torch.isnan(dloss).sum():
                logger.error('NaN loss detected! Aborting training.')
                raise NaNException

            # update step
            self.optimizer.zero_grad()
            if self.mixed_precision:
                with self.amp_handle.scale_loss(dloss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                dloss.backward()
            self.optimizer.step()
            # End of core training loop on self.device

            with torch.no_grad():
                loss = float(dloss)
            	# TODO: Evaluate performance impact of these copies and maybe avoid doing these so often
                out_class = dout.argmax(dim=1).detach().cpu()
                acc = metrics.accuracy(multi_class_target, out_class, num_classes)
                acc = np.average(acc[~np.isnan(acc)])#, weights=)
                mean_target = float(multi_class_target.to(torch.float32).mean())

                stats['tr_loss'].append(loss)
                stats['tr_accuracy'].append(acc)

                #if stats['tr_DSC_c3'][-1] < 5:
                #    IPython.embed()

                misc['mean_target'].append(mean_target)
                # if loss-loss2 == 0 and not torch.any(out_class != multi_class_target):
                #     print('grad', self.model.up_convs[0].conv2.weight.grad)
                #     IPython.embed()
                #if loss - 0.99 < 1e-3:
                #    print('asd', loss, loss2)
                #    IPython.embed()
                pbar.set_description(f'Training (loss {loss})')
                #pbar.set_description(f'Training (loss {loss} / {float(dcumloss)})')
                #pbar.set_description(f'Training (loss {loss} / {np.divide(loss, (loss-loss2))})')
                self._tracker.update_timeline([self._timer.t_passed, loss, mean_target])

            self.criterion.weight = prev_weight

            # this was changed to support ReduceLROnPlateau which does not implement get_lr
            misc['learning_rate'] = self.optimizer.param_groups[0]["lr"]  # .get_lr()[-1]
            # update schedules
            for sched in self.schedulers.values():
                # support ReduceLROnPlateau; doc. uses validation loss instead
                # http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
                if "metrics" in inspect.signature(sched.step).parameters:
                    sched.step(metrics=loss)
                else:
                    sched.step()

            running_vx_size += inp.numel()

            self.step += 1
            if self.step >= max_steps:
                logger.info(f'max_steps ({max_steps}) exceeded. Terminating...')
                self.terminate = True
            if datetime.datetime.now() >= self.end_time:
                logger.info(f'max_runtime ({max_runtime} seconds) exceeded. Terminating...')
                self.terminate = True
            if i == len(self.train_loader) - 1 or self.terminate:
                # Last step in this epoch or in the whole training
                # Preserve last training batch and network output for later visualization
                images['inp'] = inp.numpy()
                images['target'] = multi_class_target.numpy()
                images['out'] = dout.detach().cpu().numpy()

            if self.terminate:
                break

        misc['tr_speed'] = len(self.train_loader) / timer.t_passed
        misc['tr_speed_vx'] = running_vx_size / timer.t_passed / 1e6  # MVx

        return stats, misc, images

    def _validate(self) -> Dict[str, float]:
        self.model.eval()  # Set dropout and batchnorm to eval mode

        val_loss = []
        stats = {name: [] for name in self.valid_metrics.keys()}
        # TODO: Avoid unnecessary cpu -> gpu -> cpu moves, just save cpu tensors for later
        for inp, target in tqdm(self.valid_loader, 'Validating'):
            # Everything with a "d" prefix refers to tensors on self.device (i.e. probably on GPU)
            dinp = inp.to(self.device, non_blocking=True)
            dtarget = target.to(self.device, non_blocking=True)
            weight = cube_meta[0].to(device=self.device, dtype=self.criterion.weight.dtype, non_blocking=True)
            prev_weight = self.criterion.weight.clone()
            self.criterion.weight *= weight
            #self.criterion.pos_weight = self.criterion.weight

            with torch.no_grad():
                dout = self.model(dinp)
                val_loss.append(self.criterion(dout, dtarget).item())
                out = dout.detach().cpu()
                out_class = out.argmax(dim=1)
                self.criterion.weight = prev_weight
                for name, evaluator in self.valid_metrics.items():
                    stats[name].append(evaluator(multi_class_target, out_class))

        if self.tb:
            try:
                self.sample_plotting_handler(
                    self,
                    {
                        'inp': inp.numpy(),
                        'out': out.numpy(),
                        'target': multi_class_target.numpy()
                    },
                    group='val_samples'
                )
            except Exception:
                logger.exception('Error occured while logging to tensorboard:')

        stats['val_loss'] = np.mean(val_loss)
        for name in self.valid_metrics.keys():
            stats[name] = np.nanmean(stats[name])

        self.model.train()  # Reset model to training mode

        # TODO: Refactor: Remove side effects (plotting)
        return stats

    def _save_model(self, suffix: str = '', unwrap_parallel: bool = True) -> None:
        """Save/serialize trained model state to files.

        If the model uses a parallel wrapper like ``torch.nn.DataParallel``,
        this is automatically detected and the wrapped model is saved directly
        to make later deserialization easier. This can be disabled by setting
        ``unwrap_parallel=False``.

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
        # TODO: Document ScriptModule saving special cases
        # TODO: Logging
        model = self.model
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

        torch.save(model.state_dict(), state_dict_path)
        try:
            # Try saving directly as an uncompiled nn.Module
            torch.save(model, model_path)
            if self.example_input is not None and self.enable_save_trace:
                # Additionally trace and serialize the model in eval + train mode
                model_path += 's'
                traced = torch.jit.trace(model.eval(), self.example_input.to(self.device))
                traced.save(model_path)
                # Uncomment these lines if separate traces for train/eval are required:
                # traced_eval = torch.jit.trace(model.eval(), self.example_input.to(self.device))
                # traced_eval.save('eval_' + model_path)
                # traced_train = torch.jit.trace(model.train(), self.example_input.to(self.device))
                # traced_train.save('train_' + model_path)

        except (TypeError, PickleError) as exc:
            # If model is already a ScriptModule, it can't be saved with torch.save()
            # Use ScriptModule.save() instead in this case.
            # Using the file extension '.pts' to show it's a ScriptModule.
            if isinstance(model, torch.jit.ScriptModule):
                model_path += 's'
                model.save(model_path)
            else:
                raise exc

    def _tb_log_scalars(
            self,
            scalars: Dict[str, float],
            tag: str = 'default'
    ) -> None:
        for key, value in scalars.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                for i in range(len(value)):
                    self.tb.add_scalar(f'{tag}/{key}', value[i], self.step - len(value) + i)
            else:
                self.tb.add_scalar(f'{tag}/{key}', value, self.step)

    # TODO: Make more configurable
    # TODO: Inference on secondary GPU
    def _preview_inference(
            self,
            inp: np.ndarray,
            tile_shape: Optional[Tuple[int, ...]] = None,
            overlap_shape: Optional[Tuple[int, ...]] = None,
            verbose: bool = True,
    ) -> torch.Tensor:
        if self.num_classes is None:
            raise RuntimeError('Can\'t do preview prediction if Trainer.num_classes is not set.')
        out_shape = (self.num_classes, *inp.shape[2:])
        predictor = Predictor(
            model=self.model,
            device=self.device,
            batch_size=1,
            tile_shape=tile_shape,
            overlap_shape=overlap_shape,
            verbose=verbose,
            out_shape=out_shape,
            apply_softmax=self.apply_softmax_for_prediction,
        )
        out = predictor.predict(inp)
        return out


def __naive_preview_inference(  # Deprecated
        model: torch.nn.Module,
        inp_batch: torch.Tensor
) -> torch.Tensor:
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
