# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert
import datetime
import inspect
import logging
from math import nan
from pathlib import Path
from typing import Dict, List, Tuple, Union

import IPython
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from elektronn3.training import Trainer
from elektronn3.training import metrics
from elektronn3.training.train_utils import Timer
from elektronn3.training.trainer import NaNException

logger = logging.getLogger('elektronn3log')


class TrainerMulti(Trainer):
    """Experimental ``Trainer`` variant for specialized multilabel training.

    Not intended for general use. May move in the future."""

    def __init__(self, optimizer_iterations=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_iterations = optimizer_iterations
        assert(optimizer_iterations > 0)
        self.loss_crop = 16 # crop sample for loss calcuation by this amount

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
                stats, file_stats, misc, tr_sample_images = self._train(max_steps, max_runtime)
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
                self._log_to_tensorboard(stats, misc, tr_sample_images, val_sample_images, file_stats=file_stats)
                # Legacy non-tensorboard logging to files
                self._log_to_history_tracker(stats, misc)

                # Save trained model state
                self._save_model(val_loss=stats['val_loss'], verbose=False)  # Not verbose because it can get spammy.
                # TODO: Support other metrics for determining what's the "best" model?
                if stats['val_loss'] < self.best_val_loss:
                    self.best_val_loss = stats['val_loss']
                    self._save_model(suffix=f'_best{self.step}', val_loss=stats['val_loss'])
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

        out_channels = self.out_channels

        def _channel_metric(metric, c, out_channels=out_channels, mean=False):
            """Returns an evaluator that calculates the ``metric``
            and selects its value for channel ``c``."""

            def evaluator(target, out):
                #pred = metrics._argmax(out)
                m = metric(target, out, out_channels=out_channels, ignore=out_channels - 1, mean=mean)
                return m[c]

            return evaluator

        tr_evaluators = {**{
            f'tr_DSC_c{c}': _channel_metric(metrics.dice_coefficient, c=c) for c in range(out_channels)
        }, **{
            f'tr_precision_c{c}': _channel_metric(metrics.precision, c=c) for c in range(out_channels)
        }, **{
            f'tr_recall_c{c}': _channel_metric(metrics.precision, c=c) for c in range(out_channels)
        }}
        # Scalar training stats that should be logged and written to tensorboard later
        stats: Dict[str, Union[float, List[float]]] = {stat: [] for stat in ['tr_loss', 'tr_loss_mean', 'tr_accuracy']}
        stats.update({name: [] for name in tr_evaluators.keys()})
        file_stats = {}
        # Other scalars to be logged
        misc: Dict[str, Union[float, List[float]]] = {misc: [] for misc in ['mean_target']}
        # Hold image tensors for real-time training sample visualization in tensorboard
        images: Dict[str, np.ndarray] = {}

        self.model.train()
        self.optimizer.zero_grad()
        running_vx_size = 0  # Counts input sizes (number of pixels/voxels) of training batches
        timer = Timer()
        import gc
        gc.collect()
        batch_iter = tqdm(self.train_loader, 'Training', total=len(self.train_loader))
        for i, batch in enumerate(batch_iter):
            if self.step in self.extra_save_steps:
                self._save_model(f'_step{self.step}', verbose=True)
            # Everything with a "d" prefix refers to tensors on self.device (i.e. probably on GPU)
            inp, target = batch['inp'], batch['target']
            cube_meta = batch['cube_meta']
            fname = batch['fname']
            dinp = inp.to(self.device, non_blocking=True)
            dtarget = target[:,:,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop].to(self.device, non_blocking=True) if self.loss_crop else target.to(self.device, non_blocking=True)
            weight = cube_meta[0].to(device=self.device, dtype=self.criterion.weight.dtype, non_blocking=True)
            prev_weight = self.criterion.weight.clone()
            self.criterion.weight = weight

            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                ignore_mask = (1 - dtarget[0][-1]).view(1,1,*dtarget.shape[2:])
                dense_weight = self.criterion.weight.view(1,-1,1,1,1)
                positive_target_mask = (weight.view(1,-1,1,1,1) * dtarget)[0][1:-1].sum(dim=0).view(1,1,*dtarget.shape[2:]) # weighted targets w\ background and ignore
                needs_positive_target_mark = (dense_weight.sum() == 0).type(positive_target_mask.dtype)
                self.criterion.weight = ignore_mask * dense_weight + needs_positive_target_mark * positive_target_mask * prev_weight.view(1,-1,1,1,1)

            # forward pass
            dout = self.model(dinp)[:,:,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop] if self.loss_crop else self.model(dinp)

            #print(dout.dtype, dout.shape, dtarget.dtype, dtarget.shape, dout.min(), dout.max())
            dloss = self.criterion(dout, dtarget)
            #dcumloss = dloss if i == 0 else dcumloss + dloss
            #print(dloss, dloss.size())
            #dloss = (dloss * prev_weight * weight).mean()
            if torch.isnan(dloss).sum():
                logger.error('NaN loss detected! Aborting training.')
                raise NaNException

            if self.mixed_precision:
                from apex import amp
                with amp.scale_loss(dloss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # update step
                dloss.backward()

            if i % self.optimizer_iterations == self.optimizer_iterations - 1:
                self.optimizer.step()
                # TODO (lp): calling zero_grad() here makes gradients disappear from tb histograms
                self.optimizer.zero_grad()
                #loss2 = float(self.criterion(self.model(dinp), dtarget))
                #print(f'loss gain factor {np.divide(float(dloss), (float(dloss)-loss2))})')
            # End of core training loop on self.device

            with torch.no_grad():
                loss = float(dloss)
                # TODO: Evaluate performance impact of these copies and maybe avoid doing these so often
                out_class = dout.argmax(dim=1).detach().cpu()
                multi_class_target = target.argmax(1) if len(target.shape) > 4 else target  # TODO
                if self.loss_crop:
                    multi_class_target = multi_class_target[:,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop]
                acc = metrics.accuracy(multi_class_target, out_class, out_channels, mean=False).numpy()
                acc = np.average(acc[~np.isnan(acc)])#, weights=)
                mean_target = float(multi_class_target.to(torch.float32).mean())

                # import h5py
                # dsc5 = channel_metric(metrics.dice_coefficient, c=5, out_channels=out_channels)(multi_class_target, out_class)
                # after_step = '+' if i % self.optimizer_iterations == 0 else ''
                # with h5py.File(os.path.join(self.save_path, f'batch {self.step}{after_step} loss={float(dloss)} dsc5={dsc5}.h5'), "w") as f:
                #     f.create_dataset('raw', data=inp.squeeze(dim=0), compression="gzip")
                #     f.create_dataset('labels', data=multi_class_target.numpy().astype(np.uint16), compression="gzip")
                #     f.create_dataset('pred', data=dout.squeeze(dim=0).detach().cpu().numpy(), compression="gzip")

                if fname[0] not in file_stats:
                    file_stats[fname[0]] = []
                file_stats[fname[0]] += [float('nan')] * (i - len(file_stats[fname[0]])) + [loss]

                stats['tr_loss'].append(loss)
                stats['tr_loss_mean'] += [float('nan')] * (i - len(stats['tr_loss_mean']))
                if i % self.optimizer_iterations == self.optimizer_iterations - 1:
                    stats['tr_loss_mean'] += [np.mean(stats['tr_loss'][-self.optimizer_iterations:])]
                stats['tr_accuracy'].append(acc)
                for name, evaluator in tr_evaluators.items():
                    stats[name].append(evaluator(multi_class_target, out_class))

                misc['mean_target'].append(mean_target)
                # if loss-loss2 == 0 and not torch.any(out_class != multi_class_target):
                #     print('grad', self.model.up_convs[0].conv2.weight.grad)
                #     IPython.embed()
                #if loss - 0.99 < 1e-3:
                #    print('asd', loss, loss2)
                #    IPython.embed()
                batch_iter.set_description(f'Training (loss {loss:.4f})')
                #pbar.set_description(f'Training (loss {loss} / {float(dcumloss)})')
                #pbar.set_description(f'Training (loss {loss} / {np.divide(loss, (loss-loss2))})')
                self._tracker.update_timeline([self._timer.t_passed, loss, mean_target])

            self.criterion.weight = prev_weight

            # Not using .get_lr()[-1] because ReduceLROnPlateau does not implement get_lr()
            misc['learning_rate'] = self.optimizer.param_groups[0]['lr']  # LR for the this iteration
            # update schedules
            for sched in self.schedulers.values():
                # support ReduceLROnPlateau; doc. uses validation loss instead
                # http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
                if "metrics" in inspect.signature(sched.step).parameters:
                    sched.step(metrics=loss)
                else:
                    sched.step()
            # Append LR of the next iteration (after sched.step()) for local LR minima detection
            self._lr_nhood.append(self.optimizer.param_groups[0]['lr'])
            self._handle_lr()

            running_vx_size += inp.numel()

            #if stats['tr_loss_mean'][-1] < self.best_tr_loss:
            #   self.best_tr_loss = stats['tr_loss'][-1]
            #   self._save_model(suffix='_best_train', loss=stats['tr_loss'][-1])

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
                images['fname'] = Path(fname[0]).stem
                images['inp'] = inp.numpy()
                images['target'] = multi_class_target.numpy()
                images['out'] = dout.detach().cpu().numpy()

            if self.terminate:
                break

        stats['tr_loss_std'] = np.std(stats['tr_loss'])
        misc['tr_speed'] = len(self.train_loader) / timer.t_passed
        misc['tr_speed_vx'] = running_vx_size / timer.t_passed / 1e6  # MVx

        return stats, file_stats, misc, images

    @torch.no_grad()
    def _validate(self) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        self.model.eval()  # Set dropout and batchnorm to eval mode

        val_loss = []
        stats = {name: [] for name in self.valid_metrics.keys()}
        batch_iter = tqdm(self.valid_loader, 'Validating', total=len(self.valid_loader))
        for i, batch in enumerate(batch_iter):
            # Everything with a "d" prefix refers to tensors on self.device (i.e. probably on GPU)
            inp, target = batch['inp'], batch['target']
            cube_meta = batch['cube_meta']
            dinp = inp.to(self.device, non_blocking=True)
            dtarget = target[:,:,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop].to(self.device, non_blocking=True) if self.loss_crop else target.to(self.device, non_blocking=True)
            weight = cube_meta[0].to(device=self.device, dtype=self.criterion.weight.dtype, non_blocking=True)
            prev_weight = self.criterion.weight.clone()
            self.criterion.weight *= weight
            #self.criterion.pos_weight = self.criterion.weight

            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                ignore_mask = (1 - dtarget[0][-1]).view(1,1,*dtarget.shape[2:])
                dense_weight = self.criterion.weight.view(1,-1,1,1,1)
                positive_target_mask = (weight.view(1,-1,1,1,1) * dtarget)[0][1:-1].sum(dim=0).view(1,1,*dtarget.shape[2:]) # weighted targets w\ background and ignore
                needs_positive_target_mark = (dense_weight.sum() == 0).type(positive_target_mask.dtype)
                self.criterion.weight = ignore_mask * dense_weight + needs_positive_target_mark * positive_target_mask * prev_weight.view(1,-1,1,1,1)

            dout = self.model(dinp)[:,:,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop] if self.loss_crop else self.model(dinp)
            multi_class_target = target.argmax(1) if len(target.shape) > 4 else target
            if self.loss_crop:
                multi_class_target = multi_class_target[:,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop,self.loss_crop:-self.loss_crop]
            val_loss.append(self.criterion(dout, dtarget).item())
            out = dout.detach().cpu()
            out_class = out.argmax(dim=1)
            self.criterion.weight = prev_weight
            for name, evaluator in self.valid_metrics.items():
                stats[name].append(evaluator(multi_class_target, out_class))
        images = {
            'fname': Path(batch['fname'][0]).stem,
            'inp': inp.numpy(),
            'out': out.numpy(),
            'target': multi_class_target.numpy()
        }

        stats['val_loss'] = np.mean(val_loss)
        stats['val_loss_std'] = np.std(val_loss)
        for name in self.valid_metrics.keys():
            stats[name] = np.nanmean(stats[name])

        self.model.train()  # Reset model to training mode

        return stats, images
