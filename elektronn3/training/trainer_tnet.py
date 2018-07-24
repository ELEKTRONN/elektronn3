# -*- coding: utf-8 -*-
# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Martin Drawitsch
import os
import traceback
from typing import Tuple, Dict, Optional
import inspect
import IPython
import torch
from torch.autograd import Variable
import torch.utils.data
from elektronn3.training.train_utils import Timer, pretty_string_time
from elektronn3.training.trainer import Trainer, logger, NaNException
import numpy as np


class TripletNetTrainer(Trainer):
    """
    Args:
        alpha: l2-norm regularization weight of latent space results

    """
    def __init__(self, alpha=1e-3, alpha2=1, *args, **kwargs):
        if len(args) > 0:
            raise ValueError("Please provide keyword arguments for "
                             "TripletNetTrainer init only.")
        model_discr = kwargs["model"][1]
        optim_discr = kwargs["optimizer"][1]
        kwargs["model"] = kwargs["model"][0]
        kwargs["optimizer"] = kwargs["optimizer"][0]
        super().__init__(**kwargs)
        self.alpha = alpha
        self.alpha2 = alpha2
        self.model_discr = model_discr.to(self.device)
        self.optimizer_discr = optim_discr

    # overwrite train and validate method
    def train(self, max_steps: int = 1) -> None:
        """Train the network for ``max_steps`` steps.

        After each training epoch, validation performance is measured and
        visualizations are computed and logged to tensorboard."""
        while self.step < max_steps:
            try:
                # --> self.train()
                self.model.train()

                # Scalar training stats that should be logged and written to tensorboard later
                stats: Dict[str, float] = {'tr_loss': .0, 'tr_loss_distance': .0,
                                           'tr_loss_z': .0, 'tr_loss_rep': .0,
                                           'tr_loss_discr': .0}
                # Other scalars to be logged
                misc: Dict[str, float] = {}
                # Hold image tensors for real-time training sample visualization in tensorboard
                images: Dict[str, torch.Tensor] = {}

                running_error = 0
                running_mean_target = 0
                running_vx_size = 0
                timer = Timer()
                for inp in self.train_loader:
                    if inp.size()[1] != 3:
                        raise ValueError("Data must not contain targets. "
                                         "Input data shape is assumed to be "
                                         "(N, 3, ch, x, y), where the first two"
                                         " images in each sample is the similar"
                                         " pair, while the third one is the "
                                         "distant one.")
                    inp0 = Variable(inp[:, 0].to(self.device))
                    inp1 = Variable(inp[:, 1].to(self.device))
                    inp2 = Variable(inp[:, 2].to(self.device))

                    # forward pass
                    dA, dB, z0, z1, z2 = self.model(inp0, inp1, inp2)
                    target = torch.FloatTensor(dA.size()).fill_(1).to(self.device)
                    target = Variable(target)
                    loss = self.criterion(dA, dB, target)
                    stats['tr_loss_distance'] += float(loss)
                    loss_z = torch.sum(z0.norm(2, dim=1) + z1.norm(2, dim=1) +
                                       z2.norm(2, dim=1))
                    stats['tr_loss_z'] += self.alpha * float(loss_z)
                    loss = loss + self.alpha * loss_z
                    if torch.isnan(loss):
                        logger.error('NaN loss detected! Aborting training.')
                        raise NaNException

                    # Normal update step using triplet loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Adversarial part to enforce latent variable distribution
                    # to be Normal
                    # generate latent representations
                    self.model.eval()
                    # generate 3 * latent space size Gaussian samples and compare to; draw from N(0, 2)
                    z_real_gauss = Variable(torch.randn(inp0.size()[0], z0.size()[-1] * 3) * 2).to(self.device)
                    _, _, z_fake_gauss0, z_fake_gauss1, z_fake_gauss2 = self.model(inp0, inp1, inp2)
                    z_fake_gauss = torch.squeeze(torch.cat((z_fake_gauss0, z_fake_gauss1, z_fake_gauss2), dim=1))
                    # Compute discriminator outputs and loss
                    D_real_gauss = self.model_discr(z_real_gauss)
                    D_fake_gauss = self.model_discr(z_fake_gauss)
                    D_loss = -torch.mean(torch.log(D_real_gauss) + torch.log(1 - D_fake_gauss))
                    D_loss.backward()  # Backprop loss
                    self.optimizer_discr.step()  # Apply optimization step

                    # calculate loss of representation network and update weights
                    self.model.train()  # Back to use dropout
                    # rebuild graph (model output) to get clean backprop.
                    _, _, z_fake_gauss0, z_fake_gauss1, z_fake_gauss2 = self.model(inp0, inp1, inp2)
                    z_fake_gauss = torch.squeeze(torch.cat((z_fake_gauss0, z_fake_gauss1, z_fake_gauss2), dim=1))
                    D_fake_gauss = self.model_discr(z_fake_gauss)

                    R_loss = self.alpha2 * -torch.mean(torch.log(D_fake_gauss))
                    R_loss.backward()
                    self.optimizer.step()

                    # Prevent accidental autograd overheads after optimizer step
                    inp.detach_()
                    target.detach_()
                    dA.detach_()
                    dB.detach_()
                    z0.detach_()
                    z1.detach_()
                    z2.detach_()
                    loss.detach_()
                    loss_z.detach_()
                    R_loss.detach_()
                    D_loss.detach_()

                    # get training performance
                    stats['tr_loss'] += float(loss)
                    stats['tr_loss_rep'] += float(R_loss)
                    stats['tr_loss_discr'] += float(D_loss)
                    stats['tr_loss'] += float(loss)
                    error = calculate_error(dA, dB)
                    mean_target = target.to(torch.float32).mean()
                    print(f'{self.step:6d}, loss: {loss:.4f}', end='\r')
                    self._tracker.update_timeline([self._timer.t_passed, float(loss), mean_target])

                    # Preserve training batch and network output for later visualization
                    images['inp_ref'] = inp0
                    images['inp_+'] = inp1
                    images['inp_-'] = inp2
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

                    running_error += error
                    running_mean_target += mean_target
                    running_vx_size += inp.numel()

                    self.step += 1
                    if self.step >= max_steps:
                        break
                stats['tr_err'] = float(running_error) / len(self.train_loader)
                stats['tr_loss'] /= len(self.train_loader)
                stats['tr_loss_rep'] /= len(self.train_loader)
                stats['tr_loss_discr'] /= len(self.train_loader)
                stats['tr_loss_distance'] /= len(self.train_loader)
                stats['tr_loss_z'] /= len(self.train_loader)
                misc['tr_speed'] = len(self.train_loader) / timer.t_passed
                misc['tr_speed_vx'] = running_vx_size / timer.t_passed / 1e6  # MVx
                mean_target = running_mean_target / len(self.train_loader)
                if self.valid_dataset is None:
                    stats['val_loss'], stats['val_err'] = float('nan'), float('nan')
                else:
                    stats['val_loss'], stats['val_err'] = self.validate()
                # TODO: Report more metrics, e.g. dice error

                # Update history tracker (kind of made obsolete by tensorboard)
                # TODO: Decide what to do with this, now that most things are already in tensorboard.
                if self.step // len(self.train_dataset) > 1:
                    tr_loss_gain = self._tracker.history[-1][2] - stats['tr_loss']
                else:
                    tr_loss_gain = 0
                self._tracker.update_history([
                    self.step, self._timer.t_passed, stats['tr_loss'], stats['val_loss'],
                    tr_loss_gain, stats['tr_err'], stats['val_err'], misc['learning_rate'], 0, 0
                ])  # 0's correspond to mom and gradnet (?)
                t = pretty_string_time(self._timer.t_passed)
                loss_smooth = self._tracker.loss._ema

                # Logging to stdout, text log file
                text = "%05i L_m=%.3f, L=%.2f, tr=%05.2f%%, " % (self.step, loss_smooth, stats['tr_loss'], stats['tr_err'])
                text += "vl=%05.2f%s, prev=%04.1f, L_diff=%+.1e, " % (stats['val_err'], "%", mean_target * 100, tr_loss_gain)
                text += "LR=%.2e, %.2f it/s, %.2f MVx/s, %s" % (misc['learning_rate'], misc['tr_speed'], misc['tr_speed_vx'], t)
                logger.info(text)

                # Plot tracker stats to pngs in save_path
                self._tracker.plot(self.save_path)

                # Reporting to tensorboard logger
                if self.tb:
                    self.tb_log_scalars(stats, misc)
                    if self.previews_enabled:
                        self.tb_log_preview()
                    self.tb_log_sample_images(images, group='tr_samples')
                    self.tb.writer.flush()

                # Save trained model state
                torch.save(
                    self.model.state_dict(),
                    # os.path.join(self.save_path, f'model-{self.step:06d}.pth')  # Saving with different file names leads to heaps of large files,
                    os.path.join(self.save_path, 'model-checkpoint.pth')
                )
                # TODO: Also save "best" model, not only the latest one, which is often overfitted.
                #       -> "best" in which regard? Lowest validation loss, validation error?
                #          We can't blindly trust these metrics and may have to calculate
                #          additional metrics (with focus on object boundary correctness).
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
        self.model.eval()  # Set dropout and batchnorm to eval mode

        val_loss = 0.
        incorrect = 0.
        for inp in self.valid_loader:
            inp0 = inp[:, 0].to(self.device)
            inp1 = inp[:, 1].to(self.device)
            inp2 = inp[:, 2].to(self.device)
            with torch.no_grad():
                dA, dB, z0, z1, z2 = self.model(inp0, inp1, inp2)
                diff_ref = np.linalg.norm(z0-z1)
                diff_neg = np.linalg.norm(z0-z2)
                if diff_ref < 1e-7 and diff_neg < 1e-7:
                    logger.warning("Difference between reference and negative sample"
                                " is almost zero: {} and {}"
                                "".format(diff_ref, diff_neg))
                target = torch.FloatTensor(dA.size()).fill_(1).to(self.device)
                target = Variable(target)
                val_loss += float(self.criterion(dA, dB, target))
                incorrect += calculate_error(dA, dB)
        val_loss /= len(self.valid_loader)  # loss function already averages over batch size
        val_err = incorrect / len(self.valid_loader)
        self.tb_log_sample_images(
            {'inp_ref': inp0, 'inp_+': inp1, 'inp_-': inp2},
            group='val_samples'
        )

        self.model.train()  # Reset model to training mode

        return val_loss, val_err

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
        batch2img_ref = self._get_batch2img_function(images['inp_ref'], z_plane)
        batch2img_neg = self._get_batch2img_function(images['inp_-'], z_plane)
        batch2img_pos = self._get_batch2img_function(images['inp_+'], z_plane)
        inp_ref = batch2img_ref(images['inp_ref'])[0]
        inp_neg = batch2img_neg(images['inp_-'])[0]
        inp_pos = batch2img_pos(images['inp_+'])[0]
        self.tb.log_image(f'{group}/inp_ref', inp_ref, step=self.step, cmap='gray')
        self.tb.log_image(f'{group}/inp_-', inp_neg, step=self.step, cmap='gray')
        self.tb.log_image(f'{group}/inp_+', inp_pos, step=self.step, cmap='gray')


def calculate_error(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return np.array((pred <= 0).sum(), dtype=np.float32) / np.prod(dista.size()) * 100.
