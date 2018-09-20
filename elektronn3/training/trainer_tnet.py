# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Martin Drawitsch

# NOTE: This module is currently not maintained. We should probably put this somewhere else.

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
import matplotlib.pyplot as plt
import seaborn as sns
import time
from PIL import Image
import io


class TripletNetTrainer(Trainer):
    """
    Args:
        alpha: l2-norm regularization weight

    """
    def __init__(self, alpha=1e-3, alpha2=.05, latent_distr=None,
                 *args, **kwargs):
        if len(args) > 0:
            raise ValueError("Please provide keyword arguments for "
                             "TripletNetTrainer init only.")
        model_discr = kwargs["model"][1]
        optim_discr = kwargs["optimizer"][1]
        criterion_discr = kwargs["criterion"][1]
        kwargs["model"] = kwargs["model"][0]
        kwargs["optimizer"] = kwargs["optimizer"][0]
        kwargs["criterion"] = kwargs["criterion"][0]
        super().__init__(**kwargs)
        self.alpha = alpha
        self.alpha2 = alpha2
        self.model_discr = model_discr.to(self.device)
        self.optimizer_discr = optim_discr
        self.criterion_discr = criterion_discr
        if latent_distr is None:
            latent_distr = lambda n, z: torch.randn(n, z)  # draw from N(0, 1)
        self.latent_distr = latent_distr

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
                stats: Dict[str, float] = {'tr_loss_G': .0,
                                           'tr_loss_D': .0}
                # Other scalars to be logged
                misc: Dict[str, float] = {'G_loss_advreg': .0,
                                          'G_loss_tnet': .0,
                                          'G_loss_l2': .0,
                                          'D_loss_fake': .0,
                                          'D_loss_real': .0
                                          }
                # Hold image tensors for real-time training sample visualization in tensorboard
                images: Dict[str, torch.Tensor] = {}

                running_error = 0
                running_mean_target = 0
                running_vx_size = 0
                timer = Timer()
                latent_points_fake = []
                latent_points_real = []
                for inp in self.train_loader:  # ref., pos., neg. samples
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
                    self.optimizer.zero_grad()
                    # forward pass
                    dA, dB, z0, z1, z2 = self.model(inp0, inp1, inp2)
                    z_fake_gauss = torch.squeeze(torch.cat((z0, z1, z2), dim=1))
                    target = torch.FloatTensor(dA.size()).fill_(-1).to(self.device)
                    target = Variable(target)
                    loss = self.criterion(dA, dB, target)
                    L_l2 = torch.mean(torch.cat((z0.norm(2, dim=1), z1.norm(2, dim=1), z2.norm(2, dim=1)), dim=0))
                    misc['G_loss_l2'] += self.alpha * float(L_l2)
                    loss = loss + self.alpha * L_l2
                    misc['G_loss_tnet'] += (1 - self.alpha2) * float(loss)  # log actual loss
                    if torch.isnan(loss):
                        logger.error('NaN loss detected after {self.step} '
                                     'steps! Aborting training.')
                        raise NaNException


                    # Adversarial part to enforce latent variable distribution
                    # to be Normal / whatever prior is used
                    if self.alpha2 > 0:
                        self.optimizer_discr.zero_grad()
                        # adversarial labels
                        valid = Variable(torch.Tensor(inp0.size()[0], 1).fill_(1.0),
                                         requires_grad=False).to(self.device)
                        fake = Variable(torch.Tensor(inp0.shape[0], 1).fill_(0.0),
                                        requires_grad=False).to(self.device)

                        # --- Generator / TripletNet
                        self.model_discr.eval()
                        # TripletNet latent space should be classified as valid
                        L_advreg = self.criterion_discr(self.model_discr(z_fake_gauss), valid)
                        # average adversarial reg. and triplet-loss
                        loss = (1 - self.alpha2) * loss + self.alpha2 * L_advreg
                        # perform generator step
                        loss.backward()
                        self.optimizer.step()

                        # --- Discriminator
                        self.model.eval()
                        self.model_discr.train()
                        # rebuild graph (model output) to get clean backprop.
                        z_real_gauss = Variable(self.latent_distr(inp0.size()[0], z0.size()[-1] * 3)).to(self.device)
                        _, _, z_fake_gauss0, z_fake_gauss1, z_fake_gauss2 = self.model(inp0, inp1, inp2)
                        z_fake_gauss = torch.squeeze(torch.cat((z_fake_gauss0, z_fake_gauss1, z_fake_gauss2), dim=1))
                        # Compute discriminator outputs and loss
                        L_real_gauss = self.criterion_discr(self.model_discr(z_real_gauss), valid)
                        L_fake_gauss = self.criterion_discr(self.model_discr(z_fake_gauss), fake)
                        L_discr = 0.5 * (L_real_gauss + L_fake_gauss)
                        L_discr.backward()  # Backprop loss
                        self.optimizer_discr.step()  # Apply optimization step
                        self.model.train()  # set back to training mode


                        # # clean and report
                        L_discr.detach_()
                        L_advreg.detach_()
                        L_real_gauss.detach_()
                        L_fake_gauss.detach_()
                        stats['tr_loss_D'] += float(L_discr)
                        misc['G_loss_advreg'] += self.alpha2 * float(L_advreg) # log actual part of advreg
                        misc['D_loss_real'] += float(L_real_gauss)
                        misc['D_loss_fake'] += float(L_fake_gauss)
                        latent_points_real.append(z_real_gauss.detach().cpu().numpy())
                    else:
                        loss.backward()
                        self.optimizer.step()

                    latent_points_fake.append(z_fake_gauss.detach().cpu().numpy())
                    # # Prevent accidental autograd overheads after optimizer step
                    inp.detach_()
                    target.detach_()
                    dA.detach_()
                    dB.detach_()
                    z0.detach_()
                    z1.detach_()
                    z2.detach_()
                    loss.detach_()
                    L_l2.detach_()

                    # get training performance
                    stats['tr_loss_G'] += float(loss)
                    error = calculate_error(dA, dB)
                    mean_target = target.to(torch.float32).mean()
                    print(f'{self.step:6d}, loss: {loss:.4f}', end='\r')
                    self._tracker.update_timeline([self._timer.t_passed, float(loss), mean_target])

                    # Preserve training batch and network output for later visualization
                    images['inp_ref'] = inp0
                    images['inp_+'] = inp1
                    images['inp_-'] = inp2
                    # this was changed to support ReduceLROnPlateau which does not implement get_lr
                    misc['learning_rate_G'] = self.optimizer.param_groups[0]["lr"] # .get_lr()[-1]
                    misc['learning_rate_D'] = self.optimizer_discr.param_groups[0]["lr"] # .get_lr()[-1]
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
                stats['tr_err_G'] = float(running_error) / len(self.train_loader)
                stats['tr_loss_G'] /= len(self.train_loader)
                stats['tr_loss_D'] /= len(self.train_loader)
                misc['G_loss_advreg'] /= len(self.train_loader)
                misc['G_loss_tnet'] /= len(self.train_loader)
                misc['G_loss_l2'] /= len(self.train_loader)
                misc['D_loss_fake'] /= len(self.train_loader)
                misc['D_loss_real'] /= len(self.train_loader)
                misc['tr_speed'] = len(self.train_loader) / timer.t_passed
                misc['tr_speed_vx'] = running_vx_size / timer.t_passed / 1e6  # MVx
                mean_target = running_mean_target / len(self.train_loader)
                if (self.valid_dataset is None) or (1 != np.random.randint(0, 10)): # only validate 10% of the times
                    stats['val_loss_G'], stats['val_err_G'] = float('nan'), float('nan')
                else:
                    stats['val_loss_G'], stats['val_err_G'] = self.validate()
                # TODO: Report more metrics, e.g. dice error

                # Update history tracker (kind of made obsolete by tensorboard)
                # TODO: Decide what to do with this, now that most things are already in tensorboard.
                if self.step // len(self.train_dataset) > 1:
                    tr_loss_gain = self._tracker.history[-1][2] - stats['tr_loss_G']
                else:
                    tr_loss_gain = 0
                self._tracker.update_history([
                    self.step, self._timer.t_passed, stats['tr_loss_G'], stats['val_loss_G'],
                    tr_loss_gain, stats['tr_err_G'], stats['val_err_G'], misc['learning_rate_G'], 0, 0
                ])  # 0's correspond to mom and gradnet (?)
                t = pretty_string_time(self._timer.t_passed)
                loss_smooth = self._tracker.loss._ema

                # Logging to stdout, text log file
                text = "%05i L_m=%.3f, L=%.2f, tr=%05.2f%%, " % (self.step, loss_smooth, stats['tr_loss_G'], stats['tr_err_G'])
                text += "vl=%05.2f%s, prev=%04.1f, L_diff=%+.1e, " % (stats['val_err_G'], "%", mean_target * 100, tr_loss_gain)
                text += "LR=%.2e, %.2f it/s, %.2f MVx/s, %s" % (misc['learning_rate_G'], misc['tr_speed'], misc['tr_speed_vx'], t)
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

                # save histrograms
                if len(latent_points_fake) > 0:
                    fig, ax = plt.subplots()
                    sns.distplot(np.concatenate(latent_points_fake).flatten())
                    # plt.savefig(os.path.join(self.save_path,
                    #                          'latent_fake_{}.png'.format(self.step)))
                    fig.canvas.draw()
                    img_data = np.array(fig.canvas.renderer._renderer)
                    self.tb.log_image(f'latent_distr/latent_fake', img_data,
                                      step=self.step)
                    plt.close()

                if len(latent_points_real) > 0:
                    fig, ax = plt.subplots()
                    sns.distplot(np.concatenate(latent_points_real).flatten())
                    # plt.savefig(os.path.join(self.save_path,
                    #                          'latent_real_{}.png'.format(self.step)))
                    fig.canvas.draw()
                    img_data = np.array(fig.canvas.renderer._renderer)
                    self.tb.log_image(f'latent_distr/latent_real', img_data,
                                      step=self.step)
                    plt.close()




                    # grab the pixel buffer and dump it into a numpy array


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
        start = time.time()
        val_loss = 0.
        incorrect = 0.
        for inp in self.valid_loader:
            inp0 = inp[:, 0].to(self.device)
            inp1 = inp[:, 1].to(self.device)
            inp2 = inp[:, 2].to(self.device)
            with torch.no_grad():
                dA, dB, z0, z1, z2 = self.model(inp0, inp1, inp2)
                target = torch.FloatTensor(dA.size()).fill_(-1).to(self.device)
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
        dtime = time.time() - start
        print("Validation of {} samples took {}s.".format(len(self.valid_loader), dtime))
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
    pred = (distb - dista - margin).cpu().data
    return np.array((pred <= 0).sum(), dtype=np.float32) / np.prod(dista.size()) * 100.
