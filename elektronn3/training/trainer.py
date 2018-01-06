import logging
import os
import traceback
from os.path import normpath, basename

import IPython
import numpy as np
import torch
from scipy.misc import imsave
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

from elektronn3.training.train_utils import Timer, pretty_string_time
from elektronn3.training.train_utils import DelayedDataLoader
from elektronn3.training.train_utils import HistoryTracker
from elektronn3.data.image import write_overlayimg
from elektronn3.data.utils import save_to_h5py

logger = logging.getLogger('elektronn3log')

try:
    from .tensorboard import TensorBoardLogger
    tensorboard_available = True
except:
    tensorboard_available = False
    logger.exception('Tensorboard not available.')


class StoppableTrainer:
    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None,
                 save_path=None, batchsize=1, num_workers=0,
                 schedulers=None,
                 enable_tensorboard=True, tensorboard_root_path='~/tb/',
                 custom_shell=False, cuda_enabled='auto'):
        if cuda_enabled == 'auto':
            cuda_enabled = torch.cuda.is_available()
            device = 'GPU' if cuda_enabled else 'CPU'
            logger.info(f'Using {device}.')
        self.cuda_enabled = cuda_enabled
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.custom_shell = custom_shell
        self.terminate = False
        self.iterations = 0
        self.first_plot = True
        self.save_path = save_path
        if save_path is not None and not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.tracker = HistoryTracker()
        self.timer = Timer()
        if schedulers is None:
            schedulers = {"lr": ExponentialLR(optimizer, 1)}
        else:
            assert type(schedulers) == dict
        self.schedulers = schedulers
        if not tensorboard_available and enable_tensorboard:
            enable_tensorboard = False
            logger.warning('Tensorboard is not available, so it has to be disabled.')
        self.tb = None  # TensorboardX SummaryWriter
        if enable_tensorboard:
            # tb_dir = os.path.join(save_path, 'tb')
            self.tensorboard_root_path = os.path.expanduser(tensorboard_root_path)
            tb_dir = os.path.join(self.tensorboard_root_path, self.save_name)
            os.makedirs(tb_dir)
            # TODO: Make always_flush user-configurable here:
            self.tb = TensorBoardLogger(log_dir=tb_dir, always_flush=False)
        # self.enable_tensorboard = enable_tensorboard  # Using `self.tb not None` instead to check this
        try:
            self.loader = DelayedDataLoader(
                self.dataset, batch_size=self.batchsize, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.cuda_enabled,
                timeout=30  # timeout arg requires https://github.com/pytorch/pytorch/commit/1661370ac5f88ef11fedbeac8d0398e8369fc1f3
            )
        except:  # TODO: Remove this try/catch once the timeout option is in an official release
            logger.warning(
                'DataLoader doesn\'t support timeout option. This can lead to random freezes during training.\n'
                'If this is an issue for you, cherry-pick\nhttps://github.com/pytorch/pytorch/commit/1661370ac5f88ef11fedbeac8d0398e8369fc1f3\n'
                'or use a PyTorch version newer than 0.3.0'
            )
            self.loader = DelayedDataLoader(
                self.dataset, batch_size=self.batchsize, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.cuda_enabled,
            )
        self.valid_loader = None
        if self.cuda_enabled:
            self.model.cuda()
            self.criterion.cuda()

    @property
    def save_name(self):
        return basename(normpath(self.save_path)) if self.save_path is not None else None

    # Yeah I know this is an abomination, but this monolithic function makes
    # it possible to access all important locals from within the
    # KeyboardInterrupt-triggered IPython shell.
    # TODO: Try to modularize it as well as possible while keeping the
    #       above-mentioned requirement. E.g. the history tracking stuff can
    #       be refactored into smaller functions
    def train(self, epochs=1):
        while self.iterations < epochs:
            try:
                # --> self.train()
                self.model.train()
                self.dataset.train()

                tr_loss = 0
                incorrect = 0
                numel = 0
                target_sum = 0
                timer = Timer()
                inp, target, out = None, None, None  # Pre-assign to extend variable scope.
                for (inp, target) in self.loader:
                    if self.cuda_enabled:
                        inp, target = inp.cuda(), target.cuda()
                    inp = Variable(inp, requires_grad=True)
                    target = Variable(target)

                    # forward pass
                    out = self.model(inp)
                    # make channels the last axis and flatten
                    out_ = out.permute(0, 2, 3, 4, 1).contiguous()
                    out_ = out_.view(out_.numel() // 2, 2)
                    target_ = target.view(target.numel())
                    loss = self.criterion(out_, target_)  # TODO: Respect class weights

                    # update step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # get training performance
                    numel += target_.numel()
                    target_sum += target_.sum()
                    maxcl = maxclass(out_.data)
                    incorrect += maxcl.ne(target_.data).cpu().sum()
                    tr_loss += loss.data[0]
                    print(f'{self.iterations:6d}, loss: {loss.data[0]:.4f}')
                    self.tracker.update_timeline([self.timer.t_passed, loss.data[0], float(target_sum) / numel])
                    self.iterations += 1
                tr_err = 100. * incorrect / numel
                tr_loss /= len(self.loader)
                mean_target = float(target_sum) / numel
                tr_speed = len(self.loader) / timer.t_passed

                # --> self.step():
                val_loss, val_err = self.validate()
                curr_lr = self.schedulers["lr"].get_lr()[-1]
                for sched in self.schedulers.values():
                    sched.step()
                if self.iterations // self.dataset.epoch_size > 1:
                    tr_loss_gain = self.tracker.history[-1][2] - tr_loss
                else:
                    tr_loss_gain = 0
                self.tracker.update_history([self.iterations, self.timer.t_passed, tr_loss,
                                            val_loss, tr_loss_gain,
                                            tr_err, val_err, curr_lr, 0, 0])  # 0's correspond to mom and gradnet (?)
                t = pretty_string_time(self.timer.t_passed)
                loss_smooth = self.tracker.loss._ema
                text = "%05i L_m=%.3f, L=%.2f, tr=%05.2f%%, " % (self.iterations, loss_smooth, tr_loss, tr_err)
                text += "vl=%05.2f%s, prev=%04.1f, L_diff=%+.1e, " \
                    % (val_err, "%", mean_target * 100, tr_loss_gain)
                text += "LR=%.5f, %.2f it/s, %s" % (curr_lr, tr_speed, t)
                # TODO: Log voxels/s
                logger.info(text)
                if self.tb:
                    self.tb.log_scalar('stats/tr_loss', tr_loss, self.iterations)
                    self.tb.log_scalar('stats/val_loss', val_loss, self.iterations)
                    self.tb.log_scalar('stats/tr_err', tr_err, self.iterations)
                    self.tb.log_scalar('stats/val_err', val_err, self.iterations)
                    self.tb.log_scalar('misc/speed', tr_speed, self.iterations)
                    self.tb.log_scalar('misc/learning_rate', curr_lr, self.iterations)

                    # Note that the variables inp, target and out that are used here have their
                    # values from the last training iteration.
                    # The prefix "v" for variable names here signifies that
                    # the content is from the (constant) decdicated preview batch.
                    # "t" refers to training data from the last training iteration.

                    # TODO: Better variable names! And clean up in general.

                    _, v_out = preview_inference(self.model, self.dataset)

                    # TODO: Less arbitrary slicing
                    v_plane = v_out.shape[2] // 2
                    t_plane = out.shape[2] // 2

                    # Preview from constant region of preview batch data
                    vp0 = v_out[0, 0, v_plane, ...].data.cpu().numpy()  # class 0
                    vp1 = v_out[0, 1, v_plane, ...].data.cpu().numpy()  # class 1
                    vmcl = maxclass(v_out)
                    vmc = vmcl[0, v_plane, ...].data.cpu().numpy()

                    self.tb.log_image('p/vp0', vp0, step=self.iterations)
                    self.tb.log_image('p/vp1', vp1, step=self.iterations)
                    self.tb.log_image('p/vmc', vmc, self.iterations)

                    # Preview from last processed training example (random region, possibly augmented)
                    tinp = inp[0, 0, t_plane, ...].data.cpu().numpy()
                    ttarget = target[0, 0, t_plane].data.cpu().numpy()
                    tp0 = out[0, 0, t_plane, ...].data.cpu().numpy()
                    tp1 = out[0, 1, t_plane, ...].data.cpu().numpy()
                    tmcl = maxclass(out)
                    tmc = tmcl[0, t_plane, ...].data.cpu().numpy()

                    self.tb.log_image('t/tinp', tinp, step=self.iterations)
                    self.tb.log_image('t/ttarget', ttarget, step=self.iterations)
                    self.tb.log_image('t/tp0', tp0, step=self.iterations)
                    self.tb.log_image('t/tp1', tp1, step=self.iterations)
                    self.tb.log_image('t/tmc', tmc, step=self.iterations)

                    if self.first_plot:
                        preview_inp, preview_target = self.dataset.preview_batch
                        inp = preview_inp[0, 0, v_plane, ...].data.cpu().numpy()
                        target = preview_target[0, 0, v_plane, ...].data.cpu().numpy()
                        self.tb.log_image('p/gt_input', inp, step=self.iterations)
                        self.tb.log_image('p/gt_target', target, step=self.iterations)
                        self.first_plot = False

                    self.tb.writer.flush()

                if self.save_path is not None:
                    self.tracker.plot(self.save_path + "/" + self.save_name)
                if self.save_path is not None and (self.iterations // self.dataset.epoch_size) % 100 == 99:
                    # preview_inference(self.model, self.dataset, self.save_path + "/" + self.save_name + ".h5")
                    torch.save(self.model.state_dict(), "%s/%s-%d-model.pkl" % (self.save_path, self.save_name, self.iterations))
            except KeyboardInterrupt as e:
                if not isinstance(e, KeyboardInterrupt):
                    traceback.print_exc()
                    print("\nEntering Command line such that Exception can be "
                          "further inspected by user.\n\n")
                IPython.embed()
                if self.terminate:  # TODO: Somehow make this behavior more obvious
                    return
        torch.save(self.model.state_dict(), "%s/%s-final-model.pkl" % (self.save_path, self.save_name))

    def validate(self):
        if self.valid_loader is None:
            # num_workers is set to 0 here because background processes for validation sometimes
            # fail silently and stop responding, bringing down the whole training process.
            # This issue might be related to https://github.com/pytorch/pytorch/issues/1355,
            # but the deadlocks described there will only happen if timeout can't be enabled
            # (PyTorch 0.3.0).
            # The performance impact of disabling multiprocessing here is low in normal settings,
            # because the validation loader doesn't perform expensive augmentations, but just reads
            # data from hdf5s.

            try:
                self.valid_loader = DelayedDataLoader(
                    self.dataset, self.batchsize, shuffle=False, num_workers=0, pin_memory=False,
                    timeout=30  # timeout arg requires https://github.com/pytorch/pytorch/commit/1661370ac5f88ef11fedbeac8d0398e8369fc1f3
                )
            except:  # TODO: Remove this try/catch once the timeout option is in an official release
                self.valid_loader = DelayedDataLoader(
                    self.dataset, self.batchsize, shuffle=False, num_workers=0, pin_memory=False
                )

        self.dataset.validate()  # Switch dataset to validation sources
        self.model.eval()  # Set dropout and batchnorm to eval mode

        val_loss = 0
        incorrect = 0
        numel = 0
        for data, target in self.valid_loader:
            if self.cuda_enabled:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, volatile=True)
            out = self.model(data)
            out = out.permute(0, 2, 3, 4, 1).contiguous()
            out = out.view(out.numel() // 2, 2)
            target = target.view(target.numel())
            numel += target.numel()
            val_loss += self.criterion(out, target).data[0]  # TODO: Respect class weights
            maxcl = maxclass(out.data)  # get the index of the max log-probability
            incorrect += maxcl.ne(target.data).cpu().sum()
        val_loss /= len(self.valid_loader)  # loss function already averages over batch size
        val_err = 100. * incorrect / numel
        if self.save_path is not None:
            write_overlayimg(
                "%s/" % (self.save_path),
                data.data.view(data.size())[0, 0].cpu().numpy(),
                maxcl.view(data.size())[0, 0].cpu().numpy(),
                fname="%d_overlay" % self.iterations,
                nb_of_slices=2
            )
            imsave(
                "%s/%d_target.png" % (self.save_path, self.iterations),
                target.data.view(data.size())[0, 0, 8].cpu().numpy()
            )

        # Reset dataset and model to training mode
        self.dataset.train()
        self.model.train()

        return val_loss, val_err


# TODO: Move all the functions below out of trainer.py

def maxclass(class_predictions: Variable):
    """For each point in a tensor, determine the class with max. probability.

    Args:
        class_predictions: Tensor of shape (N, C, D, H, W)

    Returns:
        Tensor of shape (N, D, H, W)
    """
    maxcl = class_predictions.max(dim=1)[1]
    return maxcl


# TODO
def save_to_h5(fname: str, model_output: Variable):
    raise NotImplementedError

    maxcl = maxclass(model_output)  # TODO: Ensure correct shape
    save_to_h5py(
        [maxcl, dataset.valid_d[0][0, :shape[0], :shape[1], :shape[2]].astype(np.float32)],
        fname,
        hdf5_names=["pred", "raw"]
    )
    save_to_h5py(
        [np.exp(model_output.data.view([1, 2, shape[0], shape[1], shape[2]])[0, 1].cpu().numpy(), dtype=np.float32)],
        fname+"prob.h5",
        hdf5_names=["prob"]
    )


def preview_inference(model, dataset=None, cuda_enabled='auto'):
    model.eval()  # Set dropout and batchnorm to eval mode
    # logger.info("Starting preview prediction")
    if cuda_enabled == 'auto':
        cuda_enabled = torch.cuda.is_available()
        # device = 'GPU' if cuda_enabled else 'CPU'
        # logger.info(f'Using {device}.')
    # Attention: Inference on Variables with unexpected shapes can lead to errors!
    # Staying with multiples of 16 for lengths seems to work.
    if dataset is None:
        d, h, w = dataset.preview_shape
        inp = torch.rand(1, dataset.c_input, d, h, w)
        if cuda_enabled:
            inp = inp.cuda()
            inp = Variable(inp, volatile=True)
    else:
        inp = dataset.preview_batch[0]
    out = model(inp)
    model.train()  # Reset model to training mode

    return inp, out
