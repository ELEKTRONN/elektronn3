import os
import traceback
import signal
import numpy as np
from scipy.misc import imsave
from torch.autograd import Variable
from torch.utils.trainer import Trainer
from torch.utils.data import DataLoader
from ..data.utils import save_to_h5py
import torch
from torch.optim.lr_scheduler import ExponentialLR
import logging
from elektronn3.training.train_utils import Timer, pretty_string_time
from os.path import normpath, basename
from .train_utils import user_input, HistoryTracker
from .. import cuda_enabled
from ..data.image import write_overlayimg
from .train_utils import DelayedDataLoader
logger = logging.getLogger('elektronn3log')

try:
    import tensorboardX
    tensorboard_available = True
except:
    tensorboard_available = False

class StoppableTrainer(object):
    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None,
                 save_path=None, batchsize=1, schedulers=None, preview_freq=4,
                 enable_tensorboard=True, tensorboard_root_path='~/tb/'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.iterations = 0
        self.save_path = save_path
        if save_path is not None and not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.batchsize = batchsize
        self.tracker = HistoryTracker()
        self.timer = Timer()
        if schedulers is None:
            schedulers = {"lr": ExponentialLR(optimizer, 1)}
        else:
            assert type(schedulers) == dict
        self.schedulers = schedulers
        self.preview_freq = preview_freq
        if not tensorboard_available and enable_tensorboard:
            enable_tensorboard = False
            logger.warning('Tensorboard is not available, so it has to be disabled.')
        self.tb = None  # TensorboardX SummaryWriter
        if enable_tensorboard:
            # tb_dir = os.path.join(save_path, 'tb')
            self.tensorboard_root_path = os.path.expanduser(tensorboard_root_path)
            tb_dir = os.path.join(self.tensorboard_root_path, self.save_name)
            os.makedirs(tb_dir)
            self.tb = tensorboardX.SummaryWriter(log_dir=tb_dir)
        # self.enable_tensorboard = enable_tensorboard  # Using `self.tb not None` instead to check this

    @property
    def save_name(self):
        return basename(normpath(self.save_path)) if self.save_path is not None else None

    def run(self, epochs=1):
        while self.iterations < epochs:
            try:
                self.step()
            except (KeyboardInterrupt) as e:
                # TODO: The shell doesn't have access to the main training loops locals, so it's useless. Find out how to fix this.
                if not isinstance(e, KeyboardInterrupt):
                    traceback.print_exc()
                    print("\nEntering Command line such that Exception can be "
                          "further inspected by user.\n\n")
                # Like a command line, but cannot change singletons
                var_push = globals()
                var_push.update(locals())
                ret = user_input(var_push)
                if ret == 'kill':
                    return
        torch.save(self.model.state_dict(), "%s/%s-final-model.pkl" % (self.save_path, self.save_name))

    def step(self):
        tr_loss, tr_err, mean_target, tr_speed = self.train()
        # val_loss, val_err = self.validate()
        val_loss, val_err = 0, 0
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
        out = "%05i L_m=%.3f, L=%.2f, tr=%05.2f%%, " % (self.iterations, loss_smooth, tr_loss, tr_err)
        out += "vl=%05.2f%s, prev=%04.1f, L_diff=%+.1e, " \
               % (val_err, "%", mean_target * 100, tr_loss_gain)
        out += "LR=%.5f, %.2f it/s, %s" % (curr_lr, tr_speed, t)
        logger.info(out)
        if self.tb:
            if True:
                self.tb.add_scalars('stats/loss', {
                    'train_loss': tr_loss,
                    'valid_loss': val_loss,
                    },
                    self.iterations
                )
                self.tb.add_scalars('stats/error', {
                    'train_error': tr_err,
                    'valid_errpr': val_err,
                    },
                    self.iterations
                )
                self.tb.add_scalar('stats/train_loss_gain', tr_loss_gain, self.iterations)
                self.tb.add_scalar('perf/train_speed', tr_speed, self.iterations)
                self.tb.add_scalar('meta/learning_rate', curr_lr, self.iterations)
                if self.iterations % self.preview_freq == 0:
                    inp, pred = inference(self.dataset, self.model)
                    pred = torch.from_numpy(out)
                    self.tb.add_image('preview/input', inp, self.iterations)
                    self.tb.add_image('preview/prediction', pred, self.iterations)

            else:  # TODO: Remove later
                self.tb.add_scalar('loss/tr_loss', tr_loss, self.iterations)
                self.tb.add_scalar('error/tr_err', tr_err, self.iterations)
                self.tb.add_scalar('error/val_err', val_err, self.iterations)
                self.tb.add_scalar('tr_speed', tr_speed, self.iterations)
                self.tb.add_scalar('curr_lr', curr_lr, self.iterations)

        if self.save_path is not None:
            self.tracker.plot(self.save_path + "/" + self.save_name)
        if self.save_path is not None and (self.iterations // self.dataset.epoch_size) % 100 == 99:
            inference(self.dataset, self.model, self.save_path + "/" + self.save_name + ".h5")
            torch.save(self.model.state_dict(), "%s/%s-%d-model.pkl" % (self.save_path, self.save_name, self.iterations))

    def train(self):
        self.model.train()
        self.dataset.train()
        data_loader = DelayedDataLoader(self.dataset, batch_size=self.batchsize,
                                        shuffle=False, num_workers=4,
                                        pin_memory=cuda_enabled)
        tr_loss = 0
        incorrect = 0
        numel = 0
        target_sum = 0
        timer = Timer()
        for (data, target) in data_loader:
            if cuda_enabled:
                data, target = data.cuda(), target.cuda()
            data = Variable(data)
            data.requires_grad = True
            target = Variable(target)

            # forward pass
            out = self.model(data)
            # make channels the last axis and flatten
            out = out.permute(0, 2, 3, 4, 1).contiguous()
            out = out.view(out.numel() // 2, 2)
            target = target.view(target.numel())
            loss = self.criterion(out, target)

            # update step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # get training performance
            pred = out.data.max(1)[1]  # get the index of the max log-probability
            numel += target.numel()
            target_sum += target.sum().data.tolist()[0]
            incorrect += pred.ne(target.data).cpu().sum()
            tr_loss += loss.data[0]
            print(self.iterations, target.size(), out.size(), loss.data[0])
            self.tracker.update_timeline([self.timer.t_passed, loss.data[0], float(target_sum) / numel])
            self.iterations += 1
        tr_err = 100. * incorrect / numel
        tr_loss /= len(data_loader)
        return tr_loss, tr_err, float(target_sum) / numel, len(data_loader) / timer.t_passed

    def validate(self):
        self.model.eval()
        self.dataset.validate()
        data_loader = DelayedDataLoader(self.dataset, self.batchsize, shuffle=False,
                                 num_workers=4, pin_memory=cuda_enabled)
        val_loss = 0
        incorrect = 0
        numel = 0
        for data, target in data_loader:
            if cuda_enabled:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            target = target.view(target.numel())
            numel += target.numel()
            val_loss += self.criterion(output, target, weight=self.dataset.class_weights).data[0]
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            incorrect += pred.ne(target.data).cpu().sum()
        val_loss /= len(data_loader)  # loss function already averages over batch size
        val_err = 100. * incorrect / numel
        if self.save_path is not None:
            write_overlayimg("%s/" % (self.save_path), np.array(data.data.view(data.size()).tolist())[0, 0],
                             np.array(pred.view(data.size()).tolist())[0, 0], fname="raw%d" % self.iterations,
                             nb_of_slices=2)
            imsave("%s/target%d.png" % (self.save_path, self.iterations),
                   np.array(target.data.view(data.size()).tolist())[0, 0, 8])
        return val_loss, val_err


# TODO: Make more flexible, avoid assumptions about shapes etc.
def inference(dataset, model, fname=None):
    # logger.info("Starting preview prediction")
    model.eval()
    try:
        inp = torch.from_numpy(dataset.valid_d[0][None, :, :160, :288, :288])
    except IndexError:
        logger.warning('valid_d not accessible. Using training data for preview.')
        inp = torch.from_numpy(dataset.train_d[0][None, :, :160, :288, :288])
    if cuda_enabled:
        # inp.pin_memory()
        inp = inp.cuda()
    inp = Variable(inp, volatile=True)
    # assume single GPU / batch size 1
    out = model(inp)
    clf = out.data.max(1)[1].view(inp.size())
    pred = np.array(clf.tolist(), dtype=np.float32)[0, 0]
    if fname:
        save_to_h5py([pred, dataset.valid_d[0][0, :160, :288, :288].astype(np.float32)], fname,
                    hdf5_names=["pred", "raw"])
        save_to_h5py([np.exp(np.array(out.data.view([1, 2, 160, 288, 288]).tolist())[0, 1], dtype=np.float32)], fname+"prob.h5",
                    hdf5_names=["prob"])
    return inp, pred  # TODO: inp is Variable, but pred is ndarray. Decide on one type.
