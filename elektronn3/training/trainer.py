import os
import traceback
import signal
import numpy as np
from scipy.misc import imsave
from torch.autograd import Variable
from .train_utils import DelayedDataLoader
from torch.utils.trainer import Trainer
from torch.optim.lr_scheduler import ExponentialLR
import logging
from elektronn3.training.train_utils import Timer, pretty_string_time
from os.path import normpath, basename
from .train_utils import user_input, HistoryTracker
from .. import cuda_enabled
from ..data.image import write_overlayimg
logger = logging.getLogger('elektronn3log')


class StoppableTrainer(Trainer):
    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None, save_path=None,
                 batchsize=1, schedulers=None):
        super(StoppableTrainer, self).__init__(model=model, criterion=criterion,
                                               optimizer=optimizer, dataset=dataset)
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.batchsize = batchsize
        self.tracker = HistoryTracker()
        self.timer = Timer()
        if schedulers is None:
            schedulers = {"lr": ExponentialLR(optimizer, 1)}
        else:
            assert type(schedulers) == dict
        self.schedulers = schedulers
        self.save_name = basename(normpath(self.save_path))

    def run(self, epochs=1):
        curr_lr = self.schedulers["lr"].get_lr()[0]
        out = "%05i L_m=%.3f, L=%.2f, tr=%05.2f%%, " % (0, 0, 0, 0)
        out += "vl=%05.2f%s, prev=%04.1f, L_diff=%+.1e, " \
               % (0, "%", 0 * 100, 0)
        out += "LR=%.5f, %.2f it/s, %s" % (curr_lr, 0, "0s")
        logger.info(out)
        prev_tr_loss, tr_loss, val_loss, val_err, tr_err, mean_target, tr_speed = 0, 0, 0, 0, 0, 0, 0
        for i in range(1, epochs + 1):
            try:
                tr_loss, tr_err, mean_target, tr_speed = self.train()
                val_loss, val_err = self.validate()
                curr_lr = self.schedulers["lr"].get_lr()[-1]
                for sched in self.schedulers.values():
                    sched.step()
                loss_gain = prev_tr_loss - tr_loss
                prev_tr_loss = tr_loss
                self.tracker.update_history([i, self.timer.t_passed, tr_loss,
                                             val_loss, loss_gain,
                                             tr_err, val_err, curr_lr, 0, 0]) # 0's correspond to mom and gradnet (?)
                t = pretty_string_time(self.timer.t_passed)
                loss_smooth = self.tracker.timeline.mean()[1]
                out = "%05i L_m=%.3f, L=%.2f, tr=%05.2f%%, " % (i, loss_smooth, tr_loss, tr_err)
                out += "vl=%05.2f%s, prev=%04.1f, L_diff=%+.1e, " \
                       % (val_err, "%", mean_target * 100, loss_gain)
                out += "LR=%.5f, %.2f it/s, %s" % (curr_lr, tr_speed, t)
                logger.info(out)
                self.tracker.plot(self.save_path + "/" + self.save_name)
            except (KeyboardInterrupt) as e:
                if not isinstance(e, KeyboardInterrupt):
                    traceback.print_exc()
                    print("\nEntering Command line such that Exception can be "
                          "further inspected by user.\n\n")
                # Like a command line, but cannot change singletons
                var_push = globals()
                var_push.update(locals())
                ret = user_input(var_push)
                if ret == 'kill':
                    break

    def train(self):
        self.model.train()
        self.dataset.train()
        data_loader = DelayedDataLoader(self.dataset, batch_size=self.batchsize, shuffle=True,
                                 num_workers=4, pin_memory=cuda_enabled)
        tr_loss = 0
        incorrect = 0
        numel = 0
        target_sum = 0
        timer = Timer()
        for (data, target) in data_loader:
            # self.call_plugins('batch', i, data, target)
            if cuda_enabled:
                data, target = data.cuda(), target.cuda()
            data = Variable(data)
            target = Variable(target)
            # forward pass
            out = self.model(data)
            target = target.view(target.numel())

            # update step
            self.optimizer.zero_grad()
            loss = self.criterion(out, target, weight=self.dataset.class_weights)
            loss.backward()
            self.optimizer.step()

            # get training performance
            pred = out.data.max(1)[1]  # get the index of the max log-probability
            numel += target.numel()
            target_sum += target.sum().data.tolist()[0]
            incorrect += pred.ne(target.data).cpu().sum()
            tr_loss += loss.data[0]
            # self.call_plugins('iteration', i, data, target,
            #                   *plugin_data)
            # self.call_plugins('update', i, self.model)
            self.tracker.update_timeline([self.timer.t_passed, loss.data[0], float(target_sum) / numel])
            self.iterations += 1
        tr_err = 100. * incorrect / numel
        tr_loss /= len(data_loader)
        return tr_loss, tr_err, float(target_sum) / numel, len(data_loader) / timer.t_passed

    def validate(self):
        self.model.eval()
        self.dataset.validate()
        data_loader = DelayedDataLoader(self.dataset, 1, shuffle=True,
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
