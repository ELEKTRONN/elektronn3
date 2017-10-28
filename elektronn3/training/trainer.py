from torch.utils.trainer import Trainer
import heapq
import traceback
from .train_utils import user_input
from .. import cuda_enabled
from torch.autograd import Variable
from scipy.misc import imsave
import numpy as np
from torch.utils.data import DataLoader


class StoppableTrainer(Trainer):
    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None, dest_path=None,
                 batchsize=1):
        super(StoppableTrainer, self).__init__(model=model, criterion=criterion,
                                               optimizer=optimizer, dataset=dataset)
        self.dest_path = dest_path
        self.batchsize = batchsize

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)
        print("Running model training.")
        for i in range(1, epochs + 1):
            try:
                self.train()
                self.validate()
                self.call_plugins('epoch', i)
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
        data_loader = DataLoader(self.dataset, batch_size=self.batchsize, shuffle=True,
                                 num_workers=1, pin_memory=cuda_enabled)
        epoch_size = len(data_loader)
        for i, (data, target) in enumerate(data_loader):
            self.call_plugins('batch', i, data, target)
            if cuda_enabled:
                data, target = data.cuda(), target.cuda()
            data = Variable(data)
            target = Variable(target)
            plugin_data = [None, None]
            # forward pass
            out = self.model(data)
            target = target.view(target.numel())

            # update step
            self.optimizer.zero_grad()
            loss = self.criterion(out, target, weight=self.dataset.class_weights)
            loss.backward()
            self.optimizer.step()
            if plugin_data[0] is None:
                plugin_data[0] = out.data
                plugin_data[1] = loss.data

            # get training performance
            pred = out.data.max(1)[1]  # get the index of the max log-probability
            incorrect = pred.ne(target.data).cpu().sum()
            err = 100. * incorrect / target.numel()
            partialEpoch = self.iterations // epoch_size + i / epoch_size
            print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tError: {:.3f}'.format(
                partialEpoch, i, epoch_size, 100. * i / epoch_size,
                loss.data[0], err))
            self.call_plugins('iteration', i, data, target,
                              *plugin_data)
            self.call_plugins('update', i, self.model)
        self.iterations += i

    def validate(self):
        self.model.eval()
        self.dataset.validate()
        data_loader = DataLoader(self.dataset, 1, shuffle=True,
                                 num_workers=1, pin_memory=cuda_enabled)
        test_loss = 0
        dice_loss = 0
        incorrect = 0
        numel = 0
        for data, target in data_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            if cuda_enabled:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            target = target.view(target.numel())
            numel += target.numel()
            test_loss += self.criterion(output, target, weight=self.dataset.class_weights).data[0]
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            incorrect += pred.ne(target.data).cpu().sum()

        test_loss /= len(self.dataset)  # loss function already averages over batch size
        dice_loss /= len(self.dataset)
        err = 100. * incorrect / numel
        print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.3f}%) Dice: {:.6f}\n'.format(
            test_loss, incorrect, numel, err, dice_loss))
        imsave("/u/pschuber/vnet/raw%d.png" % (self.iterations), np.array(data.data.view(data.size()).tolist())[0, 0, 8])
        imsave("/u/pschuber/vnet/pred%d.png" % (self.iterations), np.array(pred.view(data.size()).tolist())[0, 0, 8])
        imsave("/u/pschuber/vnet/target%d.png" % (self.iterations), np.array(target.data.view(data.size()).tolist())[0, 0, 8])