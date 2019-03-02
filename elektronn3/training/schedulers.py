import math
from torch.optim.lr_scheduler import _LRScheduler

# Copied from https://github.com/pytorch/pytorch/pull/17226
# (implementation by @Kirayue)
# TODO: Delete this when it's merged and released in PyTorch.


class SGDR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{i}}\pi))
    When :math:`\T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`\T_{cut}=0`(after restart), set :math:`\eta_t=\eta_{max}`.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (float): A factor increases :math:`\T_{i}` after a restart. Default: 1.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = int(T_0)
        self.T = int(T_0)
        self.T_mult = T_mult
        self.eta_min = eta_min
        super(SGDR, self).__init__(optimizer, last_epoch)
        self.T_cur = last_epoch

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every update, i.e. if one epoch has 10 iterations(num_train / batch_size),
        we could called SGDR.step(0.1), SGDR.step(0.2), etc.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T:
                self.T_cur = self.T_cur - self.T
                self.T = int(self.T * self.T_mult)
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    T = self.T_0
                    self.T_cur = epoch
                    while self.T_cur >= T:
                        self.T_cur = self.T_cur - T
                        T = int(T * self.T_mult)
                    self.T = T
            else:
                self.T = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
