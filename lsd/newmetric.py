import torch
import sklearn
import numpy as np

from elektronn3.training import metrics


class LossFunctorMetric():

    def __init__(self, loss, channel_range):
        self.channel_range = channel_range
        self.name = (loss + "__" + str(self.channel_range)))
        if loss == 'L1':
            self.loss = torch.nn.L1Loss()
        elif loss == 'L2':
            self.loss = torch.nn.MSELoss()
        else:
            raise NotImplementedError

    def __call__(self, out, targ):
        if self.channel_range[0] != self.channel_range[1]:
            return self.loss(out[:,self.channel_range[0]:self.channel_range[1]],
                        targ[:,self.channel_range[0]:self.channel_range[1]])
        elif self.channel_range[0] == self.channel_range[1]:
            return self.loss(out[:,self.channel_range[0]], targ[:,self.channel_range[0]])
        else:
            raise NotImplementedError

#class ChannelledLoss(metrics.Evaluator):
#
#    def __init__(self, channel_range, loss = 'L1',  *args, **kwargs):
#        self.channel_range = channel_range
#        self.name = loss + '__' + str(self.channel_range)
#        self.loss_functor = LossFunctor(loss, self.channel_range)
#        super().__init__(self.loss_functor, *args, **kwargs)
