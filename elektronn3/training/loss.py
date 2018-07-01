# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

import torch
from torch.nn import functional as F

from elektronn3.training.lovasz_losses import lovasz_softmax

# TODO: Citations (V-NET and https://arxiv.org/abs/1707.03237)

def _channelwise_sum(x):
    """Sum-reduce all dimensions of a tensor except dimension 1 (C)"""
    s = x.sum(0)  # Sum over batch dimension N
    while s.dim() != 1:  # Repeatedly reduce until only the C dim remains
        s = s.sum(1)
    return s


# Simple n-dimensional dice loss. Minimalistic version for easier verification
def dice_loss(probs, target, weight=1., eps=0.0001):
    # Probs need to be softmax probabilities, not raw network outputs
    onehot_target = torch.zeros_like(probs)
    onehot_target.scatter_(1, target.unsqueeze(1), 1)

    intersection = probs * onehot_target
    numerator = 2 * _channelwise_sum(intersection)
    denominator = probs + onehot_target
    denominator = _channelwise_sum(denominator) + eps
    loss_per_channel = 1 - (numerator / denominator)
    weighted_loss_per_channel = weight * loss_per_channel
    return weighted_loss_per_channel.mean()


class DiceLoss(torch.nn.Module):
    def __init__(self, softmax=True, weight=torch.tensor(1.)):
        super().__init__()
        if softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x  # Identity (no softmax)
        self.dice = dice_loss
        self.register_buffer('weight', weight)

    def forward(self, output, target):
        probs = self.softmax(output)
        return self.dice(probs, target, weight=self.weight)


class LovaszLoss(torch.nn.Module):
    """https://arxiv.org/abs/1705.08790"""
    def __init__(self, softmax=True):
        super().__init__()
        if softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x  # Identity (no softmax)
        # lovasz_softmax works on softmax probs, so we still have to apply
        #  softmax before passing probs to it
        self.lovasz = lovasz_softmax

    def forward(self, output, target):
        probs = self.softmax(output)
        return self.lovasz(probs, target)


##### ALTERNATIVE VERSIONS OF DICE LOSS #####

# Version with features that are untested and currently not needed
# Based on https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/5
def __dice_loss_with_cool_extra_features(output, target, weights=None, ignore_index=None):
    eps = 0.0001

    encoded_target = torch.zeros_like(output)
    if ignore_index is not None:
        mask = target == ignore_index
        target = target.clone()
        target[mask] = 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, target.unsqueeze(1), 1)

    if weights is None:
        weights = 1

    intersection = output * encoded_target
    numerator = 2 * _channelwise_sum(intersection)
    denominator = output + encoded_target

    if ignore_index is not None:
        denominator[mask] = 0
    denominator = _channelwise_sum(denominator) + eps
    loss_per_channel = weights * (1 - (numerator / denominator))

    return loss_per_channel.sum() / output.shape[1]


# Very simple version. Only for binary classification. Just included for testing.
# Note that the smooth value is set to 0 and eps is introduced instead, to make it comparable.
# Based on https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398
def __dice_loss_binary(output, target, smooth=0, eps=0.0001):
    onehot_target = torch.zeros_like(output)
    onehot_target.scatter_(1, target.unsqueeze(1), 1)

    iflat = output.view(-1)
    tflat = onehot_target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2 * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth + eps))