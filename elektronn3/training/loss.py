# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

import torch
from torch.nn import functional as F
from elektronn3.training.lovasz_losses import lovasz_softmax
import numpy as np
from torch.nn import CrossEntropyLoss
from scipy.ndimage.filters import gaussian_filter
from scipy import misc
from sklearn.preprocessing import LabelBinarizer
# TODO: Citations (V-NET and https://arxiv.org/abs/1707.03237)


def _channelwise_sum(x: torch.Tensor):
    """Sum-reduce all dimensions of a tensor except dimension 1 (C)"""
    reduce_dims = tuple([0] + list(range(x.dim()))[2:])  # = (0, 2, 3, ...)
    return x.sum(dim=reduce_dims)


# Simple n-dimensional dice loss. Minimalistic version for easier verification
def dice_loss(probs, target, weight=1., eps=0.0001, onehot_target=False):
    # Probs need to be softmax probabilities, not raw network outputs
    if not onehot_target:
        onehot_target = torch.zeros_like(probs)
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
    else:
        onehot_target = target.to(torch.float32)

    intersection = probs * onehot_target
    numerator = 2 * _channelwise_sum(intersection)
    denominator = probs + onehot_target
    denominator = _channelwise_sum(denominator) + eps
    loss_per_channel = 1 - (numerator / denominator)
    weighted_loss_per_channel = weight * loss_per_channel
    return weighted_loss_per_channel.mean()


class DiceLoss(torch.nn.Module):
    def __init__(self, apply_softmax=True, weight=torch.tensor(1.),
                 onehot_target=False):
        super().__init__()
        if apply_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x  # Identity (no softmax)
        self.dice = dice_loss
        self.register_buffer('weight', weight)
        self.onehot_target = onehot_target

    def forward(self, output, target):
        probs = self.softmax(output)
        return self.dice(probs, target, weight=self.weight,
                         onehot_target=self.onehot_target)


class LovaszLoss(torch.nn.Module):
    """https://arxiv.org/abs/1705.08790"""
    def __init__(self, apply_softmax=True):
        super().__init__()
        if apply_softmax:
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


class BlurryBoarderLoss(torch.nn.Module):
    # Cross-entropy loss with per-voxel weights. Weights of voxels close
    # to label boundaries are reduced by applying weights which
    # are highest inside a class region and blurred at the boundaries
    # by Gaussian smoothing towards 0
    # TODO: GPU support ->
    def __init__(self, softmax=True, sigma=2.5):
        super().__init__()
        self.sigma = sigma
        if softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x  # Identity
        self.blurry_boarder_weights = blurry_boarder_weights

    def forward(self, output, target):
        boarder_w =  self.blurry_boarder_weights(output.size(), target,
                                                 self.sigma)
        loss = F.cross_entropy(output, target, reduce=False)
        loss = loss * boarder_w
        return loss.mean()


def blurry_boarder_weights(output_shape, target, sigma):
    boarder_w = target.cpu().numpy()
    # vigra.taggedView(target.numpy(), 'xcyz') ISSUE: gaussianSmoothing does not
    # support t-axis which should be used as batch axis
    # conflicting python 3.5/.6 dependencies of vigra vs E3
    # smoothing is applied per-channel
    n_classes = output_shape[1]
    if np.isscalar(sigma):
        sigma = [sigma] * (len(boarder_w.shape) - 1)
    else:
        assert len(sigma) == (len(boarder_w.shape) - 1)
        sigma = list(sigma)
    # add zero smoothing along channels, unfortunately this is not supported by scipy and vigra does not support python 3.6...
    sigma = [0] + sigma
    # fit to number of classes
    lb = LabelBinarizer().fit(np.arange(n_classes))
    # use probas shape because target shape does not have explicit class axis
    orig_shape = list(output_shape)
    orig_shape[1] = 1 if n_classes <= 2 else n_classes # for binary data label binarizes keeps it at length 1
    # change to shape (b, x, y, (z), C) because 'LabelBinarizer' outputs (N, C)
    orig_shape += orig_shape[1:2]
    orig_shape.pop(1)
    boarder_w = lb.transform(boarder_w.flatten())
    # now reshape to (b, x, y, (z), C) to (b, C, x, y, (z))
    boarder_w = boarder_w.reshape(orig_shape)  # (b, x, y, (z), C)
    boarder_w = boarder_w.swapaxes(-1, -2)  # (b, x, y, C, (z)) or (b, x, C, y)
    boarder_w = boarder_w.swapaxes(-2, -3)  # (b, x, C, y, (z)) or (b, C, x, y)
    if len(orig_shape) == 5:
        boarder_w = boarder_w.swapaxes(-3, -4)  # (b, C, x, y, z)
    if orig_shape[1] == 1:
        boarder_w = np.hstack((boarder_w == 0, boarder_w == 1))
    boarder_w = boarder_w.astype(np.float32)
    for ii in range(len(target)):
        curr_patch = boarder_w[ii]
        boarder_w[ii] = gaussian_filter(curr_patch, sigma=sigma)
    # choose weights according to maximum value along class axis. this leads to a low weights symmetricly spread along the boundary of classes.
    boarder_w = torch.from_numpy(np.max(boarder_w, axis=1)).float().cuda()
    boarder_w = boarder_w / boarder_w.mean()  # normalize mean
    return boarder_w
