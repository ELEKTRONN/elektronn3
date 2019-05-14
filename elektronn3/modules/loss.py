# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""Loss functions"""

import torch

from elektronn3.modules.lovasz_losses import lovasz_softmax

def _channelwise_sum(x: torch.Tensor):
    """Sum-reduce all dimensions of a tensor except dimension 1 (C)"""
    reduce_dims = tuple([0] + list(range(x.dim()))[2:])  # = (0, 2, 3, ...)
    return x.sum(dim=reduce_dims)

def nldsc_loss(probs, target, weight=1., class_weight=1.):
	return -(1 - dice_loss(probs, target, weight, class_weight) + 1e-10).log()

# TODO: Dense weight support
def dice_loss(probs, target, weight=1., class_weight=1.):
    # Probs need to be softmax probabilities, not raw network outputs
    tsh, psh = target.shape, probs.shape

    if tsh == psh:  # Already one-hot
        onehot_target = target.to(probs.dtype)
    elif tsh[0] == psh[0] and tsh[1:] == psh[2:]:  # Assume dense target storage, convert to one-hot
        onehot_target = torch.zeros_like(probs)
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
    else:
        raise ValueError(
            f'Target shape {target.shape} is not compatible with output shape {probs.shape}.'
        )
    # if weight is None:
    #     weight = torch.ones(probs.shape[0], dtype=probs.dtype)  # (C,)
    # if ignore_index is not None:
    #     weight[:, ignore_index] = 0.

    ignore_mask = (1 - onehot_target[0][-1]).view(1,1,*probs.shape[2:]) # inverse ignore

    bg_probs = 1 - probs
    bg_target = 1 - onehot_target
    dense_weight = weight.view(1,-1,1,1,1)
    positive_target_mask = onehot_target[0][1:-1].sum(dim=0).view(1,1,*probs.shape[2:]) # targets w\ background and ignore
    target_mask_empty = ((positive_target_mask * ignore_mask).sum(dim=(0,2,3,4)) == 0).type(positive_target_mask.dtype)
    target_empty = ((onehot_target * ignore_mask).sum(dim=(0,2,3,4)) == 0).type(positive_target_mask.dtype)
    # complete background for weighted classes and all class target filtered background for unweighted classes
    bg_mask = torch.ones_like(probs) * dense_weight + positive_target_mask * (1 - dense_weight)

    bg_weight = 1 - class_weight

    # make num/denom 1 for unweighted classes and classes with no target
    intersection = probs * onehot_target * ignore_mask * dense_weight  # (N, C, ...)
    intersection2 = bg_probs * bg_target * ignore_mask * bg_mask  # (N, C, ...)
    denominator = (probs + onehot_target) * ignore_mask * dense_weight  # (N, C, ...)
    denominator2 = (bg_probs + bg_target) * ignore_mask * bg_mask  # (N, C, ...)
    numerator = 2 * class_weight * _channelwise_sum(intersection) + (1 - weight) + target_empty * weight  # (C,)
    numerator2 = 2 * bg_weight * _channelwise_sum(intersection2)  # (C,)
    denominator = class_weight * _channelwise_sum(denominator) + (1 - weight) + target_empty * weight  # (C,)
    denominator2 = bg_weight * _channelwise_sum(denominator2)  # (C,)
    numerator2 += (numerator2 == 0).type(numerator2.dtype) # no tp background
    denominator2 += (denominator2 == 0).type(denominator2.dtype) # 100% tp foreground
    denominator += (denominator == 0).type(denominator.dtype) # ???¿¿¿

    #loss_per_channel = 1 - numerator / denominator  # (C,)
    #loss_per_channel = 1 - ((numerator * denominator2 + denominator * numerator2) / (2 * denominator * denominator2))  # (C,)

    loss_per_channel = 1 - (numerator * denominator2 + denominator * numerator2) / (2 * denominator * denominator2) # (C,)

    #loss_per_channel = 1 - (numerator + numerator2 + target_mask_empty * (1 - weight)) / (denominator + denominator2 + target_mask_empty * (1 - weight)) # (C,)
    #weighted_loss = 1 - (numerator.sum() + numerator2.sum())/(denominator.sum() + denominator2.sum())
    #weighted_loss = 1 - (numerator[1:-1].sum() + numerator2[1:-1].sum()) / (denominator[1:-1].sum() + denominator2[1:-1].sum()) # (C,)

    #weighted_loss = (weight[:-1] * loss_per_channel[:-1]).sum() / weight[:-1].sum()  # (C,)

    # normalize loss to [0, 1]
    weighted_loss = loss_per_channel[1:-1].sum() / ((loss_per_channel[1:-1] > 0).sum() + (loss_per_channel[1:-1].sum() == 0))  # (C,)
    weighted_loss *= 2

    if torch.isnan(weighted_loss):
        print(loss_per_channel)
        import IPython
        IPython.embed()

    return weighted_loss  # ()


class NLDSCLoss(torch.nn.Module):
    """Generalized Dice Loss, as described in https://arxiv.org/abs/1707.03237.

    Works for n-dimensional data. Assuming that the ``output`` tensor to be
    compared to the ``target`` has the shape (N, C, D, H, W), the ``target``
    can either have the same shape (N, C, D, H, W) (one-hot encoded) or
    (N, D, H, W) (with dense class indices, as in
    ``torch.nn.CrossEntropyLoss``). If the latter shape is detected, the
    ``target`` is automatically internally converted to a one-hot tensor
    for loss calculation.

    Args:
        apply_softmax: If ``True``, a softmax operation is applied to the
            ``output`` tensor before loss calculation. This is necessary if
            your model does not already apply softmax as the last layer.
            If ``False``, ``output`` is assumed to already contain softmax
            probabilities.
        weight: Weight tensor for class-wise loss rescaling.
            Has to be of shape (C,). If ``None``, classes are weighted equally.
    """
    def __init__(self, apply_softmax=True, weight=torch.tensor(1.), class_weight=torch.tensor(1.)):
        super().__init__()
        if apply_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x  # Identity (no softmax)
        self.dice = nldsc_loss
        self.register_buffer('weight', weight)
        self.register_buffer('class_weight', class_weight)

    def forward(self, output, target):
        probs = self.softmax(output)
        return self.dice(probs, target, weight=self.weight, class_weight=self.class_weight)


class DiceLoss(torch.nn.Module):
    """Generalized Dice Loss, as described in https://arxiv.org/abs/1707.03237.

    Works for n-dimensional data. Assuming that the ``output`` tensor to be
    compared to the ``target`` has the shape (N, C, D, H, W), the ``target``
    can either have the same shape (N, C, D, H, W) (one-hot encoded) or
    (N, D, H, W) (with dense class indices, as in
    ``torch.nn.CrossEntropyLoss``). If the latter shape is detected, the
    ``target`` is automatically internally converted to a one-hot tensor
    for loss calculation.

    Args:
        apply_softmax: If ``True``, a softmax operation is applied to the
            ``output`` tensor before loss calculation. This is necessary if
            your model does not already apply softmax as the last layer.
            If ``False``, ``output`` is assumed to already contain softmax
            probabilities.
        weight: Weight tensor for class-wise loss rescaling.
            Has to be of shape (C,). If ``None``, classes are weighted equally.
    """
    def __init__(self, apply_softmax=True, weight=torch.tensor(1.), class_weight=torch.tensor(1.)):
        super().__init__()
        if apply_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x  # Identity (no softmax)
        self.dice = dice_loss
        self.register_buffer('weight', weight)
        self.register_buffer('class_weight', class_weight)

    def forward(self, output, target):
        probs = self.softmax(output)
        return self.dice(probs, target, weight=self.weight, class_weight=self.class_weight)


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
