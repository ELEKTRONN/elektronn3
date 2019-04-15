# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

"""Loss functions"""

import torch

from elektronn3.modules.lovasz_losses import lovasz_softmax


def dice_loss(
        probs: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = torch.tensor(1.),
        eps: float = 0.0001,
        smooth: float = 0.0,
        reduce: bool = True
) -> torch.Tensor:
    # Probs need to be softmax probabilities, not raw network outputs
    tsh, psh = target.shape, probs.shape
    n, c = psh[:2]

    if tsh == psh:  # Already one-hot
        onehot_target = target.to(probs.dtype)
    elif tsh[0] == psh[0] and tsh[1:] == psh[2:]:  # Assume dense target storage, convert to one-hot
        onehot_target = torch.zeros_like(probs)
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
    else:
        raise ValueError(
            f'Target shape {target.shape} is not compatible with output shape {probs.shape}.'
        )
    # if ignore_index is not None:
    #     weight[:, ignore_index] = 0.

    # Reshape tensors/collapse spatial dimensions into one
    probs_flat = probs.view(n, c, -1)  # (N, C, S), where S = D*H*W (or H*W etc.)
    onehot_target_flat = onehot_target.view(n, c, -1)  # (N, C, S)
    weight_flat = weight.view(1, c, -1)  # (1, C, 1) (broadcastable to (N, C, S)

    numerator = 2 * weight_flat * (probs_flat * onehot_target_flat)  # (N, C, S)
    denominator = weight_flat * (probs_flat + onehot_target_flat)  # (N, C, S)
    if reduce:  # Sum-reduce both numerator and denominator separately
        numerator = torch.sum(numerator)
        denominator = torch.sum(denominator)
    else:
        numerator = numerator.view(psh)
        denominator = denominator.view(psh)
    loss = 1 - ((numerator + smooth) / (denominator + smooth + eps))
    return loss


class DiceLoss(torch.nn.Module):
    """Generalized Dice Loss, similar to the GDL loss described in
    https://arxiv.org/abs/1707.03237.

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
        weight: Weight tensor for loss rescaling.
            Has to be of a shape that is broadcastable to the target shape
            or scalar or a tensor of shape (C,).
            By default, no weighting is performed.
        eps: Small float value to avoid divisions by zero. Can be set to 0 if
            ``smooth`` > 0.
        smooth: Additive smoothing value. See
            https://github.com/pytorch/pytorch/issues/1249#issuecomment-337999895.
        reduce: If ``True`` (default), sum-reduce both the numerator and
            the denominator, resulting in a scalar return value (which you
            can call ``.backward()`` on). If ``False``, this step is skipped
            and the returned tensor will be of the same shape as the
            ``output``, so it can't be directly be ``.backward()``-ed.
            (Note: "mean"-reduction is not supported because it is
            mathematically equivalent to sum-reduction in this loss
            formulation.)
    """
    def __init__(
            self,
            apply_softmax: bool = True,
            weight: torch.Tensor = torch.tensor(1.),
            eps: float = 0.0001,
            smooth: float = 0.0,
            reduce: bool = True
    ):
        super().__init__()
        if apply_softmax:
            self.maybe_softmax = torch.nn.Softmax(dim=1)
        else:
            self.maybe_softmax = lambda x: x  # Identity (no softmax)
        self.dice = dice_loss
        self.eps = eps
        self.smooth = smooth
        self.reduce = reduce
        self.register_buffer('weight', weight)

    def forward(self, output, target):
        probs = self.maybe_softmax(output)
        return self.dice(
            probs,
            target,
            weight=self.weight,
            eps=self.eps,
            smooth=self.smooth,
            reduce=self.reduce
        )


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
