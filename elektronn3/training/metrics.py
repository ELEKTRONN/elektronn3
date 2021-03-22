# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

# TODO: Update docs to show Evaluator
"""Metrics and tools for evaluating neural network predictions

References:

- https://en.wikipedia.org/wiki/Confusion_matrix
- https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
- http://scikit-learn.org/stable/modules/model_evaluation.html

.. note::

    ``sklearn.metrics`` has a lot of alternative implementations that can be
    compared with these here and could be used as inspiration for future work
    (http://scikit-learn.org/stable/modules/classes.html#classification-metrics).

    For example, to get the equivalent output to
    ``elektronn3.training.metrics.recall(target, pred, num_classes=2, mean=False) / 100``,
    from scikit-learn, you can compute
    ``sklearn.metrics.recall_score(target.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy(), average=None).astype(np.float32)``.


    For most metrics, we don't use scikit-learn directly in this module for
    performance reasons:

    - PyTorch allows us to calculate metrics directly on GPU
    - We LRU-cache confusion matrices for cheap calculation of multiple metrics
"""

from functools import lru_cache
from typing import Callable, Optional

import sklearn.metrics
import torch


eps = 0.0001  # To avoid divisions by zero

# TODO: Tests would make a lot of sense here.


@lru_cache(maxsize=128)
def confusion_matrix(
        target: torch.LongTensor,
        pred: torch.LongTensor,
        num_classes: int = 2,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
        nan_when_empty: bool = True,
        ignore: Optional[int] = None,
) -> torch.Tensor:
    """ Calculate per-class confusion matrix.

    Uses an LRU cache, so subsequent calls with the same arguments are very
    cheap.

    Args:
        pred: Tensor with predicted class values
        target: Ground truth tensor with true class values
        num_classes: Number of classes that the target can assume.
            E.g. for binary classification, this is 2.
            Classes are expected to start at 0
        dtype: ``torch.dtype`` to be used for calculation and output.
            ``torch.float32`` is used as default because it is robust
            against overflows and can be used directly in true divisions
            without re-casting.
        device: PyTorch device on which to store the confusion matrix
        nan_when_empty: If ``True`` (default), the confusion matrix will
            be filled with NaN values for each channel of which there are
            no positive entries in the ``target`` tensor.
        ignore: Index to be ignored for cm calculation

    Returns:
        Confusion matrix ``cm``, with shape ``(num_classes, 4)``, where
        each row ``cm[c]`` contains (in this order) the count of
        - true positives
        - true negatives
        - false positives
        - false negatives
        of ``pred`` w.r.t. ``target`` and class ``c``.

        E.g. ``cm[1][2]`` contains the number of false positive predictions
        of class ``1``.
        If ``nan_when_empty`` is enabled and there are no positive elements
        of class ``1`` in ``target``, ``cm[1]`` will instead be filled with
        NaN values.
    """
    cm = torch.empty(num_classes, 4, dtype=dtype, device=device)
    for c in range(num_classes):
        pos_pred = pred == c
        neg_pred = ~pos_pred
        pos_target = target == c
        if ignore is not None:
            ign_target = target == ignore
        else:
            ign_target = False  # Makes `& ~ign_target` a no-op
        # Manual conversion to Tensor because of a type promotion regression in PyTorch 1.5
        ign_target = torch.tensor(ign_target, dtype=torch.bool, device=device)
        neg_target = ~pos_target

        true_pos = (pos_pred & pos_target & ~ign_target).sum(dtype=dtype)
        true_neg = (neg_pred & neg_target & ~ign_target).sum(dtype=dtype)
        false_pos = (pos_pred & neg_target & ~ign_target).sum(dtype=dtype)
        false_neg = (neg_pred & pos_target & ~ign_target).sum(dtype=dtype)

        cm[c] = torch.tensor([true_pos, true_neg, false_pos, false_neg])

        if nan_when_empty and pos_target.sum(dtype=dtype) == 0:
            cm[c] = torch.tensor([float('nan')] * 4)

    return cm


def precision(target, pred, num_classes=2, mean=True, ignore=None):
    """Precision metric (in %)"""
    cm = confusion_matrix(target, pred, num_classes=num_classes, ignore=ignore)
    tp, tn, fp, fn = cm.transpose(0, 1)  # Transposing to put class axis last
    # Compute metrics for each class simulataneously
    prec = tp / (tp + fp + eps)  # Per-class precision
    if mean:
        prec = prec.mean().item()
    return prec * 100


def recall(target, pred, num_classes=2, mean=True, ignore=None):
    """Recall metric a.k.a. sensitivity a.k.a. hit rate (in %)"""
    cm = confusion_matrix(target, pred, num_classes=num_classes, ignore=ignore)
    tp, tn, fp, fn = cm.transpose(0, 1)  # Transposing to put class axis last
    rec = tp / (tp + fn + eps)  # Per-class recall
    if mean:
        rec = rec.mean().item()
    return rec * 100


def accuracy(target, pred, num_classes=2, mean=True, ignore=None):
    """Accuracy metric (in %)"""
    cm = confusion_matrix(target, pred, num_classes=num_classes, ignore=ignore)
    tp, tn, fp, fn = cm.transpose(0, 1)  # Transposing to put class axis last
    acc = (tp + tn) / (tp + tn + fp + fn + eps)  # Per-class accuracy
    if mean:
        acc = acc.mean().item()
    return acc * 100


def dice_coefficient(target, pred, num_classes=2, mean=True, ignore=None):
    """Sørensen–Dice coefficient a.k.a. DSC a.k.a. F1 score (in %)"""
    cm = confusion_matrix(target, pred, num_classes=num_classes, ignore=ignore)
    tp, tn, fp, fn = cm.transpose(0, 1)  # Transposing to put class axis last
    dsc = 2 * tp / (2 * tp + fp + fn + eps)  # Per-class (Sørensen-)Dice similarity coefficient
    if mean:
        dsc = dsc.mean().item()
    return dsc * 100


def iou(target, pred, num_classes=2, mean=True, ignore=None):
    """IoU (Intersection over Union) a.k.a. IU a.k.a. Jaccard index (in %)"""
    cm = confusion_matrix(target, pred, num_classes=num_classes, ignore=ignore)
    tp, tn, fp, fn = cm.transpose(0, 1)  # Transposing to put class axis last
    iu = tp / (tp + fp + fn + eps)  # Per-class Intersection over Union
    if mean:
        iu = iu.mean().item()
    return iu * 100


def auroc(target, probs, mean=True):
    """ Area under Curve (AuC) of the ROC curve (in %).

    .. note::
        This implementation uses scikit-learn on the CPU to do the heavy
        lifting, so it's relatively slow (one call can take about 1 second
        for typical inputs).
    """
    assert probs.dim() == target.dim() + 1
    num_classes = probs.shape[1]
    # target: (N, [D,], H, W) -> (N*[D,]*H*W,)
    target_npflat = target.view(-1).cpu().numpy()
    # probs: (N, C, [D,], H, W) -> (C, N*[D,]*H*W)
    probs_npflat = probs.transpose(1, 0).view(num_classes, -1).cpu().numpy()
    auc = torch.empty(num_classes)
    # Direct roc_auc_score() computation with multi-class arrays can take
    #  hours, so split this into binary calculations manually here by looping
    #  through classes:
    for c in range(num_classes):
        t = target_npflat == c  # 1 where target is c, 0 everywhere else
        p = probs_npflat[c]  # probs of class c
        auc[c] = sklearn.metrics.roc_auc_score(t, p)
    if mean:
        auc = auc.mean().item()
    return auc * 100


def average_precision(target, probs, mean=True):
    """Average precision (AP) metric based on PR curves (in %).

    .. note::
        This implementation uses scikit-learn on the CPU to do the heavy
        lifting, so it's relatively slow (one call can take about 1 second
        for typical inputs).
    """
    assert probs.dim() == target.dim() + 1
    num_classes = probs.shape[1]
    # target: (N, [D,], H, W) -> (N*[D,]*H*W,)
    target_npflat = target.view(-1).cpu().numpy()
    # probs: (N, C, [D,], H, W) -> (C, N*[D,]*H*W)
    probs_npflat = probs.transpose(1, 0).view(num_classes, -1).cpu().numpy()
    ap = torch.empty(num_classes)
    # Direct average_precision_score() computation with multi-class arrays can take
    #  hours, so split this into binary calculations manually here by looping
    #  through classes:
    for c in range(num_classes):
        t = target_npflat == c  # 1 where target is c, 0 everywhere else
        p = probs_npflat[c]  # probs of class c
        ap[c] = sklearn.metrics.average_precision_score(t, p)
    if mean:
        ap = ap.mean().item()
    return ap * 100


@lru_cache(maxsize=128)
def _softmax(x, dim=1):
    return torch.nn.functional.softmax(x, dim)


@lru_cache(maxsize=128)
def _argmax(x, dim=1):
    return x.argmax(dim)


# Helper for multi-class metric construction
def channel_metric(metric, c, num_classes, argmax=True):
    """Returns an evaluator that calculates the ``metric``
    and selects its value for channel ``c``.

    Example:
        >>> from elektronn3.training import metrics
        >>> num_classes = 5  # Example. Depends on model and data set
        >>> # Metric evaluator dict that registers DSCs of all output channels.
        >>> # You can pass it to elektronn3.training.Trainer as the ``valid_metrics``
        >>> #  argument to make it log these values.
        >>> dsc_evaluators = {
        ...    f'val_DSC_c{c}': channel_metric(
        ...        metrics.dice_coefficient,
        ...        c=c, num_classes=num_classes
        ...    )
        ...    for c in range(num_classes)
        ... }
    """
    def evaluator(target, out):
        pred = _argmax(out) if argmax else out
        m = metric(target, pred, num_classes=num_classes, mean=False)
        return m[c]

    return evaluator


# Metric evaluator shortcuts for raw network outputs in binary classification
#  tasks ("bin_*"). "Raw" means not softmaxed or argmaxed.

# These are deprecated and will be removed later. Use Evaluators instead.

def bin_precision(target, out):
    pred = _argmax(out)
    return precision(
        target, pred, num_classes=2, mean=False
    )[1]  # Take only the score for class 1


def bin_recall(target, out):
    pred = _argmax(out)
    return recall(
        target, pred, num_classes=2, mean=False
    )[1]  # Take only the score for class 1


def bin_accuracy(target, out):
    pred = _argmax(out)
    return accuracy(
        target, pred, num_classes=2, mean=False
    )[1]  # Take only the score for class 1


def bin_dice_coefficient(target, out):
    pred = _argmax(out)
    return dice_coefficient(
        target, pred, num_classes=2, mean=False
    )[1]  # Take only the score for class 1


def bin_iou(target, out):
    pred = _argmax(out)
    return iou(
        target, pred, num_classes=2, mean=False
    )[1]  # Take only the score for class 1


def bin_average_precision(target, out):
    probs = _softmax(out)
    return average_precision(
        target, probs, mean=False
    )[1]  # Take only the score for class 1


def bin_auroc(target, out):
    probs = _softmax(out)
    return auroc(
        target, probs, mean=False
    )[1]  # Take only the score for class 1


class Evaluator:
    name: str = 'generic'

    def __init__(
            self,
            metric_fn: Callable,
            index: Optional[int] = None,
            ignore: Optional[int] = None,
    ):
        self.metric_fn = metric_fn
        self.index = index
        self.ignore = ignore
        self.num_classes = None

    def __call__(self, target: torch.Tensor, out: torch.Tensor) -> float:
        if self.num_classes is None:
            self.num_classes = out.shape[1]
        pred = _argmax(out)
        m = self.metric_fn(target, pred, self.num_classes, mean=False, ignore=self.ignore)
        if self.index is None:
            return m.mean().item()
        return m[self.index].item()


class Accuracy(Evaluator):
    name = 'accuracy'
    def __init__(self, *args, **kwargs): super().__init__(accuracy, *args, **kwargs)


class Precision(Evaluator):
    name = 'precision'
    def __init__(self, *args, **kwargs): super().__init__(precision, *args, **kwargs)


class Recall(Evaluator):
    name = 'recall'
    def __init__(self, *args, **kwargs): super().__init__(recall, *args, **kwargs)


class IoU(Evaluator):
    name = 'IoU'
    def __init__(self, *args, **kwargs): super().__init__(iou, *args, **kwargs)


class DSC(Evaluator):
    name = 'DSC'
    def __init__(self, *args, **kwargs): super().__init__(dice_coefficient, *args, **kwargs)
