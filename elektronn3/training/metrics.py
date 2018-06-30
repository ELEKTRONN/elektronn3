# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch

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
    ``elektronn3.training.metrics.recall(target, pred, num_classes=2, mean=False)``,
    from scikit-learn, you can compute
    ``sklearn.metrics.recall_score(target.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy(), average=None).astype(np.float32)``.


    For most metrics, we don't use scikit-learn directly in this module for
    performance reasons:

    - PyTorch allows us to calculate metrics directly on GPU
    - We LRU-cache confusion matrices for cheap calculation of multiple metrics
"""

from functools import lru_cache

import torch


# TODO: Tests would make a lot of sense here.

# TODO: Support ignoring certain classes (labels).
#       OR Support a ``labels=...`` param like sklearn, where you pass a
#        whitelist of classes that should be considered in calculations).
#        Actually although it's a bit more effort, I think that's a better
#        approach than blacklisting via an ``ignore=...` param.

# TODO:

@lru_cache(maxsize=128)
def confusion_matrix(
        target: torch.LongTensor,
        pred: torch.LongTensor,
        num_classes: int = 2,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu')
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
    """
    cm = torch.empty(num_classes, 4, dtype=dtype, device=device)
    for c in range(num_classes):
        pos_pred = pred == c
        neg_pred = ~pos_pred
        pos_target = target == c
        neg_target = ~pos_target

        true_pos = (pos_pred & pos_target).sum(dtype=dtype)
        true_neg = (neg_pred & neg_target).sum(dtype=dtype)
        false_pos = (pos_pred & neg_target).sum(dtype=dtype)
        false_neg = (neg_pred & pos_target).sum(dtype=dtype)

        cm[c] = torch.tensor([true_pos, true_neg, false_pos, false_neg])

    return cm


def precision(target, pred, num_classes=2, mean=False):
    """Precision metric"""
    cm = confusion_matrix(target, pred, num_classes=num_classes)
    tp, tn, fp, fn = cm.transpose(0, 1)  # Transposing to put class axis last
    # Compute metrics for each class simulataneously
    prec = tp / (tp + fp)  # Per-class precision
    if mean:
        prec = prec.mean().item()
    return prec


def recall(target, pred, num_classes=2, mean=False):
    """Recall metric a.k.a. sensitivity a.k.a. hit rate"""
    cm = confusion_matrix(target, pred, num_classes=num_classes)
    tp, tn, fp, fn = cm.transpose(0, 1)  # Transposing to put class axis last
    rec = tp / (tp + fn)  # Per-class recall
    if mean:
        rec = rec.mean().item()
    return rec


def accuracy(target, pred, num_classes=2, mean=False):
    """Accuracy metric"""
    cm = confusion_matrix(target, pred, num_classes=num_classes)
    tp, tn, fp, fn = cm.transpose(0, 1)  # Transposing to put class axis last
    acc = (tp + tn) / (tp + tn + fp + fn)  # Per-class accuracy
    if mean:
        acc = acc.mean().item()
    return acc


def dice_coefficient(target, pred, num_classes=2, mean=False):
    """Sørensen–Dice coefficient a.k.a. DSC a.k.a. F1 score"""
    cm = confusion_matrix(target, pred, num_classes=num_classes)
    tp, tn, fp, fn = cm.transpose(0, 1)  # Transposing to put class axis last
    dsc = 2 * tp / (2 * tp + fp + fn)  # Per-class (Sørensen-)Dice similarity coefficient
    if mean:
        dsc = dsc.mean().item()
    return dsc


def iou(target, pred, num_classes=2, mean=False):
    """IoU (Intersection over Union) a.k.a. IU a.k.a. Jaccard index"""
    cm = confusion_matrix(target, pred, num_classes=num_classes)
    tp, tn, fp, fn = cm.transpose(0, 1)  # Transposing to put class axis last
    iu = tp / (tp + fp + fn)  # Per-class Intersection over Union
    if mean:
        iu = iu.mean().item()
    return iu
