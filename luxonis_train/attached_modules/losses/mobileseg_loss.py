from typing import Literal

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from torch.nn import functional as F

from luxonis_train.tasks import Tasks

from .base_loss import BaseLoss


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.

    See Alg. 1 in paper.
    """
    gts = torch.sum(gt_sorted)
    p = len(gt_sorted)

    intersection = gts - torch.cumsum(gt_sorted, dim=0)
    union = gts + torch.cumsum(1 - gt_sorted, dim=0)
    jaccard = 1.0 - intersection.type(dtype=torch.float32) / union.type(
        dtype=torch.float32
    )

    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovasz-Softmax loss.

    Args:
        probas (Tensor): Shape is [P, C], class probabilities at each prediction (between 0 and 1).
        labels (Tensor): Shape is [P], ground truth labels (between 0 and C - 1).
        classes (str|list): 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    C = probas.shape[1]
    losses = []
    classes_to_sum = (
        list(range(C)) if classes in ["all", "present"] else classes
    )
    for c in classes_to_sum:
        fg = (labels == c).to(probas.dtype)  # foreground for class c
        if classes == "present" and fg.sum() == 0:
            continue
        fg.requires_grad = False
        if C == 1:
            if len(classes_to_sum) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = torch.abs(fg - class_pred)

        # errors_sorted, perm = torch.argsort(errors, 0, descending=True)
        perm = torch.argsort(errors, dim=0, descending=True)
        errors_sorted = torch.gather(errors, dim=0, index=perm)
        # errors_sorted.requires_grad = True

        fg_sorted = torch.gather(fg, 0, perm)
        fg_sorted.requires_grad = False

        grad = lovasz_grad(fg_sorted)
        grad.requires_grad = False
        loss = torch.sum(errors_sorted * grad)
        losses.append(loss)

    if len(classes_to_sum) == 1:
        return losses[0]

    losses_tensor = torch.stack(losses)
    mean_loss = torch.mean(losses_tensor)
    return mean_loss


def flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch."""
    if len(probas.shape) == 3:
        probas = torch.unsqueeze(probas, dim=1)
    C = probas.shape[1]
    probas = torch.permute(probas, [0, 2, 3, 1])
    probas = torch.reshape(probas, [-1, C])
    labels = torch.reshape(labels, [-1])
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    valid_mask = torch.reshape(valid, [-1, 1])
    indexs = torch.nonzero(valid_mask)
    indexs.requires_grad = False
    vprobas = torch.gather(probas, 0, indexs)
    vlabels = torch.gather(labels, 0, indexs[:, 0])
    return vprobas, vlabels


class LovaszSoftmaxLoss(nn.Module):
    """Multi-class Lovasz-Softmax loss.

    Args:
        ignore_index (int64): Specifies a target value that is ignored and does not contribute to the input gradient. Default ``255``.
        classes (str|list): 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """

    def __init__(self, ignore_index=255, classes="present"):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, logits, labels):
        r"""Forward computation.

        Args:
            logits (Tensor): Shape is [N, C, H, W], logits at each prediction (between -\infty and +\infty).
            labels (Tensor): Shape is [N, 1, H, W] or [N, H, W], ground truth labels (between 0 and C - 1).
        """
        probas = F.softmax(logits, dim=1)
        vprobas, vlabels = flatten_probas(probas, labels, self.ignore_index)
        loss = lovasz_softmax_flat(vprobas, vlabels, classes=self.classes)
        return loss


class MobileSegLoss(BaseLoss):
    """This criterion computes the cross entropy loss between input
    logits and target."""

    supported_tasks = [Tasks.SEGMENTATION, Tasks.CLASSIFICATION]

    def __init__(
        self,
        weight: list[float] | None = None,
        ignore_index: int = -100,
        reduction: Literal["none", "mean", "sum"] = "mean",
        label_smoothing: float = 0.0,
        coef: list[float | int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.coef = coef if coef is not None else [0.5, 0.5]

        self.ce_loss = nn.CrossEntropyLoss(
            weight=(torch.tensor(weight) if weight is not None else None),
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self._was_logged = False

        self.lovasz_loss = LovaszSoftmaxLoss()

    def forward(self, features: list[Tensor], target: Tensor) -> Tensor:
        loss_list: list[Tensor] = []
        for pred in features:
            if pred.ndim == target.ndim:
                ch_dim = 1 if pred.ndim > 1 else 0
                if pred.shape[ch_dim] == 1:
                    if not self._was_logged:
                        logger.warning(
                            "`CrossEntropyLoss` expects at least 2 classes. "
                            "Attempting to fix by adding a dummy channel. "
                            "If you want to be sure, use `BCEWithLogitsLoss` instead."
                        )
                        self._was_logged = True
                    pred = torch.cat(
                        [torch.zeros_like(pred), pred], dim=ch_dim
                    )
                    if target.shape[ch_dim] == 1:
                        target = torch.cat([1 - target, target], dim=ch_dim)
                target = target.argmax(dim=ch_dim)

            if target.ndim != pred.ndim - 1:
                raise RuntimeError(
                    f"Target tensor dimension should equeal to preds dimension - 1 ({pred.ndim - 1}) "
                    f"but is ({target.ndim})."
                )
            loss_list.append(self.ce_loss(pred, target) * self.coef[0])
            loss_list.append(self.lovasz_loss(pred, target) * self.coef[1])

        return torch.sum(torch.stack(loss_list), dim=0)
