"""Helpers for the confusion matrices: the merge of instance masks and
the Matthews correlation coefficient.
"""

import torch
from torch import Tensor


def preprocess_instance_masks(
    predicted_boundingbox: list[Tensor],
    predicted_instance_segmentation: list[Tensor],
    target_boundingbox: Tensor,
    target_instance_segmentation: Tensor,
    n_classes: int,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Merge the instance masks of each class into semantic masks.

    For each image, the function combines all masks of one class with a
    logical OR. A nonzero mask value marks the object. The class of a
    predicted mask comes from its box. The image and the class of a
    target mask come from its target box.

    Args:
        predicted_boundingbox (``list[Tensor]``): The predicted boxes of
            each image, of shape ``[M_i, 6]``, with the class in column
            ``5``. The length of the list is the batch size.
        predicted_instance_segmentation (``list[Tensor]``): The predicted
            masks of each image, of shape ``[M_i, height, width]``, one
            for each box.
        target_boundingbox (``Tensor``): The target boxes of the batch,
            of shape ``[N, C]``, with the batch index in column ``0`` and
            the class in column ``1``.
        target_instance_segmentation (``Tensor``): The target masks, of
            shape ``[N, height, width]``, one for each target box.
        n_classes (int): The number of classes.
        height (int): The height of the masks.
        width (int): The width of the masks.
        device (torch.device): The device of the new masks.

    Returns:
        ``tuple[Tensor, Tensor]``: The predicted and the target semantic
        masks, each of shape ``[B, n_classes, height, width]``, where
        ``B`` is the length of ``predicted_boundingbox``. They hold ``1``
        for the object and ``0`` elsewhere. The predicted masks take the
        dtype of ``predicted_instance_segmentation[0]``. The target masks
        take the dtype of ``target_instance_segmentation``.

    Example:
        The batch has one image and two classes. The two predicted masks
        of class ``1`` merge into one mask.

        >>> import torch
        >>> boxes = [
        ...     torch.tensor([[0, 0, 1, 1, 0.9, 1], [1, 0, 2, 1, 0.8, 1]])
        ... ]
        >>> masks = [torch.tensor([[[1, 0]], [[0, 1]]])]
        >>> target_boxes = torch.tensor([[0, 0, 0.0, 0.0, 0.5, 1.0]])
        >>> target_masks = torch.tensor([[[1, 0]]])
        >>> pred, target = preprocess_instance_masks(
        ...     boxes,
        ...     masks,
        ...     target_boxes,
        ...     target_masks,
        ...     n_classes=2,
        ...     height=1,
        ...     width=2,
        ...     device=torch.device("cpu"),
        ... )
        >>> pred.tolist()
        [[[[0, 0]], [[1, 1]]]]
        >>> target.tolist()
        [[[[1, 0]], [[0, 0]]]]

    """
    batch_size = len(predicted_boundingbox)
    return (
        _merge_predicted_masks(
            predicted_boundingbox,
            predicted_instance_segmentation,
            batch_size,
            n_classes,
            height,
            width,
            device,
        ),
        _merge_target_masks(
            target_boundingbox,
            target_instance_segmentation,
            batch_size,
            n_classes,
            height,
            width,
            device,
        ),
    )


def compute_mcc(cm: Tensor) -> Tensor:
    r"""Compute the Matthews correlation coefficient of a matrix.

    The function uses the multiclass form of the Matthews correlation
    coefficient (MCC). For a matrix with the total count :math:`s`, the
    sum of the diagonal :math:`c`, the row sums :math:`t_k`, and the
    column sums :math:`p_k`:

    .. math::

        \text{MCC} = \frac{c \, s - \sum_k t_k \, p_k}{\sqrt{\left(s^2 - \sum_k t_k^2\right) \left(s^2 - \sum_k p_k^2\right)}}

    The value is between ``-1`` and ``1``. It is ``1`` when all counts
    are on the diagonal and at least two classes have counts. The
    function returns ``0`` when the counts sum to ``0`` or when the
    denominator is ``0``. The transposed matrix gives the same value.

    Args:
        cm (``Tensor``): A square matrix of counts, of shape ``[K, K]``.
            The confusion matrices pass a ``float32`` copy of their
            counts.

    Returns:
        ``Tensor``: The scalar MCC as a floating point tensor, on the
        device of ``cm``.

    Example:
        >>> import torch
        >>> round(
        ...     compute_mcc(torch.tensor([[5.0, 1.0], [2.0, 2.0]])).item(), 4
        ... )
        0.3563
        >>> compute_mcc(torch.eye(3)).item()
        1.0
        >>> compute_mcc(torch.tensor([[0.0, 2.0], [2.0, 0.0]])).item()
        -1.0
        >>> compute_mcc(torch.zeros(2, 2)).item()
        0.0

    """
    N = cm.sum()
    if N == 0:
        return torch.tensor(0.0, device=cm.device)

    sum_diag = torch.diag(cm).sum()
    sum_rows = cm.sum(dim=1)
    sum_cols = cm.sum(dim=0)
    numerator = sum_diag * N - torch.dot(sum_rows, sum_cols)
    denominator = torch.sqrt(
        (N**2 - (sum_rows**2).sum()) * (N**2 - (sum_cols**2).sum())
    )

    if denominator == 0:
        return torch.tensor(0.0, device=cm.device)

    return (numerator / denominator).to(cm.device)


def _merge_predicted_masks(
    boundingbox: list[Tensor],
    instance_segmentation: list[Tensor],
    batch_size: int,
    n_classes: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    mask = torch.zeros(
        batch_size,
        n_classes,
        height,
        width,
        dtype=torch.bool,
        device=device,
    )
    for i, (bboxes, segs) in enumerate(
        zip(boundingbox, instance_segmentation, strict=True)
    ):
        for j, seg in enumerate(segs):
            class_id = bboxes[j][5].int()
            mask[i][class_id] |= seg.bool()

    return mask.to(instance_segmentation[0].dtype)


def _merge_target_masks(
    target_boundingbox: Tensor,
    target_instance_segmentation: Tensor,
    batch_size: int,
    n_classes: int,
    height: int,
    width: int,
    device: torch.device,
) -> Tensor:
    mask = torch.zeros(
        batch_size,
        n_classes,
        height,
        width,
        dtype=torch.bool,
        device=device,
    )
    for bboxes, segs in zip(
        target_boundingbox, target_instance_segmentation, strict=True
    ):
        batch_idx = bboxes[0].int()
        class_id = bboxes[1].int()
        mask[batch_idx][class_id] |= segs.bool()

    return mask.to(target_instance_segmentation.dtype)
