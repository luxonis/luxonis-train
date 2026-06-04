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
    """Merge same-class instance segmentation masks into semantic masks.

    Args:
        predicted_boundingbox (list[Tensor]): Predicted bounding boxes grouped
            by image.
        predicted_instance_segmentation (list[Tensor]): Predicted instance
            masks grouped by image.
        target_boundingbox (Tensor): Target bounding boxes.
        target_instance_segmentation (Tensor): Target instance masks.
        n_classes (int): Number of classes.
        height (int): Output mask height.
        width (int): Output mask width.
        device (torch.device): Device on which to create the masks.

    Returns:
        tuple[Tensor, Tensor]: Predicted and target semantic masks.

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
    """Compute the Matthews correlation coefficient from a confusion
    matrix.

    Args:
        cm (Tensor): Confusion matrix.

    Returns:
        Tensor: Matthews correlation coefficient.

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
