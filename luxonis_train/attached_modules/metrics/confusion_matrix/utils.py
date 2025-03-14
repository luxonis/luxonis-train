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
    """Turns an instance segmentation mask into a semantic one by
    merging the masks of the same class."""
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
