import torch
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.utils.keypoints import insert_class


def merge_bbox_kpt_targets(
    target_boundingbox: Tensor,
    target_keypoints: Tensor,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Merges the bounding box and keypoint targets into a single
    tensor.

    @param target_boundingbox: The bounding box targets.
    @param target_keypoints: The keypoint targets.
    @param device: The device to use.
    """
    target_keypoints = insert_class(target_keypoints, target_boundingbox)
    n_keypoints = (target_keypoints.shape[1] - 2) // 3
    target = torch.zeros(
        (len(target_boundingbox), n_keypoints * 3 + 6), device=device
    )
    target[:, :2] = target_boundingbox[:, :2]
    target[:, 2:6] = box_convert(target_boundingbox[:, 2:], "xywh", "xyxy")

    target[:, 6::3] = target_keypoints[:, 2::3]
    target[:, 7::3] = target_keypoints[:, 3::3]
    target[:, 8::3] = target_keypoints[:, 4::3]
    return target


def fix_empty_tensor(tensor: Tensor) -> Tensor:
    """Empty tensors can cause problems in DDP mode, this methods
    corrects them."""
    if tensor.numel() == 0 and tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor
