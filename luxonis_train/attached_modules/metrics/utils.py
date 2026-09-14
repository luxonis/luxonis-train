"""Helpers for the keypoint metrics.

`merge_bbox_kpt_targets` merges the box label and the keypoint label
into one tensor. `fix_empty_tensor` reshapes an empty one-dimensional
tensor to the shape ``[1, 0]``.

"""

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
    """Merge the bounding box and keypoint labels into one tensor.

    The function converts the boxes from ``xywh``, where ``x`` and ``y``
    are the top-left corner, to ``xyxy``. It keeps the normalized
    coordinates. Row ``i`` of the result merges row ``i`` of both
    labels.

    Args:
        target_boundingbox (``Tensor``): The box label, of shape
            ``[N, 6]``, with rows ``[batch_index, class, x, y, w, h]``.
        target_keypoints (``Tensor``): The keypoint label, of shape
            ``[N, 1 + 3K]``, with rows
            ``[batch_index, x_1, y_1, v_1, ..., x_K, y_K, v_K]``, in the
            row order of ``target_boundingbox``.
        device (torch.device | None): The device of the result. ``None``
            selects the default device of ``torch``.

    Returns:
        ``Tensor``: A new floating point tensor of shape
        ``[N, 6 + 3K]``, with rows
        ``[batch_index, class, x1, y1, x2, y2, x_1, y_1, v_1, ...]``.

    Example:
        >>> import torch
        >>> boxes = torch.tensor([[0.0, 3.0, 0.25, 0.25, 0.5, 0.25]])
        >>> keypoints = torch.tensor([[0.0, 0.5, 0.375, 2.0]])
        >>> merge_bbox_kpt_targets(boxes, keypoints).tolist()
        [[0.0, 3.0, 0.25, 0.25, 0.75, 0.5, 0.5, 0.375, 2.0]]

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
    """Reshape an empty one-dimensional tensor to the shape ``[1, 0]``.

    The keypoint metrics call this function on the tensors that they
    append to their list states. An empty tensor of shape ``[0]`` can
    cause problems in DDP mode. The function returns every other tensor
    unchanged.

    Args:
        tensor (``Tensor``): The tensor to check.

    Returns:
        ``Tensor``: A view of shape ``[1, 0]`` when ``tensor`` is empty
        and one-dimensional, otherwise ``tensor`` itself.

    Example:
        >>> import torch
        >>> fix_empty_tensor(torch.zeros(0)).shape
        torch.Size([1, 0])
        >>> fix_empty_tensor(torch.zeros(0, 3)).shape
        torch.Size([0, 3])

    """
    if tensor.numel() == 0 and tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor
