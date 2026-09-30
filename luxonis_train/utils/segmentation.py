"""Turns the raw output of a segmentation head into boolean masks."""

import torch
from torch import Tensor


def seg_output_to_bool(data: Tensor, binary_threshold: float = 0.5) -> Tensor:
    """Convert the segmentation logits of one image into boolean masks.

    With one channel, a pixel is ``True`` when its sigmoid is at least
    ``binary_threshold``. With more channels, a pixel is ``True`` only in
    the channel with the highest logit. `SegmentationVisualizer` and
    `default_annotate` use the masks.

    Args:
        data (``Tensor``): The logits of shape ``[C, H, W]``.
        binary_threshold (float): The sigmoid threshold for a single
            channel. The function ignores it for more channels.

    Returns:
        ``Tensor``: The boolean masks, with the shape and the device of
        ``data``.

    Example:
        >>> import torch
        >>> from luxonis_train.utils import seg_output_to_bool
        >>> logits = torch.tensor([[[2.0, -1.0]], [[0.0, 3.0]]])
        >>> seg_output_to_bool(logits).tolist()
        [[[True, False]], [[False, True]]]
        >>> seg_output_to_bool(torch.tensor([[[0.5, -0.5]]])).tolist()
        [[[True, False]]]

    """
    masks = torch.empty_like(data, dtype=torch.bool, device=data.device)
    if data.shape[0] == 1:
        classes = data.sigmoid()
        masks[0] = classes >= binary_threshold
    else:
        classes = data.argmax(dim=0)
        for i in range(masks.shape[0]):
            masks[i] = classes == i
    return masks
