import torch
from torch import Tensor


def seg_output_to_bool(data: Tensor, binary_threshold: float = 0.5) -> Tensor:
    """Converts seg head output to 2D boolean mask for visualization."""
    masks = torch.empty_like(data, dtype=torch.bool, device=data.device)
    if data.shape[0] == 1:
        classes = data.sigmoid()
        masks[0] = classes >= binary_threshold
    else:
        classes = data.argmax(dim=0)
        for i in range(masks.shape[0]):
            masks[i] = classes == i
    return masks
