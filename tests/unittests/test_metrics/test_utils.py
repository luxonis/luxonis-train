from torch import Size, Tensor
from torchvision.ops import box_convert


def convert_bboxes_to_xyxy_and_normalize(
    targets: Tensor, original_in_shape: Size
) -> Tensor:
    """ " Converts the targets to xyxy format and normalizes them."""
    targets = targets.float()
    targets[:, 2:] = box_convert(targets[:, 2:], "xyxy", "xywh")
    targets[:, [2, 4]] /= original_in_shape[2]
    targets[:, [3, 5]] /= original_in_shape[1]
    return targets


def normalize_kpts(kpts: Tensor, original_in_shape: Size) -> Tensor:
    kpts = kpts.float()
    kpts[:, 1::3] /= original_in_shape[2]
    kpts[:, 2::3] /= original_in_shape[1]
    return kpts
