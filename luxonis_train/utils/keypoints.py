import torch
from loguru import logger
from torch import Tensor


def get_sigmas(
    sigmas: list[float] | None,
    n_keypoints: int,
    caller_name: str | None = None,
) -> Tensor:
    """Validate or create sigma values for each keypoint.

    @type sigmas: list[float] | None
    @param sigmas: List of sigmas for each keypoint. If C{None}, then
        default sigmas are used.
    @type n_keypoints: int
    @param n_keypoints: Number of keypoints.
    @type caller_name: str | None
    @param caller_name: Name of the caller function. Used for logging.
    @rtype: Tensor
    @return: Tensor of sigmas.
    """
    if sigmas is not None:
        if len(sigmas) == n_keypoints:
            return torch.tensor(sigmas, dtype=torch.float32)
        else:
            error_msg = "The length of the sigmas list must be the same as the number of keypoints."
            if caller_name:
                error_msg = f"[{caller_name}] {error_msg}"
            raise ValueError(error_msg)
    else:
        if n_keypoints == 17:
            msg = "Default COCO sigmas are being used."
            if caller_name:
                msg = f"[{caller_name}] {msg}"
            logger.warning(msg)
            return torch.tensor(
                [
                    0.026,
                    0.025,
                    0.025,
                    0.035,
                    0.035,
                    0.079,
                    0.079,
                    0.072,
                    0.072,
                    0.062,
                    0.062,
                    0.107,
                    0.107,
                    0.087,
                    0.087,
                    0.089,
                    0.089,
                ],
                dtype=torch.float32,
            )
        else:
            msg = "Default sigma of 0.04 is being used for each keypoint."
            if caller_name:
                msg = f"[{caller_name}] {msg}"
            logger.info(msg)
            return torch.tensor([0.04] * n_keypoints, dtype=torch.float32)


def get_center_keypoints(
    bboxes: Tensor, height: int = 1, width: int = 1
) -> Tensor:
    """Get center keypoints from bounding boxes.

    @type bboxes: Tensor
    @param bboxes: Tensor of bounding boxes.
    @type height: int
    @param height: Image height. Defaults to C{1} (normalized).
    @type width: int
    @param width: Image width. Defaults to C{1} (normalized).
    @rtype: Tensor
    @return: Tensor of center keypoints.
    """
    keypoints = torch.empty(
        (bboxes.shape[0], 4), device=bboxes.device, dtype=torch.int
    )
    keypoints[:, :2] = bboxes[:, :2]
    keypoints[:, 2] = (bboxes[:, 2] + bboxes[:, 4] / 2) * width
    keypoints[:, 3] = (bboxes[:, 3] + bboxes[:, 5] / 2) * height
    return keypoints


def insert_class(keypoints: Tensor, bboxes: Tensor) -> Tensor:
    """Insert class index into keypoints tensor.

    @type keypoints: Tensor
    @param keypoints: Tensor of keypoints.
    @type bboxes: Tensor
    @param bboxes: Tensor of bounding boxes with class index.
    @rtype: Tensor
    @return: Tensor of keypoints with class index.
    """
    classes = bboxes[:, 1]
    return torch.cat(
        (
            keypoints[:, :1],
            classes.unsqueeze(-1),
            keypoints[:, 1:],
        ),
        dim=-1,
    )
