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


def compute_pose_oks(
    gt_kps: torch.Tensor,
    pred_kps: torch.Tensor,
    kp_sigmas: torch.Tensor,
    gt_bboxes: torch.Tensor | None = None,
    pose_area: torch.Tensor | None = None,
    eps: float = 1e-9,
    area_factor: float = 0.53,
    use_cocoeval_oks: bool = True,
) -> torch.Tensor:
    """Compute batched Object Keypoint Similarity (OKS) between ground
    truth and predicted keypoints.

    @type gt_kps: torch.Tensor
    @param gt_kps: Ground truth keypoints with shape [N, M1,
        num_keypoints, 3]
    @type pred_kps: torch.Tensor
    @param pred_kps: Predicted keypoints with shape [N, M2,
        num_keypoints, 3]
    @type kp_sigmas: torch.Tensor
    @param kp_sigmas: Sigmas for each keypoint, shape [num_keypoints]
    @type gt_bboxes: torch.Tensor
    @param gt_bboxes: Ground truth bounding boxes in XYXY format with
        shape [N, M1, 4]
    @type pose_area: torch.Tensor
    @param pose_area: Area of the pose, shape [N, M1, 1, 1]
    @type eps: float
    @param eps: A small constant to ensure numerical stability
    @type area_factor: float
    @param area_factor: Factor to scale the area of the pose
    @type use_cocoeval_oks: bool
    @param use_cocoeval_oks: Whether to use the same OKS formula as in
        COCOEval or use the one from the definition. Defaults to True.
    @rtype: torch.Tensor
    @return: A tensor of OKS values with shape [N, M1, M2]
    """

    if pose_area is None:
        if gt_bboxes is None:
            raise ValueError(
                "Either 'pose_area' or 'gt_bboxes' must be provided."
            )
        width = gt_bboxes[:, :, 2] - gt_bboxes[:, :, 0]
        height = gt_bboxes[:, :, 3] - gt_bboxes[:, :, 1]
        pose_area = (
            (width * height * area_factor).unsqueeze(-1).unsqueeze(-1)
        )  # shape: [N, M1, 1, 1]

    gt_xy = gt_kps[:, :, :, :2].unsqueeze(
        2
    )  # shape: [N, M1, 1, num_keypoints, 2]
    pred_xy = pred_kps[:, :, :, :2].unsqueeze(
        1
    )  # shape: [N, 1, M2, num_keypoints, 2]

    sq_diff = ((gt_xy - pred_xy) ** 2).sum(
        dim=-1
    )  # shape: [N, M1, M2, num_keypoints]

    kp_sigmas = kp_sigmas.view(1, 1, 1, -1)  # shape: [1, 1, 1, num_keypoints]

    if use_cocoeval_oks:
        # use same formula as in COCOEval script here:
        # https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L229
        exp_term = sq_diff / ((2 * kp_sigmas) ** 2) / (pose_area + eps) / 2
    else:
        # use same formula as defined here: https://cocodataset.org/#keypoints-eval
        exp_term = sq_diff / ((pose_area + eps) * kp_sigmas) ** 2 / 2

    oks_vals = torch.exp(-exp_term)  # shape: [N, M1, M2, num_keypoints]

    vis_mask = (
        gt_kps[:, :, :, 2].gt(0).float().unsqueeze(2)
    )  # shape: [N, M1, 1, num_keypoints]
    vis_count = vis_mask.sum(dim=-1)  # shape: [N, M1, M2]

    mean_oks = (oks_vals * vis_mask).sum(dim=-1) / (
        vis_count + eps
    )  # shape: [N, M1, M2]

    return mean_oks
