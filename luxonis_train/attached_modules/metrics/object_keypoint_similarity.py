import logging

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.utils.types import (
    KeypointProtocol,
    Labels,
    LabelType,
    Packet,
)

from .base_metric import BaseMetric

logger = logging.getLogger(__name__)


class ObjectKeypointSimilarity(
    BaseMetric[list[dict[str, Tensor]], list[dict[str, Tensor]]]
):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    pred_keypoints: list[Tensor]
    groundtruth_keypoints: list[Tensor]
    groundtruth_scales: list[Tensor]

    def __init__(
        self,
        n_keypoints: int | None = None,
        sigmas: list[float] | None = None,
        area_factor: float | None = None,
        use_cocoeval_oks: bool = True,
        **kwargs,
    ) -> None:
        """Object Keypoint Similarity metric for evaluating keypoint predictions.

        @type n_keypoints: int
        @param n_keypoints: Number of keypoints.
        @type sigmas: list[float] | None
        @param sigmas: Sigma for each keypoint to weigh its importance, if C{None}, then
            use COCO if possible otherwise defaults. Defaults to C{None}.
        @type area_factor: float | None
        @param area_factor: Factor by which we multiply bbox area. If None then use
            default one. Defaults to C{None}.
        @type use_cocoeval_oks: bool
        @param use_cocoeval_oks: Whether to use same OKS formula as in COCOeval or use
            the one from definition. Defaults to C{True}.
        """
        super().__init__(
            required_labels=[LabelType.KEYPOINTS], protocol=KeypointProtocol, **kwargs
        )

        if n_keypoints is None and self.node is None:
            raise ValueError(
                f"Either `n_keypoints` or `node` must be provided "
                f"to {self.__class__.__name__}."
            )
        self.n_keypoints = n_keypoints or self.node.n_keypoints

        self.sigmas = set_sigmas(sigmas, self.n_keypoints, self.__class__.__name__)
        self.area_factor = set_area_factor(area_factor, self.__class__.__name__)
        self.use_cocoeval_oks = use_cocoeval_oks

        self.add_state("pred_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_scales", default=[], dist_reduce_fx=None)

    def prepare(
        self, outputs: Packet[Tensor], labels: Labels
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        kpts_labels = labels["keypoints"][0]
        bbox_labels = labels["boundingbox"][0]
        num_keypoints = (kpts_labels.shape[1] - 2) // 3
        label = torch.zeros((len(bbox_labels), num_keypoints * 3 + 6))
        label[:, :2] = bbox_labels[:, :2]
        label[:, 2:6] = box_convert(bbox_labels[:, 2:], "xywh", "xyxy")
        label[:, 6::3] = kpts_labels[:, 2::3]  # insert kp x coordinates
        label[:, 7::3] = kpts_labels[:, 3::3]  # insert kp y coordinates
        label[:, 8::3] = kpts_labels[:, 4::3]  # insert kp visibility

        output_list_oks = []
        label_list_oks = []
        image_size = self.node.original_in_shape[1:]

        for i, pred_kpt in enumerate(outputs["keypoints"]):
            output_list_oks.append({"keypoints": pred_kpt})

            curr_label = label[label[:, 0] == i].to(pred_kpt.device)
            curr_bboxs = curr_label[:, 2:6]
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]
            curr_kpts = curr_label[:, 6:]
            curr_kpts[:, 0::3] *= image_size[1]
            curr_kpts[:, 1::3] *= image_size[0]
            curr_bboxs_widths = curr_bboxs[:, 2] - curr_bboxs[:, 0]
            curr_bboxs_heights = curr_bboxs[:, 3] - curr_bboxs[:, 1]
            curr_scales = curr_bboxs_widths * curr_bboxs_heights * self.area_factor
            label_list_oks.append({"keypoints": curr_kpts, "scales": curr_scales})

        return output_list_oks, label_list_oks

    def update(
        self, preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]]
    ) -> None:
        """Updates the inner state of the metric.

        @type preds: list[dict[str, Tensor]]
        @param preds: A list consisting of dictionaries each containing key-values for
            a single image.
            Parameters that should be provided per dict:

                - keypoints (FloatTensor): Tensor of shape (N, 3*K) and in format
                  [x, y, vis, x, y, vis, ...] where `x` an `y`
                  are unnormalized keypoint coordinates and `vis` is keypoint visibility.
        @type target: list[dict[str, Tensor]]
        @param target: A list consisting of dictionaries each containing key-values for
            a single image.
            Parameters that should be provided per dict:

                - keypoints (FloatTensor): Tensor of shape (N, 3*K) and in format
                  [x, y, vis, x, y, vis, ...] where `x` an `y`
                  are unnormalized keypoint coordinates and `vis` is keypoint visibility.
                - scales (FloatTensor): Tensor of shape (N) where each value
                  corresponds to scale of the bounding box.
                  Scale of one bounding box is defined as sqrt(width*height) where
                  width and height are unnormalized.
        """
        for item in preds:
            keypoints = fix_empty_tensors(item["keypoints"])
            self.pred_keypoints.append(keypoints)

        for item in target:
            keypoints = fix_empty_tensors(item["keypoints"])
            self.groundtruth_keypoints.append(keypoints)
            self.groundtruth_scales.append(item["scales"])

    def compute(self) -> Tensor:
        """Computes the OKS metric based on the inner state."""

        self.sigmas = self.sigmas.to(self.device)
        image_mean_oks = torch.zeros(len(self.groundtruth_keypoints))
        for i, (pred_kpts, gt_kpts, gt_scales) in enumerate(
            zip(
                self.pred_keypoints, self.groundtruth_keypoints, self.groundtruth_scales
            )
        ):
            gt_kpts = torch.reshape(gt_kpts, (-1, self.n_keypoints, 3))  # [N, K, 3]

            image_ious = compute_oks(
                pred_kpts,
                gt_kpts,
                gt_scales,
                self.sigmas,
                self.use_cocoeval_oks,
            )  # [M, N]
            gt_indices, pred_indices = linear_sum_assignment(
                image_ious.cpu().numpy(), maximize=True
            )
            matched_ious = [image_ious[n, m] for n, m in zip(gt_indices, pred_indices)]
            image_mean_oks[i] = torch.tensor(matched_ious).mean()

        final_oks = image_mean_oks.nanmean()

        return final_oks


def compute_oks(
    pred: Tensor,
    gt: Tensor,
    scales: Tensor,
    sigmas: Tensor,
    use_cocoeval_oks: bool,
) -> Tensor:
    """Compute Object Keypoint Similarity between every GT and prediction.

    @type pred: Tensor[N, K, 3]
    @param pred: Predicted keypoints.
    @type gt: Tensor[M, K, 3]
    @param gt: Groundtruth keypoints.
    @type scales: Tensor[M]
    @param scales: Scales of the bounding boxes.
    @rtype: Tensor
    @return: Object Keypoint Similarity every pred and gt [M, N]
    @type sigmas: Tensor
    @param sigmas: Sigma for each keypoint to weigh its importance, if C{None}, then use
        same weights for all.
    @type use_cocoeval_oks: bool
    @param use_cocoeval_oks: Whether to use same OKS formula as in COCOeval or use the
        one from definition.
    """
    eps = 1e-7
    distances = (gt[:, None, :, 0] - pred[..., 0]) ** 2 + (
        gt[:, None, :, 1] - pred[..., 1]
    ) ** 2
    kpt_mask = gt[..., 2] != 0  # only compute on visible keypoints
    if use_cocoeval_oks:
        # use same formula as in COCOEval script here:
        # https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L229
        oks = distances / (2 * sigmas) ** 2 / (scales[:, None, None] + eps) / 2
    else:
        # use same formula as defined here: https://cocodataset.org/#keypoints-eval
        oks = (
            distances
            / ((scales[:, None, None] + eps) * sigmas.to(scales.device)) ** 2
            / 2
        )

    return (torch.exp(-oks) * kpt_mask[:, None]).sum(-1) / (
        kpt_mask.sum(-1)[:, None] + eps
    )


def fix_empty_tensors(input_tensor: Tensor) -> Tensor:
    """Empty tensors can cause problems in DDP mode, this methods corrects them."""
    if input_tensor.numel() == 0 and input_tensor.ndim == 1:
        return input_tensor.unsqueeze(0)
    return input_tensor


def set_sigmas(
    sigmas: list[float] | None, n_keypoints: int, class_name: str | None
) -> Tensor:
    """Validate and set the sigma values."""
    if sigmas is not None:
        if len(sigmas) == n_keypoints:
            return torch.tensor(sigmas, dtype=torch.float32)
        else:
            error_msg = "The length of the sigmas list must be the same as the number of keypoints."
            if class_name:
                error_msg = f"[{class_name}] {error_msg}"
            raise ValueError(error_msg)
    else:
        if n_keypoints == 17:
            warn_msg = "Default COCO sigmas are being used."
            if class_name:
                warn_msg = f"[{class_name}] {warn_msg}"
            logger.warning(warn_msg)
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
            warn_msg = "Default sigma of 0.04 is being used for each keypoint."
            if class_name:
                warn_msg = f"[{class_name}] {warn_msg}"
            logger.warning(warn_msg)
            return torch.tensor([0.04] * n_keypoints, dtype=torch.float32)


def set_area_factor(area_factor: float | None, class_name: str | None) -> float:
    """Set the default area factor if not defined."""
    if area_factor is None:
        warn_msg = "Default area_factor of 0.53 is being used bbox area scaling."
        if class_name:
            warn_msg = f"[{class_name}] {warn_msg}"
        logger.warning(warn_msg)
        return 0.53
    else:
        return area_factor
