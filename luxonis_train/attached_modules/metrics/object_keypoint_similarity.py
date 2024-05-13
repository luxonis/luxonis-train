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


class ObjectKeypointSimilarity(
    BaseMetric[list[dict[str, Tensor]], list[dict[str, Tensor]]]
):
    """Object Keypoint Similarity metric for evaluating keypoint predictions.

    @type n_keypoints: int
    @param n_keypoints: Number of keypoints.
    @type kpt_sigmas: Tensor
    @param kpt_sigmas: Sigma for each keypoint to weigh its importance, if C{None}, then
        use same weights for all.
    @type use_cocoeval_oks: bool
    @param use_cocoeval_oks: Whether to use same OKS formula as in COCOeval or use the
        one from definition.
    """

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
        kpt_sigmas: Tensor | None = None,
        use_cocoeval_oks: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            required_labels=[LabelType.KEYPOINT], protocol=KeypointProtocol, **kwargs
        )

        if n_keypoints is None and self.node is None:
            raise ValueError(
                f"Either `n_keypoints` or `node` must be provided "
                f"to {self.__class__.__name__}."
            )
        self.n_keypoints = n_keypoints or self.node.n_keypoints
        if kpt_sigmas is not None and len(kpt_sigmas) != self.n_keypoints:
            raise ValueError("Expected kpt_sigmas to be of shape (num_keypoints).")
        self.kpt_sigmas = kpt_sigmas or torch.ones(self.n_keypoints) / self.n_keypoints
        self.use_cocoeval_oks = use_cocoeval_oks

        self.add_state("pred_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_scales", default=[], dist_reduce_fx=None)

    def prepare(
        self, outputs: Packet[Tensor], labels: Labels
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        kpts_labels = labels[LabelType.KEYPOINT]
        bbox_labels = labels[LabelType.BOUNDINGBOX]
        num_keypoints = (kpts_labels.shape[1] - 2) // 3
        label = torch.zeros((len(bbox_labels), num_keypoints * 3 + 6))
        label[:, :2] = bbox_labels[:, :2]
        label[:, 2:6] = box_convert(bbox_labels[:, 2:], "xywh", "xyxy")
        label[:, 6::3] = kpts_labels[:, 2::3]  # insert kp x coordinates
        label[:, 7::3] = kpts_labels[:, 3::3]  # insert kp y coordinates
        label[:, 8::3] = kpts_labels[:, 4::3]  # insert kp visibility

        output_list_oks = []
        label_list_oks = []
        image_size = self.node.original_in_shape[2:]

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
            curr_scales = torch.sqrt(curr_bboxs_widths * curr_bboxs_heights)
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

        self.kpt_sigmas = self.kpt_sigmas.to(self.device)
        image_mean_oks = torch.zeros(len(self.groundtruth_keypoints))
        for i, (pred_kpts, gt_kpts, gt_scales) in enumerate(
            zip(
                self.pred_keypoints, self.groundtruth_keypoints, self.groundtruth_scales
            )
        ):
            gt_kpts = torch.reshape(gt_kpts, (-1, self.n_keypoints, 3))  # [N, K, 3]

            image_ious = self._compute_oks(pred_kpts, gt_kpts, gt_scales)  # [M, N]
            gt_indices, pred_indices = linear_sum_assignment(
                image_ious.cpu().numpy(), maximize=True
            )
            matched_ious = [image_ious[n, m] for n, m in zip(gt_indices, pred_indices)]
            image_mean_oks[i] = torch.tensor(matched_ious).mean()

        final_oks = image_mean_oks.nanmean()

        return final_oks

    def _compute_oks(self, pred: Tensor, gt: Tensor, scales: Tensor) -> Tensor:
        """Compute Object Keypoint Similarity between every GT and prediction.

        @type pred: Tensor[N, K, 3]
        @param pred: Predicted keypoints.
        @type gt: Tensor[M, K, 3]
        @param gt: Groundtruth keypoints.
        @type scales: Tensor[M]
        @param scales: Scales of the bounding boxes.
        @rtype: Tensor
        @return: Object Keypoint Similarity every pred and gt [M, N]
        """
        eps = 1e-7
        distances = (gt[:, None, :, 0] - pred[..., 0]) ** 2 + (
            gt[:, None, :, 1] - pred[..., 1]
        ) ** 2
        kpt_mask = gt[..., 2] != 0  # only compute on visible keypoints
        if self.use_cocoeval_oks:
            # use same formula as in COCOEval script here:
            # https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L229
            oks = (
                distances
                / (2 * self.kpt_sigmas) ** 2
                / (scales[:, None, None] + eps)
                / 2
            )
        else:
            # use same formula as defined here: https://cocodataset.org/#keypoints-eval
            oks = (
                distances
                / ((scales[:, None, None] + eps) * self.kpt_sigmas.to(scales.device))
                ** 2
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
