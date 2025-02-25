from functools import cached_property

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchvision.ops import box_convert
from typing_extensions import override

from luxonis_train.tasks import Metadata, Tasks
from luxonis_train.utils import (
    compute_pose_oks,
    get_sigmas,
    get_with_default,
)
from luxonis_train.utils.keypoints import get_center_keypoints, insert_class

from .base_metric import BaseMetric


class ObjectKeypointSimilarity(BaseMetric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    pred_keypoints: list[Tensor]
    groundtruth_keypoints: list[Tensor]
    groundtruth_scales: list[Tensor]

    supported_tasks = [Tasks.KEYPOINTS, Tasks.INSTANCE_KEYPOINTS, Tasks.FOMO]

    def __init__(
        self,
        sigmas: list[float] | None = None,
        area_factor: float | None = None,
        use_cocoeval_oks: bool = True,
        **kwargs,
    ) -> None:
        """Object Keypoint Similarity metric for evaluating keypoint
        predictions.

        @type sigmas: list[float] | None
        @param sigmas: Sigma for each keypoint to weigh its importance,
            if C{None}, then use COCO if possible otherwise defaults.
            Defaults to C{None}.
        @type area_factor: float | None
        @param area_factor: Factor by which we multiply the bounding box
            area. If not set, the default factor of C{0.53} is used.
        @type use_cocoeval_oks: bool
        @param use_cocoeval_oks: Whether to use same OKS formula as in
            COCOeval or use the one from definition. Defaults to
            C{True}.
        """
        super().__init__(**kwargs)

        self.sigmas = get_sigmas(
            sigmas, self.n_keypoints, caller_name=self.name
        )
        self.area_factor = get_with_default(
            area_factor, "bbox area scaling", self.name, default=0.53
        )
        self.use_cocoeval_oks = use_cocoeval_oks

        self.add_state("pred_keypoints", default=[], dist_reduce_fx=None)
        self.add_state(
            "groundtruth_keypoints", default=[], dist_reduce_fx=None
        )
        self.add_state("groundtruth_scales", default=[], dist_reduce_fx=None)

    @cached_property
    @override
    def required_labels(self) -> set[str | Metadata]:
        if self.task == Tasks.FOMO:
            return Tasks.BOUNDINGBOX.required_labels
        return self.task.required_labels

    def update(
        self,
        keypoints: list[Tensor],
        target_boundingbox: Tensor,
        target_keypoints: Tensor | None = None,
    ) -> None:
        if target_keypoints is None:
            if self.task != Tasks.FOMO:
                raise ValueError(
                    "The target keypoints are not required only when used "
                    " with FOMO task."
                )
            target_keypoints = get_center_keypoints(target_boundingbox)
        target_keypoints = insert_class(target_keypoints, target_boundingbox)
        n_keypoints = (target_keypoints.shape[1] - 2) // 3
        label = torch.zeros((len(target_boundingbox), n_keypoints * 3 + 6))
        label[:, :2] = target_boundingbox[:, :2]
        label[:, 2:6] = box_convert(target_boundingbox[:, 2:], "xywh", "xyxy")
        label[:, 6::3] = target_keypoints[:, 2::3]  # insert kp x coordinates
        label[:, 7::3] = target_keypoints[:, 3::3]  # insert kp y coordinates
        label[:, 8::3] = target_keypoints[:, 4::3]  # insert kp visibility

        prediction_oks = []
        target_oks = []
        image_size = self.original_in_shape[1:]

        for i, pred_kpt in enumerate(keypoints):
            prediction_oks.append({"keypoints": pred_kpt})

            curr_label = label[label[:, 0] == i].to(pred_kpt.device)
            curr_bboxs = curr_label[:, 2:6]
            curr_bboxs[:, 0::2] = (
                (curr_bboxs[:, 0::2] * image_size[1]).round().int()
            )
            curr_bboxs[:, 1::2] = (
                (curr_bboxs[:, 1::2] * image_size[0]).round().int()
            )
            curr_kpts = curr_label[:, 6:]
            curr_kpts[:, 0::3] = (
                (curr_kpts[:, 0::3] * image_size[1]).round().int()
            )
            curr_kpts[:, 1::3] = (
                (curr_kpts[:, 1::3] * image_size[0]).round().int()
            )
            curr_bboxs_widths = curr_bboxs[:, 2] - curr_bboxs[:, 0]
            curr_bboxs_heights = curr_bboxs[:, 3] - curr_bboxs[:, 1]
            curr_scales = (
                curr_bboxs_widths * curr_bboxs_heights * self.area_factor
            )
            target_oks.append({"keypoints": curr_kpts, "scales": curr_scales})

        for item in prediction_oks:
            self.pred_keypoints.append(
                self._fix_empty_tensors(item["keypoints"])
            )

        for item in target_oks:
            self.groundtruth_keypoints.append(
                self._fix_empty_tensors(item["keypoints"])
            )
            self.groundtruth_scales.append(item["scales"])

    def compute(self) -> Tensor:
        """Computes the OKS metric based on the inner state."""

        self.sigmas = self.sigmas.to(self.device)
        image_mean_oks = torch.zeros(len(self.groundtruth_keypoints))
        for i, (pred_kpts, gt_kpts, gt_scales) in enumerate(
            zip(
                self.pred_keypoints,
                self.groundtruth_keypoints,
                self.groundtruth_scales,
            )
        ):
            gt_kpts = torch.reshape(
                gt_kpts, (-1, self.n_keypoints, 3)
            )  # [N, K, 3]

            image_ious = compute_pose_oks(
                gt_kpts.unsqueeze(0),
                pred_kpts.unsqueeze(0),
                self.sigmas,
                use_cocoeval_oks=self.use_cocoeval_oks,
                pose_area=gt_scales.unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
                gt_bboxes=None,
            ).squeeze(0)  # [N, M]
            gt_indices, pred_indices = linear_sum_assignment(
                image_ious.cpu().numpy(), maximize=True
            )
            matched_ious = [
                image_ious[n, m] for n, m in zip(gt_indices, pred_indices)
            ]
            image_mean_oks[i] = torch.tensor(matched_ious).mean()

        final_oks = image_mean_oks.nanmean()

        return final_oks

    @staticmethod
    def _fix_empty_tensors(input_tensor: Tensor) -> Tensor:
        """Empty tensors can cause problems in DDP mode, this methods
        corrects them."""
        if input_tensor.numel() == 0 and input_tensor.ndim == 1:
            return input_tensor.unsqueeze(0)
        return input_tensor
