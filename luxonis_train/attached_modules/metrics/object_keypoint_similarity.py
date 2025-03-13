from functools import cached_property
from typing import Annotated

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchvision.ops import box_convert
from typing_extensions import override

from luxonis_train.tasks import Metadata, Tasks
from luxonis_train.utils import compute_pose_oks, get_sigmas, get_with_default
from luxonis_train.utils.keypoints import get_center_keypoints, insert_class

from .base_metric import BaseMetric, State


class ObjectKeypointSimilarity(BaseMetric):
    supported_tasks = [Tasks.KEYPOINTS, Tasks.INSTANCE_KEYPOINTS, Tasks.FOMO]

    pred_keypoints: Annotated[list[Tensor], State(default=[])]
    target_keypoints: Annotated[list[Tensor], State(default=[])]
    target_scales: Annotated[list[Tensor], State(default=[])]

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

    @cached_property
    @override
    def required_labels(self) -> set[str | Metadata]:
        if self.task == Tasks.FOMO:
            return Tasks.BOUNDINGBOX.required_labels
        return self.task.required_labels

    @override
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
        targets = self._construct_targets(target_boundingbox, target_keypoints)

        h, w = self.original_in_shape[1:]

        for i, pred_kpt in enumerate(keypoints):
            target = targets[targets[:, 0] == i].to(pred_kpt.device)

            self.pred_keypoints.append(self._fix_empty_tensors(pred_kpt))

            kpts = target[:, 6:]
            kpts[:, 0::3] *= w
            kpts[:, 1::3] *= h

            self.target_keypoints.append(
                self._fix_empty_tensors(kpts.round().int())
            )

            bbox = target[:, 2:6]
            bbox[:, 0::2] *= w
            bbox[:, 1::2] *= h

            bbox_widths = bbox[:, 2] - bbox[:, 0]
            bbox_heights = bbox[:, 3] - bbox[:, 1]
            bbox_scales = bbox_widths * bbox_heights * self.area_factor

            self.target_scales.append(bbox_scales.round().int())

    @override
    def compute(self) -> Tensor:
        """Computes the OKS metric based on the inner state."""

        self.sigmas = self.sigmas.to(self.device)
        image_mean_oks = torch.zeros(len(self.target_keypoints))
        for i, (pred_keypoints, target_keypoints, target_scales) in enumerate(
            zip(
                self.pred_keypoints,
                self.target_keypoints,
                self.target_scales,
                strict=True,
            )
        ):
            target_keypoints = target_keypoints.reshape(
                -1, self.n_keypoints, 3
            )

            image_ious = compute_pose_oks(
                pred_keypoints.unsqueeze(0),
                target_keypoints.unsqueeze(0),
                self.sigmas,
                use_cocoeval_oks=self.use_cocoeval_oks,
                pose_area=target_scales.unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1),
                gt_bboxes=None,
            ).squeeze(0)  # [N, M]

            target_indices, pred_indices = linear_sum_assignment(
                image_ious.cpu().numpy(), maximize=True
            )
            matched_ious = [
                image_ious[n, m]
                for n, m in zip(target_indices, pred_indices, strict=True)
            ]
            image_mean_oks[i] = torch.tensor(matched_ious).mean()

        return image_mean_oks.nanmean()

    @staticmethod
    def _fix_empty_tensors(input_tensor: Tensor) -> Tensor:
        """Empty tensors can cause problems in DDP mode, this methods
        corrects them."""
        if input_tensor.numel() == 0 and input_tensor.ndim == 1:
            return input_tensor.unsqueeze(0)
        return input_tensor

    @staticmethod
    def _construct_targets(
        target_boundingbox: Tensor, target_keypoints: Tensor
    ) -> Tensor:
        n_keypoints = (target_keypoints.shape[1] - 2) // 3

        target = torch.zeros((len(target_boundingbox), n_keypoints * 3 + 6))
        target[:, :2] = target_boundingbox[:, :2]
        target[:, 2:6] = box_convert(target_boundingbox[:, 2:], "xywh", "xyxy")
        target[:, 6::3] = target_keypoints[:, 2::3]  # insert x coordinates
        target[:, 7::3] = target_keypoints[:, 3::3]  # insert y coordinates
        target[:, 8::3] = target_keypoints[:, 4::3]  # insert visibility
        return target
