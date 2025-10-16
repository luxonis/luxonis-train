from typing import Annotated

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Tasks
from luxonis_train.utils import (
    compute_pose_oks,
    get_sigmas,
    get_with_default,
    instances_from_batch,
)
from luxonis_train.utils.keypoints import get_center_keypoints

from .base_metric import BaseMetric, MetricState
from .utils import fix_empty_tensor


class ObjectKeypointSimilarity(BaseMetric):
    supported_tasks = [Tasks.KEYPOINTS, Tasks.INSTANCE_KEYPOINTS, Tasks.FOMO]

    pred_keypoints: Annotated[list[Tensor], MetricState()]
    target_keypoints: Annotated[list[Tensor], MetricState()]
    scales: Annotated[list[Tensor], MetricState()]

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

    @override
    def update(
        self,
        keypoints: list[Tensor],
        target_boundingbox: Tensor,
        target_keypoints: Tensor | None = None,
    ) -> None:
        keypoints, target_keypoints = self._adjust_for_fomo(
            keypoints, target_boundingbox, target_keypoints
        )

        h, w = self.original_in_shape[1:]
        bs = len(keypoints)

        for i, (bboxes, kpts) in enumerate(
            instances_from_batch(
                target_boundingbox, target_keypoints, batch_size=bs
            )
        ):
            if kpts.numel() == 0:
                # Skipping images with no keypoints annotations
                continue

            bbox_w = bboxes[:, 3] * w
            bbox_h = bboxes[:, 4] * h

            kpts = kpts[:, 1:]
            kpts[:, 0::3] *= w
            kpts[:, 1::3] *= h

            self.pred_keypoints.append(fix_empty_tensor(keypoints[i]))
            self.target_keypoints.append(fix_empty_tensor(kpts))
            self.scales.append(bbox_w * bbox_h * self.area_factor)

    @override
    def compute(self) -> Tensor:
        self.sigmas = self.sigmas.to(self.device)
        mean_oks = torch.zeros(len(self.target_keypoints), device=self.device)
        for i, (pred_kpts, target_kpts, scales) in enumerate(
            zip(
                self.pred_keypoints,
                self.target_keypoints,
                self.scales,
                strict=True,
            )
        ):
            image_ious = compute_pose_oks(
                pred_kpts.unsqueeze(0),
                target_kpts.reshape(-1, self.n_keypoints, 3).unsqueeze(0),
                sigmas=self.sigmas,
                use_cocoeval_oks=self.use_cocoeval_oks,
                pose_area=scales[None, :, None, None],
            ).squeeze(0)

            cost = image_ious.detach().cpu().numpy()
            gt_indices, pred_indices = linear_sum_assignment(
                cost, maximize=True
            )
            matched_ious = [
                image_ious[n, m]
                for n, m in zip(gt_indices, pred_indices, strict=True)
            ]

            if len(matched_ious) > 0:
                mean_oks[i] = torch.stack(matched_ious).mean()
            else:
                mean_oks[i] = torch.tensor(0.0, device=self.device)

        return mean_oks.nanmean().nan_to_num()

    def _adjust_for_fomo(
        self,
        keypoints: list[Tensor],
        target_boundingbox: Tensor,
        target_keypoints: Tensor | None,
    ) -> tuple[list[Tensor], Tensor]:
        if self.task == Tasks.FOMO:
            target_keypoints = get_center_keypoints(target_boundingbox)
        elif target_keypoints is None:
            raise ValueError(
                "The target keypoints are not required only when used "
                " with the 'FOMO' task."
            )
        return keypoints, target_keypoints
