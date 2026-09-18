"""Object keypoint similarity, the keypoint counterpart of IoU.

The metric pairs the predicted poses of each image with the target poses
and averages the similarity of the pairs. The sigma of a keypoint sets
the distance that the keypoint tolerates.

"""

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
    r"""Mean object keypoint similarity of the paired poses.

    Inputs:
        - ``keypoints`` (``list[Tensor]``): :math:`\left[M_i,
          n_{keypoints}, 3\right]` for each image, ``(x, y, conf)`` in
          pixels. `FOMOHead` gives :math:`\left[M_i, 1, 4\right]`. Only
          ``x`` and ``y`` count.
        - ``target_boundingbox`` (``Tensor``): :math:`\left[N,
          6\right]`, ``[batch, class, x, y, w, h]``, normalized
        - ``target_keypoints`` (``Tensor | None``): :math:`\left[N, 1 +
          3 n_{keypoints}\right]`, ``[batch, x, y, v, ...]``,
          normalized. ``Tasks.FOMO`` does not read it.

    Outputs:
        - ``Tensor``: scalar in ``[0, 1]``

    Formula:
        `compute_pose_oks` gives the similarity of each pair of a target
        pose and a predicted pose of one image. The pose area of a
        target is ``area_factor`` times the area of its box in pixels.
        The Hungarian algorithm pairs the poses so that the sum of the
        similarities is the largest. The score of an image is the mean
        similarity of its pairs. The metric returns the mean score of
        the images.

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        - The metric skips an image without targets.
        - A pose without a pair does not lower the score of its image.
          An image with targets and no predictions scores ``0``.
        - For ``Tasks.FOMO``, the centers of the target boxes are the
          target keypoints.

    Example:
        Attached to a ``FOMOHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: FOMOHead
              inputs: [EfficientRep]
              metrics:
                - name: ObjectKeypointSimilarity

    Compatible with:
        - Used by: `KeypointDetectionModel`
        - Nodes:

          - `EfficientKeypointBBoxHead`
          - `FOMOHead`

    """

    supported_tasks = [
        Tasks.KEYPOINTS,
        Tasks.INSTANCE_KEYPOINTS,
        Tasks.INSTANCE_SEGMENTATION_KEYPOINTS,
        Tasks.FOMO,
    ]

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
        """Initialize the metric and resolve the sigmas.

        The constructor reads `BaseAttachedModule.n_keypoints`, so the
        module needs a node. `get_sigmas` raises ``ValueError`` when
        ``sigmas`` does not have one value for each keypoint.

        Args:
            sigmas (list[float] | None): One sigma for each keypoint. A
                larger sigma tolerates a larger distance. ``None``
                selects the COCO person sigmas for ``17`` keypoints, and
                ``0.04`` for each keypoint otherwise. `get_sigmas` then
                logs the selection.
            area_factor (float | None): The factor that scales the area
                of a target box to the pose area. ``None`` selects
                ``0.53`` and logs an info message.
            use_cocoeval_oks (bool): When ``True``, use the formula of
                the COCO evaluation code. When ``False``, use the formula
                of the COCO keypoint definition. `compute_pose_oks`
                shows both formulas.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseMetric`, such as ``node``.

        """
        super().__init__(**kwargs)

        self._sigmas = get_sigmas(
            sigmas, self.n_keypoints, caller_name=self.name
        )
        self._area_factor = get_with_default(
            area_factor, "bbox area scaling", self.name, default=0.53
        )
        self._use_cocoeval_oks = use_cocoeval_oks

    @override
    def update(
        self,
        keypoints: list[Tensor],
        target_boundingbox: Tensor,
        target_keypoints: Tensor | None = None,
    ) -> None:
        """Scale the targets of one batch to pixels and store them.

        For each image with targets, the method stores the predicted
        keypoints, the target keypoints in pixels, and the pose area of
        each target. The height and the width of
        `BaseAttachedModule.original_in_shape` scale the normalized
        values.

        Args:
            keypoints (``list[Tensor]``): The predicted keypoints of each
                image, of shape ``[M_i, n_keypoints, 3]``, in pixels.
                `FOMOHead` gives the shape ``[M_i, 1, 4]``. Only ``x``
                and ``y`` count.
            target_boundingbox (``Tensor``): The target boxes of the
                batch, of shape ``[N, 6]``, as
                ``[batch, class, x, y, w, h]`` with normalized values.
            target_keypoints (``Tensor | None``): The target keypoints of
                the batch, of shape ``[N, 1 + 3 * n_keypoints]``, as
                ``[batch, x, y, v, ...]`` with normalized coordinates.
                For ``Tasks.FOMO``, the method uses the centers of the
                target boxes instead.

        Raises:
            ValueError: When ``target_keypoints`` is ``None`` and the
                task is not ``Tasks.FOMO``.

        """
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
            self.scales.append(bbox_w * bbox_h * self._area_factor)

    @override
    def compute(self) -> Tensor:
        """Pair the stored poses of each image and average the scores.

        The class docstring describes the pairing and the score. The
        method also moves ``sigmas`` to the device of the metric.

        Returns:
            ``Tensor``: The mean score of the stored images, a scalar in
            ``[0, 1]``. It is ``0`` when no image had targets.

        Example:
            The first image has two targets and one exact prediction.
            The target without a pair does not count, so the image
            scores ``1``. The second image has no prediction and scores
            ``0``:

            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import EfficientKeypointBBoxHead
            >>> head = EfficientKeypointBBoxHead(
            ...     n_heads=1,
            ...     n_classes=1,
            ...     n_keypoints=1,
            ...     input_shapes=[{"features": [Size([1, 8, 8, 8])]}],
            ...     original_in_shape=Size([3, 100, 100]),
            ... )
            >>> metric = ObjectKeypointSimilarity(
            ...     node=head, sigmas=[0.04], area_factor=0.53
            ... )
            >>> boxes = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.2, 0.2]] * 3)
            >>> boxes[2, 0] = 1.0
            >>> keypoints = torch.tensor(
            ...     [
            ...         [0.0, 0.2, 0.2, 2.0],
            ...         [0.0, 0.6, 0.6, 2.0],
            ...         [1.0, 0.2, 0.2, 2.0],
            ...     ]
            ... )
            >>> exact = torch.tensor([[[20.0, 20.0, 1.0]]])
            >>> metric.update([exact, torch.zeros(0, 1, 3)], boxes, keypoints)
            >>> metric.compute().item()
            0.5

        """
        self._sigmas = self._sigmas.to(self.device)
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
                sigmas=self._sigmas,
                use_cocoeval_oks=self._use_cocoeval_oks,
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
