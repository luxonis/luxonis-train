from typing import Annotated, Literal

import torch
from faster_coco_eval.core import COCO, COCOeval_faster
from torch import Tensor
from torchvision.ops import box_convert
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric, MetricState
from luxonis_train.attached_modules.metrics.mean_average_precision.utils import (
    add_f1_metrics,
)
from luxonis_train.attached_modules.metrics.utils import (
    fix_empty_tensor,
    merge_bbox_kpt_targets,
)
from luxonis_train.tasks import Tasks
from luxonis_train.utils import get_sigmas, get_with_default


class MeanAveragePrecisionKeypoints(BaseMetric):
    """Mean average precision metric for keypoint detections.

    Uses ``OKS`` as IoU measure.

    Metadata:
        - Module type: metric
        - Registry name: ``MeanAveragePrecisionKeypoints``
        - Task: INSTANCE_KEYPOINTS, INSTANCE_SEGMENTATION_KEYPOINTS
        - Attached node types: None
        - Inputs: ``keypoints``, ``boundingbox``, ``target_keypoints``,
          ``target_boundingbox``
        - Outputs: main keypoint mAP tensor and dictionary of keypoint AP/AR/F1
          sub-metrics
        - State: ``pred_bboxes``, ``pred_scores``, ``pred_classes``,
          ``pred_keypoints``, ``target_bboxes``, ``target_classes``,
          ``target_keypoints``

    Prediction format:
        ``keypoints`` and ``boundingbox`` are per-image prediction lists with
        keypoint triplets, boxes, scores, and class IDs.

    Target format:
        ``target_keypoints`` and ``target_boundingbox`` contain batch-indexed
        normalized keypoint and bbox targets.

    Formula:
        Converts cached predictions and targets to COCO objects and evaluates
        keypoint AP/AR using OKS.

    Provenance:
        - Source: torchmetrics mean AP adaptation
        - License: Apache License, Version 2.0
        - Implementation notes: Uses ``faster_coco_eval`` and configurable OKS
          sigmas/max detections.

    """

    supported_tasks = [
        Tasks.INSTANCE_KEYPOINTS,
        Tasks.INSTANCE_SEGMENTATION_KEYPOINTS,
    ]

    pred_bboxes: Annotated[list[Tensor], MetricState()]
    pred_scores: Annotated[list[Tensor], MetricState()]
    pred_classes: Annotated[list[Tensor], MetricState()]
    pred_keypoints: Annotated[list[Tensor], MetricState()]

    target_bboxes: Annotated[list[Tensor], MetricState()]
    target_classes: Annotated[list[Tensor], MetricState()]
    target_keypoints: Annotated[list[Tensor], MetricState()]

    @typechecked
    def __init__(
        self,
        sigmas: list[float] | None = None,
        area_factor: float | None = None,
        max_dets: int = 20,
        box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
        **kwargs,
    ):
        """Initialize the keypoint mean average precision metric.

        Args:
            sigmas (list[float] | None): Sigma for each keypoint to weigh its importance, if
                ``None``, then use COCO if possible otherwise defaults. Defaults to ``None``.
            area_factor (float | None): Factor by which we multiply the bounding box area. If not
                set, the default factor of ``0.53`` is used.
            max_dets (int): Maximum number of detections to be considered per image. Defaults to
                ``20``.
            box_format (Literal["xyxy", "xywh", "cxcywh"]): Input bounding box format. Defaults to
                ``"xyxy"``.
            **kwargs (Any): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)

        self.sigmas = get_sigmas(sigmas, self.n_keypoints, self.name)
        self.area_factor = get_with_default(
            area_factor, "bbox area scaling", self.name, default=0.53
        )
        self.max_dets = max_dets
        self.box_format = box_format

    @override
    def update(
        self,
        keypoints: list[Tensor],
        boundingbox: list[Tensor],
        target_keypoints: Tensor,
        target_boundingbox: Tensor,
    ) -> None:
        targets = merge_bbox_kpt_targets(
            target_boundingbox, target_keypoints, device=self.device
        )

        h, w = self.original_in_shape[1:]

        for i in range(len(keypoints)):
            self.pred_bboxes.append(
                self._convert_bboxes(boundingbox[i][:, :4])
            )
            self.pred_scores.append(boundingbox[i][:, 4])
            self.pred_classes.append(boundingbox[i][:, 5].int())
            self.pred_keypoints.append(
                fix_empty_tensor(
                    keypoints[i].reshape(-1, self.n_keypoints * 3)
                )
            )

            target = targets[targets[:, 0] == i]
            self.target_classes.append(target[:, 1].int())

            bboxes = target[:, 2:6]
            bboxes[:, 0::2] *= w
            bboxes[:, 1::2] *= h
            self.target_bboxes.append(self._convert_bboxes(bboxes.int()))

            kpts = target[:, 6:]
            kpts[:, 0::3] *= w
            kpts[:, 1::3] *= h
            self.target_keypoints.append(fix_empty_tensor(kpts.int()))

    @override
    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        coco_target = self._get_coco(
            self.target_bboxes,
            self.target_keypoints,
            self.target_classes,
        )
        coco_preds = self._get_coco(
            self.pred_bboxes,
            self.pred_keypoints,
            self.pred_classes,
            self.pred_scores,
        )

        self.coco_eval = COCOeval_faster(
            coco_target, coco_preds, iouType="keypoints"
        )
        self.coco_eval.params.kpt_oks_sigmas = self.sigmas.cpu().numpy()
        self.coco_eval.params.maxDets = [self.max_dets]

        self.coco_eval.run()

        stats = torch.tensor(
            self.coco_eval.stats, dtype=torch.float32, device=self.device
        )

        return stats[0], add_f1_metrics(
            {
                "kpt_map_50": stats[1],
                "kpt_map_75": stats[2],
                "kpt_map_medium": stats[3],
                "kpt_map_large": stats[4],
                "kpt_mar": stats[5],
                "kpt_mar_50": stats[6],
                "kpt_mar_75": stats[7],
                "kpt_mar_medium": stats[8],
                "kpt_mar_large": stats[9],
            }
        )

    def _get_coco(
        self,
        bboxes_list: list[Tensor],
        keypoints_list: list[Tensor],
        classes_list: list[Tensor],
        scores_list: list[Tensor] | None = None,
    ) -> COCO:
        """Transform cached targets or predictions into COCO format.

        Format is defined in the
        `COCO data format <https://cocodataset.org/#format-data>`_.

        Args:
            bboxes_list (list[Tensor]): Bounding boxes grouped by image.
            keypoints_list (list[Tensor]): Keypoints grouped by image.
            classes_list (list[Tensor]): Class IDs grouped by image.
            scores_list (list[Tensor] | None): Optional prediction scores
                grouped by image.

        Returns:
            COCO: COCO object containing the cached annotations.

        """
        annotations = []

        for i, (bboxes, keypoints, classes) in enumerate(
            zip(bboxes_list, keypoints_list, classes_list, strict=True)
        ):
            for j, (bbox, kpts, class_id) in enumerate(
                zip(bboxes, keypoints, classes, strict=False)
            ):
                annotation = {
                    "id": len(annotations) + 1,
                    "image_id": i,
                    "bbox": bbox.cpu().tolist(),
                    "area": (bbox[2] * bbox[3] * self.area_factor).item(),
                    "category_id": class_id.item(),
                    "keypoints": kpts.cpu().tolist(),
                    "num_keypoints": kpts[2::3].ne(0).sum().item(),
                    "iscrowd": 0,
                }

                if scores_list is not None:
                    annotation["score"] = scores_list[i][j].item()

                annotations.append(annotation)

        coco = COCO()
        coco.dataset = {  # type: ignore
            "annotations": annotations,
            "images": [{"id": i} for i in range(len(bboxes_list))],
            "categories": self._get_classes(),  # type: ignore
        }
        coco.createIndex()
        return coco

    def _get_classes(self) -> list[dict]:
        """Get unique classes found in ground truth and detection
        data.
        """
        return [
            {"id": i, "name": str(i)}
            for i in torch.cat(self.pred_classes + self.target_classes)
            .unique()
            .tolist()
        ]

    def _convert_bboxes(self, bboxes: Tensor) -> Tensor:
        bboxes = fix_empty_tensor(bboxes)
        if bboxes.numel() > 0:
            bboxes = box_convert(
                bboxes, in_fmt=self.box_format, out_fmt="xywh"
            )
        return bboxes
