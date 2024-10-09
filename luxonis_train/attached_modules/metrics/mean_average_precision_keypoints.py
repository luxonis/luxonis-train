import contextlib
import io
from typing import Any, Literal

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.enums import TaskType
from luxonis_train.utils import (
    Labels,
    Packet,
    get_sigmas,
    get_with_default,
)

from .base_metric import BaseMetric


class MeanAveragePrecisionKeypoints(
    BaseMetric[list[dict[str, Tensor]], list[dict[str, Tensor]]]
):
    """Mean Average Precision metric for keypoints.

    Uses C{OKS} as IoU measure.
    """

    supported_tasks: list[tuple[TaskType, ...]] = [
        (TaskType.BOUNDINGBOX, TaskType.KEYPOINTS)
    ]

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    pred_boxes: list[Tensor]
    pred_scores: list[Tensor]
    pred_labels: list[Tensor]
    pred_keypoints: list[Tensor]

    groundtruth_boxes: list[Tensor]
    groundtruth_labels: list[Tensor]
    groundtruth_area: list[Tensor]
    groundtruth_crowds: list[Tensor]
    groundtruth_keypoints: list[Tensor]

    def __init__(
        self,
        sigmas: list[float] | None = None,
        area_factor: float | None = None,
        max_dets: int = 20,
        box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy",
        **kwargs,
    ):
        """Implementation of the mean average precision metric for
        keypoint detections.

        Adapted from: U{https://github.com/Lightning-AI/torchmetrics/blob/v1.0.1/src/
        torchmetrics/detection/mean_ap.py}.

        @license: Apache License, Version 2.0

        @type sigmas: list[float] | None
        @param sigmas: Sigma for each keypoint to weigh its importance, if C{None}, then
            use COCO if possible otherwise defaults. Defaults to C{None}.
        @type area_factor: float | None
        @param area_factor: Factor by which we multiply the bounding box area.
            If not set, the default factor of C{0.53} is used.
        @type max_dets: int,
        @param max_dets: Maximum number of detections to be considered per image. Defaults to C{20}.
        @type box_format: Literal["xyxy", "xywh", "cxcywh"]
        @param box_format: Input bounding box format. Defaults to C{"xyxy"}.
        """
        super().__init__(**kwargs)

        self.sigmas = get_sigmas(
            sigmas, self.n_keypoints, caller_name=self.name
        )
        self.area_factor = get_with_default(
            area_factor, "bbox area scaling", self.name, default=0.53
        )
        self.max_dets = max_dets

        allowed_box_formats = ("xyxy", "xywh", "cxcywh")
        if box_format not in allowed_box_formats:
            raise ValueError(
                f"Expected argument `box_format` to be one of {allowed_box_formats} but got {box_format}"
            )
        self.box_format = box_format

        self.add_state("pred_boxes", default=[], dist_reduce_fx=None)
        self.add_state("pred_scores", default=[], dist_reduce_fx=None)
        self.add_state("pred_labels", default=[], dist_reduce_fx=None)
        self.add_state("pred_keypoints", default=[], dist_reduce_fx=None)

        self.add_state("groundtruth_boxes", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_labels", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_area", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_crowds", default=[], dist_reduce_fx=None)
        self.add_state(
            "groundtruth_keypoints", default=[], dist_reduce_fx=None
        )

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        assert self.node.tasks is not None
        kpts = self.get_label(labels, TaskType.KEYPOINTS)
        boxes = self.get_label(labels, TaskType.BOUNDINGBOX)

        nkpts = (kpts.shape[1] - 2) // 3
        label = torch.zeros((len(boxes), nkpts * 3 + 6))
        label[:, :2] = boxes[:, :2]
        label[:, 2:6] = box_convert(boxes[:, 2:], "xywh", "xyxy")
        label[:, 6::3] = kpts[:, 2::3]  # x
        label[:, 7::3] = kpts[:, 3::3]  # y
        label[:, 8::3] = kpts[:, 4::3]  # visiblity

        output_list_kpt_map: list[dict[str, Tensor]] = []
        label_list_kpt_map: list[dict[str, Tensor]] = []
        image_size = self.original_in_shape[1:]

        output_kpts = self.get_input_tensors(inputs, TaskType.KEYPOINTS)
        output_bboxes = self.get_input_tensors(inputs, TaskType.BOUNDINGBOX)
        for i in range(len(output_kpts)):
            output_list_kpt_map.append(
                {
                    "boxes": output_bboxes[i][:, :4],
                    "scores": output_bboxes[i][:, 4],
                    "labels": output_bboxes[i][:, 5].int(),
                    "keypoints": output_kpts[i].reshape(
                        -1, self.n_keypoints * 3
                    ),
                }
            )

            curr_label = label[label[:, 0] == i].to(output_kpts[i].device)
            curr_bboxs = curr_label[:, 2:6]
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]
            curr_kpts = curr_label[:, 6:]
            curr_kpts[:, 0::3] *= image_size[1]
            curr_kpts[:, 1::3] *= image_size[0]
            label_list_kpt_map.append(
                {
                    "boxes": curr_bboxs,
                    "labels": curr_label[:, 1].int(),
                    "keypoints": curr_kpts,
                }
            )

        return output_list_kpt_map, label_list_kpt_map

    def update(
        self, preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]]
    ) -> None:
        """Updates the metric state.

        @type preds: list[dict[str, Tensor]]
        @param preds: A list consisting of dictionaries each containing key-values for a single image.
            Parameters that should be provided per dict:

                - boxes (FloatTensor): Tensor of shape C{(N, 4)}
                  containing `N` detection boxes of the format specified in
                  the constructor. By default, this method expects `(xmin, ymin,
                  xmax, ymax)` in absolute image coordinates.
                - scores (FloatTensor): Tensor of shape C{(N)}
                  containing detection scores for the boxes.
                - labels (tIntTensor): Tensor of shape C{(N)} containing
                  0-indexed detection classes for the boxes.
                - keypoints (FloatTensor): Tensor of shape C{(N, 3*K)} and in
                  format C{[x, y, vis, x, y, vis, ...]} where C{x} an C{y} are absolute
                  keypoint coordinates and C{vis} is keypoint visibility.

        @type target: list[dict[str, Tensor]]
        @param target: A list consisting of dictionaries each containing key-values for a single image.
            Parameters that should be provided per dict:

                - boxes (FloatTensor): Tensor of shape C{(N, 4)} containing
                  `N` ground truth boxes of the format specified in the
                  constructor. By default, this method expects `(xmin, ymin, xmax, ymax)`
                  in absolute image coordinates.
                - labels: :class:`~torch.IntTensor` of shape C{(N)} containing
                  0-indexed ground truth classes for the boxes.
                - iscrow (IntTensor): Tensor of shape C{(N)} containing 0/1
                  values indicating whether the bounding box/masks indicate a crowd of
                  objects. If not provided it will automatically be set to 0.
                - area (FloatTensor): Tensor of shape C{(N)} containing the
                  area of the object. If not provided will be automatically calculated
                  based on the bounding box/masks provided. Only affects which samples
                  contribute to the C{map_small}, C{map_medium}, C{map_large} values.
                - keypoints (FloatTensor): Tensor of shape C{(N, 3*K)} in format
                  C{[x, y, vis, x, y, vis, ...]} where C{x} an C{y} are absolute keypoint
                  coordinates and `vis` is keypoint visibility.
        """
        for item in preds:
            boxes, keypoints = self._get_safe_item_values(item)
            self.pred_boxes.append(boxes)
            self.pred_keypoints.append(keypoints)
            self.pred_scores.append(item["scores"])
            self.pred_labels.append(item["labels"])

        for item in target:
            boxes, keypoints = self._get_safe_item_values(item)
            self.groundtruth_boxes.append(boxes)
            self.groundtruth_keypoints.append(keypoints)
            self.groundtruth_labels.append(item["labels"])
            self.groundtruth_area.append(
                item.get("area", torch.zeros_like(item["labels"]))
            )
            self.groundtruth_crowds.append(
                item.get("iscrowd", torch.zeros_like(item["labels"]))
            )

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Torchmetric compute function."""
        coco_target, coco_preds = COCO(), COCO()
        coco_target.dataset = self._get_coco_format(
            self.groundtruth_boxes,
            self.groundtruth_keypoints,
            self.groundtruth_labels,
            crowds=self.groundtruth_crowds,
            area=self.groundtruth_area,
        )  # type: ignore
        coco_preds.dataset = self._get_coco_format(
            self.pred_boxes,
            self.pred_keypoints,
            self.pred_labels,
            scores=self.pred_scores,
        )  # type: ignore

        with contextlib.redirect_stdout(io.StringIO()):
            coco_target.createIndex()
            coco_preds.createIndex()

            self.coco_eval = COCOeval(
                coco_target, coco_preds, iouType="keypoints"
            )
            self.coco_eval.params.kpt_oks_sigmas = self.sigmas.cpu().numpy()
            self.coco_eval.params.maxDets = [self.max_dets]

            self.coco_eval.evaluate()
            self.coco_eval.accumulate()
            self.coco_eval.summarize()
            stats = self.coco_eval.stats

        kpt_map = torch.tensor([stats[0]], dtype=torch.float32)
        return kpt_map, {
            "kpt_map_50": torch.tensor([stats[1]], dtype=torch.float32),
            "kpt_map_75": torch.tensor([stats[2]], dtype=torch.float32),
            "kpt_map_medium": torch.tensor([stats[3]], dtype=torch.float32),
            "kpt_map_large": torch.tensor([stats[4]], dtype=torch.float32),
            "kpt_mar": torch.tensor([stats[5]], dtype=torch.float32),
            "kpt_mar_50": torch.tensor([stats[6]], dtype=torch.float32),
            "kpt_mar_75": torch.tensor([stats[7]], dtype=torch.float32),
            "kpt_mar_medium": torch.tensor([stats[8]], dtype=torch.float32),
            "kpt_mar_large": torch.tensor([stats[9]], dtype=torch.float32),
        }

    def _get_coco_format(
        self,
        boxes: list[Tensor],
        keypoints: list[Tensor],
        labels: list[Tensor],
        scores: list[Tensor] | None = None,
        crowds: list[Tensor] | None = None,
        area: list[Tensor] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Transforms and returns all cached targets or predictions in
        COCO format.

        Format is defined at U{
        https://cocodataset.org/#format-data}.
        """
        images: list[dict[str, int]] = []
        annotations: list[dict[str, Any]] = []
        annotation_id = (
            1  # has to start with 1, otherwise COCOEval results are wrong
        )

        for image_id, (image_boxes, image_kpts, image_labels) in enumerate(
            zip(boxes, keypoints, labels)
        ):
            image_boxes_list: list[list[float]] = image_boxes.cpu().tolist()
            image_kpts_list: list[list[float]] = image_kpts.cpu().tolist()
            image_labels_list: list[int] = image_labels.cpu().tolist()

            images.append({"id": image_id})

            for k, (image_box, image_kpt, image_label) in enumerate(
                zip(image_boxes_list, image_kpts_list, image_labels_list)
            ):
                if len(image_box) != 4:
                    raise ValueError(
                        f"Invalid input box of sample {image_id}, element {k} "
                        f"(expected 4 values, got {len(image_box)})"
                    )

                if len(image_kpt) != 3 * self.n_keypoints:
                    raise ValueError(
                        f"Invalid input keypoints of sample {image_id}, element {k} "
                        f"(expected {3 * self.n_keypoints} values, got {len(image_kpt)})"
                    )

                if not isinstance(image_label, int):
                    raise ValueError(
                        f"Invalid input class of sample {image_id}, element {k} "
                        f"(expected value of type integer, got type {type(image_label)})"
                    )

                if area is not None and area[image_id][k].cpu().item() > 0:
                    area_stat = area[image_id][k].cpu().tolist()
                else:
                    area_stat = image_box[2] * image_box[3] * self.area_factor

                n_keypoints = len(
                    [
                        i
                        for i in range(2, len(image_kpt), 3)
                        if image_kpt[i] != 0
                    ]
                )  # number of annotated keypoints
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "bbox": image_box,
                    "area": area_stat,
                    "category_id": image_label,
                    "iscrowd": (
                        crowds[image_id][k].cpu().tolist()
                        if crowds is not None
                        else 0
                    ),
                    "keypoints": image_kpt,
                    "num_keypoints": n_keypoints,
                }

                if scores is not None:
                    score = scores[image_id][k].cpu().tolist()
                    # `tolist` returns a number for scalar tensors,
                    # the name is misleading
                    if not isinstance(score, float):
                        raise ValueError(
                            f"Invalid input score of sample {image_id}, element {k}"
                            f" (expected value of type float, got type {type(score)})"
                        )
                    annotation["score"] = score
                annotations.append(annotation)
                annotation_id += 1

        classes = [{"id": i, "name": str(i)} for i in self._get_classes()]
        return {
            "images": images,
            "annotations": annotations,
            "categories": classes,
        }

    def _get_safe_item_values(
        self, item: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Convert and return the boxes."""
        boxes = self._fix_empty_tensors(item["boxes"])
        if boxes.numel() > 0:
            boxes = box_convert(boxes, in_fmt=self.box_format, out_fmt="xywh")
        keypoints = self._fix_empty_tensors(item["keypoints"])
        return boxes, keypoints

    def _get_classes(self) -> list[int]:
        """Return a list of unique classes found in ground truth and
        detection data."""
        if len(self.pred_labels) > 0 or len(self.groundtruth_labels) > 0:
            return (
                torch.cat(self.pred_labels + self.groundtruth_labels)
                .unique()
                .cpu()
                .tolist()
            )
        return []

    @staticmethod
    def _fix_empty_tensors(input_tensor: Tensor) -> Tensor:
        """Empty tensors can cause problems in DDP mode, this methods
        corrects them."""
        if input_tensor.numel() == 0 and input_tensor.ndim == 1:
            return input_tensor.unsqueeze(0)
        return input_tensor
