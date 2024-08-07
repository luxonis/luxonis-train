from typing import cast

import torch
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.attached_modules.losses.keypoint_loss import KeypointLoss
from luxonis_train.nodes import ImplicitKeypointBBoxHead
from luxonis_train.utils.boxutils import (
    compute_iou_loss,
    match_to_anchor,
    process_bbox_predictions,
)
from luxonis_train.utils.types import IncompatibleException, Labels, LabelType, Packet

from .base_loss import BaseLoss
from .bce_with_logits import BCEWithLogitsLoss
from .smooth_bce_with_logits import SmoothBCEWithLogitsLoss

KeypointTargetType = tuple[
    list[Tensor],
    list[Tensor],
    list[Tensor],
    list[tuple[Tensor, Tensor, Tensor, Tensor]],
    list[Tensor],
]


class ImplicitKeypointBBoxLoss(BaseLoss[list[Tensor], KeypointTargetType]):
    node: ImplicitKeypointBBoxHead
    supported_labels = [(LabelType.BOUNDINGBOX, LabelType.KEYPOINTS)]

    def __init__(
        self,
        cls_pw: float = 1.0,
        viz_pw: float = 1.0,
        obj_pw: float = 1.0,
        label_smoothing: float = 0.0,
        min_objectness_iou: float = 0.0,
        bbox_loss_weight: float = 0.05,
        keypoint_visibility_loss_weight: float = 0.6,
        keypoint_regression_loss_weight: float = 0.5,
        sigmas: list[float] | None = None,
        area_factor: float | None = None,
        class_loss_weight: float = 0.6,
        objectness_loss_weight: float = 0.7,
        anchor_threshold: float = 4.0,
        bias: float = 0.5,
        balance: list[float] | None = None,
        **kwargs,
    ):
        """Joint loss for keypoint and box predictions for cases where the keypoints and
        boxes are inherently linked.

        Based on U{YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object
        Keypoint Similarity Loss<https://arxiv.org/ftp/arxiv/papers/2204/2204.06806.pdf>}.

        @type cls_pw: float
        @param cls_pw: Power for the BCE loss for classes. Defaults to C{1.0}.
        @type viz_pw: float
        @param viz_pw: Power for the BCE loss for keypoints.
        @type obj_pw: float
        @param obj_pw: Power for the BCE loss for objectness. Defaults to C{1.0}.
        @type label_smoothing: float
        @param label_smoothing: Label smoothing factor. Defaults to C{0.0}.
        @type min_objectness_iou: float
        @param min_objectness_iou: Minimum objectness iou. Defaults to C{0.0}.
        @type bbox_loss_weight: float
        @param bbox_loss_weight: Weight for the bounding box loss.
        @type keypoint_visibility_loss_weight: float
        @param keypoint_visibility_loss_weight: Weight for the keypoint visibility loss. Defaults to C{0.6}.
        @type keypoint_regression_loss_weight: float
        @param keypoint_regression_loss_weight: Weight for the keypoint regression loss. Defaults to C{0.5}.
        @type sigmas: list[float] | None
        @param sigmas: Sigmas used in KeypointLoss for OKS metric. If None then use COCO ones if possible or default ones. Defaults to C{None}.
        @type area_factor: float | None
        @param area_factor: Factor by which we multiply bbox area which is used in KeypointLoss. If None then use default one. Defaults to C{None}.
        @type class_loss_weight: float
        @param class_loss_weight: Weight for the class loss. Defaults to C{0.6}.
        @type objectness_loss_weight: float
        @param objectness_loss_weight: Weight for the objectness loss. Defaults to C{0.7}.
        @type anchor_threshold: float
        @param anchor_threshold: Threshold for matching anchors to targets. Defaults to C{4.0}.
        @type bias: float
        @param bias: Bias for matching anchors to targets. Defaults to C{0.5}.
        @type balance: list[float] | None
        @param balance: Balance for the different heads. Defaults to C{None}.
        """

        super().__init__(**kwargs)

        if not isinstance(self.node, ImplicitKeypointBBoxHead):
            raise IncompatibleException(
                f"Loss `{self.name}` is only "
                "compatible with nodes of type `ImplicitKeypointBBoxHead`."
            )
        self.n_classes = self.node.n_classes
        self.n_keypoints = self.node.n_keypoints
        self.n_anchors = self.node.n_anchors
        self.num_heads = self.node.num_heads
        self.box_offset = self.node.box_offset
        self.anchors = self.node.anchors
        self.balance = balance or [4.0, 1.0, 0.4]
        if len(self.balance) < self.num_heads:
            raise ValueError(
                f"Balance list must have at least {self.num_heads} elements."
            )

        self.min_objectness_iou = min_objectness_iou
        self.bbox_weight = bbox_loss_weight
        self.class_weight = class_loss_weight
        self.objectness_weight = objectness_loss_weight
        self.kpt_visibility_weight = keypoint_visibility_loss_weight
        self.keypoint_regression_loss_weight = keypoint_regression_loss_weight
        self.anchor_threshold = anchor_threshold

        self.bias = bias

        self.b_cross_entropy = BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw]))
        self.class_loss = SmoothBCEWithLogitsLoss(
            label_smoothing=label_smoothing,
            bce_pow=cls_pw,
        )
        self.keypoint_loss = KeypointLoss(
            n_keypoints=self.n_keypoints,
            bce_power=viz_pw,
            sigmas=sigmas,
            area_factor=area_factor,
        )

        self.positive_smooth_const = 1 - 0.5 * label_smoothing
        self.negative_smooth_const = 0.5 * label_smoothing

    def prepare(
        self, outputs: Packet[Tensor], labels: Labels
    ) -> tuple[list[Tensor], KeypointTargetType]:
        """Prepares the labels to be in the correct format for loss calculation.

        @type outputs: Packet[Tensor]
        @param outputs: Output from the forward pass.
        @type labels: L{Labels}
        @param labels: Dictionary containing the labels.
        @rtype: tuple[list[Tensor], tuple[list[Tensor], list[Tensor], list[Tensor],
            list[tuple[Tensor, Tensor, Tensor, Tensor]], list[Tensor]]]
        @return: Tuple containing the original output and the postprocessed labels. The
            processed labels are a tuple containing the class targets, box targets,
            keypoint targets, indices and anchors. Indicies are a tuple containing
            vectors of indices for batch, anchor, feature y and feature x dimensions,
            respectively. They are all of shape (n_targets,). The indices are used to
            index the output tensors of shape (batch_size, n_anchors, feature_height,
            feature_width, n_classes + box_offset + n_keypoints * 3) to get a tensor of
            shape (n_targets, n_classes + box_offset + n_keypoints * 3).
        """
        predictions = self.get_input_tensors(outputs, "features")

        kpts = self.get_label(labels, LabelType.KEYPOINTS)[0]
        boxes = self.get_label(labels, LabelType.BOUNDINGBOX)[0]

        nkpts = (kpts.shape[1] - 2) // 3
        targets = torch.zeros((len(boxes), nkpts * 3 + self.box_offset + 1))
        targets[:, :2] = boxes[:, :2]
        targets[:, 2 : self.box_offset + 1] = box_convert(
            boxes[:, 2:], "xywh", "cxcywh"
        )

        targets[:, self.box_offset + 1 :: 3] = kpts[:, 2::3]  # insert kp x coordinates
        targets[:, self.box_offset + 2 :: 3] = kpts[:, 3::3]  # insert kp y coordinates
        targets[:, self.box_offset + 3 :: 3] = kpts[:, 4::3]  # insert kp visibility

        n_targets = len(targets)

        class_targets: list[Tensor] = []
        box_targets: list[Tensor] = []
        keypoint_targets: list[Tensor] = []
        indices: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        anchors: list[Tensor] = []

        anchor_indices = (
            torch.arange(self.n_anchors, device=targets.device, dtype=torch.float32)
            .reshape(self.n_anchors, 1)
            .repeat(1, n_targets)
            .unsqueeze(-1)
        )
        targets = torch.cat((targets.repeat(self.n_anchors, 1, 1), anchor_indices), 2)

        xy_deltas = (
            torch.tensor(
                [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device
            ).float()
            * self.bias
        )

        for i in range(self.num_heads):
            anchor = self.anchors[i]
            feature_height, feature_width = predictions[i].shape[2:4]
            scaled_targets, xy_shifts = match_to_anchor(
                targets,
                anchor,
                xy_deltas,
                feature_width,
                feature_height,
                self.n_keypoints,
                self.anchor_threshold,
                self.bias,
                self.box_offset,
            )

            batch_index, cls = scaled_targets[:, :2].long().T
            box_xy = scaled_targets[:, 2:4]
            box_wh = scaled_targets[:, 4:6]
            box_xy_deltas = (box_xy - xy_shifts).long()
            feature_x_index = box_xy_deltas[:, 0].clamp_(0, feature_width - 1)
            feature_y_index = box_xy_deltas[:, 1].clamp_(0, feature_height - 1)

            anchor_indices = scaled_targets[:, -1].long()
            indices.append(
                (
                    batch_index,
                    anchor_indices,
                    feature_y_index,
                    feature_x_index,
                )
            )
            class_targets.append(cls)
            box_targets.append(torch.cat((box_xy - box_xy_deltas, box_wh), 1))
            anchors.append(anchor[anchor_indices])

            keypoint_targets.append(
                self._create_keypoint_target(scaled_targets, box_xy_deltas)
            )

        return predictions, (
            class_targets,
            box_targets,
            keypoint_targets,
            indices,
            anchors,
        )

    def forward(
        self,
        predictions: list[Tensor],
        targets: KeypointTargetType,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        device = predictions[0].device
        sub_losses = {
            "bboxes": torch.tensor(0.0, device=device),
            "objectness": torch.tensor(0.0, device=device),
            "class": torch.tensor(0.0, device=device),
            "kpt_visibility": torch.tensor(0.0, device=device),
            "kpt_regression": torch.tensor(0.0, device=device),
        }

        for pred, class_target, box_target, kpt_target, index, anchor, balance in zip(
            predictions, *targets, self.balance
        ):
            obj_targets = torch.zeros_like(pred[..., 0], device=device)
            n_targets = len(class_target)

            if n_targets > 0:
                pred_subset = pred[index]

                bbox_cx_cy, bbox_w_h, _ = process_bbox_predictions(
                    pred_subset, anchor.to(device)
                )
                bbox_loss, bbox_iou = compute_iou_loss(
                    torch.cat((bbox_cx_cy, bbox_w_h), dim=1),
                    box_target,
                    iou_type="ciou",
                    bbox_format="cxcywh",
                    reduction="mean",
                )

                sub_losses["bboxes"] += bbox_loss * self.bbox_weight

                area = box_target[:, 2] * box_target[:, 3]

                _, kpt_sublosses = self.keypoint_loss.forward(
                    pred_subset[:, self.box_offset + self.n_classes :],
                    kpt_target.to(device),
                    area.to(device),
                )

                sub_losses["kpt_regression"] += (
                    kpt_sublosses["regression"] * self.keypoint_regression_loss_weight
                )
                sub_losses["kpt_visibility"] += (
                    kpt_sublosses["visibility"] * self.kpt_visibility_weight
                )

                obj_targets[index] = (self.min_objectness_iou) + (
                    1 - self.min_objectness_iou
                ) * bbox_iou.squeeze(-1).to(obj_targets.dtype)

                if self.n_classes > 1:
                    sub_losses["class"] += (
                        self.class_loss.forward(
                            [
                                pred_subset[
                                    :,
                                    self.box_offset : self.box_offset + self.n_classes,
                                ]
                            ],
                            class_target,
                        )
                        * self.class_weight
                    )

            sub_losses["objectness"] += (
                self.b_cross_entropy.forward(pred[..., 4], obj_targets)
                * balance
                * self.objectness_weight
            )

        loss = cast(Tensor, sum(sub_losses.values())).reshape([])
        return loss, {name: loss.detach() for name, loss in sub_losses.items()}

    def _create_keypoint_target(self, scaled_targets: Tensor, box_xy_deltas: Tensor):
        keypoint_target = scaled_targets[:, self.box_offset + 1 : -1]
        for j in range(self.n_keypoints):
            idx = 3 * j
            keypoint_coords = keypoint_target[:, idx : idx + 2]
            visibility = keypoint_target[:, idx + 2]

            keypoint_mask = visibility != 0
            keypoint_coords[keypoint_mask] -= box_xy_deltas[keypoint_mask]

            keypoint_target[:, idx : idx + 2] = keypoint_coords
            keypoint_target[:, idx + 2] = visibility

        return keypoint_target
