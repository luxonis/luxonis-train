import logging

import torch
from torch import Tensor

from luxonis_train.enums import TaskType
from luxonis_train.utils import Labels, Packet

from .base_visualizer import BaseVisualizer
from .utils import (
    Color,
    draw_bounding_box_labels,
    draw_bounding_boxes,
    draw_segmentation_labels,
    get_color,
)

logger = logging.getLogger(__name__)


class InstanceSegmentationVisualizer(BaseVisualizer[Tensor, Tensor]):
    """Visualizer for instance segmentation tasks, supporting the
    visualization of predicted and ground truth bounding boxes and
    instance segmentation masks."""

    supported_tasks: list[TaskType] = [
        TaskType.INSTANCE_SEGMENTATION,
        TaskType.BOUNDINGBOX,
    ]

    def __init__(
        self,
        labels: dict[int, str] | list[str] | None = None,
        draw_labels: bool = True,
        colors: dict[str, Color] | list[Color] | None = None,
        fill: bool = False,
        width: int | None = None,
        font: str | None = None,
        font_size: int | None = None,
        alpha: float = 0.6,
        **kwargs,
    ):
        """Visualizer for instance segmentation tasks.

        @type labels: dict[int, str] | list[str] | None
        @param labels: Dictionary mapping class indices to class labels.
        @type draw_labels: bool
        @param draw_labels: Whether to draw class labels on the
            visualizations.
        @type colors: dict[str, L{Color}] | list[L{Color}] | None
        @param colors: Dicionary mapping class labels to colors.
        @type fill: bool | None
        @param fill: Whether to fill the boundingbox with color.
        @type width: int | None
        @param width: Width of the bounding box Lines.
        @type font: str | None
        @param font: Font of the clas labels.
        @type font_size: int | None
        @param font_size: Font size of the class Labels.
        @type alpha: float
        @param alpha: Alpha value of the segmentation masks. Defaults to
            C{0.6}.
        """
        super().__init__(**kwargs)

        if isinstance(labels, list):
            labels = {i: label for i, label in enumerate(labels)}

        self.bbox_labels = labels or {
            i: label for i, label in enumerate(self.class_names)
        }

        if colors is None:
            colors = {
                label: get_color(i) for i, label in self.bbox_labels.items()
            }
        if isinstance(colors, list):
            colors = {
                self.bbox_labels[i]: color for i, color in enumerate(colors)
            }

        self.colors = colors
        self.fill = fill
        self.width = width
        self.font = font
        self.font_size = font_size
        self.draw_labels = draw_labels
        self.alpha = alpha

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels | None
    ) -> tuple[Tensor, Tensor, list[Tensor], Tensor | None, Tensor | None]:
        # Override the prepare base method
        target_bboxes = labels["boundingbox"][0]
        target_masks = labels["instance_segmentation"][0]
        predicted_bboxes = inputs["boundingbox"]
        predicted_masks = inputs["instance_segmentation"]

        return target_bboxes, target_masks, predicted_bboxes, predicted_masks

    def draw_predictions(
        self,
        canvas: Tensor,
        pred_bboxes: list[Tensor],
        pred_masks: list[Tensor],
        width: int | None,
        label_dict: dict[int, str],
        color_dict: dict[str, Color],
        draw_labels: bool,
        alpha: float,
    ) -> Tensor:
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            viz[i] = canvas[i].clone()
            image_bboxes = pred_bboxes[i]
            image_masks = pred_masks[i]
            prediction_classes = image_bboxes[..., 5].int()

            cls_labels = (
                [label_dict[int(c)] for c in prediction_classes]
                if draw_labels and label_dict is not None
                else None
            )
            cls_colors = (
                [color_dict[label_dict[int(c)]] for c in prediction_classes]
                if color_dict is not None and label_dict is not None
                else None
            )

            *_, H, W = canvas.shape
            width = width or max(1, int(min(H, W) / 100))

            try:
                viz[i] = draw_segmentation_labels(
                    viz[i],
                    image_masks,
                    colors=cls_colors,
                    alpha=alpha,
                ).to(canvas.device)

                viz[i] = draw_bounding_boxes(
                    viz[i],
                    image_bboxes[:, :4],
                    width=width,
                    labels=cls_labels,
                    colors=cls_colors,
                ).to(canvas.device)
            except ValueError as e:
                logger.warning(
                    f"Failed to draw bounding boxes or masks: {e}. Skipping visualization."
                )
                viz[i] = canvas[i]

        return viz

    @staticmethod
    def draw_targets(
        canvas: Tensor,
        target_bboxes: Tensor,
        target_masks: Tensor,
        width: int | None,
        label_dict: dict[int, str],
        color_dict: dict[str, Color],
        draw_labels: bool,
        alpha: float,
    ) -> Tensor:
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            viz[i] = canvas[i].clone()
            image_bboxes = target_bboxes[target_bboxes[:, 0] == i]
            image_masks = target_masks[target_bboxes[:, 0] == i]
            target_classes = image_bboxes[:, 1].int()

            cls_labels = (
                [label_dict[int(c)] for c in target_classes]
                if draw_labels and label_dict is not None
                else None
            )
            cls_colors = (
                [color_dict[label_dict[int(c)]] for c in target_classes]
                if color_dict is not None and label_dict is not None
                else None
            )

            *_, H, W = canvas.shape
            width = width or max(1, int(min(H, W) / 100))

            viz[i] = draw_segmentation_labels(
                viz[i],
                image_masks,
                alpha=alpha,
                colors=cls_colors,
            ).to(canvas.device)
            viz[i] = draw_bounding_box_labels(
                viz[i],
                image_bboxes[:, 2:],
                width=width,
                labels=cls_labels if cls_labels else None,
                colors=cls_colors,
            ).to(canvas.device)

        return viz

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        target_bboxes: Tensor | None,
        target_masks: Tensor | None,
        predicted_bboxes: Tensor,
        predicted_masks: Tensor,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Creates visualizations of the predicted and target bounding
        boxes and instance masks.

        @type label_canvas: Tensor
        @param label_canvas: Tensor containing the target
            visualizations.
        @type prediction_canvas: Tensor
        @param prediction_canvas: Tensor containing the predicted
            visualizations.
        @type target_bboxes: Tensor | None
        @param target_bboxes: Tensor containing the target bounding
            boxes.
        @type target_masks: Tensor | None
        @param target_masks: Tensor containing the target instance
            masks.
        @type predicted_bboxes: Tensor
        @param predicted_bboxes: Tensor containing the predicted
            bounding boxes.
        @type predicted_masks: Tensor
        @param predicted_masks: Tensor containing the predicted instance
            masks.
        """
        predictions_viz = self.draw_predictions(
            prediction_canvas,
            predicted_bboxes,
            predicted_masks,
            self.width,
            self.bbox_labels,
            self.colors,
            self.draw_labels,
            self.alpha,
        )
        if target_bboxes is None or target_masks is None:
            return predictions_viz

        targets_viz = self.draw_targets(
            label_canvas,
            target_bboxes,
            target_masks,
            self.width,
            self.bbox_labels,
            self.colors,
            self.draw_labels,
            self.alpha,
        )
        return targets_viz, predictions_viz
