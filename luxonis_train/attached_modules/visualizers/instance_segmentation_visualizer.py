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
    instance masks."""

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
        """Initialize the visualizer with customization options for
        appearance.

        Parameters:
        - labels: A dictionary or list mapping class indices to labels. Defaults to None.
        - draw_labels: Whether to draw labels on bounding boxes. Defaults to True.
        - colors: Colors for each class. Can be a dictionary or list. Defaults to None.
        - fill: Whether to fill bounding boxes. Defaults to False.
        - width: Line width for bounding boxes. Defaults to None (adaptive).
        - font: Font to use for labels. Defaults to None.
        - font_size: Font size for labels. Defaults to None.
        - alpha: Transparency for instance masks. Defaults to 0.6.
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
        """
        TODO: Docstring
        """
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
        """Draw predicted bounding boxes and masks on the canvas."""
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            viz[i] = canvas[i].clone()
            prediction = pred_bboxes[i]
            masks = pred_masks[i]
            prediction_classes = prediction[..., 5].int()

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
                for j, mask in enumerate(masks):
                    print(f"mask.sum(): {mask.sum()}")
                    viz[i] = draw_segmentation_labels(
                        viz[i],
                        mask.unsqueeze(0),
                        colors=[cls_colors[j]],
                        alpha=alpha,
                    ).to(canvas.device)

                viz[i] = draw_bounding_boxes(
                    viz[i],
                    prediction[:, :4],
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
        """Draw ground truth bounding boxes and masks on the canvas."""
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            viz[i] = canvas[i].clone()
            image_targets = target_bboxes[target_bboxes[:, 0] == i]
            image_masks = target_masks[target_bboxes[:, 0] == i]
            target_classes = image_targets[:, 1].int()

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

            for j, (bbox, mask) in enumerate(
                zip(image_targets[:, 2:], image_masks)
            ):
                print(f"sum(mask): {mask.sum()}")
                viz[i] = draw_segmentation_labels(
                    viz[i],
                    mask.unsqueeze(0),
                    alpha=alpha,
                    colors=[cls_colors[j]],
                ).to(canvas.device)
                viz[i] = draw_bounding_box_labels(
                    viz[i],
                    bbox.unsqueeze(0),
                    width=width,
                    labels=[cls_labels[j]] if cls_labels else None,
                    colors=[cls_colors[j]],
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
        """Visualize predictions and ground truth."""
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
