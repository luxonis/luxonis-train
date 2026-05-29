from collections.abc import Mapping

import torch
from loguru import logger
from torch import Tensor

from luxonis_train.tasks import Tasks

from .base_visualizer import BaseVisualizer
from .utils import (
    Color,
    draw_bounding_box_labels,
    draw_bounding_boxes,
    draw_segmentation_targets,
    get_color,
    get_prediction_labels,
    potentially_upscale_masks,
)


class InstanceSegmentationVisualizer(BaseVisualizer):
    """Visualizer for instance segmentation tasks, supporting the
    visualization of predicted and ground truth bounding boxes and
    instance segmentation masks.
    """

    supported_tasks = [Tasks.INSTANCE_SEGMENTATION]

    def __init__(
        self,
        labels: dict[int, str] | list[str] | None = None,
        draw_labels: bool = True,
        draw_scores: bool = False,
        colors: dict[str, Color] | list[Color] | None = None,
        fill: bool = False,
        width: int | None = None,
        font: str | None = None,
        font_size: int | None = None,
        alpha: float = 0.6,
        **kwargs,
    ):
        """Visualizer for instance segmentation tasks.

        Args:
            labels (dict[int, str] | list[str] | None): Dictionary mapping class indices to class
                labels.
            draw_labels (bool): Whether to draw class labels on the visualizations.
            draw_scores (bool): Whether to append prediction confidence scores to the rendered
                labels. Defaults to ``False``.
            colors (dict[str, `Color`] | list[`Color`] | None): Dicionary mapping class labels to
                colors.
            fill (bool | None): Whether to fill the boundingbox with color.
            width (int | None): Width of the bounding box Lines.
            font (str | None): Font of the clas labels.
            font_size (int | None): Font size of the class Labels.
            alpha (float): Alpha value of the segmentation masks. Defaults to ``0.6``.
        """
        super().__init__(**kwargs)

        if isinstance(labels, list):
            labels = dict(enumerate(labels))

        self.bbox_labels = labels or self.classes.inverse

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
        self.draw_scores = draw_scores
        self.alpha = alpha

    @classmethod
    def draw_predictions(
        cls,
        canvas: Tensor,
        pred_bboxes: list[Tensor],
        pred_masks: list[Tensor],
        width: int | None,
        label_dict: Mapping[int, str],
        color_dict: dict[str, Color],
        draw_labels: bool,
        draw_scores: bool,
        alpha: float,
        scale: float = 1.0,
    ) -> Tensor:
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            viz[i] = canvas[i].clone()
            image_bboxes = pred_bboxes[i]
            image_masks = pred_masks[i]
            prediction_classes = image_bboxes[..., 5].int()

            if scale is not None and scale != 1:
                image_bboxes = image_bboxes.clone()
                image_bboxes[:, :4] *= scale

            image_masks = potentially_upscale_masks(image_masks, scale)

            cls_labels = get_prediction_labels(
                image_bboxes,
                label_dict,
                draw_labels,
                draw_scores,
            )
            cls_colors = (
                [color_dict[label_dict[int(c)]] for c in prediction_classes]
                if color_dict is not None and label_dict is not None
                else None
            )

            *_, H, W = canvas.shape
            width = width or max(1, int(min(H, W) / 100))

            try:
                viz[i] = draw_segmentation_targets(
                    viz[i], image_masks, colors=cls_colors, alpha=alpha
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
        label_dict: Mapping[int, str],
        color_dict: dict[str, Color],
        draw_labels: bool,
        alpha: float,
        scale: float = 1.0,
    ) -> Tensor:
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            viz[i] = canvas[i].clone()
            image_bboxes = target_bboxes[target_bboxes[:, 0] == i]
            image_masks = target_masks[target_bboxes[:, 0] == i]
            target_classes = image_bboxes[:, 1].int()

            image_masks = potentially_upscale_masks(image_masks, scale)

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

            viz[i] = draw_segmentation_targets(
                viz[i], image_masks, alpha=alpha, colors=cls_colors
            ).to(canvas.device)
            viz[i] = draw_bounding_box_labels(
                viz[i],
                image_bboxes[:, 2:],
                width=width,
                labels=cls_labels or None,
                colors=cls_colors,
            ).to(canvas.device)

        return viz

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        boundingbox: list[Tensor],
        instance_segmentation: list[Tensor],
        target_boundingbox: Tensor | None,
        target_instance_segmentation: Tensor | None,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Create visualizations of the predicted and target bounding
        boxes and instance masks.

        Args:
            target_canvas (Tensor): Tensor containing the target visualizations.
            prediction_canvas (Tensor): Tensor containing the predicted visualizations.
            target_bboxes (Tensor | None): Tensor containing the target bounding boxes.
            target_masks (Tensor | None): Tensor containing the target instance masks.
            predicted_bboxes (list[Tensor]): List of tensors containing the predicted bounding
                boxes.
            predicted_masks (list[Tensor]): List of tensors containing the predicted instance masks.
        """
        predictions_viz = self.draw_predictions(
            prediction_canvas,
            boundingbox,
            instance_segmentation,
            self.width,
            self.bbox_labels,
            self.colors,
            self.draw_labels,
            self.draw_scores,
            self.alpha,
            self.scale,
        )
        if target_boundingbox is None or target_instance_segmentation is None:
            return predictions_viz
        targets_viz = self.draw_targets(
            target_canvas,
            target_boundingbox,
            target_instance_segmentation,
            self.width,
            self.bbox_labels,
            self.colors,
            self.draw_labels,
            self.alpha,
            self.scale,
        )
        return targets_viz, predictions_viz
