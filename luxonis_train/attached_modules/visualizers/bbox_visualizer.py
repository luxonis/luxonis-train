import logging

import torch
from torch import Tensor
from torchvision.utils import draw_bounding_boxes

from luxonis_train.tasks import Tasks

from .base_visualizer import BaseVisualizer
from .utils import Color, draw_bounding_box_labels, get_color


class BBoxVisualizer(BaseVisualizer):
    supported_tasks = [Tasks.BOUNDINGBOX]

    def __init__(
        self,
        labels: dict[int, str] | list[str] | None = None,
        draw_labels: bool = True,
        colors: dict[str, Color] | list[Color] | None = None,
        fill: bool = False,
        width: int | None = None,
        font: str | None = None,
        font_size: int | None = None,
        **kwargs,
    ):
        """Visualizer for bounding box predictions.

        Creates a visualization of the bounding box predictions and
        labels.

        @type labels: dict[int, str] | list[str] | None
        @param labels: Either a dictionary mapping class indices to
            names, or a list of names. If list is provided, the label
            mapping is done by index. By default, no labels are drawn.
        @type draw_labels: bool
        @param draw_labels: Whether or not to draw labels. Defaults to
            C{True}.
        @type colors: dict[int, Color] | list[Color] | None
        @param colors: Either a dictionary mapping class indices to
            colors, or a list of colors. If list is provided, the color
            mapping is done by index. By default, random colors are
            used.
        @type fill: bool
        @param fill: Whether or not to fill the bounding boxes. Defaults
            to C{False}.
        @type width: int | None
        @param width: The width of the bounding box lines. Defaults to
            C{1}.
        @type font: str | None
        @param font: A filename containing a TrueType font. Defaults to
            C{None}.
        @type font_size: int | None
        @param font_size: The font size to use for the labels. Defaults
            to C{None}.
        """
        super().__init__(**kwargs)
        if isinstance(labels, list):
            labels = dict(enumerate(labels))

        self.label_dict = labels or self.classes.inverse

        if colors is None:
            colors = {
                label: get_color(i) for i, label in self.label_dict.items()
            }
        if isinstance(colors, list):
            colors = {
                self.label_dict[i]: color for i, color in enumerate(colors)
            }
        self.colors = colors
        self.fill = fill
        self.width = width
        self.font = font
        self.font_size = font_size
        self.draw_labels = draw_labels

    def draw_targets(self, canvas: Tensor, targets: Tensor) -> Tensor:
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            target = targets[targets[:, 0] == i]
            target_classes = target[:, 1].int()
            cls_labels = (
                [self.label_dict[int(c)] for c in target_classes]
                if self.draw_labels and self.label_dict is not None
                else None
            )
            cls_colors = (
                [self.colors[self.label_dict[int(c)]] for c in target_classes]
                if self.colors is not None and self.label_dict is not None
                else None
            )

            *_, H, W = canvas.shape
            width = self.width or max(1, int(min(H, W) / 100))
            viz[i] = draw_bounding_box_labels(
                canvas[i].clone(),
                target[:, 2:],
                width=width,
                labels=cls_labels,
                colors=cls_colors,
            ).to(canvas.device)

        return viz

    def draw_predictions(
        self, canvas: Tensor, predictions: list[Tensor], scale: float = 1.0
    ) -> Tensor:
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            prediction = predictions[i]

            boxes = prediction[:, :4].clone()
            if scale != 1.0:
                boxes *= scale

            prediction_classes = prediction[..., 5].int()
            cls_labels = (
                [self.label_dict[int(c)] for c in prediction_classes]
                if self.draw_labels and self.label_dict is not None
                else None
            )
            cls_colors = (
                [
                    self.colors[self.label_dict[int(c)]]
                    for c in prediction_classes
                ]
                if self.colors is not None and self.label_dict is not None
                else None
            )

            *_, H, W = canvas.shape
            width = self.width or max(1, int(min(H, W) / 100))
            try:
                viz[i] = draw_bounding_boxes(
                    canvas[i].clone(),
                    boxes,
                    width=width,
                    labels=cls_labels,
                    colors=cls_colors,
                )
            except ValueError as e:
                logging.getLogger(__name__).warning(
                    f"Failed to draw bounding boxes: {e}. Skipping visualization."
                )
                viz = canvas
        return viz

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        predictions: list[Tensor],
        targets: Tensor | None,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Creates a visualization of the bounding box predictions and
        labels.

        @type target_canvas: Tensor
        @param target_canvas: The canvas containing the labels.
        @type prediction_canvas: Tensor
        @param prediction_canvas: The canvas containing the predictions.
        @type prediction: Tensor
        @param prediction: The predicted bounding boxes. The shape
            should be [N, 6], where N is the number of bounding boxes
            and the last dimension is [x1, y1, x2, y2, class, conf].
        @type targets: Tensor
        @param targets: The target bounding boxes.
        """
        predictions_viz = self.draw_predictions(
            prediction_canvas, predictions, scale=self.scale
        )
        if targets is None:
            return predictions_viz

        targets_viz = self.draw_targets(target_canvas, targets)
        return targets_viz, predictions_viz
