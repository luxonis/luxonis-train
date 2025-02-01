from functools import cached_property

import torch
from loguru import logger
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Metadata, Tasks

from .base_visualizer import BaseVisualizer
from .utils import (
    Color,
    draw_segmentation_labels,
    get_color,
    seg_output_to_bool,
)

log_disable = False


class SegmentationVisualizer(BaseVisualizer):
    supported_tasks = [Tasks.SEGMENTATION, Tasks.ANOMALY_DETECTION]

    def __init__(
        self,
        colors: Color | list[Color] = "#5050FF",
        background_class: int | None = 0,
        background_color: Color = "#000000",
        alpha: float = 0.6,
        **kwargs,
    ):
        """Visualizer for segmentation tasks.

        @type colors: L{Color} | list[L{Color}]
        @param colors: Color of the segmentation masks. Defaults to C{"#5050FF"}.
        @type background_class: int | None
        @param background_class: Index of the background class. Defaults to C{0}.
          If set, the background class will be drawn with the `background_color`.
        @type background_color: L{Color} | None
        @param background_color: Color of the background class.
            Defaults to C{"#000000"}.
        @type alpha: float
        @param alpha: Alpha value of the segmentation masks. Defaults to C{0.6}.
        """
        super().__init__(**kwargs)
        if not isinstance(colors, list):
            colors = [colors]

        self.colors = colors
        self.background_class = background_class
        self.background_color = background_color
        self.alpha = alpha

    @staticmethod
    def draw_predictions(
        canvas: Tensor,
        predictions: Tensor,
        colors: list[Color] | None = None,
        background_class: int | None = None,
        background_color: Color = "#000000",
        **kwargs,
    ) -> Tensor:
        colors = SegmentationVisualizer._adjust_colors(
            predictions, colors, background_class, background_color
        )
        viz = torch.zeros_like(canvas)
        for i in range(len(canvas)):
            prediction = predictions[i]
            mask = seg_output_to_bool(prediction)
            viz[i] = draw_segmentation_labels(
                canvas[i].clone(), mask, colors=colors, **kwargs
            ).to(canvas.device)
        return viz

    @staticmethod
    def draw_targets(
        canvas: Tensor,
        targets: Tensor,
        colors: list[Color] | None = None,
        background_class: int | None = None,
        background_color: Color = "#000000",
        **kwargs,
    ) -> Tensor:
        colors = SegmentationVisualizer._adjust_colors(
            targets, colors, background_class, background_color
        )
        viz = torch.zeros_like(canvas)
        for i in range(len(viz)):
            target = targets[i]
            viz[i] = draw_segmentation_labels(
                canvas[i].clone(),
                target,
                colors=colors,
                **kwargs,
            ).to(canvas.device)

        return viz

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        predictions: Tensor,
        target: Tensor | None,
        **kwargs,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Creates a visualization of the segmentation predictions and
        labels.

        @type target_canvas: Tensor
        @param target_canvas: The canvas to draw the labels on.
        @type prediction_canvas: Tensor
        @param prediction_canvas: The canvas to draw the predictions on.
        @type predictions: Tensor
        @param predictions: The predictions to visualize.
        @type targets: Tensor
        @param targets: The targets to visualize.
        @rtype: tuple[Tensor, Tensor]
        @return: A tuple of the label and prediction visualizations.
        """

        predictions_vis = self.draw_predictions(
            prediction_canvas,
            predictions,
            colors=self.colors,
            alpha=self.alpha,
            background_class=self.background_class,
            background_color=self.background_color,
            **kwargs,
        )
        if target is None:
            return predictions_vis

        targets_vis = self.draw_targets(
            target_canvas,
            target,
            colors=self.colors,
            alpha=self.alpha,
            background_class=self.background_class,
            background_color=self.background_color,
            **kwargs,
        )
        return targets_vis, predictions_vis

    @staticmethod
    def _adjust_colors(
        data: Tensor,
        colors: list[Color] | None = None,
        background_class: int | None = None,
        background_color: Color = "#000000",
    ) -> list[Color]:
        global log_disable
        n_classes = data.size(1)
        if colors is not None and len(colors) == n_classes:
            return colors

        if not log_disable:
            if colors is None:
                logger.warning(
                    "No colors provided. Using random colors instead."
                )
            elif data.size(1) != len(colors):
                logger.warning(
                    f"Number of colors ({len(colors)}) does not match number of "
                    f"classes ({data.size(1)}). Using random colors instead."
                )
        log_disable = True
        colors = [get_color(i) for i in range(data.size(1))]
        if background_class is not None:
            colors[background_class] = background_color
        return colors

    @cached_property
    @override
    def required_labels(self) -> set[str | Metadata]:
        if self.task == Tasks.ANOMALY_DETECTION:
            return Tasks.SEGMENTATION.required_labels
        return self.task.required_labels
