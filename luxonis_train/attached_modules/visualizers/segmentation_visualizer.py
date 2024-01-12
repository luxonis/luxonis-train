import logging

import torch
from torch import Tensor

from luxonis_train.utils.types import Labels, LabelType, Packet, SegmentationProtocol

from .base_visualizer import BaseVisualizer
from .utils import (
    Color,
    draw_segmentation_labels,
    draw_segmentation_masks,
    get_color,
    seg_output_to_bool,
)

logger = logging.getLogger(__name__)
log_disable = False


class SegmentationVisualizer(BaseVisualizer[Tensor, Tensor]):
    def __init__(
        self,
        colors: Color | list[Color] = "#5050FF",
        background_class: int | None = None,
        alpha: float = 0.6,
        **kwargs,
    ):
        """Visualizer for segmentation tasks.

        @type colors: L{Color} | list[L{Color}]
        @param colors: Color of the segmentation masks. Defaults to C{"#5050FF"}.
        @type alpha: float
        @param alpha: Alpha value of the segmentation masks. Defaults to C{0.6}.
        """
        super().__init__(
            protocol=SegmentationProtocol,
            required_labels=[LabelType.SEGMENTATION],
            **kwargs,
        )
        if not isinstance(colors, list):
            colors = [colors]

        self.colors = colors
        self.background_class = background_class
        self.alpha = alpha

    def prepare(self, output: Packet[Tensor], label: Labels) -> tuple[Tensor, Tensor]:
        return output["segmentation"][0], label[LabelType.SEGMENTATION]

    @staticmethod
    def draw_predictions(
        canvas: Tensor,
        predictions: Tensor,
        colors: list[Color] | None = None,
        background_class: int | None = None,
        **kwargs,
    ) -> Tensor:
        colors = SegmentationVisualizer._adjust_colors(
            predictions, colors, background_class
        )
        viz = torch.zeros_like(canvas)
        for i in range(len(canvas)):
            prediction = predictions[i]
            mask = seg_output_to_bool(prediction)
            mask = mask.to(canvas.device)
            viz[i] = draw_segmentation_masks(
                canvas[i].clone(), mask, colors=colors, **kwargs
            )
        return viz

    @staticmethod
    def draw_targets(
        canvas: Tensor,
        targets: Tensor,
        colors: list[Color] | None = None,
        background_class: int | None = None,
        **kwargs,
    ) -> Tensor:
        colors = SegmentationVisualizer._adjust_colors(
            targets, colors, background_class
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
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        predictions: Tensor,
        targets: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Creates a visualization of the segmentation predictions and labels.

        @type label_canvas: Tensor
        @param label_canvas: The canvas to draw the labels on.
        @type prediction_canvas: Tensor
        @param prediction_canvas: The canvas to draw the predictions on.
        @type predictions: Tensor
        @param predictions: The predictions to visualize.
        @type targets: Tensor
        @param targets: The targets to visualize.
        @rtype: tuple[Tensor, Tensor]
        @return: A tuple of the label and prediction visualizations.
        """

        targets_vis = self.draw_targets(
            label_canvas,
            targets,
            colors=self.colors,
            alpha=self.alpha,
            background_class=self.background_class,
            **kwargs,
        )
        predictions_vis = self.draw_predictions(
            prediction_canvas,
            predictions,
            colors=self.colors,
            alpha=self.alpha,
            background_class=self.background_class,
            **kwargs,
        )
        return targets_vis, predictions_vis

    @staticmethod
    def _adjust_colors(
        data: Tensor,
        colors: list[Color] | None = None,
        background_class: int | None = None,
    ) -> list[Color]:
        global log_disable
        n_classes = data.size(1)
        if colors is not None and len(colors) == n_classes:
            return colors

        if not log_disable:
            if colors is None:
                logger.warning("No colors provided. Using random colors instead.")
            elif data.size(1) != len(colors):
                logger.warning(
                    f"Number of colors ({len(colors)}) does not match number of "
                    f"classes ({data.size(1)}). Using random colors instead."
                )
        log_disable = True
        colors = [get_color(i) for i in range(data.size(1))]
        if background_class is not None:
            colors[background_class] = "#000000"
        return colors
