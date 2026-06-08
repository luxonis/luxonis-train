from functools import cached_property

import torch
from loguru import logger
from luxonis_ml.data.utils.visualizations import ColorMap
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Metadata, Tasks
from luxonis_train.utils import seg_output_to_bool

from .base_visualizer import BaseVisualizer
from .utils import Color, draw_segmentation_targets, potentially_upscale_masks

log_disable = False


class SegmentationVisualizer(BaseVisualizer):
    """Visualize semantic segmentation and anomaly masks.

    Metadata:
        - Module type: visualizer
        - Registry name: ``SegmentationVisualizer``
        - Task: segmentation, anomaly_detection
        - Attached node types: None
        - Inputs: prediction and target canvases, ``segmentation``
          predictions, and optional ``segmentation`` targets.
        - Outputs: prediction visualization, or ``(targets_vis,
          predictions_vis)`` when targets are provided.
        - State: Tracks whether the color-mismatch warning has already been
          emitted.

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Converts segmentation outputs to boolean
          masks, optionally upscales masks, and overlays class colors.

    Prediction format:
        - ``predictions`` is a segmentation tensor converted by
          ``seg_output_to_bool``.

    Target format:
        - ``target`` is an optional segmentation mask tensor.

    """

    supported_tasks = [Tasks.SEGMENTATION, Tasks.ANOMALY_DETECTION]

    def __init__(
        self,
        colors: Color | list[Color] | None = None,
        background_class: int | None = 0,
        background_color: Color = "#000000",
        alpha: float = 0.6,
        **kwargs,
    ):
        """Visualizer for segmentation tasks.

        Args:
            colors (Color | list[Color] | None): Color of the segmentation masks. Defaults to
                ``"#5050FF"``.
            background_class (int | None): Index of the background class. Defaults to ``0``. If
                set, the background class will be drawn with ``background_color``.
            background_color (Color): Color of the background class. Defaults to
                ``"#000000"``.
            alpha (float): Alpha value of the segmentation masks. Defaults to ``0.6``.
            **kwargs (``Any``): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)
        if colors is not None and not isinstance(colors, list):
            colors = [colors]

        self.colors = colors
        self.background_class = background_class
        self.background_color = background_color
        self.alpha = alpha
        self.colormap = ColorMap()

        self._warn_colors = True

    @staticmethod
    def draw_predictions(
        canvas: Tensor,
        predictions: Tensor,
        alpha: float,
        colors: list[Color],
        scale: float = 1.0,
    ) -> Tensor:
        viz = torch.zeros_like(canvas)
        for i in range(len(canvas)):
            prediction = predictions[i]
            mask = seg_output_to_bool(prediction)
            mask = potentially_upscale_masks(mask, scale)
            viz[i] = draw_segmentation_targets(
                canvas[i].clone(), mask, alpha=alpha, colors=colors
            ).to(canvas.device)
        return viz

    @staticmethod
    def draw_targets(
        canvas: Tensor,
        targets: Tensor,
        alpha: float,
        colors: list[Color],
        scale: float = 1.0,
    ) -> Tensor:
        viz = torch.zeros_like(canvas)
        for i in range(len(viz)):
            target = targets[i].bool()
            target = potentially_upscale_masks(target, scale)
            viz[i] = draw_segmentation_targets(
                canvas[i].clone(), target, alpha=alpha, colors=colors
            ).to(canvas.device)

        return viz

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        predictions: Tensor,
        target: Tensor | None,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Create visualizations of segmentation predictions and labels.

        Args:
            prediction_canvas (``Tensor``): The canvas to draw the predictions on.
            target_canvas (``Tensor``): The canvas to draw the labels on.
            predictions (``Tensor``): The predictions to visualize.
            target (``Tensor | None``): The targets to visualize, or ``None`` when
                targets are unavailable.

        Returns:
            ``Tensor | tuple[Tensor, Tensor]``: Prediction visualization only when
                ``target`` is ``None``; otherwise a tuple of target and prediction
                visualizations.

        """
        colors = self._adjust_colors(
            self.colors, self.background_class, self.background_color
        )

        predictions_vis = self.draw_predictions(
            prediction_canvas,
            predictions,
            alpha=self.alpha,
            colors=colors,
            scale=self.scale,
        )
        if target is None:
            return predictions_vis

        targets_vis = self.draw_targets(
            target_canvas,
            target,
            alpha=self.alpha,
            colors=colors,
            scale=self.scale,
        )
        return targets_vis, predictions_vis

    def _adjust_colors(
        self,
        colors: list[Color] | None = None,
        background_class: int | None = None,
        background_color: Color = "#000000",
    ) -> list[Color]:
        if colors and len(colors) == self.n_classes:
            return colors

        if self._warn_colors:
            if colors is None:
                logger.warning(
                    "No colors provided. Using random colors instead."
                )
            elif len(colors) != self.n_classes:
                logger.warning(
                    f"Number of colors ({len(colors)}) does not match number of "
                    f"classes ({self.n_classes}). Using random colors instead."
                )
            self._warn_colors = False
        colors = [self.colormap[i] for i in range(self.n_classes)]
        if background_class is not None and self.n_classes > 1:
            colors[background_class] = background_color
        return colors

    @cached_property
    @override
    def required_labels(self) -> set[str | Metadata]:
        if self.task == Tasks.ANOMALY_DETECTION:
            return Tasks.SEGMENTATION.required_labels
        return self.task.required_labels
