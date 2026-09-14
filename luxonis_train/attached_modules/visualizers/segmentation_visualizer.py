"""The visualizer that blends segmentation masks into the images."""

from functools import cached_property

import torch
from loguru import logger
from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Metadata, Tasks
from luxonis_train.utils import seg_output_to_bool

from .base_visualizer import BaseVisualizer
from .utils import Color, draw_segmentation_targets, potentially_upscale_masks

log_disable = False


class SegmentationVisualizer(BaseVisualizer):
    r"""Visualizer for semantic segmentation and anomaly masks.

    .. figure::
       https://raw.githubusercontent.com/luxonis/luxonis-train/e542cf0efa20a0fc5c781ff505d699031cb0d228/media/example_viz/seg.png
       :width: 700px
       :height: 262px
       :loading: embed

       The left image shows the targets. The right image shows the
       predictions.

    Inputs:
        - ``prediction_canvas``, ``target_canvas`` (``Tensor``):
          :math:`\left[B, 3, H, W\right]`
        - ``predictions`` (``Tensor``): :math:`\left[B, n_{classes}, H,
          W\right]` logits
        - ``target`` (``Tensor | None``): :math:`\left[B, n_{classes},
          H, W\right]` one-hot

    Outputs:
        - ``Tensor | tuple[Tensor, Tensor]``: :math:`\left[B, 3, H,
          W\right]`, a pair when targets are given

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        Each predicted pixel gets the class with the highest logit. With
        one class, a pixel gets the class when the sigmoid of its logit
        is at least ``0.5``. The visualizer blends one color for each
        class into the images. On an ``anomaly_detection`` node, it
        reads the ``segmentation`` label.

    Example:
        Attached to a ``DDRNetSegmentationHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: DDRNetSegmentationHead
              inputs: [DDRNet]
              visualizers:
                - name: SegmentationVisualizer

    Compatible with:
        - Used by:

          - `AnomalyDetectionModel`
          - `SegmentationModel`

        - Nodes:

          - `BiSeNetHead`
          - `DDRNetSegmentationHead`
          - `DiscSubNetHead`
          - `SegmentationHead`
          - `TransformerSegmentationHead`

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
        """Initialize the visualizer and store the color options.

        Args:
            colors (``Color | list[Color] | None``): One color for each
                class, in the order of the class indices. A single color
                becomes a list of one color. When ``None``, or when the
                number of colors is not the number of classes, `forward`
                uses the colors of `BaseVisualizer.colormap`. It logs a
                warning on its first call.
            background_class (int | None): The index of the class that
                gets ``background_color``. It applies only when `forward`
                uses the colormap colors and the node has more than one
                class. ``None`` gives every class a colormap color.
            background_color (``Color``): The color of the background
                class.
            alpha (float): The opacity of the masks, from ``0`` for
                transparent to ``1`` for opaque.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseVisualizer`, such as ``scale`` and ``node``.

        """
        super().__init__(**kwargs)
        if colors is not None and not isinstance(colors, list):
            colors = [colors]

        self._colors = colors
        self._background_class = background_class
        self._background_color = background_color
        self._alpha = alpha

        self._warn_colors = True

    @staticmethod
    def draw_predictions(
        canvas: Tensor,
        predictions: Tensor,
        alpha: float,
        colors: list[Color],
        scale: float = 1.0,
    ) -> Tensor:
        """Draw the predicted masks of a batch on copies of the canvas.

        For each image, `seg_output_to_bool` converts the logits to one
        boolean mask for each class. With more than one class, each pixel
        gets the class with the highest logit. With one class, a pixel
        gets the class when the sigmoid of its logit is at least ``0.5``.
        `potentially_upscale_masks` resizes the masks by ``scale``, and
        `draw_segmentation_targets` blends them into the image.

        Args:
            canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]``. The method does not change them.
            predictions (``Tensor``): Logits of shape
                ``[B, n_classes, h, w]``. The masks must have the canvas
                size after the resize by ``scale``.
            alpha (float): The opacity of the masks, from ``0`` to ``1``.
            colors (``list[Color]``): One color for each class, at least
                as many colors as classes.
            scale (float): The factor that resizes the masks.

        Returns:
            ``Tensor``: A new tensor of the same shape as ``canvas``, with
            the masks drawn.

        Example:
            The first pixel column is class ``0`` and the second is class
            ``1``. The masks double in size.

            >>> import torch
            >>> canvas = torch.zeros(1, 3, 2, 4, dtype=torch.uint8)
            >>> logits = torch.tensor([[[[2.0, 0.0]], [[0.0, 2.0]]]])
            >>> viz = SegmentationVisualizer.draw_predictions(
            ...     canvas,
            ...     logits,
            ...     alpha=1.0,
            ...     colors=["red", "blue"],
            ...     scale=2.0,
            ... )
            >>> viz[0, 0].tolist()
            [[255, 255, 0, 0], [255, 255, 0, 0]]

        """
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
        """Draw the target masks of a batch on copies of the canvas.

        For each image, the method casts the target to ``bool``, so every
        non-zero value marks a pixel of the class.
        `potentially_upscale_masks` resizes the masks by ``scale``, and
        `draw_segmentation_targets` blends them into the image.

        Args:
            canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]``. The method does not change them.
            targets (``Tensor``): One-hot masks of shape
                ``[B, n_classes, h, w]``. The masks must have the canvas
                size after the resize by ``scale``.
            alpha (float): The opacity of the masks, from ``0`` to ``1``.
            colors (``list[Color]``): One color for each class, at least
                as many colors as classes.
            scale (float): The factor that resizes the masks.

        Returns:
            ``Tensor``: A new tensor of the same shape as ``canvas``, with
            the masks drawn.

        Example:
            >>> import torch
            >>> canvas = torch.zeros(1, 3, 2, 2, dtype=torch.uint8)
            >>> target = torch.tensor([[[[1, 0], [0, 0]], [[0, 1], [1, 1]]]])
            >>> viz = SegmentationVisualizer.draw_targets(
            ...     canvas, target, alpha=1.0, colors=["red", "blue"]
            ... )
            >>> viz[0, :, 0, 0].tolist(), viz[0, :, 0, 1].tolist()
            ([255, 0, 0], [0, 0, 255])

        """
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
        """Draw the predicted masks, and the target masks when given.

        The method first selects the class colors. It uses ``colors``
        when the list holds one color for each class of the node.
        Otherwise it takes the colors of `BaseVisualizer.colormap`, and
        logs a warning on the first call. With more than one class, the
        ``background_class`` then gets the ``background_color``.
        `draw_predictions` and `draw_targets` resize the masks by the
        ``scale`` factor and draw them.

        Args:
            prediction_canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]`` to draw the predictions on.
            target_canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]`` to draw the targets on.
            predictions (``Tensor``): Logits of shape
                ``[B, n_classes, h, w]``, the main output of the node.
            target (``Tensor | None``): One-hot masks of shape
                ``[B, n_classes, h, w]``, the ``segmentation`` label.
                ``None`` when the batch has no such label.

        Returns:
            ``Tensor | tuple[Tensor, Tensor]``: The predictions image when
            ``target`` is ``None``, otherwise the pair
            ``(targets, predictions)``. Each image has the shape of its
            canvas.

        """
        colors = self._adjust_colors(
            self._colors, self._background_class, self._background_color
        )

        predictions_vis = self.draw_predictions(
            prediction_canvas,
            predictions,
            alpha=self._alpha,
            colors=colors,
            scale=self._scale,
        )
        if target is None:
            return predictions_vis

        targets_vis = self.draw_targets(
            target_canvas,
            target,
            alpha=self._alpha,
            colors=colors,
            scale=self._scale,
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
        """The labels for the ``target`` parameter of `forward`.

        `BaseAttachedModule.get_parameters` reads this set for the
        ``target`` parameter of `forward`, which has no label suffix. On
        an ``anomaly_detection`` node, the set holds only
        ``"segmentation"``, so ``target`` receives the anomaly mask. On
        other nodes, it holds the labels of the task. The property
        raises ``RuntimeError`` when the visualizer has no task.

        """
        if self.task == Tasks.ANOMALY_DETECTION:
            return Tasks.SEGMENTATION.required_labels
        return self.task.required_labels
