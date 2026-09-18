"""Draws instance masks together with their boxes."""

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
    r"""Visualizer for instance segmentation predictions and targets.

    .. figure::
       https://raw.githubusercontent.com/luxonis/luxonis-train/e542cf0efa20a0fc5c781ff505d699031cb0d228/media/example_viz/instance_seg.png
       :width: 700px
       :height: 262px
       :loading: embed

       The left image shows the targets. The right image shows the
       predictions.

    Inputs:
        - ``prediction_canvas``, ``target_canvas`` (``Tensor``):
          :math:`\left[B, 3, H, W\right]`
        - ``boundingbox`` (``list[Tensor]``): :math:`\left[M_i,
          6\right]` per image, ``[x1, y1, x2, y2, conf, class]``, pixels
        - ``instance_segmentation`` (``list[Tensor]``):
          :math:`\left[M_i, H, W\right]` per image, binary
        - ``target_boundingbox`` (``Tensor | None``): :math:`\left[N,
          6\right]`, ``[batch, class, x, y, w, h]``, ``xywh`` normalized
        - ``target_instance_segmentation`` (``Tensor | None``):
          :math:`\left[N, H, W\right]`, one per target box

    Outputs:
        - ``Tensor | tuple[Tensor, Tensor]``: :math:`\left[B, 3, H,
          W\right]`, a ``(targets, predictions)`` pair when both
          targets are given

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        Overlays the masks with ``torchvision`` and draws the box of
        each mask on top. The visualizer stores the ``fill``, ``font``,
        and ``font_size`` options but does not use them.

    Example:
        Attached to a ``PrecisionSegmentBBoxHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: PrecisionSegmentBBoxHead
              inputs: [RepPANNeck]
              visualizers:
                - name: InstanceSegmentationVisualizer

    Compatible with:
        - Used by: `InstanceSegmentationModel`
        - Nodes: `PrecisionSegmentBBoxHead`

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
        """Initialize the visualizer and resolve the class names and
        colors.

        Args:
            labels (dict[int, str] | list[str] | None): Class names to
                draw. A dictionary maps a class index to a name. A list
                maps by position. When ``None`` or empty, the names come
                from the ``classes`` of the node, so the visualizer then
                needs a ``node``.
            draw_labels (bool): Whether to draw the class name next to
                each box. Applies to the predictions and the targets.
            draw_scores (bool): Whether to write the confidence of each
                predicted box, with two decimals, in its label. Applies
                to the predictions only. Without ``draw_labels``, the
                label is the confidence alone.
            colors (dict[str, Color] | list[Color] | None): Colors of
                the masks and the boxes. A dictionary maps a class name
                to a color. A list maps by class index. When ``None``,
                each class gets a distinct color from `get_color`,
                seeded with its index.
            fill (bool): The drawing methods do not read it.
            width (int | None): Line width of the boxes, in pixels. When
                ``None`` or ``0``, the width is one percent of the
                smaller canvas side, rounded down, and at least ``1``.
            font (str | None): The drawing methods do not read it.
            font_size (int | None): The drawing methods do not read it.
            alpha (float): Opacity of the masks, from ``0``
                (transparent) to ``1`` (opaque).
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseVisualizer`, such as ``scale`` and ``node``.

        """
        super().__init__(**kwargs)

        if isinstance(labels, list):
            labels = dict(enumerate(labels))

        self._bbox_labels = labels or self.classes.inverse

        if colors is None:
            colors = {
                label: get_color(i) for i, label in self._bbox_labels.items()
            }
        if isinstance(colors, list):
            colors = {
                self._bbox_labels[i]: color for i, color in enumerate(colors)
            }

        self._colors = colors
        self._fill = fill
        self._width = width
        self._font = font
        self._font_size = font_size
        self._draw_labels = draw_labels
        self._draw_scores = draw_scores
        self._alpha = alpha

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
        """Draw the predicted masks and boxes of a batch on copies of
        the canvas images.

        For each image, the method multiplies the box coordinates by
        ``scale`` and resizes the masks by ``scale`` with nearest
        interpolation. It overlays the masks with
        ``torchvision.utils.draw_segmentation_masks``. Then it draws the
        boxes on top with ``torchvision.utils.draw_bounding_boxes``.
        Each mask and box gets the color of its class.
        `get_prediction_labels` builds the box labels.

        When ``torchvision`` raises ``ValueError`` for an image, the
        method logs a warning and keeps that image as it is in
        ``canvas``. A mask size that does not match the canvas size is
        one cause.

        Args:
            canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]``. The method does not modify it.
            pred_bboxes (``list[Tensor]``): One tensor per image, of
                shape ``[M_i, 6]`` with rows
                ``[x1, y1, x2, y2, conf, class]``. The coordinates are
                pixels of the unscaled image.
            pred_masks (``list[Tensor]``): One tensor per image, of
                shape ``[M_i, H_0, W_0]``, with one binary mask for each
                box. ``H_0`` and ``W_0`` are the unscaled image size.
            width (int | None): Line width of the boxes, in pixels. When
                ``None`` or ``0``, the width is one percent of the
                smaller canvas side, rounded down, and at least ``1``.
            label_dict (``Mapping[int, str]``): Class index to class
                name. Every predicted class must have a name here.
            color_dict (dict[str, Color]): Class name to color. Every
                predicted class must have a color here.
            draw_labels (bool): Whether to write the class name in the
                label of each box.
            draw_scores (bool): Whether to write the confidence, with
                two decimals, in the label of each box.
            alpha (float): Opacity of the masks, from ``0``
                (transparent) to ``1`` (opaque).
            scale (float): Multiplier for the box coordinates and the
                mask size. Pass the factor that scaled the canvas.

        Returns:
            ``Tensor``: A new tensor of the same shape as ``canvas``
            with the masks and boxes drawn.

        Example:
            >>> import torch
            >>> canvas = torch.zeros(1, 3, 16, 16, dtype=torch.uint8)
            >>> boxes = [torch.tensor([[4.0, 4.0, 12.0, 12.0, 0.9, 0.0]])]
            >>> masks = torch.zeros(1, 16, 16, dtype=torch.uint8)
            >>> masks[:, 4:12, 4:12] = 1
            >>> viz = InstanceSegmentationVisualizer.draw_predictions(
            ...     canvas,
            ...     boxes,
            ...     [masks],
            ...     width=1,
            ...     label_dict={0: "cat"},
            ...     color_dict={"cat": (255, 0, 0)},
            ...     draw_labels=False,
            ...     draw_scores=False,
            ...     alpha=0.5,
            ... )
            >>> viz[0, :, 8, 8].tolist(), viz[0, :, 0, 0].tolist()
            ([127, 0, 0], [0, 0, 0])

        """
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
        """Draw the target masks and boxes of a batch on copies of the
        canvas images.

        The boxes and masks of image ``i`` are the rows whose batch
        index in ``target_bboxes`` equals ``i``. The method resizes the
        masks by ``scale`` with nearest interpolation and overlays them.
        Then `draw_bounding_box_labels` converts the boxes from
        normalized ``xywh`` to pixel ``xyxy`` with the canvas size and
        draws them on top. Each mask and box gets the color of its
        class, and each box gets its class name when ``draw_labels`` is
        set. Unlike `draw_predictions`, this method does not catch the
        ``ValueError`` of ``torchvision``.

        Args:
            canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]``. The method does not modify it.
            target_bboxes (``Tensor``): Boxes of shape ``[N, 6]`` with
                rows ``[batch_index, class, x, y, w, h]``. The
                coordinates are ``xywh`` normalized to ``[0, 1]``.
            target_masks (``Tensor``): Binary masks of shape
                ``[N, H_0, W_0]``, one for each row of
                ``target_bboxes``. ``H_0`` and ``W_0`` are the unscaled
                image size.
            width (int | None): Line width of the boxes, in pixels. When
                ``None`` or ``0``, the width is one percent of the
                smaller canvas side, rounded down, and at least ``1``.
            label_dict (``Mapping[int, str]``): Class index to class
                name. Every target class must have a name here.
            color_dict (dict[str, Color]): Class name to color. Every
                target class must have a color here.
            draw_labels (bool): Whether to write the class name next to
                each box.
            alpha (float): Opacity of the masks, from ``0``
                (transparent) to ``1`` (opaque).
            scale (float): Multiplier for the mask size. Pass the factor
                that scaled the canvas.

        Returns:
            ``Tensor``: A new tensor of the same shape as ``canvas``
            with the masks and boxes drawn.

        Example:
            >>> import torch
            >>> canvas = torch.zeros(1, 3, 16, 16, dtype=torch.uint8)
            >>> boxes = torch.tensor([[0, 0, 0.25, 0.25, 0.5, 0.5]])
            >>> masks = torch.zeros(1, 16, 16, dtype=torch.uint8)
            >>> masks[:, 4:12, 4:12] = 1
            >>> viz = InstanceSegmentationVisualizer.draw_targets(
            ...     canvas,
            ...     boxes,
            ...     masks,
            ...     width=1,
            ...     label_dict={0: "cat"},
            ...     color_dict={"cat": (255, 0, 0)},
            ...     draw_labels=False,
            ...     alpha=1.0,
            ... )
            >>> viz[0, :, 8, 8].tolist(), viz[0, :, 0, 0].tolist()
            ([255, 0, 0], [0, 0, 0])

        """
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
        """Draw the predicted masks and boxes, and the targets when
        given.

        `draw_predictions` draws the predictions and `draw_targets`
        draws the targets. Both use the options of the constructor and
        the ``scale`` factor.

        Args:
            prediction_canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]`` to draw the predictions on.
            target_canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]`` to draw the targets on.
            boundingbox (``list[Tensor]``): One tensor per image, of
                shape ``[M_i, 6]`` with rows
                ``[x1, y1, x2, y2, conf, class]`` in pixels.
            instance_segmentation (``list[Tensor]``): One tensor per
                image, of shape ``[M_i, H_0, W_0]``, with one binary
                mask for each box. ``H_0`` and ``W_0`` are the image
                size before the ``scale`` resize.
            target_boundingbox (``Tensor | None``): Boxes of shape
                ``[N, 6]`` with rows ``[batch_index, class, x, y, w, h]``,
                ``xywh`` normalized to ``[0, 1]``. ``None`` when the
                batch has no ``boundingbox`` labels.
            target_instance_segmentation (``Tensor | None``): Binary
                masks of shape ``[N, H_0, W_0]``, one for each target
                box. ``None`` when the batch has no
                ``instance_segmentation`` labels.

        Returns:
            ``tuple[Tensor, Tensor] | Tensor``: The pair
            ``(targets, predictions)`` of drawn images when both
            ``target_boundingbox`` and ``target_instance_segmentation``
            are set; otherwise only the predictions image.

        Example:
            >>> import torch
            >>> visualizer = InstanceSegmentationVisualizer(
            ...     labels=["cat"], colors=["red"]
            ... )
            >>> canvas = torch.zeros(1, 3, 16, 16, dtype=torch.uint8)
            >>> boxes = [torch.tensor([[4.0, 4.0, 12.0, 12.0, 0.9, 0.0]])]
            >>> masks = torch.zeros(1, 16, 16, dtype=torch.uint8)
            >>> masks[:, 4:12, 4:12] = 1
            >>> visualizer(canvas, canvas, boxes, [masks], None, None).shape
            torch.Size([1, 3, 16, 16])
            >>> targets = torch.tensor([[0, 0, 0.25, 0.25, 0.5, 0.5]])
            >>> len(visualizer(canvas, canvas, boxes, [masks], targets, masks))
            2

        """
        predictions_viz = self.draw_predictions(
            prediction_canvas,
            boundingbox,
            instance_segmentation,
            self._width,
            self._bbox_labels,
            self._colors,
            self._draw_labels,
            self._draw_scores,
            self._alpha,
            self._scale,
        )
        if target_boundingbox is None or target_instance_segmentation is None:
            return predictions_viz
        targets_viz = self.draw_targets(
            target_canvas,
            target_boundingbox,
            target_instance_segmentation,
            self._width,
            self._bbox_labels,
            self._colors,
            self._draw_labels,
            self._alpha,
            self._scale,
        )
        return targets_viz, predictions_viz
