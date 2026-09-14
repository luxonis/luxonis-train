"""Draws keypoints, the skeleton that connects them, and the boxes they
belong to.
"""

from copy import deepcopy

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

from luxonis_train.tasks import Tasks

from .bbox_visualizer import BBoxVisualizer
from .utils import Color, draw_keypoint_labels, draw_keypoints


class KeypointVisualizer(BBoxVisualizer):
    r"""Visualizer for instance keypoints and their bounding boxes.

    .. figure::
       https://raw.githubusercontent.com/luxonis/luxonis-train/e542cf0efa20a0fc5c781ff505d699031cb0d228/media/example_viz/kpts.png
       :width: 700px
       :height: 350px
       :loading: embed

       The left image shows the targets. The right image shows the
       predictions.

    Inputs:
        - ``prediction_canvas``, ``target_canvas`` (``Tensor``):
          :math:`\left[B, 3, H, W\right]`
        - ``keypoints`` (``list[Tensor]``): :math:`\left[M_i,
          n_{keypoints}, 3\right]` per image, ``(x, y, conf)``, pixels
        - ``boundingbox`` (``list[Tensor]``): :math:`\left[M_i,
          6\right]` per image, ``[x1, y1, x2, y2, conf, class]``, pixels
        - ``target_keypoints`` (``Tensor | None``): :math:`\left[N, 1 +
          3 * n_{keypoints}\right]`, ``[batch, x, y, v, ...]``,
          normalized
        - ``target_boundingbox`` (``Tensor | None``): :math:`\left[N,
          6\right]`, ``[batch, class, x, y, w, h]``, ``xywh`` normalized

    Outputs:
        - ``Tensor | tuple[Tensor, Tensor]``: :math:`\left[B, 3, H,
          W\right]`, a ``(targets, predictions)`` pair when any target
          is given

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        Draws the boxes with `BBoxVisualizer`, then the keypoints, the
        optional skeleton lines, and the optional keypoint indices on
        top. The ``FOMO`` task is in ``supported_tasks``, but `FOMOHead`
        puts no ``boundingbox`` key in its packet. On that node,
        `BaseVisualizer.run` raises ``RuntimeError``.

    Example:
        Attached to a ``EfficientKeypointBBoxHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: EfficientKeypointBBoxHead
              inputs: [RepPANNeck]
              visualizers:
                - name: KeypointVisualizer

    Compatible with:
        - Used by: `KeypointDetectionModel`
        - Nodes:

          - `EfficientKeypointBBoxHead`
          - `FOMOHead`

    """

    supported_tasks = [Tasks.INSTANCE_KEYPOINTS, Tasks.FOMO]

    def __init__(
        self,
        visibility_threshold: float = 0.5,
        connectivity: list[tuple[int, int]] | None = None,
        visible_color: Color = "red",
        nonvisible_color: Color | None = None,
        radius: int | None = None,
        draw_indices: bool = False,
        **kwargs,
    ):
        """Initialize the visualizer and store the keypoint options.

        Args:
            visibility_threshold (float): The lowest confidence of a
                visible predicted keypoint. `draw_predictions` tells how
                the visualizer draws the other keypoints.
            connectivity (list[tuple[int, int]] | None): Pairs of
                keypoint indices to connect with lines, the skeleton.
                Applies to the predictions and the targets. ``None``
                draws no lines.
            visible_color (Color): Color of the visible predicted
                keypoints, and of all target keypoints. A color name
                such as ``"red"`` or an RGB tuple.
            nonvisible_color (Color | None): Color of the predicted
                keypoints below ``visibility_threshold``. When ``None``,
                the visualizer does not draw them at their coordinates.
            radius (int | None): Radius of a keypoint, in pixels. When
                ``None``, `forward` picks it from the size of each
                canvas.
            draw_indices (bool): Whether to write the index of each
                keypoint next to it. `draw_targets` tells when this
                raises ``RuntimeError`` for the target keypoints.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BBoxVisualizer`, such as ``labels``, ``colors``,
                ``width``, ``scale``, and ``node``.

        """
        super().__init__(**kwargs)
        self._visibility_threshold = visibility_threshold
        self._connectivity = connectivity
        self._visible_color = visible_color
        self._nonvisible_color = nonvisible_color
        self._radius = radius
        self._draw_indices = draw_indices

    @staticmethod
    def _get_radius(canvas: Tensor) -> int:
        """Return the keypoint radius for the size of a canvas.

        Args:
            canvas (``Tensor``): Images whose last two dimensions are
                the height and the width.

        Returns:
            int: ``1`` when both sides are below ``96`` pixels, ``5``
            when a side is above ``512`` pixels, and ``2`` otherwise.

        """
        height = canvas.size(-2)
        width = canvas.size(-1)

        if height < 96 and width < 96:
            return 1
        if height > 512 or width > 512:
            return 5
        return 2

    @staticmethod
    def draw_predictions(
        canvas: Tensor,
        predictions: list[Tensor],
        draw_indices: bool = False,
        nonvisible_color: Color | None = None,
        visible_color: Color = "red",
        visibility_threshold: float = 0.5,
        radius: int | None = None,
        scale: float = 1.0,
        **kwargs,
    ) -> Tensor:
        """Draw the predicted keypoints of a batch on copies of the
        canvas images.

        For each image, the method multiplies the coordinates by
        ``scale``. A keypoint with a confidence below
        ``visibility_threshold`` is *not visible*. Then the method draws
        the keypoints with ``torchvision.utils.draw_keypoints`` in two
        passes:

        - The first pass draws the visible keypoints in
          ``visible_color``, clamped into the image. It moves each
          keypoint that is not visible to the **top-left corner**
          ``(0, 0)`` and draws it there too.
        - The second pass runs only when ``nonvisible_color`` is set.
          It draws the keypoints that are not visible in that color,
          and does not clamp them. It moves each visible keypoint to
          ``(0, 0)`` and draws it there too.

        When ``draw_indices`` is set, each pass also writes the keypoint
        indices at the positions it uses, with
        `draw_keypoint_indices_pil`, in the color of the pass.

        Args:
            canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]``. The method does not modify it.
            predictions (``list[Tensor]``): One tensor per image, of
                shape ``[M_i, K, 3]``. Each keypoint is
                ``(x, y, confidence)``. The coordinates are pixels of the
                unscaled image.
            draw_indices (bool): Whether to write the index of each
                keypoint next to it.
            nonvisible_color (Color | None): Color of the second pass.
                ``None`` skips the second pass.
            visible_color (Color): Color of the first pass. A ``colors``
                key in ``kwargs`` replaces it for the keypoints, but not
                for the indices.
            visibility_threshold (float): The lowest confidence of a
                visible keypoint.
            radius (int | None): Radius of a keypoint, in pixels. When
                it is ``None``, ``torchvision`` raises ``TypeError`` for
                an image with at least one keypoint.
            scale (float): Multiplier for the coordinates. Pass the
                factor that scaled the canvas.
            **kwargs (``Any``): Keyword arguments for
                ``torchvision.utils.draw_keypoints``, such as
                ``connectivity`` and ``width``.

        Returns:
            ``Tensor``: A new tensor of the same shape as ``canvas``
            with the keypoints drawn.

        Examples:
            A visible keypoint stays at its coordinates:

            >>> import torch
            >>> canvas = torch.zeros(1, 3, 16, 16, dtype=torch.uint8)
            >>> keypoints = [torch.tensor([[[8.0, 8.0, 0.9]]])]
            >>> viz = KeypointVisualizer.draw_predictions(
            ...     canvas, keypoints, radius=1
            ... )
            >>> viz[0, :, 8, 8].tolist(), viz[0, :, 0, 0].tolist()
            ([255, 0, 0], [0, 0, 0])

            A keypoint below ``visibility_threshold`` moves to the
            top-left corner:

            >>> hidden = [torch.tensor([[[8.0, 8.0, 0.1]]])]
            >>> viz = KeypointVisualizer.draw_predictions(
            ...     canvas, hidden, radius=1
            ... )
            >>> viz[0, :, 8, 8].tolist(), viz[0, :, 0, 0].tolist()
            ([0, 0, 0], [255, 0, 0])

        """
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            prediction = predictions[i]

            xy = prediction[..., :2].clone()
            v = prediction[..., 2]

            if scale != 1.0:
                xy *= scale

            not_visible = v < visibility_threshold
            visible_xy = xy * (~not_visible).unsqueeze(-1).float()

            visible_xy[..., 0] = visible_xy[..., 0].clamp(
                0, canvas.size(-1) - 1
            )
            visible_xy[..., 1] = visible_xy[..., 1].clamp(
                0, canvas.size(-2) - 1
            )

            _kwargs = deepcopy(kwargs)
            _kwargs.setdefault("radius", radius)
            _kwargs.setdefault("colors", visible_color)

            viz[i] = draw_keypoints(
                canvas[i].clone(),
                visible_xy.int(),
                **_kwargs,
            )
            if draw_indices:
                viz[i] = KeypointVisualizer.draw_keypoint_indices_pil(
                    viz[i].clone(),
                    torch.cat([visible_xy, v.unsqueeze(-1)], dim=-1),
                    colors=visible_color,
                )

            if nonvisible_color is not None:
                nonvisible_xy = xy * not_visible.unsqueeze(-1).float()

                _kwargs2 = deepcopy(kwargs)
                _kwargs2.setdefault("radius", radius)
                _kwargs2["colors"] = nonvisible_color

                viz[i] = draw_keypoints(
                    viz[i].clone(),
                    nonvisible_xy,
                    **_kwargs2,
                )

                if draw_indices:
                    viz[i] = KeypointVisualizer.draw_keypoint_indices_pil(
                        viz[i].clone(),
                        torch.cat([nonvisible_xy, v.unsqueeze(-1)], dim=-1),
                        colors=nonvisible_color,
                    )

        return viz

    @staticmethod
    def draw_keypoint_indices_pil(
        canvas: Tensor,
        keypoints: Tensor,
        offset: tuple[int, int] = (7, 7),
        colors: Color = "red",
    ) -> Tensor:
        """Write the index of each keypoint next to it with PIL.

        The method flattens ``keypoints`` to rows of
        ``(x, y, visibility)`` and numbers the rows from ``0`` in one
        sequence. With several instances, the numbers do not restart
        for each instance. It does not read the visibility.

        The method centers each label on its keypoint and then shifts it
        by ``offset``. The direction of the shift cycles from one row to
        the next: down-left, down-right, up-right, and up-left.

        Args:
            canvas (``Tensor``): One ``uint8`` image of shape
                ``[3, H, W]``. PIL raises ``TypeError`` for a floating
                point image.
            keypoints (``Tensor``): Pixel keypoints with three values per
                keypoint, such as ``[M, K, 3]`` or ``[M, 3 * K]``. The
                method calls ``view``, so a tensor that it cannot view
                as ``[-1, 3]`` raises ``RuntimeError``.
            offset (tuple[int, int]): The vertical and the horizontal
                shift of a label, in pixels.
            colors (Color): Text color.

        Returns:
            ``Tensor``: A new ``float32`` image of shape ``[3, H, W]`` on
            the CPU, with values from ``0`` to ``255`` and the indices
            drawn.

        Example:
            >>> import torch
            >>> canvas = torch.zeros(3, 32, 32, dtype=torch.uint8)
            >>> keypoints = torch.tensor([[16.0, 16.0, 1.0]])
            >>> viz = KeypointVisualizer.draw_keypoint_indices_pil(
            ...     canvas, keypoints
            ... )
            >>> viz.dtype, viz.shape
            (torch.float32, torch.Size([3, 32, 32]))
            >>> bool(viz[0].any()), bool(viz[1].any())
            (True, False)

        """
        ndarr = canvas.permute(1, 2, 0).detach().cpu().numpy()
        img = Image.fromarray(ndarr)
        draw = ImageDraw.Draw(img)

        kp = keypoints.view(-1, 3)
        oy, ox = offset

        offset_modes = [
            (+oy, -ox),
            (+oy, +ox),
            (-oy, +ox),
            (-oy, -ox),
        ]

        for idx, (x, y, _v) in enumerate(kp):
            x, y = int(x.item()), int(y.item())
            label = str(idx)

            # Get text size
            bbox = draw.textbbox((0, 0), label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # Center text on keypoint
            cx = x - text_w // 2
            cy = y - text_h // 2

            # Apply cycled offset
            dy, dx = offset_modes[idx % len(offset_modes)]
            tx = cx + dx
            ty = cy + dy

            draw.text((tx, ty), label, fill=colors)

        out = np.asarray(img).astype(np.float32)
        return torch.from_numpy(out).permute(2, 0, 1)

    @staticmethod
    def draw_targets(
        canvas: Tensor,
        targets: Tensor,
        draw_indices: bool = False,
        colors: Color = "red",
        **kwargs,
    ) -> Tensor:
        """Draw the target keypoints of a batch on copies of the canvas
        images.

        The keypoints of image ``i`` are the rows of ``targets`` whose
        first column equals ``i``. `draw_keypoint_labels` converts them
        from normalized to pixel coordinates with the canvas size and
        draws them with ``torchvision.utils.draw_keypoints``. It draws
        every keypoint, whatever its visibility.

        Args:
            canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]``. The method does not modify it.
            targets (``Tensor``): Keypoints of shape ``[N, 1 + 3 * K]``
                with rows ``[batch_index, x_1, y_1, v_1, ..., v_K]``. The
                coordinates are normalized to ``[0, 1]``.
            draw_indices (bool): Whether to write the index of each
                keypoint next to it with `draw_keypoint_indices_pil`.
                The method **fails** with ``RuntimeError`` when an image
                has more than one instance and ``K`` is more than ``1``.
            colors (Color): Color of the keypoints and the indices.
            **kwargs (``Any``): Keyword arguments for
                ``torchvision.utils.draw_keypoints``, such as ``radius``
                and ``connectivity``.

        Returns:
            ``Tensor``: A new tensor of the same shape as ``canvas``
            with the keypoints drawn.

        Example:
            >>> import torch
            >>> canvas = torch.zeros(1, 3, 16, 16, dtype=torch.uint8)
            >>> targets = torch.tensor([[0, 0.5, 0.5, 2.0]])
            >>> viz = KeypointVisualizer.draw_targets(
            ...     canvas, targets, radius=1
            ... )
            >>> viz[0, :, 8, 8].tolist(), viz[0, :, 0, 0].tolist()
            ([255, 0, 0], [0, 0, 0])

        """
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            target = targets[targets[:, 0] == i][:, 1:]
            viz[i] = draw_keypoint_labels(
                canvas[i].clone(),
                target,
                colors=colors,
                **kwargs,
            )
            if draw_indices:
                viz[i] = KeypointVisualizer.draw_keypoint_indices_pil(
                    viz[i].clone(), target, colors=colors
                )

        return viz

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        keypoints: list[Tensor],
        boundingbox: list[Tensor],
        target_keypoints: Tensor | None,
        target_boundingbox: Tensor | None,
        **kwargs,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Draw the predicted boxes and keypoints, and the targets when
        given.

        `BBoxVisualizer.draw_predictions` draws the boxes, and
        `draw_predictions` draws the keypoints on top. When
        ``target_boundingbox`` is set, `BBoxVisualizer.draw_targets`
        draws the target boxes. When ``target_keypoints`` is set,
        `draw_targets` draws the target keypoints on top, in
        ``visible_color``.

        When ``radius`` is ``None``, the radius comes from the size
        of each canvas. It is ``1`` when both sides are below ``96``
        pixels, ``5`` when a side is above ``512`` pixels, and ``2``
        otherwise.

        Args:
            prediction_canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]`` to draw the predictions on.
            target_canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]`` to draw the targets on.
            keypoints (``list[Tensor]``): One tensor per image, of shape
                ``[M_i, K, 3]``. Each keypoint is ``(x, y, confidence)``,
                with ``x`` and ``y`` in pixels. `draw_predictions` scales
                the coordinates by the ``scale`` factor.
            boundingbox (``list[Tensor]``): One tensor per image, of
                shape ``[M_i, 6]`` with rows
                ``[x1, y1, x2, y2, conf, class]`` in pixels.
                `BBoxVisualizer.draw_predictions` scales them by the
                ``scale`` factor.
            target_keypoints (``Tensor | None``): Keypoints of shape
                ``[N, 1 + 3 * K]`` with rows
                ``[batch_index, x_1, y_1, v_1, ..., v_K]``. The
                coordinates are normalized to ``[0, 1]``. ``None`` when
                the batch has no ``keypoints`` labels.
            target_boundingbox (``Tensor | None``): Boxes of shape
                ``[N, 6]`` with rows ``[batch_index, class, x, y, w, h]``,
                ``xywh`` normalized to ``[0, 1]``. ``None`` when the
                batch has no ``boundingbox`` labels.
            **kwargs (``Any``): Keyword arguments for
                ``torchvision.utils.draw_keypoints``, such as ``width``.
                The method passes them to `draw_predictions` and to
                `draw_targets`. A key that the method also passes by
                name, such as ``radius`` or ``connectivity``, raises
                ``TypeError``. `BaseVisualizer.run` does not pass any.

        Returns:
            ``tuple[Tensor, Tensor] | Tensor``: The pair
            ``(targets, predictions)`` of drawn images when
            ``target_keypoints`` or ``target_boundingbox`` is set;
            otherwise only the predictions image.

        Example:
            >>> import torch
            >>> visualizer = KeypointVisualizer(
            ...     labels=["person"], colors=["red"]
            ... )
            >>> canvas = torch.zeros(1, 3, 16, 16, dtype=torch.uint8)
            >>> boxes = [torch.tensor([[2.0, 2.0, 14.0, 14.0, 0.9, 0.0]])]
            >>> keypoints = [torch.tensor([[[8.0, 8.0, 0.9]]])]
            >>> visualizer(canvas, canvas, keypoints, boxes, None, None).shape
            torch.Size([1, 3, 16, 16])
            >>> targets = torch.tensor([[0, 0.5, 0.5, 2.0]])
            >>> len(
            ...     visualizer(canvas, canvas, keypoints, boxes, targets, None)
            ... )
            2

        """
        pred_viz = super().draw_predictions(
            prediction_canvas, boundingbox, self._scale
        )

        prediction_radius = (
            KeypointVisualizer._get_radius(prediction_canvas)
            if self._radius is None
            else self._radius
        )
        target_radius = (
            KeypointVisualizer._get_radius(target_canvas)
            if self._radius is None
            else self._radius
        )

        pred_viz = self.draw_predictions(
            pred_viz,
            keypoints,
            self._draw_indices,
            connectivity=self._connectivity,
            nonvisible_color=self._nonvisible_color,
            visible_color=self._visible_color,
            visibility_threshold=self._visibility_threshold,
            radius=prediction_radius,
            scale=self._scale,
            **kwargs,
        )

        if target_keypoints is None and target_boundingbox is None:
            return pred_viz

        if target_boundingbox is not None:
            target_viz = super().draw_targets(
                target_canvas, target_boundingbox
            )
        else:
            target_viz = target_canvas

        if target_keypoints is not None:
            target_viz = self.draw_targets(
                target_viz,
                target_keypoints,
                self._draw_indices,
                radius=target_radius,
                colors=self._visible_color,
                connectivity=self._connectivity,
                **kwargs,
            )

        return target_viz, pred_viz
