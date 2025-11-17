from copy import deepcopy

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

from luxonis_train.tasks import Tasks

from .bbox_visualizer import BBoxVisualizer
from .utils import Color, draw_keypoint_labels, draw_keypoints


class KeypointVisualizer(BBoxVisualizer):
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
        """Visualizer for keypoints.

        @type visibility_threshold: float
        @param visibility_threshold: Threshold for visibility of
            keypoints. If the visibility of a keypoint is below this
            threshold, it is considered as not visible. Defaults to
            C{0.5}.
        @type connectivity: list[tuple[int, int]] | None
        @param connectivity: List of tuples of keypoint indices that
            define the connections in the skeleton. Defaults to C{None}.
        @type visible_color: L{Color}
        @param visible_color: Color of visible keypoints. Either a
            string or a tuple of RGB values. Defaults to C{"red"}.
        @type nonvisible_color: L{Color} | None
        @param nonvisible_color: Color of nonvisible keypoints. If
            C{None}, nonvisible keypoints are not drawn. Defaults to
            C{None}.
        @type radius: int | None
        @param radius: the radius of drawn keypoints
        """
        super().__init__(**kwargs)
        self.visibility_threshold = visibility_threshold
        self.connectivity = connectivity
        self.visible_color = visible_color
        self.nonvisible_color = nonvisible_color
        self.radius = radius
        self.draw_indices = draw_indices

    @staticmethod
    def _get_radius(canvas: Tensor) -> int:
        """Determine keypoint radius based on image size.

        If the image is under 128 in both height and width: 1.
        If the image is more than 512 in either height or width: 6.
        Otherwise: 3.
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
        visibility_threshold: float = 0.5,
        radius: int | None = None,
        **kwargs,
    ) -> Tensor:
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            prediction = predictions[i]
            mask = prediction[..., 2] < visibility_threshold
            visible_kpts = prediction[..., :2] * (~mask).unsqueeze(-1).float()
            visible_kpts[..., 0] = visible_kpts[..., 0].clamp(
                0, canvas.size(-1) - 1
            )
            visible_kpts[..., 1] = visible_kpts[..., 1].clamp(
                0, canvas.size(-2) - 1
            )

            _kwargs = deepcopy(kwargs)
            _kwargs.setdefault("radius", radius)

            viz[i] = draw_keypoints(
                canvas[i].clone(),
                visible_kpts[..., :2].int(),
                **_kwargs,
            )

            if nonvisible_color is not None:
                _kwargs = deepcopy(kwargs)
                _kwargs.setdefault("radius", radius)
                _kwargs["colors"] = nonvisible_color
                nonvisible_kpts = (
                    prediction[..., :2] * mask.unsqueeze(-1).float()
                )
                viz[i] = draw_keypoints(
                    viz[i].clone(),
                    nonvisible_kpts[..., :2],
                    **_kwargs,
                )

        return viz

    @staticmethod
    def draw_keypoint_indices_pil(
        canvas: Tensor,
        keypoints: Tensor,
        offset: tuple[int, int] = (5, 5),
        color: tuple[int, int, int] = (255, 0, 0),
    ) -> Tensor:
        """Draw keypoint indices using PIL, and cycle text offsets to
        reduce overlap.

        canvas: Tensor (3, H, W) in [0,255]
        keypoints: Tensor (1, N*3) containing (x, y, v) triplets
        offset: (dy, dx)
        color: RGB tuple
        """
        ndarr = canvas.permute(1, 2, 0).detach().cpu().numpy()
        img = Image.fromarray(ndarr)
        draw = ImageDraw.Draw(img)

        kp = keypoints.view(-1, 3)
        oy, ox = offset

        offset_modes = [
            (-oy, +ox),
            (+oy, -ox),
            (-oy, -ox),
        ]

        for idx, (x, y, v) in enumerate(kp):
            if v < 1:
                continue

            x, y = int(x.item()), int(y.item())

            # Pick one of the three positions based on keypoint index (cycle in a modulo way)
            dx, dy = offset_modes[idx % 3]

            tx, ty = x + dx, y + dy

            draw.text((tx, ty), str(idx), fill=color)

        out = np.asarray(img).astype(np.float32)
        return torch.from_numpy(out).permute(2, 0, 1)

    @staticmethod
    def draw_targets(
        canvas: Tensor, targets: Tensor, draw_indices: bool = False, **kwargs
    ) -> Tensor:
        viz = torch.zeros_like(canvas)

        for i in range(len(canvas)):
            target = targets[targets[:, 0] == i][:, 1:]
            viz[i] = draw_keypoint_labels(
                canvas[i].clone(),
                target,
                **kwargs,
            )
            if draw_indices:
                viz[i] = KeypointVisualizer.draw_keypoint_indices_pil(
                    viz[i].clone(),
                    target,
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
        pred_viz = super().draw_predictions(prediction_canvas, boundingbox)

        prediction_radius = (
            KeypointVisualizer._get_radius(prediction_canvas)
            if self.radius is None
            else self.radius
        )
        target_radius = (
            KeypointVisualizer._get_radius(target_canvas)
            if self.radius is None
            else self.radius
        )

        pred_viz = self.draw_predictions(
            pred_viz,
            keypoints,
            self.draw_indices,
            connectivity=self.connectivity,
            colors=self.visible_color,
            nonvisible_color=self.nonvisible_color,
            visibility_threshold=self.visibility_threshold,
            radius=prediction_radius,
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
                self.draw_indices,
                radius=target_radius,
                colors=self.visible_color,
                connectivity=self.connectivity,
                **kwargs,
            )

        return target_viz, pred_viz
