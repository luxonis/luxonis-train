from copy import deepcopy

import torch
from torch import Tensor

from luxonis_train.enums import TaskType

from .base_visualizer import BaseVisualizer
from .utils import Color, draw_keypoint_labels, draw_keypoints


class KeypointVisualizer(BaseVisualizer[list[Tensor], Tensor]):
    supported_tasks: list[TaskType] = [TaskType.KEYPOINTS]

    def __init__(
        self,
        visibility_threshold: float = 0.5,
        connectivity: list[tuple[int, int]] | None = None,
        visible_color: Color = "red",
        nonvisible_color: Color | None = None,
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
        """
        super().__init__(**kwargs)
        self.visibility_threshold = visibility_threshold
        self.connectivity = connectivity
        self.visible_color = visible_color
        self.nonvisible_color = nonvisible_color

    @staticmethod
    def draw_predictions(
        canvas: Tensor,
        predictions: list[Tensor],
        nonvisible_color: Color | None = None,
        visibility_threshold: float = 0.5,
        **kwargs,
    ) -> Tensor:
        viz = torch.zeros_like(canvas)
        for i in range(len(canvas)):
            prediction = predictions[i]
            mask = prediction[..., 2] < visibility_threshold
            visible_kpts = prediction[..., :2] * (~mask).unsqueeze(-1).float()
            visible_kpts[..., 0] = torch.clamp(
                visible_kpts[..., 0], 0, canvas.size(-1) - 1
            )
            visible_kpts[..., 1] = torch.clamp(
                visible_kpts[..., 1], 0, canvas.size(-2) - 1
            )
            viz[i] = draw_keypoints(
                canvas[i].clone(),
                visible_kpts[..., :2],
                **kwargs,
            )
            if nonvisible_color is not None:
                _kwargs = deepcopy(kwargs)
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
    def draw_targets(canvas: Tensor, targets: Tensor, **kwargs) -> Tensor:
        viz = torch.zeros_like(canvas)
        for i in range(len(canvas)):
            target = targets[targets[:, 0] == i][:, 1:]
            viz[i] = draw_keypoint_labels(
                canvas[i].clone(),
                target,
                **kwargs,
            )

        return viz

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        predictions: list[Tensor],
        targets: Tensor | None,
        **kwargs,
    ) -> tuple[Tensor, Tensor] | Tensor:
        pred_viz = self.draw_predictions(
            prediction_canvas,
            predictions,
            connectivity=self.connectivity,
            colors=self.visible_color,
            nonvisible_color=self.nonvisible_color,
            visibility_threshold=self.visibility_threshold,
            **kwargs,
        )
        if targets is None:
            return pred_viz

        target_viz = self.draw_targets(
            label_canvas,
            targets,
            colors=self.visible_color,
            connectivity=self.connectivity,
            **kwargs,
        )
        return target_viz, pred_viz
