from torch import Tensor

from luxonis_train.attached_modules.visualizers import BBoxVisualizer
from luxonis_train.tasks import Tasks

from .keypoint_visualizer import KeypointVisualizer


class FOMOVisualizer(BBoxVisualizer):
    supported_tasks = [Tasks.FOMO]

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        keypoints: list[Tensor],
        target_boundingbox: Tensor | None,
    ) -> tuple[Tensor, Tensor] | Tensor:
        pred_viz = KeypointVisualizer.draw_predictions(
            prediction_canvas, keypoints, colors="red", radius=5
        )
        if target_boundingbox is None:
            return pred_viz

        target_viz = super().draw_targets(target_canvas, target_boundingbox)
        return target_viz, pred_viz
