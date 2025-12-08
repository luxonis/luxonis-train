import torch
from torch import Tensor
from torchvision.utils import draw_keypoints

from luxonis_train.attached_modules.visualizers import BBoxVisualizer
from luxonis_train.tasks import Tasks

from .keypoint_visualizer import KeypointVisualizer


class FOMOVisualizer(BBoxVisualizer):
    supported_tasks = [Tasks.FOMO]

    def __init__(self, visibility_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.visibility_threshold = visibility_threshold

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        keypoints: list[Tensor],
        target_boundingbox: Tensor | None,
    ) -> tuple[Tensor, Tensor] | Tensor:
        single_class = self._determine_single_class(keypoints)
        if single_class:
            pred_viz = KeypointVisualizer.draw_predictions(
                prediction_canvas, keypoints, colors="red", radius=5
            )
        else:
            pred_viz = self.draw_predictions_per_class(
                prediction_canvas, keypoints
            )
        if target_boundingbox is None:
            return pred_viz

        target_viz = super().draw_targets(target_canvas, target_boundingbox)
        return target_viz, pred_viz

    def _determine_single_class(self, predictions: list[Tensor]) -> bool:
        return all(x.shape[2] == 3 for x in predictions)

    def draw_predictions_per_class(
        self, canvas: Tensor, predictions: list[Tensor]
    ) -> Tensor:
        viz = canvas.clone()

        for i in range(len(canvas)):
            prediction = predictions[i]

            xy = prediction[..., :2].clone()
            v = prediction[..., 2]
            keypoint_class = prediction[..., 3].long()

            if self.scale and self.scale != 1.0:
                xy *= self.scale

            visible = v >= self.visibility_threshold
            visible_xy = xy[visible]
            visible_classes = keypoint_class[visible]

            if visible_xy.numel() == 0:
                continue

            visible_xy[:, 0] = visible_xy[:, 0].clamp(0, canvas.size(-1) - 1)
            visible_xy[:, 1] = visible_xy[:, 1].clamp(0, canvas.size(-2) - 1)

            for cls in torch.unique(visible_classes):
                cls = cls.item()
                cls_mask = visible_classes == cls
                cls_points = visible_xy[cls_mask]

                if cls_points.numel() == 0:
                    continue

                label = (
                    self.label_dict.get(cls, str(cls))
                    if self.label_dict
                    else str(cls)
                )
                color = (
                    self.colors[label]
                    if self.colors and label in self.colors
                    else (255, 255, 255)
                )

                viz[i] = draw_keypoints(
                    image=viz[i],
                    keypoints=cls_points.int().unsqueeze(1),
                    radius=5,
                    colors=color,
                )

        return viz
