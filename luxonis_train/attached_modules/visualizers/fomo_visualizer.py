import torch
from torch import Tensor
from torchvision.utils import draw_keypoints

from luxonis_train.attached_modules.visualizers import BBoxVisualizer
from luxonis_train.tasks import Tasks

from .keypoint_visualizer import KeypointVisualizer


class FOMOVisualizer(BBoxVisualizer):
    supported_tasks = [Tasks.FOMO]

    def __init__(
        self, visibility_threshold: float = 0.5, radius: int = 5, **kwargs
    ):
        super().__init__(**kwargs)
        self._visibility_threshold = visibility_threshold
        self._radius = radius

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
                prediction_canvas, keypoints, colors="red", radius=self._radius
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
            viz[i] = self._draw_image_predictions(viz[i], predictions[i])
        return viz

    def _draw_image_predictions(
        self, image: Tensor, prediction: Tensor
    ) -> Tensor:
        xy = prediction[..., :2].clone()
        if self._scale and self._scale != 1.0:
            xy *= self._scale
        visible = prediction[..., 2] >= self._visibility_threshold
        xy, classes = xy[visible], prediction[..., 3].long()[visible]
        if xy.numel() == 0:
            return image
        xy[:, 0] = xy[:, 0].clamp(0, image.size(-1) - 1)
        xy[:, 1] = xy[:, 1].clamp(0, image.size(-2) - 1)
        for class_id in torch.unique(classes):
            image = self._draw_class_keypoints(
                image, xy[classes == class_id], int(class_id)
            )
        return image

    def _draw_class_keypoints(
        self, image: Tensor, points: Tensor, class_id: int
    ) -> Tensor:
        label = (
            self._label_dict.get(class_id, str(class_id))
            if self._label_dict
            else str(class_id)
        )
        color = (
            self._colors[label]
            if self._colors and label in self._colors
            else (255, 255, 255)
        )
        return draw_keypoints(
            image=image,
            keypoints=points.int().unsqueeze(1),
            radius=self._radius,
            colors=color,
        )
