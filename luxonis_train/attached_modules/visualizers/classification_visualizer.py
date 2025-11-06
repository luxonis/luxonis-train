import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from luxonis_train.tasks import Tasks

from .base_visualizer import BaseVisualizer
from .utils import (
    dynamically_determine_font_scale,
    figure_to_torch,
    numpy_to_torch_img,
    torch_img_to_numpy,
)


class ClassificationVisualizer(BaseVisualizer):
    supported_tasks = [Tasks.CLASSIFICATION]

    def __init__(
        self,
        include_plot: bool = True,
        font_scale: float | None = None,
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
        multilabel: bool = False,
        **kwargs,
    ):
        """Visualizer for classification tasks.

        @type include_plot: bool
        @param include_plot: Whether to include a plot of the class
            probabilities in the visualization. Defaults to C{True}.
        @type font_scale: float | None = None,
        @param font_scale: Font scale for text. If None, scales
            proportionally to the image height and width
        """
        super().__init__(**kwargs)
        self.include_plot = include_plot
        self.font_scale = font_scale
        self.color = color
        self.thickness = thickness
        self.multilabel = multilabel

    def _get_class_name(self, pred: Tensor) -> str:
        """Handles both single-label and multi-label classification."""
        if self.multilabel:
            idxs = (pred > 0.5).nonzero(as_tuple=True)[0].tolist()
            return ", ".join([self.classes.inverse[idx] for idx in idxs])
        return self.classes.inverse[int(pred.argmax().item())]

    def _generate_plot(
        self, prediction: Tensor, width: int, height: int
    ) -> Tensor:
        prediction = prediction.to(torch.float32)
        if self.multilabel:
            pred = prediction.sigmoid().detach().cpu().numpy()
        else:
            pred = prediction.softmax(-1).detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.bar(np.arange(len(pred)), pred)
        ax.set_xticks(np.arange(len(pred)))
        ax.set_xticklabels(self.classes.keys(), rotation=90)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.grid(True)
        return figure_to_torch(fig, width, height)

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        predictions: Tensor,
        target: Tensor | None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        overlay = torch.zeros_like(target_canvas)
        plots = torch.zeros_like(prediction_canvas)
        for i in range(len(overlay)):
            prediction = predictions[i]
            arr = torch_img_to_numpy(target_canvas[i].clone())
            height, width = arr.shape[:2]

            if not self.font_scale:
                font_scale, thickness = dynamically_determine_font_scale(
                    height, width, self.thickness, self.font_scale
                )
                base_y: int = int(height * 0.15)
                line_spacing: int = int(height * 0.1)

                y_gt, y_pred = base_y, base_y + line_spacing
            else:
                font_scale, thickness = self.font_scale, self.thickness
                y_gt, y_pred = 50, 75

            curr_class = self._get_class_name(prediction)
            if target is not None:
                gt = self._get_class_name(target[i])
                arr = cv2.putText(
                    arr,
                    f"GT: {gt}",
                    (5, y_gt),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    self.color,
                    thickness,
                )
            arr = cv2.putText(
                arr,
                f"Pred: {curr_class}",
                (5, y_pred),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                self.color,
                thickness,
            )

            overlay[i] = numpy_to_torch_img(arr)

            if self.include_plot:
                plots[i] = self._generate_plot(
                    prediction,
                    prediction_canvas.shape[3],
                    prediction_canvas.shape[2],
                )

        if self.include_plot:
            return overlay, plots
        return overlay
