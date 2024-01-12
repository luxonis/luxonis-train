import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .base_visualizer import BaseVisualizer
from .utils import (
    figure_to_torch,
    numpy_to_torch_img,
    torch_img_to_numpy,
)


class ClassificationVisualizer(BaseVisualizer[Tensor, Tensor]):
    def __init__(
        self,
        include_plot: bool = True,
        font_scale: float = 1.0,
        color: tuple[int, int, int] = (255, 0, 0),
        thickness: int = 1,
        **kwargs,
    ):
        """Visualizer for classification tasks.

        @type include_plot: bool
        @param include_plot: Whether to include a plot of the class probabilities in the
            visualization. Defaults to C{True}.
        """
        super().__init__(**kwargs)
        self.include_plot = include_plot
        self.font_scale = font_scale
        self.color = color
        self.thickness = thickness

    def _get_class_name(self, pred: Tensor) -> str:
        idx = int((pred.argmax()).item())
        if self.node.class_names is None:
            return str(idx)
        return self.node.class_names[idx]

    def _generate_plot(self, prediction: Tensor, width: int, height: int) -> Tensor:
        prediction = prediction.softmax(-1).detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.bar(np.arange(len(prediction)), prediction)
        ax.set_xticks(np.arange(len(prediction)))
        if self.node.class_names is not None:
            ax.set_xticklabels(self.node.class_names, rotation=90)
        else:
            ax.set_xticklabels(np.arange(1, len(prediction) + 1))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.grid(True)
        return figure_to_torch(fig, width, height)

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        predictions: Tensor,
        labels: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        overlay = torch.zeros_like(label_canvas)
        plots = torch.zeros_like(prediction_canvas)
        for i in range(len(overlay)):
            prediction = predictions[i]
            gt = self._get_class_name(labels[i])
            arr = torch_img_to_numpy(label_canvas[i].clone())
            curr_class = self._get_class_name(prediction)
            arr = cv2.putText(
                arr,
                f"GT: {gt}",
                (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.color,
                self.thickness,
            )
            arr = cv2.putText(
                arr,
                f"Pred: {curr_class}",
                (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.color,
                self.thickness,
            )
            overlay[i] = numpy_to_torch_img(arr)
            if self.include_plot:
                plots[i] = self._generate_plot(
                    prediction, prediction_canvas.shape[3], prediction_canvas.shape[2]
                )

        if self.include_plot:
            return overlay, plots
        return overlay
