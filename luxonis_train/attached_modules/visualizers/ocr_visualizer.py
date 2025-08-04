import cv2
import numpy as np
import torch
from torch import Tensor

from luxonis_train.nodes import OCRCTCHead

from .base_visualizer import BaseVisualizer
from .utils import numpy_to_torch_img, torch_img_to_numpy


class OCRVisualizer(BaseVisualizer):
    """Visualizer for OCR tasks."""

    node: OCRCTCHead

    def __init__(
        self,
        font_scale: float = 0.5,
        color: tuple[int, int, int] = (0, 0, 0),
        thickness: int = 1,
        **kwargs,
    ):
        """Initializes the OCR visualizer.

        @type font_scale: float
        @param font_scale: Font scale of the text. Defaults to C{0.5}.
        @type color: tuple[int, int, int]
        @param color: Color of the text. Defaults to C{(0, 0, 0)}.
        @type thickness: int
        @param thickness: Thickness of the text. Defaults to C{1}.
        """
        super().__init__(**kwargs)
        self.font_scale = font_scale
        self.color = color
        self.thickness = thickness

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        predictions: Tensor,
        targets: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Creates a visualization of the OCR predictions and labels.

        @type label_canvas: Tensor
        @param label_canvas: The canvas to draw the labels on.
        @type prediction_canvas: Tensor
        @param prediction_canvas: The canvas to draw the predictions on.
        @type predictions: list[str]
        @param predictions: The predictions to visualize.
        @type targets: list[str]
        @param targets: The targets to visualize.
        @rtype: tuple[Tensor, Tensor]
        @return: A tuple of the label and prediction visualizations.
        """
        decoded_predictions = self.node.decoder(predictions)

        target_strings = []
        if targets is not None:
            for target in targets:
                target = target[target != 0]
                target = [chr(int(char.item())) for char in target]
                target = "".join(target)
                target_strings.append(target)

        overlay = torch.zeros_like(target_canvas)
        preds_targets = torch.zeros_like(prediction_canvas)

        for i in range(len(overlay)):
            pred_text, probability = decoded_predictions[i]
            arr = torch_img_to_numpy(target_canvas[i].clone())
            pred_img = np.full_like(arr, 255)

            if targets is not None:
                gt_text = target_strings[i]
                pred_img = cv2.putText(
                    pred_img,
                    f"GT: {gt_text}",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    self.color,
                    self.thickness,
                )

            pred_img = cv2.putText(
                pred_img,
                f"Pred: {pred_text} {probability:.2f}",
                (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.color,
                self.thickness,
            )

            overlay[i] = numpy_to_torch_img(arr)
            preds_targets[i] = numpy_to_torch_img(pred_img)

        return overlay, preds_targets
