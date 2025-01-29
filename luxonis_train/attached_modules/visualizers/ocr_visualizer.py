import cv2
import numpy as np
import torch
from torch import Tensor

from luxonis_train.nodes import OCRCTCHead
from luxonis_train.utils import Labels, Packet

from .base_visualizer import BaseVisualizer
from .utils import numpy_to_torch_img, torch_img_to_numpy


class OCRVisualizer(BaseVisualizer[Tensor, Tensor]):
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

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels
    ) -> tuple[list[tuple[str, float]], list[str]]:
        """Prepares the predictions and targets for visualization.

        @type inputs: Packet[Tensor]
        @param inputs: A packet containing input tensors, typically
            network predictions.
        @type labels: Labels
        @param labels: A dictionary containing text labels and
            corresponding lengths.
        @rtype: tuple[Tensor, list[str]]
        @return: A tuple of predictions and targets.
        """

        preds = inputs["/classification"][0]

        preds = self.node.decoder(preds)
        targets = labels["/metadata/text"]

        target_strings = []
        for target in targets:
            target = target[target != 0]
            target = [chr(int(char.item())) for char in target]
            target = "".join(target)
            target_strings.append(target)

        return (preds, target_strings)

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        predictions: list[str],
        targets: list[str] | None,
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

        overlay = torch.zeros_like(label_canvas)
        preds_targets = torch.zeros_like(prediction_canvas)

        for i in range(len(overlay)):
            prediction_text = predictions[i][0]
            prediction_prob = predictions[i][1]
            arr = torch_img_to_numpy(label_canvas[i].clone())
            pred_img = np.full_like(arr, 255)

            if targets is not None:
                gt_text = targets[i]
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
                f"Pred: {prediction_text} {prediction_prob:.2f}",
                (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.color,
                self.thickness,
            )

            overlay[i] = numpy_to_torch_img(arr)
            preds_targets[i] = numpy_to_torch_img(pred_img)

        return overlay, preds_targets
