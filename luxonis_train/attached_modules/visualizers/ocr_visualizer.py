import cv2
import numpy as np
import torch
from torch import Tensor

from luxonis_train.nodes import OCRCTCHead

from .base_visualizer import BaseVisualizer
from .utils import numpy_to_torch_img, torch_img_to_numpy


class OCRVisualizer(BaseVisualizer):
    """Visualize OCR predictions and optional text targets.

    Metadata:
        - Module type: visualizer
        - Registry name: ``OCRVisualizer``
        - Task: ocr
        - Attached node types: ``OCRCTCHead``
        - Inputs: prediction and target canvases, OCR predictions, and
          optional text targets.
        - Outputs: ``(overlay, preds_targets)`` visualizations.

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Decodes predictions with the attached
          ``OCRCTCHead`` decoder and renders text with OpenCV.

    Prediction format:
        - ``predictions`` is the tensor expected by ``OCRCTCHead.decoder``.

    Target format:
        - ``targets`` is an optional padded tensor of character codes with
          zeros ignored.

    """

    node: OCRCTCHead

    def __init__(
        self,
        font_scale: float = 0.5,
        color: tuple[int, int, int] = (0, 0, 0),
        thickness: int = 1,
        **kwargs,
    ):
        """Initialize the OCR visualizer.

        Args:
            font_scale (float): Font scale of the text. Defaults to ``0.5``.
            color (tuple[int, int, int]): Color of the text. Defaults to ``(0, 0, 0)``.
            thickness (int): Thickness of the text. Defaults to ``1``.
            **kwargs (``Any``): Keyword arguments forwarded to the parent class.

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
        """Create a visualization of the OCR predictions and labels.

        Args:
            prediction_canvas (``Tensor``): The canvas to draw the predictions on.
            target_canvas (``Tensor``): The canvas to draw the labels on.
            predictions (``Tensor``): The predictions to visualize.
            targets (``Tensor | None``): The targets to visualize, or ``None``
                when targets are unavailable.

        Returns:
            ``tuple[Tensor, Tensor]``: A tuple of the label and prediction visualizations.

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
