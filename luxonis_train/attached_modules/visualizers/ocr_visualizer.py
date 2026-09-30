"""The visualizer that writes the recognized text on a white panel."""

import cv2
import numpy as np
import torch
from torch import Tensor

from luxonis_train.nodes import OCRCTCHead

from .base_visualizer import BaseVisualizer
from .utils import numpy_to_torch_img, torch_img_to_numpy


class OCRVisualizer(BaseVisualizer):
    r"""Visualizer for the text predictions of an OCR head.

    The visualizer does not draw on the input images. It returns them
    with a white panel for each image. The panel shows the target text
    and the predicted text with its mean probability.

    .. figure::
       https://raw.githubusercontent.com/luxonis/luxonis-train/e542cf0efa20a0fc5c781ff505d699031cb0d228/media/example_viz/ocr.png
       :width: 700px
       :height: 52px
       :loading: embed

       An input image and its text panel.

    Inputs:
        - ``prediction_canvas``, ``target_canvas`` (``Tensor``):
          :math:`\left[B, 3, H, W\right]`
        - ``predictions`` (``Tensor``): :math:`\left[B, T, C\right]`
          logits
        - ``targets`` (``Tensor | None``): :math:`\left[B,
          T_{max}\right]` Unicode code points, padded with ``0``

    Outputs:
        - ``tuple[Tensor, Tensor]``: :math:`\left[B, 3, H, W\right]`,
          the input images and the text panels

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        The ``decoder`` of the attached `OCRCTCHead` converts the
        predictions to text. OpenCV writes the text on the panels.

    Example:
        Attached to a ``OCRCTCHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: OCRCTCHead
              inputs: [SVTRNeck]
              visualizers:
                - name: OCRVisualizer

    Compatible with:
        - Used by: `OCRRecognitionModel`
        - Nodes: `OCRCTCHead`

    """

    node: OCRCTCHead

    def __init__(
        self,
        font_scale: float = 0.5,
        color: tuple[int, int, int] = (0, 0, 0),
        thickness: int = 1,
        **kwargs,
    ):
        """Initialize the visualizer and store the text options.

        Args:
            font_scale (float): The OpenCV font scale of the text.
            color (tuple[int, int, int]): The color of the text, one value
                in ``[0, 255]`` for each channel, in the channel order of
                the canvas. The default is black.
            thickness (int): The line thickness of the text, in pixels.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseVisualizer`, such as ``scale`` and ``node``. The
                ``node`` must be an `OCRCTCHead`, because `forward` uses
                its ``decoder``. Another node type raises
                ``IncompatibleError``.

        """
        super().__init__(**kwargs)
        self._font_scale = font_scale
        self._color = color
        self._thickness = thickness

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        predictions: Tensor,
        targets: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Write the target text and the predicted text on white panels.

        The ``decoder`` of the node converts ``predictions`` to a text
        and a mean probability for each image. For each image, the method
        fills a panel of the canvas size with white. When ``targets`` is
        given, it writes ``"GT: <target text>"`` with the bottom-left
        corner of the text at the pixel ``(5, 20)``. It writes
        ``"Pred: <predicted text> <probability>"`` at ``(5, 40)``. The
        probability has two decimals and is ``nan`` for an empty text.

        Args:
            prediction_canvas (``Tensor``): Images of shape
                ``[B, 3, H, W]``. The method uses only the shape, the
                dtype, and the device of this tensor.
            target_canvas (``Tensor``): ``uint8`` images of shape
                ``[B, 3, H, W]``. The panels get the size of these images.
            predictions (``Tensor``): The logits of the node, of shape
                ``[B, T, n_classes]``.
            targets (``Tensor | None``): The ``metadata/text`` label, of
                shape ``[B, T_max]``. Each row holds the Unicode code
                points of one text, padded with ``0``. ``None`` when the
                batch has no text labels.

        Returns:
            ``tuple[Tensor, Tensor]``: A copy of ``target_canvas``, and
            the panels in a tensor of the same shape.

        Example:
            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import OCRCTCHead
            >>> shapes = [{"features": [Size([1, 8, 1, 4])]}]
            >>> head = OCRCTCHead(alphabet=["a", "b"], input_shapes=shapes)
            >>> visualizer = OCRVisualizer(node=head)
            >>> canvas = torch.zeros(1, 3, 48, 96, dtype=torch.uint8)
            >>> targets = torch.tensor([[97, 98, 0]])
            >>> images, panels = visualizer(
            ...     canvas, canvas, torch.zeros(1, 4, 3), targets
            ... )
            >>> bool((images == canvas).all()), panels[0, :, 0, 0].tolist()
            (True, [255, 255, 255])

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
                    self._font_scale,
                    self._color,
                    self._thickness,
                )

            pred_img = cv2.putText(
                pred_img,
                f"Pred: {pred_text} {probability:.2f}",
                (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._font_scale,
                self._color,
                self._thickness,
            )

            overlay[i] = numpy_to_torch_img(arr)
            preds_targets[i] = numpy_to_torch_img(pred_img)

        return overlay, preds_targets
