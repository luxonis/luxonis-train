"""The base class every visualizer inherits."""

from abc import abstractmethod
from functools import cached_property
from inspect import Parameter

import torch.nn.functional as F
from luxonis_ml.data.utils import ColorMap
from torch import Tensor
from typing_extensions import TypeVarTuple, Unpack, override

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.registry import VISUALIZERS
from luxonis_train.typing import Labels, Packet
from luxonis_train.utils import get_signature

Ts = TypeVarTuple("Ts")


class BaseVisualizer(BaseAttachedModule, register=False, registry=VISUALIZERS):
    """Base class for all visualizers.

    A visualizer draws the predictions of a node, and the labels of the
    batch, on copies of the input images. Every subclass registers
    itself in the `VISUALIZERS` registry under its class name, so a
    config names it as a string.

    A subclass implements `forward`. `BaseAttachedModule.get_parameters`
    describes how `run` fills its non-canvas parameters.

    """

    def __init__(self, *args, scale: float = 1.0, **kwargs) -> None:
        """Initialize the visualizer and store the canvas scale.

        Args:
            *args (``Any``): Positional arguments forwarded to
                `BaseAttachedModule`.
            scale (float): Factor that `run` applies to both canvases
                with `scale_canvas` before it calls `forward`. Defaults
                to ``1.0``.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseAttachedModule`, such as ``node``.

        """
        super().__init__(*args, **kwargs)
        self._scale = scale

    @override
    def __getstate__(self) -> dict:
        """Return the state to pickle, without the cached ``colormap``.

        A ``ColorMap`` holds a generator, and ``pickle`` cannot
        serialize a generator. The unpickled module creates a new map
        on its next access to `colormap`.

        Returns:
            dict: The state of the module without the ``colormap`` key.

        """
        state = super().__getstate__()
        if "colormap" in state:
            del state["colormap"]
        return state

    @staticmethod
    def scale_canvas(canvas: Tensor, scale: float = 1.0) -> Tensor:
        """Resize a batch of images by a factor with bilinear
        interpolation.

        Args:
            canvas (``Tensor``): Images of shape ``[B, C, H, W]``.
            scale (float): Multiplier for the height and the width.
                Defaults to ``1.0``.

        Returns:
            ``Tensor``: Images of shape
            ``[B, C, floor(H * scale), floor(W * scale)]``.

        Example:
            >>> import torch
            >>> canvas = torch.zeros(1, 3, 4, 6)
            >>> BaseVisualizer.scale_canvas(canvas, scale=0.5).shape
            torch.Size([1, 3, 2, 3])
            >>> BaseVisualizer.scale_canvas(canvas, scale=2.0).shape
            torch.Size([1, 3, 8, 12])

        """
        return F.interpolate(
            canvas,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )

    @cached_property
    def colormap(self) -> ColorMap:
        """A ``ColorMap`` that gives each label a distinct RGB color.

        The map assigns a color on the first access to a label and
        returns the same color afterwards. This property creates the map
        on its first access and caches it.

        """
        return ColorMap()

    @abstractmethod
    def forward(
        self,
        target_canvas: Tensor,
        prediction_canvas: Tensor,
        *args: Unpack[Ts],
    ) -> (
        Tensor
        | tuple[Tensor, Tensor]
        | tuple[Tensor, list[Tensor]]
        | list[Tensor]
    ):
        """Draw the labels and the predictions on the canvases.

        Implementations return one of:

        - One image, as `ClassificationVisualizer` does when
          ``include_plot`` is ``False``.
        - A tuple ``(labels, predictions)`` of two images, as
          `BBoxVisualizer` does.
        - A tuple of the labels image and a list of images.
        - A list of unrelated images.

        Args:
            target_canvas (``Tensor``): Images to draw the labels on, of
                shape ``[B, 3, H, W]``.
            prediction_canvas (``Tensor``): Images to draw the
                predictions on, of shape ``[B, 3, H, W]``.
            *args (``Unpack[Ts]``): The predictions and labels that
                `run` resolves from the parameter names of the
                implementation.

        Returns:
            ``Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]] | list[Tensor]``:
            The visualizations, in one of the four forms above.

        """
        ...

    @cached_property
    def _signature(self) -> dict[str, Parameter]:
        signature = get_signature(self.forward)
        for key in list(signature.keys()):
            if "canvas" in key:
                del signature[key]
        return signature

    # TODO: Canvases not required if remove `MultiVisualizer`
    def run(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        inputs: Packet[Tensor],
        labels: Labels | None,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]]:
        """Scale the canvases, resolve the inputs, and call `forward`.

        `BaseAttachedModule.get_parameters` documents how the remaining
        `forward` parameters select predictions and labels.

        Args:
            prediction_canvas (``Tensor``): Images to draw the
                predictions on, of shape ``[B, 3, H, W]``.
            target_canvas (``Tensor``): Images to draw the labels on, of
                shape ``[B, 3, H, W]``.
            inputs (``Packet[Tensor]``): The output packet of the node.
            labels (``Labels | None``): The labels of the batch, keyed
                ``<task_name>/<label>``, or ``None`` when the batch has
                none. Then every optional ``target`` parameter receives
                ``None``, and a required one raises ``RuntimeError``.

        Returns:
            ``Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]]``:
            What `forward` returns.

        """
        prediction_canvas = self.scale_canvas(prediction_canvas, self._scale)
        target_canvas = self.scale_canvas(target_canvas, self._scale)

        return self(
            target_canvas,
            prediction_canvas,
            **self.get_parameters(inputs, labels),
        )
