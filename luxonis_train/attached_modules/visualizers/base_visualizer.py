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

    A subclass implements `forward`. The name of each parameter that
    does not contain ``canvas`` selects the data it receives:

    - ``predictions``, or another name that starts with ``pred`` and
      has no underscore, selects the main output of the task of the
      visualizer.
    - ``pred_<key>`` selects the key ``<key>`` of the node packet.
    - ``target``, or another name that starts with ``target`` and has
      no underscore, selects the single label the task requires. When
      the task requires several labels, this raises ``RuntimeError``.
      ``target_<label>`` selects the label ``<label>``. Both look the
      label up as ``<task_name>/<label>``, with the ``task_name`` of
      the node.
    - Any other name selects the packet key of that name.
    - A parameter annotated with ``| None`` receives ``None`` when the
      data is not available, even when it has a default value. A
      parameter without ``| None`` but with a default value keeps the
      default. A parameter with neither raises ``RuntimeError``.

    `run` resolves these parameters with
    `BaseAttachedModule.get_parameters` before it calls `forward`.

    """

    def __init__(self, *args, scale: float = 1.0, **kwargs) -> None:
        """Initialize the visualizer and store the canvas scale.

        Args:
            *args (``Any``): Positional arguments forwarded to
                `BaseAttachedModule`. It accepts none, so any value
                raises ``TypeError``.
            scale (float): Factor that `run` applies to both canvases
                with `scale_canvas` before it calls `forward`. The
                visualizer stores the value as ``self.scale``, so a
                subclass can scale the pixel coordinates of its
                predictions the same way. Defaults to ``1.0``.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseAttachedModule`, such as ``node``.

        """
        super().__init__(*args, **kwargs)
        self.scale = scale

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

        An implementation receives the two canvases and the data that
        `run` resolves from the names of its remaining parameters. It
        returns one of:

        - One image, as `ClassificationVisualizer` does when
          ``include_plot`` is ``False``.
        - A tuple ``(labels, predictions)`` of two images, as
          `BBoxVisualizer` does.
        - A tuple of the labels image and a list of images.
        - A list of unrelated images.

        `combine_visualizations` accepts the first two forms only. It
        resizes a pair to the larger height, with the aspect ratios
        kept, and puts it side by side, with the labels on the left.
        It raises ``NotImplementedError`` for the third form. It
        treats a list of exactly two images as a pair, and raises
        ``ValueError`` for a list of any other length.

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
        """The `forward` parameters that `run` must resolve.

        `get_signature` drops ``self`` and ``kwargs``. This property
        also drops every parameter whose name contains ``canvas``.

        """
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

        The trainer calls this method with the same batch of images for
        both canvases. `scale_canvas` resizes both by ``self.scale``.
        Then `BaseAttachedModule.get_parameters` picks the predictions
        and labels that the `forward` parameters name. It clones every
        tensor it picks, so `forward` cannot change ``inputs`` or
        ``labels``. It raises ``TypeError`` when a value does not match
        the annotation of its parameter. Finally `forward` runs with
        ``target_canvas`` as the first positional argument,
        ``prediction_canvas`` as the second, and the resolved data as
        keyword arguments.

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
        prediction_canvas = self.scale_canvas(prediction_canvas, self.scale)
        target_canvas = self.scale_canvas(target_canvas, self.scale)

        return self(
            target_canvas,
            prediction_canvas,
            **self.get_parameters(inputs, labels),
        )
