from abc import abstractmethod
from functools import cached_property
from inspect import Parameter

import torch.nn.functional as F
from torch import Tensor
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.registry import VISUALIZERS
from luxonis_train.typing import Labels, Packet, get_signature

Ts = TypeVarTuple("Ts")


class BaseVisualizer(BaseAttachedModule, register=False, registry=VISUALIZERS):
    """A base class for all visualizers.

    This class defines the basic interface for all visualizers. It
    utilizes automatic registration of defined subclasses to the
    L{VISUALIZERS} registry.
    """

    def __init__(self, *args, scale: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale = scale

    @staticmethod
    def scale_canvas(canvas: Tensor, scale: float = 1.0) -> Tensor:
        return F.interpolate(
            canvas,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )

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
        """Forward pass of the visualizer.

        Takes an image and the prepared inputs from the `prepare` method and
        produces visualizations. Visualizations can be either:

            - A single image (I{e.g.} for classification, weight visualization).
            - A tuple of two images, representing (labels, predictions) (I{e.g.} for
              bounding boxes, keypoints).
            - A tuple of an image and a list of images,
              representing (labels, multiple visualizations) (I{e.g.} for segmentation,
              depth estimation).
            - A list of images, representing unrelated visualizations.

        @type target_canvas: Tensor
        @param target_canvas: An image to draw the labels on.
        @type prediction_canvas: Tensor
        @param prediction_canvas: An image to draw the predictions on.
        @type args: Unpack[Ts]
        @param args: Prepared inputs from the `prepare` method.

        @rtype: Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]] | list[Tensor]
        @return: Visualizations.

        @raise IncompatibleError: If the inputs are not compatible with the module.
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
        prediction_canvas = self.scale_canvas(prediction_canvas, self.scale)
        target_canvas = self.scale_canvas(target_canvas, self.scale)

        return self(
            target_canvas,
            prediction_canvas,
            **self.get_parameters(inputs, labels),
        )
