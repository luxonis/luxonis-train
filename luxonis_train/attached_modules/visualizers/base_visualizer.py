from abc import abstractmethod

from torch import Tensor
from typing_extensions import TypeVarTuple, Unpack

from luxonis_train.attached_modules import BaseAttachedModule
from luxonis_train.utils.registry import VISUALIZERS
from luxonis_train.utils.types import Labels, Packet

Ts = TypeVarTuple("Ts")


class BaseVisualizer(
    BaseAttachedModule[Unpack[Ts]],
    register=False,
    registry=VISUALIZERS,
):
    """A base class for all visualizers.

    This class defines the basic interface for all visualizers. It utilizes automatic
    registration of defined subclasses to the L{VISUALIZERS} registry.
    """

    @abstractmethod
    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        *args: Unpack[Ts],
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]] | list[Tensor]:
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

        @type label_canvas: Tensor
        @param label_canvas: An image to draw the labels on.
        @type prediction_canvas: Tensor
        @param prediction_canvas: An image to draw the predictions on.
        @type args: Unpack[Ts]
        @param args: Prepared inputs from the `prepare` method.

        @rtype: Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]] | list[Tensor]
        @return: Visualizations.

        @raise IncompatibleException: If the inputs are not compatible with the module.
        """
        ...

    def run(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        inputs: Packet[Tensor],
        labels: Labels,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]]:
        self.validate(inputs, labels)
        return self(label_canvas, prediction_canvas, *self.prepare(inputs, labels))
