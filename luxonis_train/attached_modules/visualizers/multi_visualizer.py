from torch import Tensor

from luxonis_train.utils.registry import VISUALIZERS
from luxonis_train.utils.types import (
    Kwargs,
    Labels,
    Packet,
)

from .base_visualizer import BaseVisualizer


class MultiVisualizer(BaseVisualizer[Packet[Tensor], Labels]):
    """Special type of visualizer that combines multiple visualizers together.

    All the visualizers are applied in the order they are provided and they all draw on
    the same canvas.

    @type visualizers: list[Kwargs]
    @param visualizers: List of visualizers to combine.
        Each item in the list is a dictionary with the following keys::

            >>> {"name": "name_of_the_visualizer",
                 "params": {"param1": value1, "param2": value2, ...}}
    """

    def __init__(self, visualizers: list[Kwargs], **kwargs):
        super().__init__(**kwargs)
        self.visualizers = []
        for item in visualizers:
            visualizer_params = item.get("params", {})
            visualizer = VISUALIZERS.get(item["name"])(**visualizer_params, **kwargs)
            self.visualizers.append(visualizer)

    def prepare(
        self, output: Packet[Tensor], label: Labels, idx: int = 0
    ) -> tuple[Packet[Tensor], Labels]:
        self._idx = idx
        return output, label

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        outputs: Packet[Tensor],
        labels: Labels,
    ) -> tuple[Tensor, Tensor]:
        for visualizer in self.visualizers:
            match visualizer.run(label_canvas, prediction_canvas, outputs, labels):
                case Tensor(data=prediction_viz):
                    prediction_canvas = prediction_viz
                case (Tensor(data=label_viz), Tensor(data=prediction_viz)):
                    label_canvas = label_viz
                    prediction_canvas = prediction_viz
                case _:
                    raise NotImplementedError
        return label_canvas, prediction_canvas
