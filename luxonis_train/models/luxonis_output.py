from dataclasses import dataclass, field
from pprint import pformat

from torch import Tensor

from luxonis_train.typing import Packet
from luxonis_train.utils import to_shape_packet


@dataclass
class LuxonisOutput:
    outputs: dict[str, Packet[Tensor]]
    losses: dict[str, dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]]
    visualizations: dict[str, dict[str, Tensor]] = field(default_factory=dict)
    metrics: dict[str, dict[str, Tensor]] = field(default_factory=dict)

    def __str__(self) -> str:
        outputs = {
            node_name: to_shape_packet(packet)
            for node_name, packet in self.outputs.items()
        }
        viz = {
            f"{node_name}.{viz_name}": viz_value.shape
            for node_name, viz in self.visualizations.items()
            for viz_name, viz_value in viz.items()
        }
        string = pformat(
            {"outputs": outputs, "visualizations": viz, "losses": self.losses}
        )
        return f"{self.__class__.__name__}(\n{string}\n)"

    def __repr__(self) -> str:
        return str(self)
