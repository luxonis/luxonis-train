"""The result of one forward pass of the node graph."""

from dataclasses import dataclass, field
from pprint import pformat

from torch import Tensor

from luxonis_train.typing import Packet
from luxonis_train.utils import to_shape_packet


@dataclass
class LuxonisOutput:
    """The result of one forward pass.

    `LuxonisLightningModule.full_forward` creates it. The node name is
    the key of each dictionary. The string form shows the shapes of the
    outputs and of the visualizations, and the values of the losses. It
    does not show the metrics.

    Attributes:
        outputs (``dict[str, Packet[Tensor]]``): The output packet of each
            output node.
        losses (``dict[str, dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]]``):
            The value of each loss of a node, keyed by loss name. A
            value is a tensor, or a tuple of the tensor and its
            sub-losses.
        visualizations (``dict[str, dict[str, Tensor]]``): The image
            batch of each visualizer of a node, keyed by visualizer
            name.
        metrics (``dict[str, dict[str, Tensor]]``): The metric values of
            each node, keyed by metric name.
            `LuxonisLightningModule.full_forward` leaves it empty.

    Example:
        >>> import torch
        >>> from luxonis_train.lightning import LuxonisOutput
        >>> output = LuxonisOutput(
        ...     outputs={"head": {"boxes": torch.zeros(2, 4)}}, losses={}
        ... )
        >>> print(output)
        LuxonisOutput(
        {'losses': {},
         'outputs': {'head': {'boxes': torch.Size([2, 4])}},
         'visualizations': {}}
        )

    """

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
