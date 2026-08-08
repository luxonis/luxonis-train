"""Runtime contract cases for `luxonis_train.nodes.base_node`.

`BaseNode.forward` is abstract and its contract is deliberately generic:
a single tensor argument named `inputs` with an opaque node-specific
shape, and a single tensor output with an opaque node-specific shape.
The contract can therefore only be executed through a concrete
subclass. The two minimal nodes below exercise both readings of the
contract: one that changes the tensor shape and one that preserves it.
"""

from collections.abc import Callable

import torch
from torch import Size, Tensor, nn

from luxonis_train.nodes.base_node import BaseNode

from .case import ContractCase


class ProjectionNode(BaseNode, register=False):
    """Concrete node projecting the input onto other channels."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        """Initialize the node.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            **kwargs: Forwarded to `BaseNode`.

        """
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Project the input features."""
        return self.conv(inputs)


class ScalingNode(BaseNode, register=False):
    """Concrete node preserving the shape of its input."""

    def forward(self, inputs: Tensor) -> Tensor:
        """Scale the input features."""
        return inputs * 2.0


def _base_node_cases() -> list[ContractCase]:
    torch.manual_seed(0)

    projection = ProjectionNode(
        4,
        6,
        input_shapes=[{"features": [Size((4, 8, 8))]}],
        original_in_shape=Size((3, 8, 8)),
    )
    projection.eval()

    scaling = ScalingNode(
        input_shapes=[{"features": [Size((5, 16, 12))]}],
        original_in_shape=Size((3, 16, 12)),
    )
    scaling.eval()

    return [
        ContractCase(
            module=projection,
            inputs={"inputs": torch.randn(2, 4, 8, 8)},
            bindings={
                "input_shape": (2, 4, 8, 8),
                "output_shape": (2, 6, 8, 8),
            },
        ),
        ContractCase(
            module=scaling,
            inputs={"inputs": torch.randn(1, 5, 16, 12)},
            bindings={
                "input_shape": (1, 5, 16, 12),
                "output_shape": (1, 5, 16, 12),
            },
        ),
    ]


CASES: dict[
    type[nn.Module], Callable[[], ContractCase | list[ContractCase]]
] = {
    BaseNode: _base_node_cases,
}

UNSUPPORTED: dict[type[nn.Module], str] = {}
