import torch
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.loaders import BaseLoaderTorch, LuxonisLoaderTorchOutput
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet


class CustomMultiInputLoader(BaseLoaderTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._height = 224
        self._width = 224

    @property
    def input_shapes(self):
        return {
            "left": torch.Size([3, 224, 224]),
            "right": torch.Size([3, 224, 224]),
            "disparity": torch.Size([1, 224, 224]),
            "pointcloud": torch.Size([1000, 3]),
        }

    def get(self, _: int) -> LuxonisLoaderTorchOutput:
        left = torch.rand(3, 224, 224, dtype=torch.float32)
        right = torch.rand(3, 224, 224, dtype=torch.float32)
        disparity = torch.rand(1, 224, 224, dtype=torch.float32)
        pointcloud = torch.rand(1000, 3, dtype=torch.float32)
        inputs = {
            "left": left,
            "right": right,
            "disparity": disparity,
            "pointcloud": pointcloud,
        }

        segmap = torch.zeros(1, 224, 224, dtype=torch.float32)
        segmap[0, 100:150, 100:150] = 1
        labels = {"/segmentation": segmap}

        return inputs, labels

    def __len__(self):
        return 10

    @override
    def get_classes(self) -> dict[str, dict[str, int]]:
        return {"": {"square": 0}}


class MultiInputTestBaseNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scalar = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, inputs: list[Tensor]):
        return [self.scalar * inp for inp in inputs]

    def unwrap(self, inputs: list[dict[str, list[Tensor]]]):
        return [item for inp in inputs for key in inp for item in inp[key]]


class FullBackbone(MultiInputTestBaseNode): ...


class RGBDBackbone(MultiInputTestBaseNode): ...


class PointcloudBackbone(MultiInputTestBaseNode): ...


class FusionNeck(MultiInputTestBaseNode): ...


class FusionNeck2(MultiInputTestBaseNode): ...


class CustomSegHead1(MultiInputTestBaseNode):
    task = Tasks.SEGMENTATION
    attach_index = -1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def unwrap(self, inputs: list[Packet[Tensor]]) -> Tensor:
        assert len(inputs) == 1
        return inputs[0]["features"][-1]

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class CustomSegHead2(MultiInputTestBaseNode):
    task = Tasks.SEGMENTATION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def unwrap(self, inputs: list[Packet[Tensor]]):
        return [packet["features"][-1] for packet in inputs]

    def forward(self, inputs: list[Tensor]) -> Tensor:
        fn1, _, disp = inputs
        return self.conv(fn1 + disp)
