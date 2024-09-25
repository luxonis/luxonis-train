import torch
from torch import Tensor, nn

from luxonis_train.enums import TaskType
from luxonis_train.loaders import BaseLoaderTorch
from luxonis_train.nodes import BaseNode
from luxonis_train.utils import Packet


class CustomMultiInputLoader(BaseLoaderTorch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def input_shapes(self):
        return {
            "left": torch.Size([3, 224, 224]),
            "right": torch.Size([3, 224, 224]),
            "disparity": torch.Size([1, 224, 224]),
            "pointcloud": torch.Size([1000, 3]),
        }

    def __getitem__(self, _):  # pragma: no cover
        # Fake data
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

        # Fake labels
        segmap = torch.zeros(1, 224, 224, dtype=torch.float32)
        segmap[0, 100:150, 100:150] = 1
        labels = {
            "segmentation": (segmap, TaskType.SEGMENTATION),
        }

        return inputs, labels

    def __len__(self):
        return 10

    def get_classes(self) -> dict[TaskType, list[str]]:
        return {TaskType.SEGMENTATION: ["square"]}


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
    tasks = {TaskType.SEGMENTATION: "segmentation"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def unwrap(self, inputs: list[Packet[Tensor]]) -> Tensor:
        assert len(inputs) == 1
        return inputs[0]["features"][-1]

    def forward(self, inputs: Tensor):
        return [self.conv(inputs)]


class CustomSegHead2(MultiInputTestBaseNode):
    tasks = {TaskType.SEGMENTATION: "segmentation"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def unwrap(self, inputs: list[Packet[Tensor]]):
        return [packet["features"][-1] for packet in inputs]

    def forward(self, inputs: list[Tensor]):
        fn1, _, disp = inputs
        x = fn1 + disp
        return [self.conv(x)]
