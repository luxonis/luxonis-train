import os
import shutil
from pathlib import Path
from typing import Annotated

import pytest
import torch
from pydantic import Field
from torch import Tensor
from torch.nn.parameter import Parameter

from luxonis_train.core import Exporter, Inferer, Trainer
from luxonis_train.nodes import BaseNode
from luxonis_train.utils.loaders import BaseLoaderTorch
from luxonis_train.utils.registry import LOADERS
from luxonis_train.utils.types import BaseProtocol, FeaturesProtocol, LabelType

LOADERS.register_module()


class CustomMultiInputLoader(BaseLoaderTorch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def input_shape(self):
        return {
            "left": torch.Size([3, 224, 224]),
            "right": torch.Size([3, 224, 224]),
            "disparity": torch.Size([1, 224, 224]),
            "pointcloud": torch.Size([1000, 3]),
        }

    def __getitem__(self, idx):
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
        labels = {
            "default": {
                LabelType.SEGMENTATION: segmap,
            }
        }

        return inputs, labels

    def __len__(self):
        return 10

    def get_classes(self) -> dict[LabelType, list[str]]:
        return {LabelType.SEGMENTATION: ["square"]}


class FullCustomMultiInputProtocol(BaseProtocol):
    left: Annotated[list[Tensor], Field(min_length=1)]
    right: Annotated[list[Tensor], Field(min_length=1)]
    disparity: Annotated[list[Tensor], Field(min_length=1)]
    pointcloud: Annotated[list[Tensor], Field(min_length=1)]


class RGBDCustomMultiInputProtocol(BaseProtocol):
    left: Annotated[list[Tensor], Field(min_length=1)]
    right: Annotated[list[Tensor], Field(min_length=1)]
    disparity: Annotated[list[Tensor], Field(min_length=1)]


class PointcloudCustomMultiInputProtocol(BaseProtocol):
    pointcloud: Annotated[list[Tensor], Field(min_length=1)]


class DisparityCustomMultiInputProtocol(BaseProtocol):
    disparity: Annotated[list[Tensor], Field(min_length=1)]


class MultiInputTestBaseNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scalar = Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, inputs):
        return [self.scalar * inp for inp in inputs]

    def unwrap(self, inputs: list[dict[str, list[Tensor]]]):
        return [item for inp in inputs for key in inp for item in inp[key]]


class FullBackbone(MultiInputTestBaseNode):
    def __init__(self, **kwargs):
        in_protocols = [FullCustomMultiInputProtocol]
        super().__init__(**kwargs)
        self.in_protocols = in_protocols


class RGBDBackbone(MultiInputTestBaseNode):
    def __init__(self, **kwargs):
        in_protocols = [RGBDCustomMultiInputProtocol]
        super().__init__(**kwargs)
        self.in_protocols = in_protocols


class PointcloudBackbone(MultiInputTestBaseNode):
    def __init__(self, **kwargs):
        in_protocols = [PointcloudCustomMultiInputProtocol]
        super().__init__(**kwargs)
        self.in_protocols = in_protocols


class FusionNeck(MultiInputTestBaseNode):
    def __init__(self, **kwargs):
        in_protocols = [
            DisparityCustomMultiInputProtocol,
            FeaturesProtocol,
            FeaturesProtocol,
        ]
        super().__init__(**kwargs)
        self.in_protocols = in_protocols


class FusionNeck2(MultiInputTestBaseNode):
    def __init__(self, **kwargs):
        in_protocols = [FeaturesProtocol, FeaturesProtocol, FeaturesProtocol]
        super().__init__(**kwargs)
        self.in_protocols = in_protocols


class CustomSegHead1(MultiInputTestBaseNode):
    def __init__(self, **kwargs):
        in_protocols = [FeaturesProtocol]
        super().__init__(**kwargs)
        self.in_protocols = in_protocols

    def wrap(self, outputs: list[Tensor]):
        return {"segmentation": outputs}


class CustomSegHead2(MultiInputTestBaseNode):
    def __init__(self, **kwargs):
        in_protocols = [
            DisparityCustomMultiInputProtocol,
            FeaturesProtocol,
            FeaturesProtocol,
        ]
        super().__init__(**kwargs)
        self.in_protocols = in_protocols

    def wrap(self, outputs: list[Tensor]):
        return {"segmentation": outputs}


@pytest.fixture(scope="function", autouse=True)
def clear_output():
    shutil.rmtree("output", ignore_errors=True)


@pytest.mark.parametrize(
    "config_file", [path for path in os.listdir("configs") if "multi_input" in path]
)
def test_sanity(config_file):
    # Test training
    trainer = Trainer(f"configs/{config_file}")
    trainer.train()
    # Test evaluation
    trainer.test(view="val")

    # Test export
    Exporter(f"configs/{config_file}").export("test_export_multi_input.onnx")
    # Cleanup after exporter
    assert os.path.exists("test_export_multi_input.onnx")
    os.remove("test_export_multi_input.onnx")

    # Test inference
    Inferer(
        f"configs/{config_file}",
        opts=None,
        view="train",
        save_dir=Path("infer_save_dir"),
    ).infer()
    # Cleanup after inferer
    assert os.path.exists("infer_save_dir")
    shutil.rmtree("infer_save_dir")
