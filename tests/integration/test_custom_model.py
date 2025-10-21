import json
import tarfile
from pathlib import Path

import torch
from luxonis_ml.typing import Params
from pytest_subtests import SubTests
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.core import LuxonisModel
from luxonis_train.loaders import BaseLoaderTorch, LuxonisLoaderTorchOutput
from luxonis_train.nodes import BaseNode
from luxonis_train.nodes.heads.base_head import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet


class CustomMultiInputLoader(BaseLoaderTorch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_points = 100

    @property
    def input_shapes(self):
        return {
            "left": torch.Size([3, self.height, self.width]),
            "right": torch.Size([3, self.height, self.width]),
            "disparity": torch.Size([1, self.height, self.width]),
            "pointcloud": torch.Size([self.n_points, 3]),
        }

    def get(self, _: int) -> LuxonisLoaderTorchOutput:
        left = torch.rand(3, self.height, self.width, dtype=torch.float32)
        right = torch.rand(3, self.height, self.width, dtype=torch.float32)
        disparity = torch.rand(1, self.height, self.width, dtype=torch.float32)
        pointcloud = torch.rand(self.n_points, 3, dtype=torch.float32)
        inputs = {
            "left": left,
            "right": right,
            "disparity": disparity,
            "pointcloud": pointcloud,
        }

        labels = {
            "/segmentation": torch.zeros(
                1, self.height, self.width, dtype=torch.float32
            )
        }

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

    def forward(self, inputs: list[Packet[Tensor]]) -> list[Tensor]:
        return [
            item * self.scalar
            for inp in inputs
            for key in inp
            for item in inp[key]
        ]


class FullBackbone(MultiInputTestBaseNode): ...


class RGBDBackbone(MultiInputTestBaseNode): ...


class PointcloudBackbone(MultiInputTestBaseNode): ...


class FusionNeck(MultiInputTestBaseNode): ...


class FusionNeck2(MultiInputTestBaseNode): ...


class CustomSegHead1(BaseHead):
    task = Tasks.SEGMENTATION
    attach_index = -1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class CustomSegHead2(BaseHead):
    task = Tasks.SEGMENTATION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, inputs: list[Packet[Tensor]]) -> Tensor:
        fn1, _, disp = [packet["features"][-1] for packet in inputs]
        return self.conv(fn1 + disp)

    def get_custom_head_config(self) -> Params:
        return {"custom_param": "value"}


def test_custom_model(opts: Params, tmp_path: Path, subtests: SubTests):
    model = LuxonisModel(get_config(), opts)
    with subtests.test("train"):
        model.train()

    with subtests.test("test"):
        model.test(view="val")

    with subtests.test("export"):
        model.export()
        assert (
            (model.run_save_dir / "export" / model.cfg.model.name)
            .with_suffix(".onnx")
            .exists()
        )

    with subtests.test("archive"):
        model.archive()

        archive_path = Path(
            model.run_save_dir, "archive", model.cfg.model.name
        ).with_suffix(".onnx.tar.xz")
        assert archive_path.exists()
        correct_archive_config = json.loads(
            Path("tests/files/custom_archive_config.json").read_text()
        )
        with tarfile.open(archive_path) as tar:
            extracted_cfg = tar.extractfile("config.json")

            assert extracted_cfg is not None, (
                "Config JSON not found in the archive."
            )
            generated_config = json.loads(extracted_cfg.read().decode())

        keys_to_sort = ["inputs", "outputs", "heads"]
        sort_by_name(generated_config, keys_to_sort)
        sort_by_name(correct_archive_config, keys_to_sort)
        assert generated_config == correct_archive_config

    with subtests.test("infer"):
        model.infer(view="val", save_dir=tmp_path)
        assert (
            len(list(tmp_path.glob("*.png")))
            == len(model.pytorch_loaders["val"].dataset) * 2  # type: ignore
        )


def sort_by_name(config: dict, keys: list[str]) -> None:
    for key in keys:
        if key in config["model"]:
            config["model"][key] = sorted(
                config["model"][key], key=lambda x: x["name"]
            )


def get_config() -> Params:
    return {
        "model": {
            "name": "custom_model",
            "nodes": [
                {"name": "FullBackbone"},
                {
                    "name": "RGBDBackbone",
                    "input_sources": ["left", "right", "disparity"],
                },
                {
                    "name": "PointcloudBackbone",
                    "input_sources": ["pointcloud"],
                },
                {
                    "name": "FusionNeck",
                    "inputs": ["RGBDBackbone", "PointcloudBackbone"],
                    "input_sources": ["disparity"],
                },
                {
                    "name": "FusionNeck2",
                    "inputs": [
                        "RGBDBackbone",
                        "PointcloudBackbone",
                        "FullBackbone",
                    ],
                },
                {
                    "name": "CustomSegHead1",
                    "inputs": ["FusionNeck"],
                    "losses": [{"name": "BCEWithLogitsLoss"}],
                    "metrics": [
                        {"name": "JaccardIndex", "is_main_metric": True}
                    ],
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
                {
                    "name": "CustomSegHead2",
                    "inputs": ["FusionNeck", "FusionNeck2"],
                    "input_sources": ["disparity"],
                    "losses": [{"name": "CrossEntropyLoss"}],
                    "metrics": [{"name": "JaccardIndex"}],
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                },
            ],
        },
        "loader": {
            "name": "CustomMultiInputLoader",
            "image_source": "left",
        },
    }
