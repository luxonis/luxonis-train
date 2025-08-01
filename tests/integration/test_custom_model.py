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

        labels = {
            "/segmentation": torch.zeros(1, 224, 224, dtype=torch.float32)
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

    def forward(self, inputs: list[Tensor]):
        return [self.scalar * inp for inp in inputs]

    def unwrap(self, inputs: list[dict[str, list[Tensor]]]):
        return [item for inp in inputs for key in inp for item in inp[key]]


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

    def unwrap(self, inputs: list[Packet[Tensor]]) -> Tensor:
        assert len(inputs) == 1
        return inputs[0]["features"][-1]

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class CustomSegHead2(BaseHead):
    task = Tasks.SEGMENTATION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def unwrap(self, inputs: list[Packet[Tensor]]):
        return [packet["features"][-1] for packet in inputs]

    def forward(self, inputs: list[Tensor]) -> Tensor:
        fn1, _, disp = inputs
        return self.conv(fn1 + disp)

    def get_custom_head_config(self) -> Params:
        return {"custom_param": "value"}


def test_custom_model(opts: Params, tempdir: Path, subtests: SubTests):
    cfg = "tests/configs/multi_input.yaml"
    model = LuxonisModel(cfg, opts)
    with subtests.test("train"):
        model.train()

    with subtests.test("test"):
        model.test(view="val")

    with subtests.test("export"):
        model.export()
        assert (
            model.run_save_dir / "export" / "example_multi_input.onnx"
        ).exists()

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
        model.infer(view="val", save_dir=tempdir)
        assert (
            len(list(tempdir.glob("*.png")))
            == len(model.pytorch_loaders["val"].dataset) * 2  # type: ignore
        )


def sort_by_name(config: dict, keys: list[str]) -> None:
    for key in keys:
        if key in config["model"]:
            config["model"][key] = sorted(
                config["model"][key], key=lambda x: x["name"]
            )
