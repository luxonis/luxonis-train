from typing import Any

import pytest
import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.losses import BaseDistillationLoss
from luxonis_train.config import Config
from luxonis_train.config.config import PreprocessingConfig
from luxonis_train.lightning.distillation.teacher import InputAdapter
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet


class FeatureLoss(BaseDistillationLoss, register=False):
    def forward(self, features: Tensor, teacher_features: Tensor) -> Tensor:
        return (features - teacher_features).abs().mean()


class MainOutputLoss(BaseDistillationLoss, register=False):
    supported_tasks = [Tasks.CLASSIFICATION]

    def forward(self, predictions: Tensor, teacher: Tensor) -> Tensor:
        return (predictions - teacher).abs().mean()


class InPlaceLoss(BaseDistillationLoss, register=False):
    def forward(self, features: Tensor, teacher_features: Tensor) -> Tensor:
        teacher_features.add_(1)
        return (features - teacher_features).abs().mean()


def test_teacher_parameters_read_the_teacher_packet():
    student: Packet[Tensor] = {"features": torch.zeros(2)}
    teacher: Packet[Tensor] = {"features": torch.ones(2)}
    loss = FeatureLoss().run(student, {}, teacher)
    assert isinstance(loss, Tensor)
    assert loss.item() == 1


def test_bare_teacher_parameter_reads_the_main_output():
    student: Packet[Tensor] = {"classification": torch.zeros(2, 3)}
    teacher: Packet[Tensor] = {"classification": torch.ones(2, 3)}
    loss = MainOutputLoss().run(student, {}, teacher)
    assert isinstance(loss, Tensor)
    assert loss.item() == 1


def test_teacher_tensors_are_copies():
    # A second loss on the same teacher node must see the original.
    features = torch.ones(2)
    InPlaceLoss().run({"features": torch.zeros(2)}, {}, {"features": features})
    assert torch.equal(features, torch.ones(2))


def test_setup_names_a_missing_teacher_output():
    shapes: Packet[Size] = {"features": Size([2, 4])}
    with pytest.raises(ValueError, match="teacher output 'features'"):
        FeatureLoss().setup(shapes, {"classification": Size([2, 3])})


def test_a_config_round_trip_drops_only_the_teacher():
    cfg = Config.model_validate(
        {
            "rich_logging": False,
            "model": {
                "nodes": [
                    {"name": "ResNet", "distillation": False},
                    {
                        "name": "ClassificationHead",
                        "inputs": ["ResNet"],
                        "distillation": [
                            {
                                "name": "LogitDistillationLoss",
                                "teacher_node": "Head",
                            }
                        ],
                    },
                ],
                "teacher": {"weights": "does/not/exist.ckpt"},
            },
        }
    )
    reloaded = Config.model_validate(cfg.model_dump())
    assert reloaded.model.teacher is None
    assert reloaded.model.nodes == cfg.model.nodes


def preprocessing(**fields: Any) -> PreprocessingConfig:
    return PreprocessingConfig.model_validate(fields)


def test_input_adapter_is_identity_for_equal_preprocessing():
    adapter = InputAdapter("image", preprocessing(), preprocessing())
    inputs = {"image": torch.randn(1, 3, 4, 4)}
    assert adapter(inputs) is inputs


def test_input_adapter_renormalizes_and_flips_channels():
    student = preprocessing()
    teacher = preprocessing(color_space="BGR", normalize={"active": False})
    adapter = InputAdapter("image", student, teacher)
    pixels = torch.rand(1, 3, 4, 4) * 255
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (pixels / 255 - mean) / std

    converted = adapter({"image": normalized, "depth": torch.zeros(1)})

    assert torch.allclose(converted["image"], pixels.flip(1), atol=1e-3)
    assert torch.equal(converted["depth"], torch.zeros(1))


def test_input_adapter_rejects_gray_against_color():
    with pytest.raises(ValueError, match="GRAY"):
        InputAdapter(
            "image", preprocessing(color_space="GRAY"), preprocessing()
        )
