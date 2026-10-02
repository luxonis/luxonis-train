from typing import Any

import pytest
import torch
from torch import Tensor

from luxonis_train.attached_modules.losses import BaseDistillationLoss
from luxonis_train.config import Config, NodeConfig
from luxonis_train.config.config import PreprocessingConfig
from luxonis_train.distillation.teacher import InputAdapter
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet


class FeatureLoss(BaseDistillationLoss, register=False):
    def forward(self, features: Tensor, teacher_features: Tensor) -> Tensor:
        return (features - teacher_features).abs().mean()


class MainOutputLoss(BaseDistillationLoss, register=False):
    supported_tasks = [Tasks.CLASSIFICATION]

    def forward(self, predictions: Tensor, teacher: Tensor) -> Tensor:
        return (predictions - teacher).abs().mean()


def test_teacher_parameters_read_the_teacher_packet():
    student: Packet[Tensor] = {"features": torch.zeros(2)}
    teacher: Packet[Tensor] = {"features": torch.ones(2)}
    kwargs = FeatureLoss().get_parameters(student, {}, teacher)
    # Teacher tensors carry no gradient, so they are not copied.
    assert kwargs["teacher_features"] is teacher["features"]
    assert kwargs["features"] is not student["features"]
    loss = FeatureLoss().run(student, {}, teacher)
    assert isinstance(loss, Tensor)
    assert loss.item() == 1


def test_bare_teacher_parameter_reads_the_main_output():
    packet: Packet[Tensor] = {"classification": torch.ones(2, 3)}
    kwargs = MainOutputLoss().get_parameters(packet, {}, packet)
    assert kwargs["teacher"] is packet["classification"]


def test_missing_teacher_output_names_the_teacher():
    student: Packet[Tensor] = {"features": torch.zeros(2)}
    with pytest.raises(RuntimeError, match="teacher outputs"):
        FeatureLoss().get_parameters(student, {}, {})


@pytest.mark.parametrize(
    ("value", "expected"),
    [(False, "off"), (None, "off"), (True, "auto"), ("auto", "auto")],
)
def test_node_distillation_reads_yaml_booleans(value: object, expected: str):
    node = NodeConfig.model_validate({"name": "ResNet", "distillation": value})
    assert node.distillation == expected


def config(nodes: list[dict[str, Any]], **model: Any) -> Config:
    return Config.model_validate(
        {"rich_logging": False, "model": {"nodes": nodes, **model}}
    )


def config_with_two_kd_losses() -> Config:
    return config(
        [
            {"name": "ResNet"},
            {
                "name": "ClassificationHead",
                "inputs": ["ResNet"],
                "losses": [{"name": "CrossEntropyLoss", "alias": "kd"}],
                "distillation": [{"name": "LogitKDLoss", "alias": "kd"}],
            },
        ],
        teacher={"weights": "does/not/exist.ckpt"},
    )


def test_distillation_names_are_unique_together_with_losses():
    head = config_with_two_kd_losses().model.nodes[1]
    names = [
        loss.identifier for loss in [*head.losses, *head.distillation_losses]
    ]
    assert len(set(names)) == 2


def test_config_round_trip_keeps_the_distillation_fields():
    cfg = config_with_two_kd_losses()
    reloaded = Config.model_validate(cfg.model_dump())
    assert reloaded.model.teacher == cfg.model.teacher
    assert reloaded.model.nodes == cfg.model.nodes


def test_distillation_alias_rejects_slash():
    node = {
        "name": "ResNet",
        "distillation": [{"name": "CWDDistillationLoss", "alias": "a/b"}],
    }
    with pytest.raises(ValueError, match="'/'"):
        config([node])


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
