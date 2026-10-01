"""Distill a light classification model from a heavy one.

The test trains, so it runs on a machine with a GPU and the test data,
not on a laptop.

"""

from pathlib import Path

import torch
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel
from tests.conftest import LuxonisTestDataset


def classification_model(
    variant: str, opts: Params, dataset: LuxonisTestDataset
) -> Params:
    return opts | {
        "model.name": f"kd_{variant}",
        "model.predefined_model.variant": variant,
        "loader.params.dataset_name": dataset.identifier,
        "tracker.run_name": f"kd_{variant}",
    }


def test_light_student_distills_from_heavy_teacher(
    opts: Params, cifar10_dataset: LuxonisTestDataset, tmp_path: Path
):
    config = "luxonis_train/configs/classification_light_model.yaml"
    teacher = LuxonisModel(
        config, classification_model("heavy", opts, cifar10_dataset)
    )
    teacher.train()
    teacher_file = tmp_path / "teacher.ckpt"
    teacher.save_checkpoint(teacher_file)
    teacher_state = torch.load(teacher_file, map_location="cpu")["state_dict"]

    # The teacher path is the only distillation setting.
    student = LuxonisModel(
        config,
        classification_model("light", opts, cifar10_dataset)
        | {"model.teacher.weights": str(teacher_file)},
    )
    module = student.lightning_module
    module.attach_distillation(student._input_shapes)
    controller = module.distillation
    assert controller is not None
    connectors_before = {
        key: value.clone()
        for key, value in controller.connectors.state_dict().items()
    }
    student.train()

    # The automatic recipe distills the head and the backbone stage it
    # reads; ResNet18 has 512 channels and ResNet50 has 2048.
    assert set(module.nodes["ClassificationHead"].distillation) == {
        "LogitKDLoss"
    }
    assert set(module.nodes["ResNet"].distillation) == {"CWDDistillationLoss"}
    assert set(controller.connectors) == {"ResNet/CWDDistillationLoss"}

    # The teacher did not train.
    teacher_nodes = controller._teachers["teacher"].nodes
    for name, node in teacher_nodes.items():
        for key, value in node.module.state_dict().items():
            saved = teacher_state[f"nodes.{name}.module.{key}"]
            assert torch.equal(value.cpu(), saved), key

    # The connectors trained, and the teacher was released.
    assert any(
        not torch.equal(value.cpu(), connectors_before[key].cpu())
        for key, value in controller.connectors.state_dict().items()
    )
    assert not controller.active

    # Distillation runs in training steps only.
    logged = set(student.pl_trainer.callback_metrics)
    assert any(
        key.startswith("train/loss/") and "LogitKD" in key for key in logged
    )
    assert not any(
        key.startswith("val/") and "LogitKD" in key for key in logged
    )

    # The checkpoint holds the connectors but no teacher weights.
    checkpoint = student.get_min_loss_checkpoint_path()
    assert checkpoint is not None
    state = torch.load(checkpoint, map_location="cpu")["state_dict"]
    assert any(key.startswith("distillation.connectors.") for key in state)
    assert not any("teacher" in key for key in state)

    # Export needs neither the teacher file nor the connectors.
    teacher_file.unlink()
    exported = LuxonisModel(weights=checkpoint)
    assert exported.export(save_path=tmp_path / "export").exists()
