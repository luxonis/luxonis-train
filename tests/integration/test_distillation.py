"""Distill a light classification model from a heavy one.

The test trains, so it runs on a machine with a GPU, not on a laptop.
Its dataset comes from the public CIFAR-10 download of ``torchvision``,
so it needs no cloud credentials.

"""

from pathlib import Path

import pytest
import torch
import torchvision
from luxonis_ml.data import DatasetIterator
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel
from luxonis_train.nodes.blocks import ConvBlock
from tests.conftest import LuxonisTestDataset

N_IMAGES = 40


@pytest.fixture(scope="module")
def cifar10_subset(data_dir: Path) -> LuxonisTestDataset:
    root = data_dir / "cifar10_public"
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    cifar = torchvision.datasets.CIFAR10(root=root, train=False, download=True)

    def generator() -> DatasetIterator:
        for i in range(N_IMAGES):
            image, label = cifar[i]
            path = images / f"cifar_{i}.png"
            image.save(path)
            yield {"file": path, "annotation": {"class": cifar.classes[label]}}

    dataset = LuxonisTestDataset(
        "cifar10_kd_test", delete_local=True, source_path=images
    )
    dataset.add(generator())
    dataset.make_splits()
    return dataset


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
    opts: Params, cifar10_subset: LuxonisTestDataset, tmp_path: Path
):
    config = "luxonis_train/configs/classification_light_model.yaml"
    teacher = LuxonisModel(
        config, classification_model("heavy", opts, cifar10_subset)
    )
    teacher.train()
    teacher_file = tmp_path / "teacher.ckpt"
    teacher.save_checkpoint(teacher_file)

    # The teacher path is the only distillation setting.
    student = LuxonisModel(
        config,
        classification_model("light", opts, cifar10_subset)
        | {"model.teacher.weights": str(teacher_file)},
    )
    module = student.lightning_module
    student.train()

    # The automatic recipe distills the head and the backbone stage it
    # reads; ResNet18 has 512 channels and ResNet50 has 2048.
    distiller = module.distiller
    assert distiller is not None
    assert set(distiller.losses("ClassificationHead")) == {
        "LogitDistillationLoss"
    }
    assert set(distiller.losses("ResNet")) == {"ChannelWiseDistillationLoss"}
    assert set(distiller.connectors) == {"ResNet/ChannelWiseDistillationLoss"}

    # The adapter ran in training steps, and the teacher was released.
    adapters = distiller.connectors["ResNet/ChannelWiseDistillationLoss"]
    assert isinstance(adapters, torch.nn.ModuleList)
    adapter = adapters[0]
    assert isinstance(adapter, ConvBlock)
    assert isinstance(adapter.bn, torch.nn.BatchNorm2d)
    assert adapter.bn.num_batches_tracked is not None
    assert adapter.bn.num_batches_tracked.item() > 0
    assert not distiller.has_teacher

    # Distillation runs in training steps only.
    logged = set(student.pl_trainer.callback_metrics)
    assert any(
        key.startswith("train/loss/") and "LogitDistillation" in key
        for key in logged
    )
    assert not any(
        key.startswith("val/") and "LogitDistillation" in key for key in logged
    )

    # The checkpoint holds the connectors, but no teacher weights and no
    # teacher path.
    checkpoint = student.get_min_loss_checkpoint_path()
    assert checkpoint is not None
    saved = torch.load(checkpoint, map_location="cpu")
    state = saved["state_dict"]
    assert any(key.startswith("distiller.connectors.") for key in state)
    assert not any("teacher" in key for key in state)
    assert "teacher" not in saved["config"]["model"]

    # Export needs neither the teacher file nor the connectors.
    teacher_file.unlink()
    exported = LuxonisModel(weights=checkpoint)
    assert exported.export(save_path=tmp_path / "export").exists()
