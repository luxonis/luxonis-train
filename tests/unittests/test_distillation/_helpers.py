from pathlib import Path
from typing import Any

import torch
from torch import Size, Tensor, nn

import luxonis_train
from luxonis_train.config import Config
from luxonis_train.lightning.utils import Nodes
from luxonis_train.nodes import BaseNode
from luxonis_train.nodes.heads.base_head import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import DatasetMetadata

INPUT_SHAPES = {"image": Size([3, 32, 32])}
CLASSES = {"": {"cat": 0, "dog": 1}}


class KDBackbone(BaseNode):
    def __init__(self, width: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 2, 1), nn.BatchNorm2d(width), nn.ReLU()
        )
        self.stage = nn.Conv2d(width, 2 * width, 3, 2, 1)

    def forward(self, x: Tensor) -> list[Tensor]:
        first = self.stem(x)
        return [first, self.stage(first)]


class KDHead(BaseHead):
    task = Tasks.CLASSIFICATION
    attach_index = -1
    distillation_loss = {"name": "LogitKDLoss", "params": {"temperature": 4.0}}
    in_channels: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fc = nn.Linear(self.in_channels, self.n_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x.mean((2, 3)))


def make_config(
    width: int,
    *,
    teacher: Path | None = None,
    backbone: dict[str, Any] | None = None,
    head: dict[str, Any] | None = None,
    normalize: bool = True,
) -> Config:
    model: dict[str, Any] = {
        "name": "kd",
        "nodes": [
            {"name": "KDBackbone", "params": {"width": width}}
            | (backbone or {}),
            {
                "name": "KDHead",
                "inputs": ["KDBackbone"],
                "losses": [{"name": "CrossEntropyLoss"}],
            }
            | (head or {}),
        ],
    }
    if teacher is not None:
        model["teacher"] = {"weights": str(teacher)}
    return Config.model_validate(
        {
            "rich_logging": False,
            "model": model,
            "trainer": {
                "preprocessing": {
                    "train_image_size": [32, 32],
                    "normalize": {"active": normalize},
                }
            },
        }
    )


def save_teacher(
    path: Path, *, width: int = 8, classes: dict | None = None
) -> tuple[Path, Nodes]:
    """Save a checkpoint in the layout of a luxonis-train checkpoint."""
    cfg = make_config(width)
    metadata = DatasetMetadata(classes=classes or CLASSES)
    nodes = Nodes(cfg, metadata, INPUT_SHAPES)
    state_dict = {
        f"nodes.{name}.module.{key}": value
        for name, node in nodes.items()
        for key, value in node.module.state_dict().items()
    }
    ckpt = {
        "config": cfg.model_dump(),
        "state_dict": state_dict,
        "dataset_metadata": metadata.dump(),
        "version": luxonis_train.__version__,
    }
    file = path / "teacher.ckpt"
    torch.save(ckpt, file)
    return file, nodes


def student_nodes(cfg: Config) -> Nodes:
    return Nodes(cfg, DatasetMetadata(classes=CLASSES), INPUT_SHAPES)
