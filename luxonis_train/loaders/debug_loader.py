from collections import defaultdict
from typing import Literal

import torch
from torch import Size, Tensor
from typing_extensions import override

from luxonis_train.config import Config
from luxonis_train.registry import NODES
from luxonis_train.tasks import Metadata
from luxonis_train.typing import Labels

from .base_loader import BaseLoaderTorch


class DebugLoader(BaseLoaderTorch):
    """A dummy data loader for testing purposes.

    It serves as a placeholder in place of C{LuxonisLoaderTorch} when no
    real data is available.

    It can be extended to be used instead of custom loaders as well by
    overriding the C{get_label_shapes} method.
    """

    def __init__(
        self,
        cfg: Config,
        view: list[str],
        height: int | None = None,
        width: int | None = None,
        image_source: str = "image",
        color_space: Literal["RGB", "BGR", "GRAY"] = "RGB",
        n_keypoints: int = 3,
    ):
        super().__init__(
            view=view,
            height=height,
            width=width,
            image_source=image_source,
            color_space=color_space,
        )
        self.n_keypoints = n_keypoints
        self.batch_size = cfg.trainer.batch_size
        self.labels: dict[str, set[str | Metadata]] = defaultdict(set)
        for node in cfg.model.nodes:
            Node = NODES.get(node.name)
            if Node.task is not None:
                for label in Node.task.required_labels:
                    self.labels[f"{node.task_name or ''}"].add(label)
        self.n_channels = 1 if color_space == "GRAY" else 3

    @property
    @override
    def input_shapes(self) -> dict[str, Size]:
        return {
            self.image_source: Size(
                [
                    self.n_channels,
                    self.height,
                    self.width,
                ]
            )
        }

    @override
    def __len__(self) -> int:
        return self.batch_size * 10

    @override
    def get(self, idx: int) -> tuple[Tensor | dict[str, Tensor], Labels]:
        img = torch.zeros(self.n_channels, self.height, self.width)
        label_shapes = self.get_label_shapes(self.labels)
        labels = {
            f"{task_name}/{task_type}": torch.zeros(
                label_shapes[f"{task_name}/{task_type}"]
            )
            for task_name, task_types in self.labels.items()
            for task_type in task_types
        }
        return img, labels

    @override
    def get_classes(self) -> dict[str, dict[str, int]]:
        return {task_name: {"x": 0} for task_name in self.labels}

    @override
    def get_n_keypoints(self) -> dict[str, int] | None:
        return dict.fromkeys(self.labels, self.n_keypoints)

    def get_label_shapes(
        self, labels: dict[str, set[str | Metadata]]
    ) -> dict[str, tuple[int, ...]]:
        """Creates a dictionary with shape information for each label
        based on the task type.

        Handles all LDF-native labels by default, but needs to be
        extended for custom loaders.
        """
        shapes = {}
        for task_name, task_types in labels.items():
            for task_type in task_types:
                name = f"{task_name}/{task_type}"
                match task_type:
                    case "boundingbox":
                        shapes[name] = (1, 5)
                    case "keypoints":
                        shapes[name] = (1, self.n_keypoints * 3)
                    case "segmentation" | "instance_segmentation":
                        shapes[name] = (1, self.height, self.width)
                    case _:
                        shapes[name] = (2,)

        return shapes
